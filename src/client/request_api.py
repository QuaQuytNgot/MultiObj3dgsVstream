"""Stable networking contracts and facade; consumer modules need no curl types.

Only constructing NetworkClient imports the low-level transport implementation.
The contracts, samples and interface can be used with a different transport.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit
from uuid import uuid4

__all__ = [
    "BandwidthSample", "CompletionCallback", "ErrorKind", "NetworkClient",
    "ProtocolUnavailableError", "RequestSpec", "TransferClient", "TransferError",
    "TransferResult", "to_bandwidth_sample",
]

logger = logging.getLogger(__name__)


class ProtocolUnavailableError(RuntimeError):
    """The selected transport backend cannot use the requested HTTP protocol."""


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """A planner's resolved GET request. Range endpoints are inclusive."""

    request_id: str
    url: str
    object_id: str | None = None
    representation_id: str | None = None
    segment_id: str | int | None = None
    layer_id: str | int | None = None
    byte_range: tuple[int, int] | None = None
    priority: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(self.url, str) or any(c.isspace() for c in self.url):
            raise ValueError("url must be an absolute HTTP(S) URL without whitespace")
        try:
            parsed = urlsplit(self.url)
            valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
            parsed.port  # Also validate a malformed or out-of-bounds port.
        except ValueError as exc:
            raise ValueError("url must be a valid absolute HTTP(S) URL") from exc
        if not valid:
            raise ValueError("url must be an absolute HTTP(S) URL")
        for name in ("object_id", "representation_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string or None")
        for name in ("segment_id", "layer_id"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (str, int))
                or (isinstance(value, str) and not value.strip())
                or (isinstance(value, int) and value < 0)
            ):
                raise ValueError(f"{name} must be a nonempty string, nonnegative int or None")
        if self.byte_range is not None:
            value = self.byte_range
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
                or value[0] < 0
                or value[1] < value[0]
            ):
                raise ValueError("byte_range must be an inclusive (start, end) integer tuple")
        if self.priority is not None and (
            isinstance(self.priority, bool) or not isinstance(self.priority, int)
        ):
            raise ValueError("priority must be an integer or None")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        # Snapshot the outer mapping; nested context remains caller-owned.
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def range_header(self) -> str | None:
        if self.byte_range is None:
            return None
        return f"bytes={self.byte_range[0]}-{self.byte_range[1]}"


class ErrorKind(str, Enum):
    TIMEOUT = "timeout"
    HTTP = "http"
    CONNECTION = "connection"
    INCOMPLETE = "incomplete"
    RANGE_IGNORED = "range_ignored"
    INVALID_RANGE = "invalid_range"
    PROTOCOL = "protocol"
    FILE = "file"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TransferError:
    kind: ErrorKind
    message: str
    curl_code: int | None = None


@dataclass(frozen=True, slots=True)
class TransferResult:
    """Network measurements. Times are seconds; goodput values are bits/second."""

    spec: RequestSpec
    status_code: int | None
    body: bytes | None
    output_path: Path | None
    received_bytes: int
    start_time: float
    first_byte_time: float | None
    end_time: float
    ttfb: float | None
    transfer_duration: float | None
    total_duration: float
    body_goodput_bps: float | None
    effective_goodput_bps: float | None
    http_version: str | None
    response_headers: Mapping[str, str]
    success: bool
    error: TransferError | None
    range_honored: bool | None
    timestamp: float
    timing_source: str

    @property
    def request_id(self) -> str:
        return self.spec.request_id

    @property
    def url(self) -> str:
        return self.spec.url

    @property
    def object_id(self) -> str | None:
        return self.spec.object_id

    @property
    def representation_id(self) -> str | None:
        return self.spec.representation_id

    @property
    def segment_id(self) -> str | int | None:
        return self.spec.segment_id

    @property
    def layer_id(self) -> str | int | None:
        return self.spec.layer_id

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.spec.metadata

    @property
    def requested_range(self) -> tuple[int, int] | None:
        return self.spec.byte_range

    @property
    def content_range(self) -> str | None:
        return self.response_headers.get("content-range")

    @property
    def content_length(self) -> int | None:
        value = self.response_headers.get("content-length")
        if value is None or not re.fullmatch(r"[0-9]+", value):
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @property
    def accept_ranges(self) -> str | None:
        return self.response_headers.get("accept-ranges")


@dataclass(frozen=True, slots=True)
class BandwidthSample:
    """Raw transfer observations, without payload, context or estimation logic."""

    timestamp: float
    received_bytes: int
    ttfb: float | None
    transfer_duration: float | None
    total_duration: float
    body_goodput_bps: float | None
    effective_goodput_bps: float | None
    http_version: str | None
    success: bool


def to_bandwidth_sample(result: TransferResult) -> BandwidthSample:
    """Copy measurements unchanged, including failures and unavailable timings."""
    return BandwidthSample(
        timestamp=result.timestamp,
        received_bytes=result.received_bytes,
        ttfb=result.ttfb,
        transfer_duration=result.transfer_duration,
        total_duration=result.total_duration,
        body_goodput_bps=result.body_goodput_bps,
        effective_goodput_bps=result.effective_goodput_bps,
        http_version=result.http_version,
        success=result.success,
    )


@runtime_checkable
class TransferClient(Protocol):
    """Minimal memory-transfer interface for schedulers and alternate backends."""

    async def fetch(self, spec: RequestSpec) -> TransferResult:
        """Execute a resolved request; cancellation propagates to the caller."""
        ...


CompletionCallback = Callable[[TransferResult], Awaitable[None] | None]


class NetworkClient:
    """Own a reusable RequestHandler behind transport-independent public types.

    Use one client per asyncio event loop and close it with async with or aclose.
    Helpers build RequestSpec values; they do not resolve content or plan requests.
    """

    def __init__(
        self,
        *,
        http_version: str = "h2",
        concurrency: int = 1,
        connect_timeout_s: float = 5,
        total_timeout_s: float = 30,
        verify_tls: bool = True,
        on_transfer_complete: CompletionCallback | None = None,
    ) -> None:
        self._subscribers: dict[int, CompletionCallback] = {}
        self._next_subscription = 0
        self._closed = False
        if on_transfer_complete is not None:
            self.subscribe(on_transfer_complete)
        # Import here to keep contracts usable without importing curl_cffi and to
        # let the low-level implementation consume these same canonical models.
        from .request_handler import RequestHandler

        self._handler = RequestHandler(
            http_version=http_version,
            concurrency=concurrency,
            connect_timeout_s=connect_timeout_s,
            total_timeout_s=total_timeout_s,
            verify_tls=verify_tls,
            on_transfer_complete=self._dispatch_complete,
        )

    async def __aenter__(self) -> NetworkClient:
        await self._handler.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the owned transport, including pending request cleanup."""
        self._closed = True
        await self._handler.aclose()
        self._subscribers.clear()

    async def fetch(
        self, spec: RequestSpec, *, output_path: str | Path | None = None
    ) -> TransferResult:
        """Execute a scheduler-provided spec, optionally publishing a file."""
        return await self._handler.fetch(spec, output_path=output_path)

    async def get(
        self,
        url: str,
        *,
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TransferResult:
        """Fetch an already resolved resource into memory."""
        return await self.fetch(self._request(url, request_id, metadata))

    async def get_segment(
        self,
        url: str,
        *,
        object_id: str,
        representation_id: str,
        segment_id: str | int,
        request_id: str | None = None,
        layer_id: str | int | None = None,
        byte_range: tuple[int, int] | None = None,
        priority: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TransferResult:
        """Attach caller-resolved segment identity; no index lookup or scheduling."""
        return await self.fetch(RequestSpec(
            request_id=self._request_id(request_id),
            url=url,
            object_id=object_id,
            representation_id=representation_id,
            segment_id=segment_id,
            layer_id=layer_id,
            byte_range=byte_range,
            priority=priority,
            metadata={} if metadata is None else metadata,
        ))

    async def get_range(
        self,
        url: str,
        start: int,
        end: int,
        *,
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TransferResult:
        """Fetch one inclusive byte range; response validation stays in transport."""
        return await self.fetch(RequestSpec(
            request_id=self._request_id(request_id),
            url=url,
            byte_range=(start, end),
            metadata={} if metadata is None else metadata,
        ))

    async def download_to_file(
        self,
        request: RequestSpec | str,
        output_path: str | Path,
        *,
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TransferResult:
        """Stream a spec or resolved URL to an atomically published file."""
        if isinstance(request, RequestSpec):
            if request_id is not None or metadata is not None:
                raise ValueError("Set request_id and metadata on the supplied RequestSpec")
            spec = request
        else:
            spec = self._request(request, request_id, metadata)
        return await self.fetch(spec, output_path=output_path)

    def subscribe(self, callback: CompletionCallback) -> Callable[[], None]:
        """Register a sync/async result observer; return an idempotent unsubscribe.

        Each registration is independent, even for the same callable. Delivery
        uses a snapshot in registration order; mutations affect the next event.
        """
        if self._closed:
            raise RuntimeError("NetworkClient is closed")
        if not callable(callback):
            raise ValueError("Transfer subscriber must be callable")
        token = self._next_subscription
        self._next_subscription += 1
        self._subscribers[token] = callback

        def unsubscribe() -> None:
            self._subscribers.pop(token, None)

        return unsubscribe

    async def _dispatch_complete(self, result: TransferResult) -> None:
        for callback in tuple(self._subscribers.values()):
            try:
                returned = callback(result)
                if inspect.isawaitable(returned):
                    await returned
            except Exception:
                logger.exception("Transfer subscriber failed request=%s", result.request_id)

    @staticmethod
    def _request_id(request_id: str | None) -> str:
        return f"request-{uuid4().hex}" if request_id is None else request_id

    @classmethod
    def _request(
        cls, url: str, request_id: str | None, metadata: Mapping[str, Any] | None
    ) -> RequestSpec:
        return RequestSpec(
            request_id=cls._request_id(request_id),
            url=url,
            metadata={} if metadata is None else metadata,
        )

"""Async HTTP transport for resolved content requests, without planning or decoding."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO

from curl_cffi import Curl, CurlError, CurlHttpVersion, CurlInfo, CurlOpt
from curl_cffi.requests import AsyncSession, Response

# Keep the original imports compatible; contracts belong to the public layer.
from .request_api import (
    ErrorKind, ProtocolUnavailableError, RequestSpec, TransferError, TransferResult,
)

__all__ = [
    "ErrorKind", "HttpVersion", "ProtocolCapabilities", "ProtocolUnavailableError",
    "RequestHandler", "RequestSpec", "TransferError", "TransferResult",
]

logger = logging.getLogger(__name__)


class HttpVersion(str, Enum):
    """Protocol preference; only H3_ONLY prohibits negotiated fallback."""

    H1 = "h1"
    H2 = "h2"
    H3 = "h3"
    H3_ONLY = "h3-only"

    @property
    def curl_version(self) -> CurlHttpVersion:
        return {
            HttpVersion.H1: CurlHttpVersion.V1_1,
            HttpVersion.H2: CurlHttpVersion.V2_0,
            HttpVersion.H3: CurlHttpVersion.V3,
            HttpVersion.H3_ONLY: CurlHttpVersion.V3ONLY,
        }[self]


@dataclass(frozen=True, slots=True)
class ProtocolCapabilities:
    libcurl_version: str
    http2: bool
    http3: bool
    http3_only: bool

    @classmethod
    def detect(cls) -> ProtocolCapabilities:
        """Probe compiled support through setopt, without making a network request."""
        curl = Curl()
        try:
            version = curl.version().decode("utf-8", errors="replace")

            def supports(mode: CurlHttpVersion) -> bool:
                try:
                    curl.setopt(CurlOpt.HTTP_VERSION, mode)
                except CurlError:
                    return False
                return True

            return cls(
                version,
                supports(CurlHttpVersion.V2_0),
                supports(CurlHttpVersion.V3),
                supports(CurlHttpVersion.V3ONLY),
            )
        finally:
            curl.close()


def _positive_seconds(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a positive finite number of seconds")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number of seconds")
    return float(value)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class _BodySink:
    """Consume curl body callbacks without a stream queue or a second body buffer."""

    def __init__(self, output_path: Path | None) -> None:
        self.output_path = output_path
        self.buffer = bytearray() if output_path is None else None
        self.received_bytes = 0
        self.write_error: OSError | None = None
        self.temp_path: Path | None = None
        self.file: BinaryIO | None = None
        if output_path is not None:
            fd, name = tempfile.mkstemp(
                prefix=f".{output_path.name}.", suffix=".part", dir=output_path.parent
            )
            self.temp_path = Path(name)
            try:
                self.file = os.fdopen(fd, "wb")
            except BaseException:
                os.close(fd)
                self.temp_path.unlink(missing_ok=True)
                raise

    def write(self, chunk: bytes) -> int:
        self.received_bytes += len(chunk)
        try:
            if self.file is not None:
                self.file.write(chunk)
            elif self.buffer is not None:
                self.buffer.extend(chunk)
        except OSError as exc:
            self.write_error = exc
            return 0  # Abort curl; never let exceptions escape an FFI callback.
        return len(chunk)

    def publish(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.temp_path is not None and self.output_path is not None:
            os.replace(self.temp_path, self.output_path)
            self.temp_path = None

    def cleanup(self) -> None:
        try:
            if self.file is not None:
                self.file.close()
        finally:
            self.file = None
            if self.temp_path is not None:
                self.temp_path.unlink(missing_ok=True)
                self.temp_path = None


@dataclass(slots=True)
class _TransferState:
    sink: _BodySink | None = None
    response: Response | None = None
    started: float | None = None
    finished: float | None = None
    error: TransferError | None = None
    range_honored: bool | None = None
    published_path: Path | None = None


CompletionHook = Callable[[TransferResult], Awaitable[None] | None]


class RequestHandler:
    """Reusable async GET transport. Use one handler per asyncio event loop.

    No internal retries, redirect following, request prioritization or scheduling.
    Transfer failures are returned; cancellation propagates after cleanup.
    """

    def __init__(
        self,
        http_version: str | HttpVersion = "h2",
        concurrency: int = 1,
        connect_timeout_s: float = 5,
        total_timeout_s: float = 30,
        verify_tls: bool = True,
        on_transfer_complete: CompletionHook | None = None,
    ) -> None:
        try:
            self._http_version = HttpVersion(http_version)
        except (ValueError, TypeError) as exc:
            raise ValueError("http_version must be h1, h2, h3 or h3-only") from exc
        self.concurrency = _positive_integer("concurrency", concurrency)
        self.connect_timeout_s = _positive_seconds("connect_timeout_s", connect_timeout_s)
        self.total_timeout_s = _positive_seconds("total_timeout_s", total_timeout_s)
        if not isinstance(verify_tls, bool):
            raise ValueError("verify_tls must be a bool")
        if on_transfer_complete is not None and not callable(on_transfer_complete):
            raise ValueError("on_transfer_complete must be callable or None")
        self.verify_tls = verify_tls
        self.on_transfer_complete = on_transfer_complete
        self.capabilities = ProtocolCapabilities.detect()
        available = {
            HttpVersion.H1: True,
            HttpVersion.H2: self.capabilities.http2,
            HttpVersion.H3: self.capabilities.http3,
            HttpVersion.H3_ONLY: self.capabilities.http3_only,
        }[self.http_version]
        if not available:
            raise ProtocolUnavailableError(
                f"{self.http_version.value} is unavailable in installed curl_cffi/libcurl: "
                f"{self.capabilities.libcurl_version}. Install a curl_cffi build with "
                "the requested HTTP backend; no substitute protocol was selected."
            )
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._session: AsyncSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._active: set[asyncio.Task[Any]] = set()
        self._batches: dict[asyncio.Task[Any], set[asyncio.Task[Any]]] = {}
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def http_version(self) -> HttpVersion:
        return self._http_version

    @property
    def curl_http_version(self) -> CurlHttpVersion:
        return self.http_version.curl_version

    def _ensure_session(self) -> AsyncSession:
        if self._closed:
            raise RuntimeError("RequestHandler is closed")
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("RequestHandler must be used on its original asyncio loop")
        if self._session is None:
            self._loop = loop
            self._session = AsyncSession(
                max_clients=self.concurrency,
                http_version=self.curl_http_version,
                verify=self.verify_tls,
                timeout=None,
                retry=0,
                allow_redirects=False,
                trust_env=False,
                curl_infos=[CurlInfo.TOTAL_TIME, CurlInfo.STARTTRANSFER_TIME],
                curl_options={
                    CurlOpt.CONNECTTIMEOUT_MS: max(1, math.ceil(self.connect_timeout_s * 1000)),
                    CurlOpt.TIMEOUT_MS: max(1, math.ceil(self.total_timeout_s * 1000)),
                    CurlOpt.HTTP_CONTENT_DECODING: 0,
                },
            )
        return self._session

    async def __aenter__(self) -> RequestHandler:
        self._ensure_session()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Cancel pending/active fetches, clean files and close the connection pool."""
        current = asyncio.current_task()
        if not self._closed:
            self._closed = True
            children = set().union(*self._batches.values()) if self._batches else set()
            tasks = [
                task for task in self._active | self._batches.keys()
                if task is not current and task not in children
            ]

            async def close_resources() -> None:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if self._session is not None:
                    await self._session.close()
                    self._session = None

            self._close_task = asyncio.create_task(close_resources())
        elif current in self._active or current in self._batches:
            # A cancellation hook may call aclose while close_resources awaits it.
            return
        if self._close_task is not None:
            await asyncio.shield(self._close_task)

    async def fetch(self, spec: RequestSpec, output_path: str | Path | None = None) -> TransferResult:
        if not isinstance(spec, RequestSpec):
            raise TypeError("spec must be a RequestSpec")
        session = self._ensure_session()
        path = Path(output_path) if output_path is not None else None
        state = _TransferState()
        task = asyncio.current_task()
        assert task is not None
        self._active.add(task)
        try:
            try:
                async with self._semaphore:
                    if self._closed:
                        raise RuntimeError("RequestHandler is closed")
                    result = await self._perform(session, spec, path, state)
            except asyncio.CancelledError:
                state.finished = time.monotonic()
                state.error = TransferError(ErrorKind.CANCELLED, "Request cancelled by caller")
                await self._notify(self._result(spec, state))
                raise
            await self._notify(result)
            return result
        finally:
            self._active.discard(task)

    async def fetch_many(
        self, specs: Iterable[RequestSpec], concurrency: int | None = None
    ) -> list[TransferResult]:
        """Return individual results in input order; no throughput aggregation."""
        self._ensure_session()
        limit = self.concurrency if concurrency is None else _positive_integer("concurrency", concurrency)
        # Materialize and validate before starting tasks, including generator failures.
        requests = list(specs)
        if any(not isinstance(spec, RequestSpec) for spec in requests):
            raise TypeError("Every item in specs must be a RequestSpec")
        batch = asyncio.Semaphore(min(limit, self.concurrency))

        async def run(spec: RequestSpec) -> TransferResult:
            async with batch:
                return await self.fetch(spec)

        tasks = [asyncio.create_task(run(spec)) for spec in requests]
        current = asyncio.current_task()
        assert current is not None
        self._batches[current] = set(tasks)
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            self._batches.pop(current, None)

    async def _perform(
        self, session: AsyncSession, spec: RequestSpec, path: Path | None, state: _TransferState
    ) -> TransferResult:
        headers = {"Accept-Encoding": "identity"}
        if spec.range_header is not None:
            headers["Range"] = spec.range_header
        logger.debug(
            "Starting request=%s object=%s representation=%s segment=%s url=%s range=%s protocol=%s",
            spec.request_id, spec.object_id, spec.representation_id, spec.segment_id,
            spec.url, spec.range_header, self.http_version.value,
        )
        try:
            state.sink = _BodySink(path)
            state.started = time.monotonic()
            try:
                state.response = await session.get(
                    spec.url, headers=headers, accept_encoding="identity",
                    content_callback=state.sink.write,
                )
            except CurlError as exc:
                state.response = getattr(exc, "response", None)
                code = int(exc.code)
                kind = {
                    28: ErrorKind.TIMEOUT,
                    18: ErrorKind.INCOMPLETE,
                    1: ErrorKind.PROTOCOL,
                    4: ErrorKind.PROTOCOL,
                }.get(code, ErrorKind.CONNECTION)
                state.error = TransferError(kind, str(exc), code)
            finally:
                state.finished = time.monotonic()
            if spec.byte_range is not None and state.response is not None and state.response.status_code == 200:
                state.range_honored = False
            if state.sink.write_error is not None:
                state.error = TransferError(ErrorKind.FILE, str(state.sink.write_error))
            if state.error is None:
                state.error, honored = self._validate_response(spec, state)
                if honored is not None:
                    state.range_honored = honored
            if state.error is None:
                state.sink.publish()
                state.published_path = path
        except OSError as exc:
            state.error = TransferError(ErrorKind.FILE, str(exc))
        finally:
            if state.sink is not None:
                try:
                    state.sink.cleanup()
                except OSError as exc:
                    logger.error("Temporary file cleanup failed request=%s: %s", spec.request_id, exc)
                    if state.error is None:
                        state.error = TransferError(ErrorKind.FILE, f"Temporary file cleanup failed: {exc}")
        return self._result(spec, state)

    def _validate_response(
        self, spec: RequestSpec, state: _TransferState
    ) -> tuple[TransferError | None, bool | None]:
        response = state.response
        assert response is not None and state.sink is not None
        status = response.status_code
        headers = {key.lower(): value for key, value in response.headers.items()}
        received = state.sink.received_bytes
        if not 200 <= status < 300:
            return TransferError(ErrorKind.HTTP, f"HTTP {status}"), None
        if self.http_version == HttpVersion.H3_ONLY and self._protocol(response) != "h3":
            return TransferError(ErrorKind.PROTOCOL, "h3-only request did not negotiate HTTP/3"), None
        if headers.get("content-encoding", "identity").lower() != "identity":
            return TransferError(ErrorKind.PROTOCOL, "Server returned non-identity Content-Encoding"), None
        length = headers.get("content-length")
        if length is not None:
            if not re.fullmatch(r"[0-9]+", length):
                return TransferError(ErrorKind.PROTOCOL, "Invalid Content-Length"), None
            try:
                expected_length = int(length)
            except ValueError:
                return TransferError(ErrorKind.PROTOCOL, "Content-Length exceeds numeric parsing limits"), None
            if expected_length != received:
                return TransferError(
                    ErrorKind.INCOMPLETE, f"Content-Length={length} but received {received} body bytes"
                ), None
        if spec.byte_range is None:
            if status == 206:
                return TransferError(ErrorKind.INVALID_RANGE, "Unrequested partial response (206)"), None
            return None, None
        if status == 200:
            return TransferError(ErrorKind.RANGE_IGNORED, "Server ignored Range and returned HTTP 200"), False
        if status != 206:
            return TransferError(ErrorKind.INVALID_RANGE, f"Range request expected 206, received {status}"), False
        value = headers.get("content-range", "")
        match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+|\*)", value, re.IGNORECASE)
        if match is None:
            return TransferError(ErrorKind.INVALID_RANGE, "Missing or invalid Content-Range"), False
        try:
            start, end = int(match[1]), int(match[2])
            total = None if match[3] == "*" else int(match[3])
        except ValueError:
            return TransferError(ErrorKind.INVALID_RANGE, "Content-Range exceeds numeric parsing limits"), False
        requested_start, requested_end = spec.byte_range
        expected_end = requested_end if total is None else min(requested_end, total - 1)
        if (
            start != requested_start or end != expected_end or end < start
            or (total is not None and (total <= 0 or end >= total))
        ):
            return TransferError(ErrorKind.INVALID_RANGE, f"Content-Range {value!r} does not match requested range"), False
        if received != end - start + 1:
            return TransferError(ErrorKind.INCOMPLETE, "Body byte count does not match Content-Range"), False
        return None, True

    @staticmethod
    def _protocol(response: Response | None) -> str | None:
        if response is None:
            return None
        return {
            int(CurlHttpVersion.V1_0): "h1.0",
            int(CurlHttpVersion.V1_1): "h1",
            int(CurlHttpVersion.V2_0): "h2",
            int(CurlHttpVersion.V3): "h3",
        }.get(int(response.http_version))

    @staticmethod
    def _result(spec: RequestSpec, state: _TransferState) -> TransferResult:
        response, sink = state.response, state.sink
        finished = state.finished if state.finished is not None else time.monotonic()
        infos = response.infos if response is not None else {}
        native_total = infos.get(CurlInfo.TOTAL_TIME)
        timing_source = "libcurl" if native_total is not None else "monotonic"
        total = max(0.0, float(native_total)) if native_total is not None else (
            max(0.0, finished - state.started) if state.started is not None else 0.0
        )
        # Anchor native durations to observed completion; absolute timestamps are estimates.
        start = finished - total
        native_ttfb = infos.get(CurlInfo.STARTTRANSFER_TIME)
        ttfb = min(total, max(0.0, float(native_ttfb))) if native_ttfb and native_ttfb > 0 else None
        interval = max(0.0, total - ttfb) if ttfb is not None else None
        received = sink.received_bytes if sink is not None else 0

        def goodput(duration: float | None) -> float | None:
            return received * 8 / duration if duration is not None and duration > 0 else None

        return TransferResult(
            spec=spec,
            status_code=(response.status_code or None) if response is not None else None,
            body=bytes(sink.buffer) if sink is not None and sink.buffer is not None else None,
            output_path=state.published_path,
            received_bytes=received,
            start_time=start,
            first_byte_time=start + ttfb if ttfb is not None else None,
            end_time=finished,
            ttfb=ttfb,
            transfer_duration=interval,
            total_duration=total,
            body_goodput_bps=goodput(interval),
            effective_goodput_bps=goodput(total),
            http_version=RequestHandler._protocol(response),
            response_headers=MappingProxyType({key.lower(): value for key, value in response.headers.items()})
            if response is not None else MappingProxyType({}),
            success=state.error is None,
            error=state.error,
            range_honored=state.range_honored,
            timestamp=time.time(),
            timing_source=timing_source,
        )

    async def _notify(self, result: TransferResult) -> None:
        logger.log(
            logging.INFO if result.success else logging.WARNING,
            "Completed request=%s object=%s representation=%s segment=%s url=%s range=%s "
            "protocol=%s status=%s bytes=%s ttfb=%s body_duration=%s body_goodput_bps=%s "
            "effective_goodput_bps=%s success=%s error=%s",
            result.request_id, result.object_id, result.representation_id, result.segment_id,
            result.url, result.requested_range, result.http_version, result.status_code,
            result.received_bytes, result.ttfb, result.transfer_duration, result.body_goodput_bps,
            result.effective_goodput_bps, result.success, result.error,
        )
        if self.on_transfer_complete is not None:
            try:
                returned = self.on_transfer_complete(result)
                if inspect.isawaitable(returned):
                    await returned
            except Exception:
                logger.exception("Transfer completion hook failed request=%s", result.request_id)

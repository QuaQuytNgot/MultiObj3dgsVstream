"""Public API integration with the real RequestHandler and an offline HTTP double.

No web server is created here. The double replaces AsyncSession at the transport
boundary; these tests make no claims about HTTP/2 or HTTP/3 on the wire.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

# Consumer examples import only the public layer.
from src.client.request_api import (
    BandwidthSample,
    ErrorKind,
    NetworkClient,
    ProtocolUnavailableError,
    RequestSpec,
    TransferClient,
    TransferError,
    TransferResult,
    to_bandwidth_sample,
)

PAYLOAD = b"binary segment payload\x00\xff" * 32
URL = "https://static.example/segment.bin"


@pytest.fixture
def transport(monkeypatch):
    # Backend-specific imports are confined to this transport test double.
    from curl_cffi import CurlInfo
    from src.client import request_handler as implementation

    sessions = []

    class Session:
        def __init__(self, **options):
            self.options = options
            self.requests = []
            self.closed = False
            self.started = asyncio.Event()
            sessions.append(self)

        async def get(self, url, *, headers, accept_encoding, content_callback):
            self.requests.append((url, dict(headers)))
            assert accept_encoding == "identity"
            status, payload = 200, PAYLOAD
            response_headers = {"accept-ranges": "bytes"}
            if url.endswith("/error"):
                status, payload = 503, b"unavailable"
            elif headers.get("Range") and not url.endswith("/ignore-range"):
                match = re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"])
                start, end = map(int, match.groups())
                end = min(end, len(PAYLOAD) - 1)
                status, payload = 206, PAYLOAD[start:end + 1]
                response_headers["content-range"] = f"bytes {start}-{end}/{len(PAYLOAD)}"
            response_headers["content-length"] = str(len(payload))
            split = max(1, len(payload) // 2)
            content_callback(payload[:split])
            self.started.set()
            if url.endswith("/wait"):
                await asyncio.sleep(3600)
            await asyncio.sleep(0)
            content_callback(payload[split:])
            return SimpleNamespace(
                status_code=status,
                headers=response_headers,
                http_version=3,  # Simulated HTTP/2 result, not a network assertion.
                infos={CurlInfo.TOTAL_TIME: 0.04, CurlInfo.STARTTRANSFER_TIME: 0.01},
            )

        async def close(self):
            self.closed = True

    monkeypatch.setattr(implementation, "AsyncSession", Session)
    monkeypatch.setattr(
        implementation.ProtocolCapabilities,
        "detect",
        classmethod(lambda cls: cls("offline backend double", True, True, True)),
    )
    return sessions


async def _scheduler_like(client: TransferClient, spec: RequestSpec) -> TransferResult:
    """A caller depends on a structural interface and a resolved request only."""
    return await client.fetch(spec)


def _payload_consumer(result: TransferResult) -> bytes:
    """Illustrate parser input selection without implementing a payload codec."""
    if not result.success:
        raise ValueError("Cannot parse a failed transfer")
    if result.body is not None:
        return result.body
    assert result.output_path is not None
    return result.output_path.read_bytes()


@pytest.mark.asyncio
async def test_scheduler_to_real_handler_and_public_result(transport):
    spec = RequestSpec(
        request_id="ld-q1-1", url=URL, object_id="longdress",
        representation_id="q1", segment_id=1,
        metadata={"quality_profile_uri": "profiles/q1.json"},
    )
    async with NetworkClient(concurrency=1) as client:
        assert isinstance(client, TransferClient)
        result = await _scheduler_like(client, spec)
        second = await _scheduler_like(client, replace(spec, request_id="ld-q1-2"))
    assert isinstance(result, TransferResult)
    assert result.spec is spec
    assert result.success and second.success
    assert result.metadata["quality_profile_uri"] == "profiles/q1.json"
    assert _payload_consumer(result) == PAYLOAD
    # One owned RequestHandler creates and reuses one backend session.
    assert len(transport) == 1
    assert len(transport[0].requests) == 2
    assert transport[0].options["max_clients"] == 1
    assert transport[0].closed


@pytest.mark.asyncio
async def test_get_unique_ids_metadata_and_explicit_id_validation(transport):
    async with NetworkClient() as client:
        first = await client.get(URL, metadata={"purpose": "index"})
        second = await client.get(URL)
        explicit = await client.get(URL, request_id="known")
        with pytest.raises(ValueError, match="request_id"):
            await client.get(URL, request_id="")
    assert first.request_id != second.request_id
    assert explicit.request_id == "known"
    assert first.metadata == {"purpose": "index"}
    assert first.body == PAYLOAD


@pytest.mark.asyncio
async def test_get_segment_preserves_resolved_context(transport):
    async with NetworkClient() as client:
        result = await client.get_segment(
            URL, object_id="longdress", representation_id="q1", segment_id=9,
            request_id="segment-9", layer_id="enhancement", priority=7,
            byte_range=(10, 29), metadata={"duration_s": 1.0},
        )
    assert result.request_id == "segment-9"
    assert result.object_id == "longdress"
    assert result.representation_id == "q1"
    assert result.segment_id == 9
    assert result.layer_id == "enhancement"
    assert result.spec.priority == 7
    assert result.metadata == {"duration_s": 1.0}
    assert result.body == PAYLOAD[10:30]
    assert result.range_honored


@pytest.mark.asyncio
async def test_get_range_and_ignored_range(transport):
    async with NetworkClient() as client:
        result = await client.get_range(URL, 10, 29, request_id="range")
        ignored = await client.get_range("https://static.example/ignore-range", 10, 29)
        with pytest.raises(ValueError, match="byte_range"):
            await client.get_range(URL, 29, 10)
    assert result.success
    assert result.requested_range == (10, 29)
    assert result.received_bytes == 20
    assert result.body == PAYLOAD[10:30]
    assert transport[0].requests[0][1]["Range"] == "bytes=10-29"
    assert not ignored.success
    assert ignored.error.kind == ErrorKind.RANGE_IGNORED


@pytest.mark.asyncio
@pytest.mark.parametrize("use_spec", [False, True])
async def test_download_to_file_and_payload_consumer(transport, tmp_path, use_spec):
    path = tmp_path / "segment.bin"
    request = RequestSpec(request_id="file", url=URL) if use_spec else URL
    async with NetworkClient() as client:
        result = await client.download_to_file(request, path)
    assert result.success
    assert result.body is None
    assert result.output_path == path
    assert _payload_consumer(result) == PAYLOAD
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.asyncio
async def test_scheduler_spec_can_download_range_without_losing_identity(transport, tmp_path):
    spec = RequestSpec(
        request_id="chunk", url=URL, object_id="longdress",
        representation_id="q1", segment_id=1, byte_range=(5, 14),
    )
    async with NetworkClient() as client:
        result = await client.download_to_file(spec, tmp_path / "chunk.bin")
        with pytest.raises(ValueError, match="supplied RequestSpec"):
            await client.download_to_file(spec, tmp_path / "other.bin", request_id="override")
        with pytest.raises(ValueError, match="supplied RequestSpec"):
            await client.download_to_file(spec, tmp_path / "other.bin", metadata={})
    assert result.spec is spec
    assert result.received_bytes == 10
    assert _payload_consumer(result) == PAYLOAD[5:15]


@pytest.mark.asyncio
async def test_failed_download_does_not_publish_or_parse(transport, tmp_path):
    path = tmp_path / "segment.bin"
    path.write_bytes(b"previous complete file")
    async with NetworkClient() as client:
        result = await client.download_to_file("https://static.example/error", path)
    assert not result.success
    assert result.error.kind == ErrorKind.HTTP
    assert result.body is None and result.output_path is None
    assert path.read_bytes() == b"previous complete file"
    assert not list(tmp_path.glob("*.part"))
    with pytest.raises(ValueError, match="failed transfer"):
        _payload_consumer(result)


@pytest.mark.asyncio
async def test_bandwidth_sample_copies_only_requested_metrics(transport):
    async with NetworkClient() as client:
        result = await client.get(URL)
    sample = to_bandwidth_sample(result)
    names = [
        "timestamp", "received_bytes", "ttfb", "transfer_duration", "total_duration",
        "body_goodput_bps", "effective_goodput_bps", "http_version", "success",
    ]
    assert [field.name for field in fields(BandwidthSample)] == names
    for name in names:
        assert getattr(sample, name) == getattr(result, name)
    assert sample.body_goodput_bps == pytest.approx(len(PAYLOAD) * 8 / 0.03)
    assert sample.effective_goodput_bps == pytest.approx(len(PAYLOAD) * 8 / 0.04)
    with pytest.raises(FrozenInstanceError):
        sample.success = False


@pytest.mark.asyncio
async def test_failure_and_missing_timing_samples_are_not_filtered_or_invented(transport):
    async with NetworkClient() as client:
        failed = await client.get("https://static.example/error")
    result = replace(
        failed, ttfb=None, transfer_duration=None, total_duration=0,
        body_goodput_bps=None, effective_goodput_bps=None, http_version=None,
    )
    sample = to_bandwidth_sample(result)
    assert not sample.success
    assert sample.received_bytes == len(b"unavailable")
    assert sample.ttfb is None and sample.transfer_duration is None
    assert sample.body_goodput_bps is None and sample.effective_goodput_bps is None
    assert sample.total_duration == 0 and sample.http_version is None


@pytest.mark.asyncio
async def test_sync_async_subscribers_and_unsubscribe(transport):
    initial, synchronous, samples = [], [], []

    async def observe(result):
        await asyncio.sleep(0)
        samples.append(to_bandwidth_sample(result))

    async with NetworkClient(on_transfer_complete=initial.append) as client:
        stop = client.subscribe(synchronous.append)
        client.subscribe(observe)
        first = await client.get(URL)
        stop()
        stop()
        second = await client.get("https://static.example/error")
    assert initial == [first, second]
    assert synchronous == [first]
    assert samples == [to_bandwidth_sample(first), to_bandwidth_sample(second)]


@pytest.mark.asyncio
async def test_bad_subscriber_is_isolated_and_reported(transport, caplog):
    events = []

    async def broken(_result):
        raise RuntimeError("consumer failed")

    async with NetworkClient() as client:
        client.subscribe(broken)
        client.subscribe(events.append)
        result = await client.get(URL)
    assert result.success
    assert events == [result]
    assert "Transfer subscriber failed" in caplog.text


@pytest.mark.asyncio
async def test_subscriber_changes_take_effect_on_next_event(transport):
    calls = []
    stop_other = None

    def first(result):
        nonlocal stop_other
        calls.append(("first", result.request_id))
        if result.request_id == "one":
            stop_other()
            client.subscribe(lambda event: calls.append(("new", event.request_id)))

    async with NetworkClient() as client:
        client.subscribe(first)
        stop_other = client.subscribe(lambda event: calls.append(("other", event.request_id)))
        await client.get(URL, request_id="one")
        await client.get(URL, request_id="two")
    assert calls == [("first", "one"), ("other", "one"), ("first", "two"), ("new", "two")]


@pytest.mark.asyncio
async def test_duplicate_subscriptions_are_independent(transport):
    events = []
    async with NetworkClient() as client:
        stop = client.subscribe(events.append)
        client.subscribe(events.append)
        first = await client.get(URL)
        stop()
        second = await client.get(URL)
    assert events == [first, first, second]


@pytest.mark.asyncio
async def test_cancellation_cleanup_and_sample_notification(transport, tmp_path):
    events = []
    path = tmp_path / "cancelled.bin"
    async with NetworkClient(on_transfer_complete=events.append) as client:
        task = asyncio.create_task(client.download_to_file("https://static.example/wait", path))
        while not transport:
            await asyncio.sleep(0)
        await transport[0].started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        recovered = await client.get(URL)
    assert len(events) == 2 and recovered.success
    cancelled = events[0]
    assert cancelled.error.kind == ErrorKind.CANCELLED
    assert not to_bandwidth_sample(cancelled).success
    assert cancelled.received_bytes > 0
    assert cancelled.output_path is None
    assert not path.exists() and not list(tmp_path.glob("*.part"))


@pytest.mark.asyncio
async def test_context_cleanup_and_closed_client(transport):
    async with NetworkClient() as client:
        await client.get(URL)
        with pytest.raises(ValueError, match="callable"):
            client.subscribe(None)
    await client.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await client.get(URL)
    with pytest.raises(RuntimeError, match="closed"):
        client.subscribe(lambda _result: None)
    assert transport[0].closed


def test_low_level_imports_keep_the_same_contract_identity():
    from src.client import request_handler as low_level

    for name, value in {
        "RequestSpec": RequestSpec, "TransferResult": TransferResult,
        "TransferError": TransferError, "ErrorKind": ErrorKind,
        "ProtocolUnavailableError": ProtocolUnavailableError,
    }.items():
        assert getattr(low_level, name) is value


def test_public_contracts_import_without_transport_dependency():
    code = """
import importlib.abc
import sys

class DenyCurl(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'curl_cffi' or fullname.startswith('curl_cffi.'):
            raise AssertionError('Public contracts imported curl_cffi')

sys.meta_path.insert(0, DenyCurl())
from src.client.request_api import RequestSpec, TransferClient, BandwidthSample
RequestSpec(request_id='public', url='https://static.example/segment.bin')
assert 'src.client.request_handler' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr

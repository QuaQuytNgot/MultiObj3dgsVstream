"""Offline transport tests.

The fixture serves HTTP/1.1. These tests do not claim to test HTTP/2 or HTTP/3
on the wire; protocol configuration and libcurl capabilities are checked apart.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import socketserver
import threading
import time
from urllib.parse import urlsplit

from curl_cffi import CurlHttpVersion
import pytest

from src.client.request_handler import (
    ErrorKind,
    ProtocolCapabilities,
    ProtocolUnavailableError,
    RequestHandler,
    RequestSpec,
)


PAYLOAD = b"0123456789abcdefghijklmnopqrstuvwxyz" * 2048


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.lock = threading.Lock()
        self.records: list[dict[str, object]] = []
        self.active = 0
        self.max_active = 0
        self.stream_started = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *args):
        pass

    def _reply(self, status, payload, **headers):
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Client-Port", str(self.client_address[1]))
        for key, value in headers.items():
            self.send_header(key.replace("_", "-"), str(value))
        self.end_headers()
        if payload:
            self.wfile.write(payload)
            self.wfile.flush()

    def do_GET(self):
        path = urlsplit(self.path).path
        server = self.server
        with server.lock:
            server.records.append({"path": path, "range": self.headers.get("Range")})
        try:
            if path == "/range":
                match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
                if not match:
                    self._reply(400, b"closed byte range required")
                    return
                start, end = map(int, match.groups())
                if start >= len(PAYLOAD):
                    self._reply(416, b"", Content_Range=f"bytes */{len(PAYLOAD)}")
                    return
                end = min(end, len(PAYLOAD) - 1)
                self._reply(
                    206,
                    PAYLOAD[start : end + 1],
                    Content_Range=f"bytes {start}-{end}/{len(PAYLOAD)}",
                    Accept_Ranges="bytes",
                )
            elif path == "/bad-range":
                self._reply(206, PAYLOAD[:20], Content_Range=f"bytes 1-20/{len(PAYLOAD)}")
            elif path == "/missing-range":
                self._reply(206, PAYLOAD[:20])
            elif path == "/huge-range":
                self._reply(206, PAYLOAD[:20], Content_Range=f"bytes 0-19/{'9' * 5000}")
            elif path == "/truncated":
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(PAYLOAD[:123])
                self.wfile.flush()
                self.close_connection = True
            elif path == "/error":
                self._reply(503, b"temporarily unavailable")
            elif path == "/redirect":
                self._reply(302, b"redirect body", Location="/payload")
            elif path == "/empty":
                self._reply(200, b"")
            elif path == "/slow-headers":
                time.sleep(0.25)
                self._reply(200, PAYLOAD)
            elif path == "/slow-body":
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.end_headers()
                self.wfile.flush()
                time.sleep(0.25)
                self.wfile.write(PAYLOAD)
                self.wfile.flush()
            elif path == "/stream":
                self.send_response(200)
                self.send_header("Content-Length", "1000000")
                self.end_headers()
                self.wfile.write(b"x" * 4096)
                self.wfile.flush()
                server.stream_started.set()
                for _ in range(100):
                    time.sleep(0.03)
                    self.wfile.write(b"x" * 4096)
                    self.wfile.flush()
            elif path == "/concurrent":
                with server.lock:
                    server.active += 1
                    server.max_active = max(server.active, server.max_active)
                try:
                    time.sleep(0.10)
                    self._reply(200, PAYLOAD)
                finally:
                    with server.lock:
                        server.active -= 1
            else:
                self._reply(200, PAYLOAD, Accept_Ranges="bytes")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Cancellation/timeout deliberately closes a client connection.
            pass


@pytest.fixture
def server():
    server = _Server()
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _spec(server, path="/payload", request_id="request-1", **kwargs):
    return RequestSpec(
        request_id=request_id,
        url=f"{server.url}{path}",
        object_id="longdress",
        representation_id="q1",
        segment_id=1,
        **kwargs,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": ""},
        {"request_id": "   "},
        {"url": "not-a-url"},
        {"url": "ftp://example.invalid/payload"},
        {"url": "https:///payload"},
        {"byte_range": (-1, 10)},
        {"byte_range": (10, 9)},
        {"byte_range": (1,)},
        {"byte_range": (0, 1.5)},
        {"byte_range": (False, 10)},
    ],
)
def test_request_spec_validation(changes):
    values = {"request_id": "r1", "url": "https://example.invalid/segment"}
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        RequestSpec(**values)


def test_request_spec_metadata_and_immutability():
    spec = RequestSpec(
        request_id="r1",
        url="https://example.invalid/segment",
        byte_range=(0, 0),
        layer_id="enhancement-1",
        priority=3,
        metadata={"quality_profile_uri": "profiles/q1.json", "duration_s": 1.0},
    )
    assert spec.metadata["quality_profile_uri"] == "profiles/q1.json"
    with pytest.raises((FrozenInstanceError, AttributeError)):
        spec.request_id = "changed"


@pytest.mark.asyncio
async def test_get_bytes_and_response_headers(server):
    spec = _spec(server)
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(spec)
    assert result.success
    assert result.error is None
    assert result.status_code == 200
    assert result.body == PAYLOAD
    assert result.received_bytes == len(PAYLOAD)
    assert result.output_path is None
    assert result.request_id == spec.request_id
    assert result.url == spec.url
    assert result.object_id == "longdress"
    assert result.representation_id == "q1"
    assert result.segment_id == 1
    assert result.requested_range is None
    assert result.range_honored is None
    assert result.http_version == "h1"
    assert result.content_length == len(PAYLOAD)
    assert result.accept_ranges == "bytes"
    assert result.response_headers["content-length"] == str(len(PAYLOAD))


@pytest.mark.asyncio
async def test_byte_range_header_and_206(server):
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server, "/range", byte_range=(1000, 1999)))
    assert server.records[-1]["range"] == "bytes=1000-1999"
    assert result.success
    assert result.status_code == 206
    assert result.range_honored is True
    assert result.requested_range == (1000, 1999)
    assert result.received_bytes == 1000
    assert result.body == PAYLOAD[1000:2000]
    assert result.content_range == f"bytes 1000-1999/{len(PAYLOAD)}"
    assert result.content_length == 1000
    assert result.accept_ranges == "bytes"


@pytest.mark.asyncio
async def test_range_end_is_clamped_to_representation_length(server):
    start = len(PAYLOAD) - 10
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server, "/range", byte_range=(start, len(PAYLOAD) + 100)))
    assert result.success and result.range_honored
    assert result.body == PAYLOAD[-10:]
    assert result.received_bytes == 10
    assert result.content_range == f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"


@pytest.mark.asyncio
async def test_server_ignores_range(server):
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server, "/ignore-range", byte_range=(10, 19)))
    assert not result.success
    assert result.status_code == 200
    assert result.range_honored is False
    assert result.error.kind == ErrorKind.RANGE_IGNORED
    assert result.received_bytes == len(PAYLOAD)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/bad-range", "/missing-range", "/huge-range"])
async def test_invalid_partial_content_is_failure(server, path):
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server, path, byte_range=(0, 19)))
    assert not result.success
    assert result.status_code == 206
    assert result.error.kind == ErrorKind.INVALID_RANGE


@pytest.mark.asyncio
async def test_timing_fields_and_goodput(server):
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server))
    assert result.success
    assert result.start_time <= result.first_byte_time <= result.end_time
    assert result.ttfb >= 0
    assert result.transfer_duration >= 0
    assert result.total_duration >= result.ttfb
    assert result.total_duration == pytest.approx(result.end_time - result.start_time)
    assert result.ttfb == pytest.approx(result.first_byte_time - result.start_time)
    assert result.transfer_duration == pytest.approx(result.end_time - result.first_byte_time)
    assert result.body_goodput_bps == pytest.approx(
        8 * result.received_bytes / result.transfer_duration
    )
    assert result.effective_goodput_bps == pytest.approx(
        8 * result.received_bytes / result.total_duration
    )


@pytest.mark.asyncio
async def test_ttfb_distinguishes_headers_from_body_wait(server):
    async with RequestHandler(http_version="h1") as handler:
        headers = await handler.fetch(_spec(server, "/slow-headers", request_id="headers"))
        body = await handler.fetch(_spec(server, "/slow-body", request_id="body"))
    assert headers.success and body.success
    assert headers.ttfb >= 0.20
    assert body.transfer_duration >= 0.20
    assert body.transfer_duration > body.ttfb


@pytest.mark.asyncio
async def test_empty_body_metrics(server):
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server, "/empty"))
    assert result.success
    assert result.body == b""
    assert result.received_bytes == 0
    assert result.effective_goodput_bps == 0
    assert result.body_goodput_bps in (None, 0)


@pytest.mark.asyncio
async def test_atomic_file_output_without_memory_copy(server, tmp_path):
    output = tmp_path / "segment.bin"
    output.write_bytes(b"previous segment")
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server), output_path=output)
    assert result.success
    assert result.body is None
    assert result.output_path == output
    assert result.received_bytes == len(PAYLOAD)
    assert output.read_bytes() == PAYLOAD
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.asyncio
async def test_range_file_output(server, tmp_path):
    output = tmp_path / "range.bin"
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server, "/range", byte_range=(17, 300)), output_path=output)
    assert result.success and result.range_honored
    assert output.read_bytes() == PAYLOAD[17:301]
    assert result.body is None


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/error", "/truncated", "/ignore-range"])
async def test_failed_file_transfer_preserves_existing_destination(server, tmp_path, path):
    output = tmp_path / "segment.bin"
    output.write_bytes(b"old segment")
    kwargs = {"byte_range": (0, 10)} if path == "/ignore-range" else {}
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server, path, **kwargs), output_path=output)
    assert not result.success
    assert result.output_path is None
    assert result.body is None
    assert output.read_bytes() == b"old segment"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.asyncio
async def test_total_timeout(server, tmp_path):
    output = tmp_path / "timeout.bin"
    async with RequestHandler(http_version="h1", total_timeout_s=0.05) as handler:
        result = await handler.fetch(_spec(server, "/slow-headers"), output_path=output)
    assert not result.success
    assert result.error.kind == ErrorKind.TIMEOUT
    assert result.total_duration >= 0
    assert not output.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_connect_timeout_during_local_tls_handshake():
    class StallHandshake(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(65536)  # Receive ClientHello, then omit ServerHello.
            time.sleep(0.30)

    class StallServer(socketserver.ThreadingTCPServer):
        daemon_threads = True

    stalled = StallServer(("127.0.0.1", 0), StallHandshake)
    thread = threading.Thread(target=lambda: stalled.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        async with RequestHandler(
            http_version="h1", connect_timeout_s=0.05, total_timeout_s=2, verify_tls=False
        ) as handler:
            result = await handler.fetch(
                RequestSpec(request_id="tls-timeout", url=f"https://127.0.0.1:{stalled.server_address[1]}")
            )
        assert not result.success
        assert result.error.kind == ErrorKind.TIMEOUT
        assert result.received_bytes == 0
        assert result.total_duration < 1
    finally:
        stalled.shutdown()
        stalled.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_http_error_does_not_retry(server):
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server, "/error"))
    assert not result.success
    assert result.status_code == 503
    assert result.error.kind == ErrorKind.HTTP
    assert len(server.records) == 1


@pytest.mark.asyncio
async def test_redirect_is_reported_without_following(server):
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server, "/redirect"))
    assert not result.success
    assert result.status_code == 302
    assert result.error.kind == ErrorKind.HTTP
    assert len(server.records) == 1
    assert result.response_headers["location"] == "/payload"


@pytest.mark.asyncio
async def test_file_error_is_structured_and_cleans_temporary_file(server, tmp_path):
    destination = tmp_path / "existing-directory"
    destination.mkdir()
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server), output_path=destination)
    assert not result.success
    assert result.error.kind == ErrorKind.FILE
    assert result.output_path is None
    assert destination.is_dir()
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.asyncio
async def test_incomplete_response_accounts_received_bytes(server):
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(_spec(server, "/truncated"))
    assert not result.success
    assert result.error.kind == ErrorKind.INCOMPLETE
    assert result.status_code == 200
    assert result.received_bytes == 123
    assert result.content_length == len(PAYLOAD)


@pytest.mark.asyncio
async def test_connection_error_is_structured(server):
    # Closing a bound server makes refusal deterministic without external DNS.
    closed = _Server()
    url = closed.url
    closed.server_close()
    async with RequestHandler(http_version="h1", connect_timeout_s=0.5) as handler:
        result = await handler.fetch(RequestSpec(request_id="connection", url=url))
    assert not result.success
    assert result.error.kind == ErrorKind.CONNECTION
    assert result.received_bytes == 0
    assert result.http_version is None


@pytest.mark.asyncio
async def test_session_reuses_connection(server):
    async with RequestHandler(http_version="h1") as handler:
        first = await handler.fetch(_spec(server, request_id="first"))
        second = await handler.fetch(_spec(server, request_id="second"))
    assert first.success and second.success
    assert first.response_headers["x-client-port"] == second.response_headers["x-client-port"]


@pytest.mark.asyncio
async def test_async_context_manager_and_idempotent_close(server):
    handler = RequestHandler(http_version="h1")
    async with handler as active:
        assert active is handler
        assert (await active.fetch(_spec(server))).success
    await handler.aclose()
    await handler.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await handler.fetch(_spec(server))


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2])
async def test_fetch_many_preserves_order_and_concurrency_bound(server, concurrency):
    specs = [_spec(server, "/concurrent", request_id=f"request-{i}") for i in range(5)]
    async with RequestHandler(http_version="h1", concurrency=concurrency) as handler:
        results = await handler.fetch_many(specs, concurrency=3)
    assert [r.request_id for r in results] == [s.request_id for s in specs]
    assert all(r.success and r.received_bytes == len(PAYLOAD) for r in results)
    assert server.max_active == concurrency


@pytest.mark.asyncio
async def test_fetch_many_override_can_lower_limit(server):
    specs = [_spec(server, "/concurrent", request_id=str(i)) for i in range(3)]
    async with RequestHandler(http_version="h1", concurrency=3) as handler:
        results = await handler.fetch_many(specs, concurrency=1)
        assert await handler.fetch_many([]) == []
    assert all(r.success for r in results)
    assert server.max_active == 1


@pytest.mark.asyncio
async def test_concurrent_individual_fetches_share_instance_limit(server):
    async with RequestHandler(http_version="h1", concurrency=2) as handler:
        results = await asyncio.gather(
            *(handler.fetch(_spec(server, "/concurrent", request_id=str(i))) for i in range(5))
        )
    assert all(result.success for result in results)
    assert server.max_active == 2


@pytest.mark.asyncio
async def test_cancellation_cleans_temporary_output_and_allows_next_request(server, tmp_path):
    output = tmp_path / "cancelled.bin"
    output.write_bytes(b"existing segment")
    events = []
    async with RequestHandler(http_version="h1", on_transfer_complete=events.append) as handler:
        task = asyncio.create_task(handler.fetch(_spec(server, "/stream", request_id="cancel"), output_path=output))
        started = await asyncio.to_thread(server.stream_started.wait, 2)
        assert started
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert output.read_bytes() == b"existing segment"
        assert list(tmp_path.iterdir()) == [output]
        recovered = await handler.fetch(_spec(server, request_id="recovered"))
    assert recovered.success
    cancelled = [event for event in events if event.request_id == "cancel"]
    assert len(cancelled) == 1
    assert not cancelled[0].success
    assert cancelled[0].error.kind == ErrorKind.CANCELLED


@pytest.mark.asyncio
async def test_aclose_cancels_active_and_queued_requests_without_publishing_files(server, tmp_path):
    output = tmp_path / "active.bin"
    events = []
    handler = RequestHandler(http_version="h1", concurrency=1, on_transfer_complete=events.append)
    await handler.__aenter__()
    active = asyncio.create_task(
        handler.fetch(_spec(server, "/stream", request_id="active"), output_path=output)
    )
    try:
        assert await asyncio.to_thread(server.stream_started.wait, 2)
        queued = asyncio.create_task(handler.fetch(_spec(server, request_id="queued")))
        await asyncio.sleep(0)  # Let the request reach the handler's semaphore.
        await asyncio.wait_for(handler.aclose(), timeout=2)
        for task in (active, queued):
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not list(tmp_path.iterdir())
        assert {event.request_id for event in events} == {"active", "queued"}
        assert all(event.error.kind == ErrorKind.CANCELLED for event in events)
        assert len(server.records) == 1
        await handler.aclose()
    finally:
        active.cancel()
        await handler.aclose()


@pytest.mark.asyncio
async def test_aclose_cancels_fetch_many_and_its_queued_children(server):
    handler = RequestHandler(http_version="h1", concurrency=1)
    specs = [_spec(server, "/stream", request_id=f"batch-{i}") for i in range(5)]
    task = asyncio.create_task(handler.fetch_many(specs, concurrency=3))
    try:
        assert await asyncio.to_thread(server.stream_started.wait, 2)
        await asyncio.wait_for(handler.aclose(), timeout=2)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(server.records) == 1
    finally:
        task.cancel()
        await handler.aclose()


@pytest.mark.asyncio
async def test_fetch_many_validates_entire_batch_before_network_requests(server):
    async with RequestHandler(http_version="h1") as handler:
        with pytest.raises(TypeError, match="RequestSpec"):
            await handler.fetch_many([_spec(server), "invalid specification"])
        await asyncio.sleep(0)
    assert server.records == []


@pytest.mark.asyncio
async def test_fetch_many_iterable_failure_does_not_leave_background_requests(server):
    def broken_specs():
        yield _spec(server)
        raise RuntimeError("content index iteration failed")

    async with RequestHandler(http_version="h1") as handler:
        with pytest.raises(RuntimeError, match="content index iteration failed"):
            await handler.fetch_many(broken_specs())
        await asyncio.sleep(0)
    assert server.records == []


@pytest.mark.asyncio
async def test_sync_completion_hook_receives_result_and_metadata(server):
    events = []
    metadata = {"quality_profile_uri": "q1/profile.json", "dependencies": ["base"]}
    async with RequestHandler(http_version="h1", on_transfer_complete=events.append) as handler:
        result = await handler.fetch(_spec(server, metadata=metadata))
    assert events == [result]
    assert result.metadata == metadata


@pytest.mark.asyncio
async def test_async_completion_hook_includes_failed_transfer(server):
    events = []

    async def complete(result):
        await asyncio.sleep(0)
        events.append(result)

    async with RequestHandler(http_version="h1", on_transfer_complete=complete) as handler:
        result = await handler.fetch(_spec(server, "/error"))
    assert events == [result]
    assert not result.success


@pytest.mark.parametrize(
    "options",
    [
        {"http_version": "h4"},
        {"http_version": ""},
        {"concurrency": 0},
        {"concurrency": -1},
        {"concurrency": True},
        {"connect_timeout_s": 0},
        {"total_timeout_s": -1},
    ],
)
def test_protocol_and_transport_config_validation(options):
    with pytest.raises((TypeError, ValueError)):
        RequestHandler(**options)


def test_protocol_capability_introspection():
    # This reports the local libcurl build; it does not make a network request.
    handler = RequestHandler(http_version="h1")
    capabilities = handler.capabilities
    assert isinstance(capabilities.http2, bool)
    assert isinstance(capabilities.http3, bool)
    assert isinstance(capabilities.http3_only, bool)
    assert isinstance(capabilities.libcurl_version, str)
    assert capabilities.libcurl_version


@pytest.mark.parametrize(
    "protocol, expected, capability",
    [
        ("h1", CurlHttpVersion.V1_1, None),
        ("h2", CurlHttpVersion.V2_0, "http2"),
        ("h3", CurlHttpVersion.V3, "http3"),
        ("h3-only", CurlHttpVersion.V3ONLY, "http3_only"),
    ],
)
def test_protocol_configuration_mapping_or_explicit_unavailable(protocol, expected, capability):
    capabilities = RequestHandler(http_version="h1").capabilities
    if capability is not None and not getattr(capabilities, capability):
        with pytest.raises(ProtocolUnavailableError, match="(?i)(HTTP|h[23]|libcurl)"):
            RequestHandler(http_version=protocol)
    else:
        handler = RequestHandler(http_version=protocol)
        assert handler.http_version == protocol
        assert handler.curl_http_version == expected


@pytest.mark.parametrize("protocol", ["h2", "h3", "h3-only"])
def test_missing_protocol_capability_has_diagnostic_and_no_fallback(monkeypatch, protocol):
    capabilities = ProtocolCapabilities(
        libcurl_version="test build without multiplexed HTTP",
        http2=False,
        http3=False,
        http3_only=False,
    )
    monkeypatch.setattr(ProtocolCapabilities, "detect", classmethod(lambda cls: capabilities))
    with pytest.raises(ProtocolUnavailableError) as failure:
        RequestHandler(http_version=protocol)
    diagnostic = str(failure.value)
    assert protocol in diagnostic
    assert capabilities.libcurl_version in diagnostic


@pytest.mark.asyncio
async def test_h2_preference_reports_actual_h1_for_local_h1_server(server):
    # An H2 preference may negotiate H1; the result must report the wire protocol.
    capabilities = RequestHandler(http_version="h1").capabilities
    if not capabilities.http2:
        pytest.skip("local libcurl build has no HTTP/2 capability")
    async with RequestHandler(http_version="h2") as handler:
        result = await handler.fetch(_spec(server))
    assert result.success
    assert result.http_version == "h1"

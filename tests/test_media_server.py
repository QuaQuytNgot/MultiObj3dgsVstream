"""Real HTTP tests of the portable development media origin."""

from __future__ import annotations

from contextlib import contextmanager
import http.client
from pathlib import Path
import threading

import pytest

from src.server import create_server


PAYLOAD = b"0123456789abcdefghijklmnopqrstuvwxyz"


@contextmanager
def _running(root: Path):
    server = create_server(root, port=0)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def media(tmp_path):
    (tmp_path / "manifest.mpd").write_text("<MPD />", encoding="utf-8")
    (tmp_path / "longdress" / "media").mkdir(parents=True)
    (tmp_path / "longdress" / "media" / "segment.cpseg").write_bytes(PAYLOAD)
    (tmp_path / "empty.cpgs").touch()
    (tmp_path / "incomplete.cpseg.filepart").write_bytes(b"partial")
    (tmp_path / ".secret").write_bytes(b"hidden")
    with _running(tmp_path) as server:
        yield server


def _request(server, path="/longdress/media/segment.cpseg", *, method="GET", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_manifest_and_complete_segment(media):
    status, headers, body = _request(media, "/manifest.mpd?cache=1")
    assert status == 200
    assert headers["Content-Type"] == "application/dash+xml"
    assert body == b"<MPD />"
    status, headers, body = _request(media)
    assert status == 200
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Length"] == str(len(PAYLOAD))
    assert body == PAYLOAD


@pytest.mark.parametrize(
    ("value", "start", "end"),
    [("bytes=0-0", 0, 0), ("bytes=4-8", 4, 8), ("bytes=32-", 32, 35),
     ("bytes=-4", 32, 35), ("bytes=30-100", 30, 35), ("bytes=-100", 0, 35)],
)
def test_byte_ranges(media, value, start, end):
    status, headers, body = _request(media, headers={"Range": value})
    assert status == 206
    assert headers["Content-Range"] == f"bytes {start}-{end}/{len(PAYLOAD)}"
    assert headers["Content-Length"] == str(end - start + 1)
    assert body == PAYLOAD[start:end + 1]


@pytest.mark.parametrize("value", ["bytes=100-", "bytes=8-4", "bytes=-0", "bytes=-", "bytes=0-1,4-5", "items=0-1", "bytes=x-y"])
def test_unsatisfiable_ranges(media, value):
    status, headers, body = _request(media, headers={"Range": value})
    assert status == 416
    assert headers["Content-Range"] == f"bytes */{len(PAYLOAD)}"
    assert body == b""


def test_head_and_empty_file(media):
    status, headers, body = _request(media, method="HEAD", headers={"Range": "bytes=4-8"})
    assert status == 206
    assert headers["Content-Length"] == "5"
    assert body == b""
    assert _request(media, "/empty.cpgs")[2] == b""
    assert _request(media, "/empty.cpgs", headers={"Range": "bytes=0-0"})[0] == 416


def test_if_range_uses_last_modified(media):
    _, headers, _ = _request(media)
    assert _request(media, headers={"Range": "bytes=0-1", "If-Range": headers["Last-Modified"]})[0] == 206
    status, _, body = _request(media, headers={"Range": "bytes=0-1", "If-Range": "Wed, 01 Jan 2020 00:00:00 GMT"})
    assert status == 200
    assert body == PAYLOAD


@pytest.mark.parametrize("path", ["/", "/longdress/", "/missing", "/incomplete.cpseg.filepart", "/.secret", "/../outside", "/%2e%2e/outside", "/longdress/%2e%2e/%2e%2e/outside", "/bad%00name"])
def test_unavailable_and_unsafe_paths(media, path):
    assert _request(media, path)[0] == 404


def test_symlink_cannot_escape_root(media, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.write_bytes(b"outside root")
    try:
        (tmp_path / "escaped.cpgs").symlink_to(outside)
        assert _request(media, "/escaped.cpgs")[0] == 404
    finally:
        outside.unlink()


@pytest.mark.asyncio
async def test_existing_request_handler_accepts_media_range(media):
    from src.client.request_handler import RequestHandler, RequestSpec

    request = RequestSpec(
        request_id="range-smoke", url=f"http://127.0.0.1:{media.server_port}/longdress/media/segment.cpseg",
        byte_range=(4, 8),
    )
    async with RequestHandler(http_version="h1") as handler:
        result = await handler.fetch(request)
    assert result.success, result.error
    assert result.range_honored is True
    assert result.body == PAYLOAD[4:9]


def test_root_and_port_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_server(tmp_path / "missing", port=0)
    with pytest.raises(ValueError):
        create_server(tmp_path, port=-1)

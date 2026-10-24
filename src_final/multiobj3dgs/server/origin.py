"""Small HTTP/1.1 origin for existing MPDs and encoded media.

Only complete files below the configured root are exposed. A single byte range
is supported, including open-ended and suffix ranges; directory listings and
unfinished transfer files are never served. This development origin deliberately
does not claim HTTP/2 or HTTP/3 support.
"""

from __future__ import annotations

from datetime import timezone
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def _byte_range(value: str, size: int) -> tuple[int, int]:
    """Resolve an inclusive, single HTTP byte range or raise ValueError."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if match is None or not any(match.groups()) or size == 0:
        raise ValueError("unsupported or unsatisfiable byte range")
    start_text, end_text = match.groups()
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError("empty suffix range")
            return max(0, size - suffix), size - 1
        start = int(start_text)
        end = min(int(end_text), size - 1) if end_text else size - 1
    except ValueError:
        raise ValueError("invalid byte range") from None
    if start >= size or end < start:
        raise ValueError("unsatisfiable byte range")
    return start, end


class MediaHTTPServer(ThreadingHTTPServer):
    """Read-only HTTP origin; use ``create_server`` and ``serve_forever``.

    Port zero selects an available local port for integration tests. Call
    ``shutdown`` from another thread before ``server_close`` when serving.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, root: str | Path, host: str, port: int, *, pacer=None) -> None:
        self.pacer = pacer
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be a nonempty string")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer between 0 and 65535")
        super().__init__((host, port), _MediaHandler)


def create_server(
    root: str | Path, host: str = "127.0.0.1", port: int = 8080, *, pacer=None,
) -> MediaHTTPServer:
    """Create a bound server without starting a worker or changing the cwd."""
    return MediaHTTPServer(root, host, port, pacer=pacer)


class _MediaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: MediaHTTPServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def do_GET(self) -> None:
        self._serve_file(send_body=True)

    def do_HEAD(self) -> None:
        self._serve_file(send_body=False)

    def _resolve_file(self) -> Path | None:
        try:
            decoded = unquote(urlsplit(self.path).path, errors="strict")
            parts = decoded.split("/")
            if "\x00" in decoded or "\\" in decoded or ".." in parts:
                return None
            if any(part.startswith(".") and part not in ("", ".") for part in parts):
                return None
            path = self.server.root.joinpath(*(part for part in parts if part)).resolve()
            if not path.is_relative_to(self.server.root):
                return None
            if path.name.endswith((".filepart", ".part")) or not path.is_file():
                return None
            return path
        except (ValueError, UnicodeDecodeError, OSError, RuntimeError):
            return None

    def _serve_file(self, *, send_body: bool) -> None:
        path = self._resolve_file()
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Complete media file not found")
            return
        try:
            source = path.open("rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Media file unavailable")
            return
        with source:
            # fstat keeps Content-Length tied to the open descriptor.
            stat = os.fstat(source.fileno())
            size = stat.st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            if range_header and self._if_range_matches(stat.st_mtime):
                try:
                    start, end = _byte_range(range_header, size)
                except ValueError:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
            else:
                status = HTTPStatus.OK
            length = max(0, end - start + 1)
            content_type = {
                ".mpd": "application/dash+xml",
                ".cpgs": "application/octet-stream",
                ".cpseg": "application/octet-stream",
            }.get(path.suffix.lower(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not send_body:
                return
            if self.server.pacer:
                self.server.pacer.latency()
            source.seek(start)
            remaining = length
            try:
                while remaining:
                    chunk = source.read(min(16384 if self.server.pacer else 1024 * 1024, remaining))
                    if not chunk:
                        self.close_connection = True
                        break
                    if self.server.pacer:
                        self.server.pacer.pace(len(chunk))
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                self.close_connection = True

    def _if_range_matches(self, modified: float) -> bool:
        value = self.headers.get("If-Range")
        if value is None:
            return True
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return int(modified) <= int(date.timestamp())
        except (ValueError, TypeError, OverflowError):
            # No ETag is advertised, so an entity tag cannot match.
            return False

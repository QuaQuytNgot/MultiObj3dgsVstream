#!/usr/bin/env python3
"""Bounded, resumable downloads of official 8i point-cloud ZIP members.

No content-preparation or training stage is invoked. HTTPS verification remains
on. HTTP is considered only with --allow-http-fallback after a certificate
verification failure, and the resulting provenance states its limitations.

Examples:
  python tools/download_8i_object.py --object longdress --frames 1051:1080 --dry-run
  python tools/download_8i_object.py --object soldier --all-frames --output output/datasets/8i
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import ssl
import struct
import tempfile
import time
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
HOST = "plenodb.jpeg.org"
BASE_URL = f"https://{HOST}/pc/8ilabs/"
OBJECTS = ("longdress", "soldier", "loot")
CHUNK_BYTES = 1024*1024
MAX_RANGE_BYTES = 8*1024*1024
MAX_LICENSE_BYTES = 4*1024*1024
TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}
INVENTORY_SCHEMA = "official-8i-download-v1"


class DownloadError(RuntimeError):
    pass


class RangeUnsupported(DownloadError):
    pass


class ArchiveChanged(DownloadError):
    pass


class TruncatedResponse(DownloadError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def verified_tls_opener(ca_file=None):
    """Use verified TLS, optionally with a configured/system CA bundle."""
    if ca_file is None and not os.environ.get("SSL_CERT_FILE"):
        system_bundle = Path("/etc/ssl/certs/ca-certificates.crt")
        if system_bundle.is_file():
            ca_file = system_bundle
    context = ssl.create_default_context(cafile=None if ca_file is None else str(ca_file))
    def opener(request, *, timeout):
        return urllib.request.urlopen(request, timeout=timeout, context=context)
    return opener


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(data)
    return digest.hexdigest()


def file_integrity(path):
    digest, crc, count = hashlib.sha256(), 0, 0
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(data)
            crc = zlib.crc32(data, crc)
            count += len(data)
    return {"bytes": count, "sha256": digest.hexdigest(), "zip_crc32": f"{crc & 0xffffffff:08x}"}


def _cert_failure(exc):
    reason = exc.reason if isinstance(exc, URLError) else exc
    return isinstance(reason, ssl.SSLCertVerificationError)


def _retryable(exc):
    if _cert_failure(exc) or isinstance(exc, (ArchiveChanged, RangeUnsupported)):
        return False
    if isinstance(exc, HTTPError):
        return exc.code in TRANSIENT_HTTP
    return isinstance(exc, (URLError, TimeoutError, socket.timeout, ConnectionError,
                            http.client.IncompleteRead, TruncatedResponse, OSError))


def _status(response):
    return getattr(response, "status", None) or response.getcode()


def _verify_origin(response, url):
    final = response.geturl() if hasattr(response, "geturl") else url
    before, after = urlsplit(url), urlsplit(final)
    if after.hostname != HOST or after.scheme != before.scheme:
        raise DownloadError(f"Unexpected redirect to {final!r}; refusing an origin change or transport downgrade.")
    return final


def _length(headers, *, required=False):
    value = headers.get("Content-Length")
    if value is None:
        if required:
            raise DownloadError("Official source response lacks Content-Length.")
        return None
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise DownloadError("Invalid Content-Length in official source response.") from exc
    if result < 0:
        raise DownloadError("Negative Content-Length in official source response.")
    return result


def _open_with_retry(opener, request, timeout, retries, counters, retry_delay):
    for attempt in range(retries+1):
        counters["requests"] += 1
        try:
            return opener(request, timeout=timeout)
        except Exception as exc:
            if attempt == retries or not _retryable(exc):
                raise
            time.sleep(min(retry_delay*2**attempt, 8.))


class RangeReader(io.RawIOBase):
    """Seekable HTTP ZIP input with bounded reads and pinned archive identity.

    A 200 response to a Range request is rejected before reading its body. Every
    206 must identify exactly the requested byte interval and pinned ETag/size.
    Retries restart only the current range, never commit partial member files,
    and count all received bytes, including discarded truncated attempts.
    """
    def __init__(self, url, *, timeout=60, retries=3, opener=None, retry_delay=.5,
                 max_range_bytes=MAX_RANGE_BYTES):
        super().__init__()
        self.url = url
        self.pos = 0
        self.transferred = 0
        self.failed_attempt_bytes = 0
        self.counters = {"requests": 0, "ranges": 0, "cache_hits": 0,
                         "prefetched_members": 0, "prefetched_bytes": 0}
        self._member_cache = None
        self.timeout, self.retries, self.retry_delay = float(timeout), int(retries), float(retry_delay)
        self.max_range_bytes = int(max_range_bytes)
        if self.timeout <= 0 or self.retries < 0 or self.max_range_bytes <= 0 or self.retry_delay < 0:
            raise ValueError("Timeout/max_range_bytes must be positive; retries/retry_delay nonnegative.")
        self.opener = opener or verified_tls_opener()
        request = urllib.request.Request(url, method="HEAD", headers={"Accept-Encoding": "identity"})
        try:
            with _open_with_retry(self.opener, request, self.timeout, self.retries, self.counters, self.retry_delay) as response:
                _verify_origin(response, url)
                if _status(response) != 200:
                    raise DownloadError(f"Unexpected HEAD status {_status(response)}.")
                self.length = _length(response.headers, required=True)
                self.etag = response.headers.get("ETag")
                self.last_modified = response.headers.get("Last-Modified")
        except HTTPError as exc:
            if exc.code not in {405, 501}:
                raise
            # Some otherwise range-capable servers reject HEAD. A one-byte
            # probe supplies the same identity metadata without fetching ZIP data.
            request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"})
            with _open_with_retry(self.opener, request, self.timeout, self.retries, self.counters, self.retry_delay) as response:
                _verify_origin(response, url)
                if _status(response) != 206:
                    raise RangeUnsupported("Server does not support byte ranges; refusing a full ZIP download.")
                match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
                if not match:
                    raise DownloadError("One-byte probe returned an invalid Content-Range.")
                self.length = int(match[1])
                self.etag = response.headers.get("ETag")
                self.last_modified = response.headers.get("Last-Modified")
                probe = response.read(2)
                self.transferred += len(probe)
                if len(probe) != 1:
                    raise TruncatedResponse("One-byte archive probe length mismatch.")
                self.counters["ranges"] += 1
        if self.length < 22:
            raise DownloadError("Remote file is too small to be a ZIP archive.")
        if not self.etag:
            raise DownloadError("Archive ETag is unavailable; refusing an unpinned resumable archive. No full ZIP was downloaded.")

    @property
    def requests(self):
        return self.counters["requests"]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence not in {io.SEEK_SET, io.SEEK_CUR, io.SEEK_END}:
            raise ValueError("Unknown seek origin.")
        new_pos = int(offset)+(0 if whence == io.SEEK_SET else self.pos if whence == io.SEEK_CUR else self.length)
        if not 0 <= new_pos <= self.length:
            raise ValueError("Remote ZIP seek is outside archive bounds.")
        self.pos = new_pos
        return self.pos

    def read(self, size=-1):
        self._checkClosed()
        size = int(size)
        if size < -1:
            raise ValueError("Read size must be nonnegative or -1.")
        available = self.length-self.pos
        size = available if size == -1 else min(size, available)
        if size == 0:
            return b""
        if self._member_cache is not None:
            cache_start, cache_data = self._member_cache
            if cache_start <= self.pos and self.pos+size <= cache_start+len(cache_data):
                offset = self.pos-cache_start
                self.pos += size
                self.counters["cache_hits"] += 1
                return cache_data[offset:offset+size]
        if size > self.max_range_bytes:
            raise RangeUnsupported(f"Refusing a {size:,}-byte range read. Stream selected ZIP members in chunks <= {self.max_range_bytes:,}; full archive downloads are disabled.")
        start, end = self.pos, self.pos+size-1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if not self.etag.startswith("W/"):
            headers["If-Match"] = self.etag
        elif self.last_modified:
            headers["If-Unmodified-Since"] = self.last_modified
        request = urllib.request.Request(self.url, headers=headers)
        for attempt in range(self.retries+1):
            received_this_attempt = 0
            try:
                # Opening retries are governed by this encompassing attempt so
                # truncated reads and connection failures share the same budget.
                self.counters["requests"] += 1
                with self.opener(request, timeout=self.timeout) as response:
                    _verify_origin(response, self.url)
                    if _status(response) != 206:
                        raise RangeUnsupported("Server ignored Range (expected HTTP 206); refusing to read a full ZIP response body.")
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.length}":
                        raise ArchiveChanged("Content-Range does not match the requested interval/pinned archive size.")
                    if response.headers.get("ETag") != self.etag:
                        raise ArchiveChanged("Archive ETag changed during download; use a new inventory/output after inspecting the source.")
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise DownloadError("Range responses must not use HTTP content encoding.")
                    declared = _length(response.headers)
                    if declared is not None and declared != size:
                        raise TruncatedResponse("Range Content-Length differs from requested interval.")
                    result = bytearray()
                    while len(result) < size:
                        block = response.read(min(CHUNK_BYTES, size-len(result)))
                        if not block:
                            raise TruncatedResponse("Range response ended before the requested interval was complete.")
                        self.transferred += len(block)
                        received_this_attempt += len(block)
                        if len(block) > size-len(result):
                            raise DownloadError("Range response returned bytes beyond its requested interval.")
                        result.extend(block)
                    self.counters["ranges"] += 1
                    self.pos += size
                    return bytes(result)
            except HTTPError as exc:
                self.failed_attempt_bytes += received_this_attempt
                if exc.code == 412:
                    raise ArchiveChanged("Archive conditional request failed; pinned ETag changed.") from exc
                if attempt == self.retries or not _retryable(exc):
                    raise
                time.sleep(min(self.retry_delay*2**attempt, 8.))
            except Exception as exc:
                self.failed_attempt_bytes += received_this_attempt
                if attempt == self.retries or not _retryable(exc):
                    raise
                time.sleep(min(self.retry_delay*2**attempt, 8.))
        raise AssertionError("Unreachable range retry state")

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def clear_member_cache(self):
        """Release the one-member compressed-byte cache immediately after use."""
        self._member_cache = None

    def prefetch_member(self, member, max_bytes=MAX_RANGE_BYTES):
        """Cache exactly one selected ZIP local header + compressed payload.

        Two pinned range requests suffice for the usual 5-6 MiB 8i member:
        its fixed 30-byte local header, then name/extra/compressed bytes. The
        normal ZipExtFile still performs decompression and CRC verification;
        all of its small reads are served from this bounded input cache. Large
        members retain the streaming path, without a full-archive request.
        """
        max_bytes = int(max_bytes)
        if max_bytes < 0 or max_bytes > self.max_range_bytes:
            raise ValueError(f"Prefetch limit must lie in [0,{self.max_range_bytes}].")
        self.clear_member_cache()
        if max_bytes == 0 or member.compress_size+30 > max_bytes:
            return False
        previous_position = self.pos
        try:
            self.seek(member.header_offset)
            fixed_header = self.read(30)
            if len(fixed_header) != 30:
                raise DownloadError("Selected ZIP member has a truncated local header.")
            (signature, version, flags, method, mtime, mdate, crc,
             compressed_size, uncompressed_size, name_length, extra_length) = struct.unpack("<4s5H3I2H", fixed_header)
            if signature != b"PK\x03\x04" or method != member.compress_type or flags & 1:
                raise DownloadError("Selected ZIP member local header is incompatible with its central directory.")
            if not flags & 8:
                if crc != member.CRC or compressed_size not in {member.compress_size, 0xffffffff} or uncompressed_size not in {member.file_size, 0xffffffff}:
                    raise DownloadError("Selected ZIP member local CRC/size differs from its central directory.")
            total = 30+name_length+extra_length+member.compress_size
            if total > max_bytes:
                return False
            end = member.header_offset+total
            member_end = getattr(member, "_end_offset", None)
            if end > self.length or member_end is not None and member_end < end:
                raise DownloadError("Selected ZIP compressed payload overlaps another member or exceeds archive bounds.")
            remainder = self.read(total-30)
            encoding = "utf-8" if flags & 0x800 else "cp437"
            try:
                local_name = remainder[:name_length].decode(encoding)
            except UnicodeDecodeError as exc:
                raise DownloadError("Invalid selected ZIP local filename encoding.") from exc
            if local_name != member.orig_filename:
                raise DownloadError("Selected ZIP local filename differs from its central directory.")
            self._member_cache = (member.header_offset, fixed_header+remainder)
            self.counters["prefetched_members"] += 1
            self.counters["prefetched_bytes"] += total
            return True
        finally:
            self.seek(previous_position)


def parse_frames(value):
    """Inclusive integer range (START:END) or a comma-separated selection."""
    if re.fullmatch(r"\d+:\d+", value):
        start, end = map(int, value.split(":"))
        if end < start:
            raise argparse.ArgumentTypeError("Frame range end must be >= start.")
        return list(range(start, end+1))
    if re.fullmatch(r"\d+(,\d+)*", value):
        frames = [int(frame) for frame in value.split(",")]
        if len(set(frames)) != len(frames):
            raise argparse.ArgumentTypeError("Frame IDs must be unique.")
        return sorted(frames)
    raise argparse.ArgumentTypeError("Use inclusive START:END or comma-separated integer frame IDs.")


def _archive_members(archive, object_name):
    pattern = re.compile(rf"{re.escape(object_name)}_vox\d+_(\d+)\.ply", re.IGNORECASE)
    members = {}
    for member in archive.infolist():
        parts = PurePosixPath(member.filename).parts
        if not parts or "__MACOSX" in parts or member.is_dir():
            continue
        match = pattern.fullmatch(parts[-1])
        if not match:
            continue
        if member.flag_bits & 1:
            raise DownloadError("Encrypted ZIP members are unsupported.")
        if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}:
            raise DownloadError(f"Unsupported member compression method: {member.compress_type}")
        frame = int(match[1])
        if frame in members:
            raise DownloadError(f"Archive has duplicate frame ID {frame}: ambiguous point-cloud members.")
        members[frame] = member
    if not members:
        raise DownloadError(f"No {object_name} PLY frame members in official ZIP.")
    return dict(sorted(members.items()))


def _source_allowed(url, object_name):
    parsed = urlsplit(url)
    return parsed.scheme in {"https", "http"} and parsed.hostname == HOST and parsed.path == f"/pc/8ilabs/{object_name}.zip" and not parsed.query and not parsed.fragment


def _load_cache(cache, object_name):
    if cache is None:
        return None
    root = Path(cache).expanduser().resolve()
    candidates = [root/"provenance.json", root/object_name/"inventory.json", root/"inventory.json"]
    for path in candidates:
        if not path.is_file():
            continue
        document = json.loads(path.read_text())
        if not _source_allowed(document.get("source_url", ""), object_name):
            continue
        return {"root": root, "path": path, "document": document, "sha256": sha256(path)}
    return None


def _cached_path(cache, record):
    raw = Path(record["path"])
    path = raw.resolve() if raw.is_absolute() else (cache["path"].parent/raw).resolve()
    if not path.is_relative_to(cache["root"]):
        raise DownloadError("Cache provenance points outside its specified cache directory.")
    return path


def _matching_cache_frame(cache, frame, member, reader):
    if cache is None:
        return None
    document = cache["document"]
    archive = document.get("archive", {})
    length = document.get("archive_bytes", archive.get("bytes"))
    etag = document.get("archive_etag", archive.get("etag"))
    if length is not None and length != reader.length or etag is not None and etag != reader.etag:
        return None
    record = next((r for r in document.get("frames", []) if r["frame"] == frame), None)
    if not record:
        return None
    if record.get("member") != member.filename or record.get("bytes") != member.file_size or record.get("zip_crc32", "").lower() != f"{member.CRC:08x}":
        return None
    path = _cached_path(cache, record)
    if not path.is_file():
        return None
    integrity = file_integrity(path)
    if integrity != {"bytes": record["bytes"], "sha256": record.get("sha256"), "zip_crc32": record["zip_crc32"].lower()}:
        raise DownloadError(f"Cached frame fails prior SHA256/CRC/size provenance: {path}")
    return {"path": path, "integrity": integrity, "provenance": cache}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".", suffix=".tmp", mode="w", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _output_lock(output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output/".download.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DownloadError(f"Another dataset download uses {output}; wait for that invocation.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _copy_atomic(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name+".", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with Path(source).open("rb") as origin:
                shutil.copyfileobj(origin, stream, CHUNK_BYTES)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if file_integrity(temporary) != expected:
            raise DownloadError("Cached member changed while being imported.")
        if target.exists():
            if file_integrity(target) != expected:
                raise FileExistsError(f"Preserving existing artifact with different bytes: {target}")
        else:
            os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_atomic(archive, member, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name+".", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        digest, crc, count = hashlib.sha256(), 0, 0
        try:
            with archive.open(member) as origin:
                while True:
                    data = origin.read(CHUNK_BYTES)
                    if not data:
                        break
                    stream.write(data)
                    digest.update(data)
                    crc = zlib.crc32(data, crc)
                    count += len(data)
                    if count > member.file_size:
                        raise DownloadError("ZIP member exceeds declared uncompressed size.")
            if count != member.file_size or crc & 0xffffffff != member.CRC:
                raise DownloadError("ZIP member CRC/uncompressed size verification failed.")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    integrity = {"bytes": count, "sha256": digest.hexdigest(), "zip_crc32": f"{crc & 0xffffffff:08x}"}
    try:
        if target.exists():
            if file_integrity(target) != integrity:
                raise FileExistsError(f"Preserving existing artifact with different bytes: {target}")
        else:
            os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return integrity


class _ParallelMembers:
    """Network-only workers; each thread owns its pinned ZIP reader/cache."""
    def __init__(self, source, *, workers, opener, timeout, retries, retry_delay, prefetch_bytes):
        self.url, self.etag, self.length = source.url, source.etag, source.length
        self.workers, self.opener = workers, opener
        self.timeout, self.retries, self.retry_delay = timeout, retries, retry_delay
        self.prefetch_bytes = prefetch_bytes
        self.local = threading.local()
        self.lock = threading.Lock()
        self.contexts = []
        self.active, self.peak_active = 0, 0
        self.peak_pending = 0

    def _context(self):
        context = getattr(self.local, "context", None)
        if context is not None:
            return context
        source = RangeReader(self.url, opener=self.opener, timeout=self.timeout,
                             retries=self.retries, retry_delay=self.retry_delay)
        context = {"reader": source, "archive": None, "metadata_bytes": 0}
        with self.lock:
            self.contexts.append(context)
        try:
            if source.etag != self.etag or source.length != self.length:
                raise ArchiveChanged("Worker archive identity differs from the initial pinned ETag/size.")
            context["archive"] = zipfile.ZipFile(source)
            context["metadata_bytes"] = source.transferred
            self.local.context = context
            return context
        except BaseException:
            context["metadata_bytes"] = source.transferred
            source.close()
            raise

    def _download(self, job):
        frame, expected, target = job
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        context = None
        try:
            context = self._context()
            reader, archive = context["reader"], context["archive"]
            member = archive.getinfo(expected.filename)
            if any(getattr(member, name) != getattr(expected, name)
                   for name in ("CRC", "file_size", "compress_size", "compress_type", "header_offset")):
                raise ArchiveChanged("Worker ZIP member differs from the initial pinned central directory.")
            reader.prefetch_member(member, self.prefetch_bytes)
            integrity = _extract_atomic(archive, member, target)
            return frame, expected, target, integrity
        finally:
            if context is not None:
                context["reader"].clear_member_cache()
            with self.lock:
                self.active -= 1

    def statistics(self):
        with self.lock:
            contexts = list(self.contexts)
            peak_active, peak_pending = self.peak_active, self.peak_pending
        result = {"transferred": 0, "metadata_bytes": 0, "requests": 0,
                  "failed_attempt_bytes": 0, "prefetched_members": 0,
                  "prefetched_bytes": 0, "cache_hits": 0,
                  "peak_active": peak_active, "peak_pending": peak_pending}
        for context in contexts:
            source = context["reader"]
            result["transferred"] += source.transferred
            result["metadata_bytes"] += context["metadata_bytes"]
            result["requests"] += source.requests
            result["failed_attempt_bytes"] += source.failed_attempt_bytes
            for name in ("prefetched_members", "prefetched_bytes", "cache_hits"):
                result[name] += source.counters[name]
        return result

    def run(self, jobs, on_result):
        """Bound pending work; only the calling thread invokes on_result."""
        iterator = iter(jobs)
        executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="8i-network")
        pending = {}
        failure = None

        def submit_one():
            job = next(iterator, None)
            if job is None:
                return False
            pending[executor.submit(self._download, job)] = job
            self.peak_pending = max(self.peak_pending, len(pending))
            return True

        try:
            for _ in range(self.workers):
                if not submit_one():
                    break
            while pending and failure is None:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Accept successful completions before propagating a failure in
                # the same set. Inventory frame ordering remains chronological.
                for future in sorted(done, key=lambda f: pending[f][0]):
                    pending.pop(future)
                    try:
                        on_result(*future.result())
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
                if failure is None:
                    for _ in done:
                        if not submit_one():
                            break
        except BaseException as exc:
            failure = exc
        finally:
            if failure is not None:
                for future in pending:
                    future.cancel()
            executor.shutdown(wait=True, cancel_futures=failure is not None)
            # A running request may commit atomically after another worker fails.
            # Checkpoint every such success on the main thread before rethrowing.
            for future, job in sorted(pending.items(), key=lambda item: item[1][0]):
                if future.cancelled():
                    continue
                try:
                    on_result(*future.result())
                except BaseException as exc:
                    if failure is None:
                        failure = exc
            for context in self.contexts:
                if context["archive"] is not None:
                    context["archive"].close()
                context["reader"].close()
        if failure is not None:
            raise failure


def _license_cache(cache):
    if cache is None:
        return None
    document = cache["document"]
    record = document.get("license", {})
    if isinstance(record, str):
        record = {"path": record, "sha256": document.get("license_sha256")}
    if not record.get("path") or not record.get("sha256"):
        return None
    path = _cached_path(cache, record)
    if not path.is_file():
        return None
    if sha256(path) != record["sha256"]:
        raise DownloadError("Cached license fails its prior SHA256 provenance.")
    if not path.read_bytes().startswith(b"%PDF-"):
        raise DownloadError("Cached license is not a PDF.")
    return path, record["sha256"]


def _download_license(output, cache, allow_http_fallback, opener, timeout, retries, retry_delay):
    target = output/"license.pdf"
    previous_record = output/"license.inventory.json"
    cached = _license_cache(cache)
    if cached:
        source, digest = cached
        expected = file_integrity(source)
        _copy_atomic(source, target, expected)
        record = {"path": "../license.pdf", "bytes": target.stat().st_size, "sha256": digest,
                "source_url": BASE_URL+"license.pdf", "acquisition": "verified_local_cache",
                "cache_provenance_sha256": cache["sha256"],
                "transport": cache["document"].get("transport", cache["document"].get("transport_provenance")),
                "publisher_signature_verified": False}
        if not previous_record.exists():
            _atomic_json(previous_record, record)
        return record, 0, 0
    if target.exists():
        if not previous_record.exists():
            raise FileExistsError(f"Preserving untracked existing license: {target}. Supply a cache with verified license provenance.")
        record = json.loads(previous_record.read_text())
        if sha256(target) != record["sha256"] or target.stat().st_size != record["bytes"]:
            raise DownloadError("Existing license fails recorded SHA256/size provenance.")
        return dict(record, path="../license.pdf", acquisition="resumed_verified_inventory"), 0, 0
    counters = {"requests": 0}
    transferred = 0
    url = BASE_URL+"license.pdf"
    transport = "HTTPS with certificate verification"
    fallback_reason = None
    try:
        response = _open_with_retry(opener, urllib.request.Request(url, headers={"Accept-Encoding": "identity"}), timeout, retries, counters, retry_delay)
    except Exception as exc:
        if not allow_http_fallback or not _cert_failure(exc):
            raise
        fallback_reason = str(exc)
        url = url.replace("https://", "http://", 1)
        transport = "HTTP explicitly allowed after HTTPS certificate verification failure"
        response = _open_with_retry(opener, urllib.request.Request(url, headers={"Accept-Encoding": "identity"}), timeout, retries, counters, retry_delay)
    with response:
        _verify_origin(response, url)
        if _status(response) != 200:
            raise DownloadError("License response is not HTTP 200.")
        declared = _length(response.headers)
        if declared is not None and declared > MAX_LICENSE_BYTES:
            raise DownloadError("License response exceeds the bounded 4 MiB PDF limit.")
        data = bytearray()
        while True:
            block = response.read(min(CHUNK_BYTES, MAX_LICENSE_BYTES+1-len(data)))
            if not block:
                break
            data.extend(block)
            transferred += len(block)
            if len(data) > MAX_LICENSE_BYTES:
                raise DownloadError("License response exceeds the bounded 4 MiB PDF limit.")
        if declared is not None and len(data) != declared:
            raise TruncatedResponse("License response is truncated.")
        if not data.startswith(b"%PDF-"):
            raise DownloadError("Official license response is not a PDF.")
    with tempfile.NamedTemporaryFile(dir=output, prefix="license.", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    record = {"path": "../license.pdf", "bytes": len(data), "sha256": sha256(target),
              "source_url": url, "acquisition": "official_host_download", "transport": transport,
              "https_certificate_failure": fallback_reason, "publisher_signature_verified": False}
    _atomic_json(previous_record, record)
    return record, transferred, counters["requests"]


def _open_archive(object_name, allow_http_fallback, opener, timeout, retries, retry_delay):
    url = BASE_URL+f"{object_name}.zip"
    fallback_reason = None
    try:
        reader = RangeReader(url, opener=opener, timeout=timeout, retries=retries, retry_delay=retry_delay)
        archive = zipfile.ZipFile(reader)
    except Exception as exc:
        if "reader" in locals():
            reader.close()
        if not allow_http_fallback or not _cert_failure(exc):
            raise
        fallback_reason = str(exc)
        url = url.replace("https://", "http://", 1)
        reader = RangeReader(url, opener=opener, timeout=timeout, retries=retries, retry_delay=retry_delay)
        try:
            archive = zipfile.ZipFile(reader)
        except BaseException:
            reader.close()
            raise
    return reader, archive, fallback_reason


def _existing_inventory(path, object_name, reader):
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    if document.get("schema") != INVENTORY_SCHEMA or document.get("object") != object_name:
        raise DownloadError("Existing inventory is incompatible; preserving it.")
    if document["archive"]["bytes"] != reader.length or document["archive"]["etag"] != reader.etag:
        raise ArchiveChanged("Official archive differs from existing inventory; preserving artifacts. Inspect the source before choosing another output directory.")
    for record in document.get("frames", []):
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise DownloadError("Existing inventory frame paths must be relative.")
        target = path.parent/relative
        if file_integrity(target) != {name: record[name] for name in ("bytes", "sha256", "zip_crc32")}:
            raise DownloadError(f"Existing frame fails resume integrity/provenance: {target}")
    return document


def download_object(object_name, frames=None, *, all_frames=False, output=None, cache=None,
                    dry_run=False, allow_http_fallback=False, timeout=60, retries=3,
                    opener=None, retry_delay=.5, progress=None, ca_file=None,
                    prefetch_bytes=MAX_RANGE_BYTES, download_workers=1):
    """Inspect/download selected official ZIP frames; return a JSON-safe inventory.

    Dry-run performs HEAD/central-directory ranges and local integrity checks,
    creates no outputs, and fetches neither frame member data nor the license.
    The ``opener`` dependency is injectable for deterministic no-network tests.
    """
    object_name = object_name.lower()
    if object_name not in OBJECTS:
        raise ValueError(f"Object must be one of {', '.join(OBJECTS)}.")
    if bool(all_frames) == (frames is not None):
        raise ValueError("Specify exactly one of frames or all_frames=True.")
    if frames is not None and (not frames or any(not isinstance(f, int) or isinstance(f, bool) or f < 0 for f in frames) or len(set(frames)) != len(frames)):
        raise ValueError("Frame selection must contain unique nonnegative integer IDs.")
    if not isinstance(prefetch_bytes, int) or isinstance(prefetch_bytes, bool) or not 0 <= prefetch_bytes <= MAX_RANGE_BYTES:
        raise ValueError(f"prefetch_bytes must be an integer in [0,{MAX_RANGE_BYTES}].")
    if not isinstance(download_workers, int) or isinstance(download_workers, bool) or not 1 <= download_workers <= 8:
        raise ValueError("download_workers must be an integer in [1,8]; network concurrency never changes GPU execution.")
    output = Path(output or ROOT/"output/datasets/8i").expanduser().resolve()
    cache = _load_cache(cache, object_name)
    opener = opener or verified_tls_opener(ca_file)
    reader, archive, fallback_reason = _open_archive(object_name, allow_http_fallback, opener, timeout, retries, retry_delay)
    try:
        members = _archive_members(archive, object_name)
        selected = list(members) if all_frames else sorted(frames)
        absent = set(selected)-set(members)
        if absent:
            raise DownloadError(f"Requested frames are missing from official archive: {sorted(absent)}")
        inventory_path = output/object_name/"inventory.json"
        previous = _existing_inventory(inventory_path, object_name, reader)
        existing = {r["frame"]: r for r in previous.get("frames", [])} if previous else {}
        cached = {frame: _matching_cache_frame(cache, frame, members[frame], reader) for frame in selected if frame not in existing}
        metadata_bytes = reader.transferred
        plan_rows = [{"frame": frame, "member": members[frame].filename, "bytes": members[frame].file_size,
                      "compressed_bytes": members[frame].compress_size, "zip_crc32": f"{members[frame].CRC:08x}",
                      "acquisition": "resume_inventory" if frame in existing else "verified_cache" if cached.get(frame) else "download",
                      "estimated_member_range_requests": 0 if frame in existing or cached.get(frame) else 2 if prefetch_bytes and members[frame].compress_size+30+len(members[frame].filename.encode("utf-8"))+len(members[frame].extra) <= prefetch_bytes else 2+math.ceil(members[frame].compress_size/CHUNK_BYTES)} for frame in selected]
        plan = {"dry_run": bool(dry_run), "object": object_name, "source_url": reader.url,
                "preferred_source_url": BASE_URL+f"{object_name}.zip", "archive_bytes": reader.length,
                "archive_etag": reader.etag, "archive_frame_ids": list(members), "archive_frame_count": len(members),
                "selected_frame_ids": selected, "selected_frame_count": len(selected), "frames": plan_rows,
                "download_compressed_bytes_estimate": sum(r["compressed_bytes"] for r in plan_rows if r["acquisition"] == "download"),
                "member_range_requests_estimate": sum(r["estimated_member_range_requests"] for r in plan_rows),
                "metadata_transferred_bytes": metadata_bytes, "metadata_requests": reader.requests,
                "member_prefetch_limit_bytes": prefetch_bytes,
                "download_workers": download_workers,
                "license_requested": False, "output": str(output),
                "request_estimate_note": "Bounded member prefetch normally needs two ranges. Estimates exclude metadata/retries/license; local header extras and streaming decompression can change request counts.",
                "transport_provenance": {"https_certificate_verified": urlsplit(reader.url).scheme == "https",
                                         "explicit_http_fallback": fallback_reason is not None,
                                         "https_certificate_failure": fallback_reason,
                                         "authenticity_limit": "ZIP CRC and local SHA256 establish corruption detection/reproducibility; no publisher cryptographic signature is verified. HTTP additionally lacks authenticated encrypted transport."}}
        if dry_run:
            return plan
        with _output_lock(output):
            # Recheck after acquiring the process lock; metadata planning itself
            # is read-only and may safely occur before another invocation ends.
            previous = _existing_inventory(inventory_path, object_name, reader)
            known = {r["frame"]: r for r in previous.get("frames", [])} if previous else {}
            license_record, license_bytes, license_requests = _download_license(output, cache, allow_http_fallback, opener, timeout, retries, retry_delay)
            invocation_start = utc_now()
            previous_total = previous.get("transfer", {}).get("cumulative_received_bytes", 0) if previous else 0
            pool = _ParallelMembers(reader, workers=download_workers, opener=opener, timeout=timeout,
                                    retries=retries, retry_delay=retry_delay, prefetch_bytes=prefetch_bytes) if download_workers > 1 else None
            document = {"schema": INVENTORY_SCHEMA, "object": object_name, "source_url": reader.url,
                        "preferred_source_url": BASE_URL+f"{object_name}.zip",
                        "archive": {"bytes": reader.length, "etag": reader.etag, "last_modified": reader.last_modified,
                                    "frame_ids": list(members), "frame_count": len(members)},
                        "capture": {"fps": 30., "fps_basis": "official_JPEG_8i_dataset_description; not inferred from ZIP timestamps",
                                    "description_url": "https://plenodb.jpeg.org/pc/8ilabs",
                                    "sample_count": len(members), "samples_duration_seconds": len(members)/30.,
                                    "timestamp_span_seconds": (max(members)-min(members))/30.,
                                    "duration_convention": "N samples / FPS; last-minus-first timestamp span is separately reported"},
                        "selection": {"requested_frame_ids": selected, "all_archive_frames": bool(all_frames),
                                      "sample_count": len(selected), "samples_duration_seconds": len(selected)/30.,
                                      "timestamp_span_seconds": (max(selected)-min(selected))/30.},
                        "frames": list(known.values()), "license": license_record,
                        "created_at_utc": previous["created_at_utc"] if previous else invocation_start,
                        "updated_at_utc": invocation_start, "transport_provenance": plan["transport_provenance"],
                        "tool": {"path": "tools/download_8i_object.py", "sha256": sha256(__file__)},
                        "download_settings": {"member_prefetch_limit_bytes": prefetch_bytes,
                                              "concurrent_members": download_workers,
                                              "gpu_work": False},
                        "cache": None if cache is None else {"provenance_sha256": cache["sha256"],
                                  "source_url": cache["document"]["source_url"],
                                  "transport": cache["document"].get("transport", cache["document"].get("transport_provenance"))}}

            def checkpoint(complete):
                worker = pool.statistics() if pool else {name: 0 for name in (
                    "transferred", "metadata_bytes", "requests", "failed_attempt_bytes", "prefetched_members",
                    "prefetched_bytes", "cache_hits", "peak_active", "peak_pending")}
                received = reader.transferred+worker["transferred"]+license_bytes
                combined_metadata = metadata_bytes+worker["metadata_bytes"]
                document["frames"] = [known[f] for f in sorted(known)]
                document["complete_for_selection"] = complete
                document["updated_at_utc"] = utc_now()
                document["transfer"] = {"this_invocation_received_bytes": received,
                                        "accounting_scope": "Observed HTTP response-body bytes persisted in invocation checkpoints; cumulative_received_bytes sums persisted invocation counters. Traffic after the last checkpoint on hard kill or older interrupted versions may be omitted. HTTP headers/TLS overhead are excluded; this is not an exact wire-byte total. Verified payload file sizes/SHA256/ZIP CRC are accounted independently.",
                                        "verified_payload_files_bytes": sum(record["bytes"] for record in known.values()),
                                        "metadata_bytes": combined_metadata, "member_received_bytes": received-combined_metadata-license_bytes,
                                        "license_bytes": license_bytes, "failed_range_attempt_bytes": reader.failed_attempt_bytes+worker["failed_attempt_bytes"],
                                        "http_requests": reader.requests+worker["requests"]+license_requests,
                                        "prefetched_members": reader.counters["prefetched_members"]+worker["prefetched_members"],
                                        "prefetched_bytes": reader.counters["prefetched_bytes"]+worker["prefetched_bytes"],
                                        "member_cache_hits": reader.counters["cache_hits"]+worker["cache_hits"],
                                        "cumulative_received_bytes": previous_total+received}
                document["download_settings"]["peak_concurrent_network_members"] = worker["peak_active"] if pool else 1 if reader.counters["prefetched_members"] else 0
                document["download_settings"]["peak_pending_jobs"] = worker["peak_pending"] if pool else 1 if reader.counters["prefetched_members"] else 0
                _atomic_json(inventory_path, document)

            def record_frame(frame, member, target, integrity, acquisition="official_zip_range_download", from_cache=None):
                record = {"frame": frame, "path": f"Ply/{target.name}", "member": member.filename,
                          "compressed_bytes": member.compress_size, **integrity,
                          "timestamp_seconds_from_archive_start": (frame-min(members))/30.,
                          "acquisition": acquisition, "verified_at_utc": utc_now(),
                          "source_url": reader.url, "archive_etag": reader.etag}
                if from_cache:
                    record["cache_provenance_sha256"] = cache["sha256"]
                    record["original_acquisition_source_url"] = cache["document"]["source_url"]
                known[frame] = record
                checkpoint(False)
                if progress:
                    progress(f"Prepared {target.name}: {integrity['bytes']:,} bytes; {acquisition}; SHA256/ZIP CRC verified")

            checkpoint(False)
            network_jobs = []
            for frame in selected:
                member = members[frame]
                if frame in known:
                    if progress:
                        progress(f"Resume verified frame {frame}")
                    continue
                target = output/object_name/"Ply"/PurePosixPath(member.filename).name
                from_cache = cached.get(frame)
                if from_cache:
                    integrity = from_cache["integrity"]
                    _copy_atomic(from_cache["path"], target, integrity)
                    acquisition = "verified_local_cache"
                elif target.exists():
                    # A crash can leave a committed member before its inventory
                    # checkpoint. Adopt it only after current ZIP CRC/size match.
                    integrity = file_integrity(target)
                    if integrity["bytes"] != member.file_size or integrity["zip_crc32"] != f"{member.CRC:08x}":
                        raise FileExistsError(f"Preserving existing artifact incompatible with official ZIP member: {target}")
                    acquisition = "recovered_crc_verified_atomic_member"
                elif pool:
                    network_jobs.append((frame, member, target))
                    continue
                else:
                    try:
                        reader.prefetch_member(member, prefetch_bytes)
                        integrity = _extract_atomic(archive, member, target)
                    except BaseException:
                        checkpoint(False)
                        raise
                    finally:
                        reader.clear_member_cache()
                    acquisition = "official_zip_range_download"
                record_frame(frame, member, target, integrity, acquisition, from_cache)
            if network_jobs:
                try:
                    pool.run(network_jobs, record_frame)
                except BaseException:
                    checkpoint(False)
                    raise
            checkpoint(True)
            return document
    finally:
        archive.close()
        reader.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", choices=OBJECTS, required=True, dest="object_name")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--frames", type=parse_frames, help="Inclusive START:END or comma-separated frame IDs")
    choice.add_argument("--all-frames", action="store_true", help="Select every object PLY frame listed in official ZIP metadata")
    parser.add_argument("--output", type=Path, default=ROOT/"output/datasets/8i")
    parser.add_argument("--cache", type=Path, default=ROOT/"output/progressive_gap_real/raw")
    parser.add_argument("--dry-run", action="store_true", help="Read remote ZIP metadata and estimate; fetch no frame/license data and write no files")
    parser.add_argument("--allow-http-fallback", action="store_true", help="Explicitly allow HTTP only after HTTPS certificate verification failure; TLS verification stays enabled")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--ca-file", type=Path, help="Verified TLS CA bundle; default uses system CA if available. Never disables certificate verification.")
    parser.add_argument("--prefetch-bytes", type=int, default=MAX_RANGE_BYTES,
                        help="One selected compressed-member cache limit (default 8 MiB); 0 disables, maximum 8 MiB. Requests remain pinned/bounded.")
    parser.add_argument("--download-workers", type=int, default=1,
                        help="Concurrent network members only (default 1, max 8); independent pinned readers and <=8 MiB compressed cache each. No GPU work.")
    args = parser.parse_args(argv)
    try:
        document = download_object(args.object_name, args.frames, all_frames=args.all_frames, output=args.output,
                                   cache=args.cache, dry_run=args.dry_run, allow_http_fallback=args.allow_http_fallback,
                                   timeout=args.timeout, retries=args.retries, ca_file=args.ca_file,
                                   prefetch_bytes=args.prefetch_bytes, download_workers=args.download_workers,
                                   progress=lambda message: print(message, file=__import__("sys").stderr, flush=True))
    except Exception as exc:
        parser.exit(1, f"Download failed: {exc}\n")
    print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

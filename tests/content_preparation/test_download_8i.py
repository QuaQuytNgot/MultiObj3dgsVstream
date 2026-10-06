"""Downloader tests use mocked HTTP responses and tiny in-memory ZIP archives."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import random
import ssl
import threading
import time
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError
import zipfile

from tools.download_8i_object import (
    ArchiveChanged, BASE_URL, DownloadError, RangeReader, RangeUnsupported,
    download_object, file_integrity, parse_frames,
)

LICENSE = b"%PDF-1.4\nmock official license\n"


def make_zip(object_name="longdress", frames=(1051, 1052, 1053)):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for frame in frames:
            archive.writestr(f"{object_name}/Ply/{object_name}_vox10_{frame:04d}.ply",
                             b"ply\nformat ascii 1.0\ncomment fixture " + str(frame).encode()+b"\nend_header\n"+bytes([frame % 256])*10000)
    return stream.getvalue()


class Response:
    def __init__(self, data, status, headers, url):
        self.stream = io.BytesIO(data)
        self.status, self.headers, self.url = status, headers, url
        self.read_calls = 0
        self.returned_bytes = 0
    def read(self, size=-1):
        self.read_calls += 1
        data = self.stream.read(size)
        self.returned_bytes += len(data)
        return data
    def geturl(self):
        return self.url
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.stream.close()


class MockHTTP:
    def __init__(self, data):
        self.data = data
        self.calls, self.responses = [], []
        self.ignore_ranges = False
        self.bad_etag = False
        self.bad_content_range = False
        self.truncate_once = False
        self.certificate_failure = False
        self.other_tls_failure = False
    def __call__(self, request, *, timeout):
        url, method = request.full_url, request.get_method()
        interval = request.get_header("Range")
        self.calls.append((method, url, interval))
        if url.startswith("https://") and self.certificate_failure:
            raise URLError(ssl.SSLCertVerificationError("certificate verify failed"))
        if url.startswith("https://") and self.other_tls_failure:
            raise URLError("connection refused")
        if url.endswith("license.pdf"):
            response = Response(LICENSE, 200, {"Content-Length": str(len(LICENSE))}, url)
        elif method == "HEAD":
            response = Response(b"", 200, {"Content-Length": str(len(self.data)), "ETag": '"fixture-zip"'}, url)
        elif interval:
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", interval).groups())
            if self.ignore_ranges:
                response = Response(self.data, 200, {"Content-Length": str(len(self.data))}, url)
            else:
                block = self.data[start:end+1]
                if self.truncate_once:
                    block = block[:max(0, len(block)//2)]
                    self.truncate_once = False
                response = Response(block, 206, {"Content-Length": str(end-start+1),
                     "ETag": '"changed"' if self.bad_etag else '"fixture-zip"',
                     "Content-Range": "bytes 0-1/2" if self.bad_content_range else f"bytes {start}-{end}/{len(self.data)}"}, url)
        else:
            raise AssertionError("Archive full GET must never be made.")
        self.responses.append(response)
        return response


class Download8iTests(unittest.TestCase):
    def test_sequential_interrupt_persists_observed_partial_range_bytes(self):
        data = make_zip()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            member = archive.infolist()[0]
            payload_start = member.header_offset+30
        class InterruptedBody(Response):
            def read(self, size=-1):
                if self.read_calls:
                    raise KeyboardInterrupt("mock interrupt during member range body")
                return super().read(min(7, size))
        class InterruptedHTTP(MockHTTP):
            def __call__(self, request, *, timeout):
                response = super().__call__(request, timeout=timeout)
                interval = request.get_header("Range")
                if interval and int(interval.split("=")[1].split("-")[0]) == payload_start:
                    interrupted = InterruptedBody(response.stream.getvalue(), response.status, response.headers, response.url)
                    self.responses[-1] = interrupted
                    return interrupted
                return response
        remote = InterruptedHTTP(data)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(KeyboardInterrupt):
                download_object("longdress", all_frames=True, output=output, opener=remote, retry_delay=0)
            partial = json.loads((output/"longdress/inventory.json").read_text())
            self.assertFalse(partial["complete_for_selection"])
            self.assertEqual(partial["frames"], [])
            self.assertEqual(partial["transfer"]["verified_payload_files_bytes"], 0)
            # Fixed local header plus the seven response-body bytes delivered
            # before interruption are checkpointed even though no frame commits.
            self.assertEqual(partial["transfer"]["member_received_bytes"], 37)
            self.assertEqual(partial["transfer"]["this_invocation_received_bytes"], sum(r.returned_bytes for r in remote.responses))
            self.assertIn("not an exact wire-byte total", partial["transfer"]["accounting_scope"])
            self.assertIn("hard kill", partial["transfer"]["accounting_scope"])
            self.assertFalse(list(output.rglob("*.part")))
            resumed = download_object("longdress", all_frames=True, output=output, opener=MockHTTP(data), retry_delay=0)
            self.assertTrue(resumed["complete_for_selection"])
            self.assertEqual(resumed["transfer"]["verified_payload_files_bytes"], sum(r["bytes"] for r in resumed["frames"]))
            self.assertEqual(resumed["transfer"]["cumulative_received_bytes"],
                             partial["transfer"]["cumulative_received_bytes"]+resumed["transfer"]["this_invocation_received_bytes"])

    def test_parallel_workers_have_private_readers_bounded_jobs_and_exact_accounting(self):
        data = make_zip(frames=(1051, 1052, 1053, 1054))
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            slow_offset = archive.infolist()[0].header_offset
        class ReorderedHTTP(MockHTTP):
            def __call__(self, request, *, timeout):
                interval = request.get_header("Range")
                if interval and int(interval.split("=")[1].split("-")[0]) == slow_offset:
                    time.sleep(.04)
                return super().__call__(request, timeout=timeout)
        observed = []
        class PrivateReader(RangeReader):
            def __init__(self, *args, **kwargs):
                self.owner = threading.get_ident()
                observed.append(self)
                super().__init__(*args, **kwargs)
            def read(self, *args, **kwargs):
                if threading.get_ident() != self.owner:
                    raise AssertionError("A reader was shared across network workers.")
                return super().read(*args, **kwargs)
        remote, completions = ReorderedHTTP(data), []
        main_thread = threading.get_ident()
        def progress(message):
            self.assertEqual(threading.get_ident(), main_thread)
            if message.startswith("Prepared"):
                completions.append(int(re.search(r"vox10_(\d+)\.ply", message)[1]))
        with tempfile.TemporaryDirectory() as directory, patch("tools.download_8i_object.RangeReader", PrivateReader):
            result = download_object("longdress", all_frames=True, output=directory, opener=remote, retry_delay=0,
                                     download_workers=2, progress=progress)
            self.assertTrue(result["complete_for_selection"])
            self.assertEqual([r["frame"] for r in result["frames"]], [1051, 1052, 1053, 1054])
            self.assertNotEqual(completions[0], 1051)
            self.assertEqual(result["download_settings"]["peak_concurrent_network_members"], 2)
            self.assertLessEqual(result["download_settings"]["peak_pending_jobs"], 2)
            self.assertEqual(result["transfer"]["this_invocation_received_bytes"], sum(r.returned_bytes for r in remote.responses))
            self.assertEqual(result["transfer"]["http_requests"], len(remote.calls))
            self.assertEqual(len({reader.owner for reader in observed}), 3)  # Main metadata reader plus two worker readers.
            self.assertTrue(all(reader.closed for reader in observed))
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for record in result["frames"]:
                    self.assertEqual(record["sha256"], hashlib.sha256(archive.read(record["member"])).hexdigest())

    def test_parallel_failure_checkpoints_successes_then_resume_preserves_artifacts(self):
        data = make_zip()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            fail_offset = archive.infolist()[1].header_offset
            slow_offset = archive.infolist()[0].header_offset
        class FailedHTTP(MockHTTP):
            def __call__(self, request, *, timeout):
                interval = request.get_header("Range")
                start = int(interval.split("=")[1].split("-")[0]) if interval else None
                if start == slow_offset:
                    time.sleep(.03)
                response = super().__call__(request, timeout=timeout)
                if start == fail_offset:
                    response.headers["ETag"] = '"changed-while-worker-read"'
                return response
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(ArchiveChanged):
                download_object("longdress", all_frames=True, output=output, opener=FailedHTTP(data),
                                retry_delay=0, download_workers=2)
            partial = json.loads((output/"longdress/inventory.json").read_text())
            self.assertFalse(partial["complete_for_selection"])
            self.assertEqual([r["frame"] for r in partial["frames"]], [1051])
            path = output/"longdress"/partial["frames"][0]["path"]
            before = path.stat().st_mtime_ns
            self.assertFalse((output/"longdress/Ply/longdress_vox10_1052.ply").exists())
            self.assertFalse(list(output.rglob("*.part")))
            resumed = download_object("longdress", all_frames=True, output=output, opener=MockHTTP(data),
                                      retry_delay=0, download_workers=2)
            self.assertTrue(resumed["complete_for_selection"])
            self.assertEqual(len(resumed["frames"]), 3)
            self.assertEqual(path.stat().st_mtime_ns, before)

    def test_parallel_resume_adopts_committed_orphan_only_after_crc_check(self):
        data = make_zip()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            orphan = output/"longdress/Ply/longdress_vox10_1051.ply"
            orphan.parent.mkdir(parents=True)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                orphan.write_bytes(archive.read("longdress/Ply/longdress_vox10_1051.ply"))
            before = orphan.stat().st_mtime_ns
            result = download_object("longdress", all_frames=True, output=output, opener=MockHTTP(data),
                                     retry_delay=0, download_workers=2)
            self.assertTrue(result["complete_for_selection"])
            self.assertEqual(result["frames"][0]["acquisition"], "recovered_crc_verified_atomic_member")
            self.assertEqual(orphan.stat().st_mtime_ns, before)
            with self.assertRaises(ValueError):
                download_object("longdress", all_frames=True, output=output, opener=MockHTTP(data), download_workers=0)

    def test_bounded_member_prefetch_reduces_requests_and_preserves_output(self):
        # About 16 MiB uncompressed / 4 MiB compressed, resembling one real PLY
        # member. The mocked server never touches the network.
        rng = random.Random(7)
        payload = b"ply\ncomment prefetch fixture\nend_header\n"+b"".join(rng.randbytes(16384)*4 for _ in range(256))
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("longdress/Ply/longdress_vox10_1051.ply", payload)
        remote_streaming, remote_prefetch = MockHTTP(stream.getvalue()), MockHTTP(stream.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            streaming = download_object("longdress", all_frames=True, output=root/"streaming", opener=remote_streaming,
                                        prefetch_bytes=0, retry_delay=0)
            prefetched = download_object("longdress", all_frames=True, output=root/"prefetched", opener=remote_prefetch,
                                         retry_delay=0)
            before_ranges = sum(interval is not None for _, _, interval in remote_streaming.calls)
            after_ranges = sum(interval is not None for _, _, interval in remote_prefetch.calls)
            self.assertLess(after_ranges, before_ranges/2)
            self.assertEqual(prefetched["transfer"]["prefetched_members"], 1)
            self.assertGreater(prefetched["transfer"]["member_cache_hits"], 4)
            self.assertLessEqual(prefetched["transfer"]["prefetched_bytes"], 8*1024*1024)
            self.assertEqual(streaming["frames"][0]["sha256"], prefetched["frames"][0]["sha256"])
            self.assertEqual(streaming["frames"][0]["zip_crc32"], prefetched["frames"][0]["zip_crc32"])
            self.assertEqual(streaming["transfer"]["member_received_bytes"], prefetched["transfer"]["member_received_bytes"])
            with self.assertRaises(ValueError):
                download_object("longdress", all_frames=True, output=root/"bad", opener=remote_prefetch, prefetch_bytes=9*1024*1024)

    def test_range_zero_bounds_readinto_and_pinned_etag(self):
        remote = MockHTTP(make_zip())
        with RangeReader(BASE_URL+"longdress.zip", opener=remote, retry_delay=0) as reader:
            count = len(remote.calls)
            self.assertEqual(reader.read(0), b"")
            self.assertEqual(len(remote.calls), count)
            buffer = bytearray(4)
            self.assertEqual(reader.readinto(buffer), 4)
            self.assertEqual(bytes(buffer), remote.data[:4])
            reader.seek(0, io.SEEK_END)
            count = len(remote.calls)
            self.assertEqual(reader.read(), b"")
            self.assertEqual(len(remote.calls), count)
            with self.assertRaises(ValueError):
                reader.seek(-1)
            with self.assertRaises(ValueError):
                reader.seek(reader.length+1)
            reader.seek(0)
            remote.bad_etag = True
            with self.assertRaises(ArchiveChanged):
                reader.read(3)
        with self.assertRaises(ValueError):
            reader.read(1)

    def test_refuses_full_archive_response_before_reading_body(self):
        remote = MockHTTP(make_zip())
        with RangeReader(BASE_URL+"longdress.zip", opener=remote, retry_delay=0) as reader:
            remote.ignore_ranges = True
            with self.assertRaises(RangeUnsupported):
                reader.read(2)
            self.assertEqual(remote.responses[-1].read_calls, 0)
        with RangeReader(BASE_URL+"longdress.zip", opener=MockHTTP(make_zip()), max_range_bytes=5) as reader:
            with self.assertRaises(RangeUnsupported):
                reader.read(6)

    def test_retry_truncation_without_position_drift_or_lost_accounting(self):
        remote = MockHTTP(make_zip())
        with RangeReader(BASE_URL+"longdress.zip", opener=remote, retry_delay=0, retries=1) as reader:
            remote.truncate_once = True
            self.assertEqual(reader.read(10), remote.data[:10])
            self.assertEqual(reader.tell(), 10)
            self.assertEqual(reader.transferred, 15)
            self.assertEqual(reader.failed_attempt_bytes, 5)
            self.assertEqual(len(remote.calls), 3)  # HEAD plus two attempts
            remote.bad_content_range = True
            with self.assertRaises(ArchiveChanged):
                reader.read(1)

    def test_dry_run_reads_only_metadata_and_writes_nothing(self):
        remote = MockHTTP(make_zip())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"absent"
            result = download_object("longdress", [1051, 1052], output=output, dry_run=True, opener=remote, retry_delay=0)
            self.assertFalse(output.exists())
            self.assertEqual(result["selected_frame_ids"], [1051, 1052])
            self.assertTrue(result["download_compressed_bytes_estimate"] > 0)
            self.assertTrue(result["metadata_transferred_bytes"] < len(remote.data))
            self.assertFalse(any(url.endswith("license.pdf") for _, url, _ in remote.calls))

    def test_end_to_end_all_frames_inventory_roundtrip_and_resume(self):
        remote = MockHTTP(make_zip())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = download_object("longdress", all_frames=True, output=output, opener=remote, retry_delay=0)
            self.assertTrue(result["complete_for_selection"])
            self.assertEqual(result["capture"]["sample_count"], 3)
            self.assertEqual(result["capture"]["fps"], 30)
            self.assertEqual(result["capture"]["samples_duration_seconds"], .1)
            self.assertEqual(result["capture"]["timestamp_span_seconds"], 2/30)
            self.assertTrue((output/"license.pdf").is_file())
            self.assertEqual(json.loads((output/"longdress/inventory.json").read_text()), result)
            paths = [output/"longdress"/record["path"] for record in result["frames"]]
            timestamps = [path.stat().st_mtime_ns for path in paths]
            remote.calls.clear()
            resumed = download_object("longdress", all_frames=True, output=output, opener=remote, retry_delay=0)
            self.assertEqual(timestamps, [path.stat().st_mtime_ns for path in paths])
            self.assertTrue(resumed["complete_for_selection"])
            self.assertEqual(resumed["transfer"]["member_received_bytes"], 0)
            self.assertFalse(any(url.endswith("license.pdf") for _, url, _ in remote.calls))
            paths[0].write_bytes(b"corrupt")
            with self.assertRaisesRegex(DownloadError, "resume integrity"):
                download_object("longdress", all_frames=True, output=output, opener=remote, retry_delay=0)
            self.assertEqual(paths[0].read_bytes(), b"corrupt")

    def test_verified_legacy_cache_import_avoids_member_network_reads(self):
        data = make_zip()
        remote = MockHTTP(data)
        with tempfile.TemporaryDirectory() as directory:
            cache, output = Path(directory)/"raw", Path(directory)/"out"
            cache.mkdir()
            records = []
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.infolist():
                    frame = int(Path(member.filename).stem.rsplit("_", 1)[1])
                    path = cache/member.filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(member))
                    records.append({"frame": frame, "path": str(path), "member": member.filename,
                                    "compressed_bytes": member.compress_size, **file_integrity(path)})
            license_path = cache/"license.pdf"
            license_path.write_bytes(LICENSE)
            (cache/"provenance.json").write_text(json.dumps({"source_url": BASE_URL.replace("https://", "http://")+"longdress.zip",
                   "archive_bytes": len(data), "archive_etag": '"fixture-zip"', "frames": records,
                   "license": str(license_path), "license_sha256": hashlib.sha256(LICENSE).hexdigest(),
                   "transport": "legacy HTTP; no publisher signature"}))
            result = download_object("longdress", all_frames=True, output=output, cache=cache, opener=remote, retry_delay=0)
            self.assertTrue(all(r["acquisition"] == "verified_local_cache" for r in result["frames"]))
            self.assertEqual(result["transfer"]["member_received_bytes"], 0)
            self.assertEqual(result["transfer"]["license_bytes"], 0)
            self.assertTrue((output/"license.inventory.json").exists())
            self.assertEqual(result["cache"]["transport"], "legacy HTTP; no publisher signature")
            self.assertFalse(any(url.endswith("license.pdf") for _, url, _ in remote.calls))

    def test_checkpoint_survives_stop_after_one_frame_then_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            def stop_after_one(message):
                raise RuntimeError("test interruption after committed frame")
            with self.assertRaisesRegex(RuntimeError, "test interruption"):
                download_object("longdress", all_frames=True, output=output, opener=MockHTTP(make_zip()), retry_delay=0, progress=stop_after_one)
            interrupted = json.loads((output/"longdress/inventory.json").read_text())
            self.assertFalse(interrupted["complete_for_selection"])
            self.assertEqual(len(interrupted["frames"]), 1)
            first = output/"longdress"/interrupted["frames"][0]["path"]
            before = first.stat().st_mtime_ns
            completed = download_object("longdress", all_frames=True, output=output, opener=MockHTTP(make_zip()), retry_delay=0)
            self.assertTrue(completed["complete_for_selection"])
            self.assertEqual(first.stat().st_mtime_ns, before)
            self.assertFalse(list(output.rglob("*.part")))

    def test_https_fallback_requires_explicit_flag_and_only_cert_failure(self):
        remote = MockHTTP(make_zip())
        remote.certificate_failure = True
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(URLError):
                download_object("longdress", all_frames=True, output=output, dry_run=True, opener=remote, retries=0)
            self.assertFalse(any(url.startswith("http://") for _, url, _ in remote.calls))
            result = download_object("longdress", all_frames=True, output=output, dry_run=True, opener=remote,
                                     retries=0, allow_http_fallback=True)
            self.assertTrue(result["source_url"].startswith("http://"))
            self.assertTrue(result["transport_provenance"]["explicit_http_fallback"])
            self.assertFalse(result["transport_provenance"]["https_certificate_verified"])
            remote = MockHTTP(make_zip())
            remote.other_tls_failure = True
            with self.assertRaises(URLError):
                download_object("longdress", all_frames=True, output=output, dry_run=True, opener=remote,
                                 retries=0, allow_http_fallback=True)
            self.assertFalse(any(url.startswith("http://") for _, url, _ in remote.calls))

    def test_generic_soldier_and_loot_and_inclusive_selection(self):
        self.assertEqual(parse_frames("1051:1053"), [1051, 1052, 1053])
        self.assertEqual(parse_frames("3,1,2"), [1, 2, 3])
        for object_name in ["soldier", "loot"]:
            with tempfile.TemporaryDirectory() as directory:
                result = download_object(object_name, all_frames=True, output=directory, dry_run=True,
                                         opener=MockHTTP(make_zip(object_name)), retry_delay=0)
                self.assertEqual(result["object"], object_name)
                self.assertEqual(result["archive_frame_count"], 3)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DownloadError, "missing"):
                download_object("longdress", [9999], output=directory, dry_run=True, opener=MockHTTP(make_zip()), retry_delay=0)


if __name__ == "__main__":
    unittest.main()

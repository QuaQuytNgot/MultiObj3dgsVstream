"""Experimental lossless stream-reference package, independent of the MPD.

Packets retain their original prefix, checksum and ordered compressed streams.
Each distinct stream is stored once. A client can join at any frame by loading
the ordinary same-frame Q0..Qn dependency chain; no previous frame is needed.
"""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .optimized_codec import ByteCache

SCHEMA = "cp-stream-references-v1"


class StaticDynamicStore:
    def __init__(self, path, cache_bytes=128*1024**2):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.document = json.loads(self.path.read_text())
        if self.document.get("schema") != SCHEMA:
            raise ValueError("Unsupported stream-reference package")
        self.cache = ByteCache(cache_bytes)
        self.stats = Counter()

    def read(self, payload_id, expected_checksum, expected_bytes):
        record = self.document["payloads"][payload_id]
        if record["sha256"] != expected_checksum or record["bytes"] != expected_bytes:
            raise ValueError("Stream-reference identity differs from MPD")
        prefix = base64.b64decode(record["prefix"], validate=True)
        suffix = base64.b64decode(record["suffix"], validate=True)
        parts = [prefix]
        size = len(prefix)+len(suffix)
        for reference in record["streams"]:
            digest, length = reference["sha256"], reference["bytes"]
            if not isinstance(digest,str) or re.fullmatch("[0-9a-f]{64}",digest) is None:
                raise ValueError("Invalid stream digest")
            if type(length) is not int or length < 0 or size+length > expected_bytes:
                raise ValueError("Oversized stream reference")
            data = self.cache.get((digest,length))
            if data is None:
                path = (self.root/"streams"/digest).resolve()
                if not path.is_relative_to(self.root) or path.stat().st_size != length:
                    raise ValueError("Invalid stream path/size")
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError("Stream checksum mismatch")
                self.cache.put((digest,length),data,len(data))
                self.stats["stream_reads"] += 1
                self.stats["read_bytes"] += len(data)
            else:
                self.stats["cache_hits"] += 1
            parts.append(data)
            size += length
        if size != expected_bytes:
            raise ValueError("Packet reference size mismatch")
        parts.append(suffix)
        blob = b"".join(parts)
        if hashlib.sha256(blob).hexdigest() != expected_checksum:
            raise ValueError("Reassembled packet checksum mismatch")
        return blob

    def close(self):
        self.cache.clear()

    async def read_http(self, payload_id, expected_checksum, expected_bytes, client, base_url):
        """Fetch missing content-addressed streams through the existing transport.

        The local experimental index is checked against the original MPD packet
        digest after reassembly. No segment-cache/planner credit is fabricated.
        """
        from .request_api import RequestSpec
        record = self.document["payloads"][payload_id]
        if record["sha256"] != expected_checksum or record["bytes"] != expected_bytes:
            raise ValueError("Stream-reference identity differs from MPD")
        prefix = base64.b64decode(record["prefix"],validate=True)
        suffix = base64.b64decode(record["suffix"],validate=True)
        parts, size = [prefix], len(prefix)+len(suffix)
        for reference in record["streams"]:
            digest,length = reference["sha256"],reference["bytes"]
            if not isinstance(digest,str) or re.fullmatch("[0-9a-f]{64}",digest) is None:
                raise ValueError("Invalid stream digest")
            if type(length) is not int or length < 0 or size+length > expected_bytes:
                raise ValueError("Oversized stream reference")
            data = self.cache.get((digest,length))
            if data is None:
                result = await client.fetch(RequestSpec("stream:"+digest,base_url.rstrip("/")+"/streams/"+digest,
                                                       metadata={"experimental_package":SCHEMA}))
                if not result.success or result.body is None:
                    raise RuntimeError(f"Stream transfer failed: {digest}: {result.error}")
                data = result.body
                if len(data)!=length or hashlib.sha256(data).hexdigest()!=digest:
                    raise ValueError("Stream size/checksum mismatch")
                self.cache.put((digest,length),data,len(data))
                self.stats["stream_reads"] += 1
                self.stats["read_bytes"] += len(data)
            else:
                self.stats["cache_hits"] += 1
            parts.append(data); size += length
        if size != expected_bytes: raise ValueError("Packet reference size mismatch")
        parts.append(suffix)
        blob = b"".join(parts)
        if hashlib.sha256(blob).hexdigest()!=expected_checksum:
            raise ValueError("Reassembled packet checksum mismatch")
        return blob

"""Metadata-only adapter for the project's static progressive Gaussian MPD.

Local loading requires an explicit HTTP media base for RequestSpec construction.
HTTP loading reuses TransferClient. Neither route reads media or imports a
renderer/decoder. Source manifests/package indexes remain authoritative.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import url2pathname
import math
import re
import xml.etree.ElementTree as ET

from src.content_metadata import (
    MAX_DOCUMENT_BYTES, absolute_uri, checksum, density_reference, digest,
    integer, json_bytes, profile_reference, relative_path, resolve_uri,
    validate_object_metadata,
)
from src.content_mpd import (
    DENSITY_PROPERTY, INDEX_SCHEMA, NAMESPACE, PROFILE, PROPERTY, REVISION_PROPERTY,
    VERSION, build_mpd,
)
from .adaptation_inputs import (
    ContentIndexSnapshot, DensityValue, PayloadDescriptor, RequestUnit, freeze,
)
from .request_api import RequestSpec, TransferClient


class MetadataError(ValueError):
    """Malformed, unsupported, unsafe or inconsistent project metadata."""


@dataclass(frozen=True, slots=True)
class ProfileReference:
    uri: str
    checksum: str
    size_bytes: int
    metric: str
    reference: str
    normalization: str
    metadata: Mapping


@dataclass(frozen=True, slots=True)
class RepresentationMetadata:
    object_id: str
    quality: int
    quality_id: str
    layer: str
    representation_id: str
    dependency_ids: tuple[str, ...]
    required_bandwidth_bps: int  # Descriptive cumulative rate, never request cost.


@dataclass(frozen=True, slots=True)
class SegmentInfo:
    representation: RepresentationMetadata
    request_unit: RequestUnit
    frames: tuple[int, ...]
    presentation_start_s: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class PayloadLocation:
    """Member location after receiving its complete requestable container."""
    request_unit_id: str
    offset: int
    length: int
    checksum: str


class QualityProfileMetadata:
    """Immutable measured-row lookup; no implicit interpolation or calibration.

    This is not a DistortionDensityProvider: a producer-validated, calibrated
    adapter remains necessary before planner use on actual content.
    """
    def __init__(self, reference: ProfileReference, document: dict, quality_ids: list[str]):
        self.reference = reference
        self.metadata = freeze(document["metadata"])
        rows = {}
        for row in document["rows"]:
            key = (quality_ids.index(row["quality"]), float(row["azimuth"]) % 360,
                   float(row["elevation"]), float(row["scale"]), float(row["timestamp"]))
            if key in rows:
                raise MetadataError("Ambiguous measured profile coordinates")
            rows[key] = freeze(row)
        self.rows = MappingProxyType(rows)

    def _row(self, quality: int, azimuth: float, elevation: float, scale: float, timestamp: float):
        if isinstance(quality, bool) or not isinstance(quality, int) or not 0 <= quality <= 3:
            raise MetadataError("Quality must be an integer Q0..Q3 target")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (azimuth, elevation, scale, timestamp)) or scale <= 0 or timestamp < 0:
            raise MetadataError("Invalid profile coordinates")
        return self.rows.get((quality, float(azimuth) % 360, float(elevation), float(scale), float(timestamp)))

    def lookup_mse(self, quality: int, *, azimuth: float, elevation: float,
                   scale: float, timestamp: float) -> float | None:
        row = self._row(quality, azimuth, elevation, scale, timestamp)
        return None if row is None else row["metrics"]["mse"]

    def lookup_density(self, quality: int, *, azimuth: float, elevation: float,
                       scale: float, timestamp: float) -> DensityValue:
        if self.reference.metric != "q0-normalized-distortion-density":
            raise MetadataError("Raw MSE is not normalized distortion density")
        row = self._row(quality, azimuth, elevation, scale, timestamp)
        if row is None:
            return DensityValue(None, None, "density_context_unavailable")
        return DensityValue(row["error_density"], row["solo_area"],
                            None if row["error_density"] is not None else "q0_support_below_guard")


@dataclass(frozen=True, slots=True)
class _Document:
    uri: str
    kind: str = "metadata"


def _xml(raw: bytes):
    if not isinstance(raw, bytes) or len(raw) > MAX_DOCUMENT_BYTES:
        raise MetadataError("MPD exceeds metadata size limit")
    if re.search(br"<!\s*(DOCTYPE|ENTITY)", raw, re.I):
        raise MetadataError("DTD/entity declarations are not supported")
    try:
        if "\0" in raw.decode("utf-8"):
            raise MetadataError("MPD must use UTF-8 encoding")
    except UnicodeError as exc:
        raise MetadataError("MPD must use UTF-8 encoding") from exc
    try:
        root = ET.fromstring(raw)
    except (ET.ParseError, RecursionError) as exc:
        raise MetadataError("Malformed MPD/XML") from exc
    if root.tag != f"{{{NAMESPACE}}}MPD" or root.get("profiles") != PROFILE or root.get("type") != "static":
        raise MetadataError("Unsupported MPD namespace/profile/type")
    if len(root.findall(f"{{{NAMESPACE}}}Period")) != 1:
        raise MetadataError("This profile requires exactly one Period")
    known_essential = {"object-id", "codec", "container", REVISION_PROPERTY,
                       "extended-bandwidth-bits-per-second:1"}
    for node in root.iter(f"{{{NAMESPACE}}}EssentialProperty"):
        if node.get("schemeIdUri") not in {PROPERTY + n for n in known_essential}:
            raise MetadataError("Unsupported essential extension/version")
    for node in root.iter():
        if node is not root and node.find(f"{{{NAMESPACE}}}BaseURL") is not None:
            raise MetadataError("Only a root BaseURL is supported by this project profile")
    return root


def _descriptor(node, name):
    matches = [child for child in node if child.tag in
               (f"{{{NAMESPACE}}}SupplementalProperty", f"{{{NAMESPACE}}}EssentialProperty")
               and child.get("schemeIdUri") == PROPERTY + name]
    if len(matches) != 1 or not matches[0].get("value"):
        raise MetadataError(f"Missing/duplicate descriptor: {name}")
    return matches[0].get("value")


def _verified(raw: bytes, size: int, sha: str, *, parse_json=True):
    integer(size, "metadata bytes", 1)
    checksum(sha)
    if len(raw) != size or digest(raw) != sha:
        raise MetadataError("Referenced metadata bytes/hash mismatch")
    return json_bytes(raw) if parse_json else None


def _semantic(node, base):
    attributes = dict(node.attrib)
    if node.tag == f"{{{NAMESPACE}}}SegmentURL":
        attributes["media"] = resolve_uri(base, attributes["media"])
    uri_descriptors = {PROPERTY + k for k in
                       ("content-index", "manifest", "quality-profile", "proxy", DENSITY_PROPERTY)}
    if attributes.get("schemeIdUri") in uri_descriptors:
        attributes["value"] = resolve_uri(base, attributes["value"])
    return (node.tag, attributes, (node.text or "").strip(),
            [_semantic(child, base) for child in node if child.tag != f"{{{NAMESPACE}}}BaseURL"])


class MPDParser:
    """Project MPD → existing ContentIndexSnapshot and typed profile references."""
    def __init__(self, *, media_base_url: str | None = None,
                 metadata_loader: Callable[[str], bytes] | None = None):
        if media_base_url is not None:
            absolute_uri(media_base_url)
            if urlsplit(media_base_url).scheme not in ("http", "https") or not media_base_url.endswith("/"):
                raise MetadataError("media_base_url must be an HTTP(S) directory URI ending in /")
        self.media_base_url = media_base_url
        self._loader = metadata_loader
        self._snapshot = None

    @classmethod
    def from_local(cls, path: str | Path, *, media_base_url: str) -> MPDParser:
        source = Path(path).resolve()
        root = source.parent

        def read(uri):
            parsed = urlsplit(uri)
            if parsed.scheme != "file":
                raise MetadataError("Local loader will not fetch HTTP references")
            candidate = Path(url2pathname(parsed.path)).resolve()
            if not candidate.is_relative_to(root):
                raise MetadataError("Metadata path/symlink escapes local content root")
            if candidate.suffix.lower() not in (".json", ".mpd"):
                raise MetadataError("Local parser reads only JSON/MPD files")
            if candidate.stat().st_size > MAX_DOCUMENT_BYTES:
                raise MetadataError("Metadata exceeds size limit")
            with candidate.open("rb") as stream:
                return stream.read(MAX_DOCUMENT_BYTES + 1)

        parser = cls(media_base_url=media_base_url, metadata_loader=read)
        return parser.parse_mpd(read(source.as_uri()), source_uri=source.as_uri())

    @classmethod
    async def from_http(cls, uri: str, client: TransferClient) -> MPDParser:
        """Fetch metadata via the existing transport; observer counts it separately."""
        absolute_uri(uri)
        if urlsplit(uri).scheme not in ("http", "https"):
            raise MetadataError("HTTP loading requires an HTTP(S) MPD URI")

        async def fetch(document):
            result = await client.fetch(RequestSpec(
                "mpd-metadata:" + digest(document.uri.encode()), document.uri,
                metadata={"traffic_kind": document.kind, "bandwidth_eligible": False}))
            if not result.success or result.body is None:
                raise MetadataError(f"Metadata fetch failed: {document.uri}")
            if len(result.body) > MAX_DOCUMENT_BYTES:
                raise MetadataError("Fetched metadata exceeds size limit")
            return result.body

        parser = cls()
        raw = await fetch(_Document(uri))
        generator = parser._parse(raw, uri)
        try:
            request = next(generator)
            while True:
                request = generator.send(await fetch(request))
        except StopIteration:
            return parser
        except (KeyError, TypeError, ValueError, RecursionError, OverflowError) as exc:
            raise MetadataError(str(exc)) from exc
        finally:
            generator.close()

    def parse_mpd(self, source: bytes, *, source_uri: str,
                  documents: Mapping[str, bytes] | None = None) -> MPDParser:
        """Parse supplied XML and referenced metadata bytes; never fetch implicitly.

        documents keys are resolved absolute metadata URIs. Alternatively supply
        a metadata_loader; from_local/from_http are convenience I/O adapters.
        """
        generator = self._parse(source, source_uri)
        try:
            request = next(generator)
            while True:
                if documents is not None:
                    raw = documents[request.uri]
                elif self._loader is not None:
                    raw = self._loader(request.uri)
                else:
                    raise MetadataError(f"Referenced metadata not supplied: {request.uri}")
                request = generator.send(raw)
        except StopIteration:
            return self
        except (OSError, KeyError, TypeError, ValueError, RecursionError, OverflowError) as exc:
            raise MetadataError(str(exc)) from exc
        finally:
            generator.close()

    def _parse(self, raw: bytes, source_uri: str):
        self._snapshot = None  # A failed reparse must not expose stale state.
        absolute_uri(source_uri)
        root = _xml(raw)
        directory = urljoin(source_uri, "./")
        base_nodes = root.findall(f"{{{NAMESPACE}}}BaseURL")
        if len(base_nodes) > 1:
            raise MetadataError("Ambiguous root BaseURL")
        base = resolve_uri(directory, base_nodes[0].text.strip()) if base_nodes else directory
        if urlsplit(source_uri).scheme == "file" and self.media_base_url is None:
            raise MetadataError("Local metadata needs an explicit HTTP media_base_url")
        index_uri = resolve_uri(base, _descriptor(root, "content-index"))
        index_raw = yield _Document(index_uri)
        index = json_bytes(index_raw)
        if index.get("schema_version") != INDEX_SCHEMA or index.get("writer_version") != VERSION:
            raise MetadataError("Unsupported content index schema/writer version; regenerate MPD")
        _verified(raw, index["mpd"]["bytes"], index["mpd"]["sha256"], parse_json=False)
        if index["mpd"]["path"] != unquote(urlsplit(source_uri).path).rsplit("/", 1)[-1]:
            raise MetadataError("MPD identity/path mismatch")
        records, profiles, density_profiles = [], {}, {}
        total = len(raw) + len(index_raw)

        def request(path, size, kind="metadata"):
            integer(size, "referenced metadata bytes", 1)
            if size > MAX_DOCUMENT_BYTES or total + size > 128 * 1024 * 1024:
                raise MetadataError("Referenced metadata exceeds size budget")
            return _Document(resolve_uri(base, relative_path(path)), kind)

        for obj in index["objects"]:
            manifest_path = relative_path(obj["manifest_path"])
            package_path = relative_path(obj["package_index_path"])
            manifest_raw = yield request(manifest_path, obj["manifest_bytes"])
            manifest = _verified(manifest_raw, obj["manifest_bytes"], obj["manifest_sha256"])
            total += len(manifest_raw)
            package_raw = yield request(package_path, obj["package_index_bytes"])
            package = _verified(package_raw, obj["package_index_bytes"], obj["package_index_sha256"])
            total += len(package_raw)
            validate_object_metadata(manifest, package)
            if obj["id"] != manifest["object"]["id"]:
                raise MetadataError("Content index/object identity mismatch")
            object_directory = str(Path(manifest_path).parent.as_posix())
            if relative_path(obj["package_index_path"]) != object_directory + "/package_index.json":
                raise MetadataError("Package index is outside its object root")
            record = {"manifest": manifest, "manifest_path": manifest_path,
                      "manifest_sha256": digest(manifest_raw), "manifest_bytes": len(manifest_raw),
                      "directory": object_directory, "package_path": obj["package_index_path"],
                      "package_sha256": digest(package_raw), "package_bytes": len(package_raw)}
            if manifest["object"].get("quality_profile_path"):
                reference = obj["quality_profile"]
                if reference["path"] != object_directory + "/" + relative_path(manifest["object"]["quality_profile_path"]):
                    raise MetadataError("Invalid quality profile reference")
                profile_raw = yield request(reference["path"], reference["bytes"], "profile")
                profile = _verified(profile_raw, reference["bytes"], reference["sha256"])
                assets = {a["path"]: a for a in manifest["assets"]}
                asset = assets[manifest["object"]["quality_profile_path"]]
                _verified(profile_raw, asset["bytes"], asset["sha256"])
                record["quality_profile"] = profile_reference(reference["path"], profile_raw, profile, manifest)
                profiles[obj["id"]] = profile
                total += len(profile_raw)
            if "density_profile" in obj:
                if "quality_profile" not in record:
                    raise MetadataError("Density requires raw MSE source profile")
                reference = obj["density_profile"]
                density_raw = yield request(reference["path"], reference["bytes"], "profile")
                density = _verified(density_raw, reference["bytes"], reference["sha256"])
                record["density_profile"] = density_reference(reference["path"], density_raw, density,
                                                              profile, record["quality_profile"], manifest)
                density_profiles[obj["id"]] = density
                total += len(density_raw)
            if total > 128 * 1024 * 1024:
                raise MetadataError("Metadata bundle exceeds total size limit")
            records.append(record)
        expected_xml, expected_index = build_mpd(records)
        expected_index["mpd"] = index["mpd"]
        if index != expected_index:
            raise MetadataError("Content index conflicts with authoritative source metadata/revision")
        if _semantic(root, base) != _semantic(ET.fromstring(expected_xml), base):
            raise MetadataError("MPD conflicts with source timing/URLs/dependencies/bandwidth/extensions")
        if _descriptor(root, REVISION_PROPERTY) != index["content_revision"]:
            raise MetadataError("Content revision mismatch")
        self._construct(index, records, profiles, density_profiles, directory, base)

    def _construct(self, index, records, profiles, density_profiles, directory, base):
        objects, units, payloads, representations, segments, locations = [], {}, {}, {}, {}, {}
        references, parsed_profiles = {}, {}

        def media_url(relative):
            if urlsplit(base).scheme in ("http", "https"):
                return resolve_uri(base, relative)
            local_base = Path(url2pathname(urlsplit(base).path))
            local_root = Path(url2pathname(urlsplit(directory).path))
            prefix = local_base.relative_to(local_root).as_posix()
            return resolve_uri(self.media_base_url, relative if prefix == "." else prefix + "/" + relative)

        for obj, record in zip(index["objects"], records):
            object_id = obj["id"]
            if object_id in objects:
                raise MetadataError("Duplicate object identity")
            objects.append(object_id)
            manifest = record["manifest"]
            qualities = manifest["quality_order"]
            if not 1 <= len(qualities) <= 4:
                raise MetadataError("Client contracts support one to four progressive targets Q0..Q3")
            reps = sorted(obj["representations"], key=lambda r: r["nominal_quality"])
            for quality, rep in enumerate(reps):
                layer = "Base" if quality == 0 else f"E{quality}"
                if rep["nominal_quality"] != quality or rep["quality"] != qualities[quality] or rep["layer"] != layer:
                    raise MetadataError("Noncanonical progressive quality/layer mapping")
                info = RepresentationMetadata(object_id, quality, rep["quality"], layer, rep["id"],
                                               tuple(rep["required_representation_ids"]),
                                               rep["required_bandwidth_bits_per_second"])
                representations[(object_id, quality)] = info
                segment_infos = []
                for segment in rep["segments"]:
                    identifier = object_id + ":" + segment["id"]
                    spec = RequestSpec(identifier, media_url(segment["path"]), object_id=object_id,
                                       representation_id=rep["id"], segment_id=segment["id"], layer_id=layer,
                                       metadata={"traffic_kind": "base" if quality == 0 else "enhancement",
                                                 "content_revision": index["content_revision"],
                                                 "expected_bytes": segment["bytes"], "sha256": segment["sha256"]})
                    units[identifier] = RequestUnit(identifier, object_id, segment["bytes"], spec, segment["sha256"])
                    first_tick = round(obj["frames"][0]["timestamp"] * obj["timescale"])
                    segment_infos.append(SegmentInfo(info, units[identifier], tuple(segment["frames"]),
                                                     (segment["start_tick"] - first_tick) / obj["timescale"],
                                                     segment["duration_ticks"] / obj["timescale"]))
                segments[(object_id, quality)] = tuple(segment_infos)
            for payload in manifest["package"]["payloads"]:
                identifier = object_id + ":" + payload["id"]
                unit_id = object_id + ":" + payload["segment_id"]
                codec = payload["codec"]
                payloads[identifier] = PayloadDescriptor(
                    identifier, object_id, payload["frame"], qualities.index(payload["quality"]),
                    payload["epoch"], payload["timestamp"], unit_id, payload["decoded_state_hash"],
                    tuple(object_id + ":" + d for d in payload["dependencies"]),
                    payload.get("parent_state_hash"), payload.get("decoded_state_version", "gaussian-state-v1"),
                    codec["name"] + "/" + codec["version"])
                locations[identifier] = PayloadLocation(unit_id, payload["segment_offset"],
                                                        payload["bytes"], payload["sha256"])
            for name, documents in (("quality_profile", profiles), ("density_profile", density_profiles)):
                if name not in obj:
                    continue
                ref = obj[name]
                typed = ProfileReference(resolve_uri(base, ref["path"]), ref["sha256"], ref["bytes"],
                                         ref["metric"], ref["reference"], ref["normalization"], freeze(ref))
                references[(object_id, name)] = typed
                parsed_profiles[(object_id, name)] = QualityProfileMetadata(typed, documents[object_id], qualities)
        # Commit only after every source and normalized record has passed.
        self._representations = MappingProxyType(representations)
        self._segments = MappingProxyType(segments)
        self._locations = MappingProxyType(locations)
        self._references = MappingProxyType(references)
        self._profiles = MappingProxyType(parsed_profiles)
        self._metadata = freeze(index)
        self._snapshot = ContentIndexSnapshot(index["content_revision"], tuple(objects), payloads, units,
                                              {obj: ref.uri for (obj, name), ref in references.items()
                                               if name == "quality_profile"})

    def get_content_index(self) -> ContentIndexSnapshot:
        if self._snapshot is None:
            raise MetadataError("No successfully parsed MPD")
        return self._snapshot

    def get_objects(self) -> tuple[str, ...]:
        return self.get_content_index().object_ids

    def get_object_metadata(self, object_id: str) -> Mapping:
        """Frozen timing, transform, epochs and provenance from the content index."""
        self.get_quality_levels(object_id)
        return next(obj for obj in self._metadata["objects"] if obj["id"] == object_id)

    def get_quality_levels(self, object_id: str) -> tuple[int, ...]:
        self.get_content_index()
        levels = tuple(q for obj, q in self._representations if obj == object_id)
        if not levels:
            raise KeyError(object_id)
        return levels

    def get_representation(self, object_id: str, quality: int) -> RepresentationMetadata:
        self.get_content_index()
        integer(quality, "quality target")
        return self._representations[(object_id, quality)]

    def get_segment_info(self, object_id: str, quality: int, time_s: float) -> SegmentInfo:
        """Containing target-layer container at presentation time [start,end).

        This does not resolve parent containers or remaining cost. Use the
        existing RequirementsResolver for that, including cache/state credit.
        """
        self.get_representation(object_id, quality)
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s) or time_s < 0:
            raise MetadataError("Invalid media presentation time")
        candidates = self._segments[(object_id, quality)]
        position = bisect_right([s.presentation_start_s for s in candidates], time_s) - 1
        # Use the next boundary / declared terminal time directly. Adding two
        # rounded floats can otherwise admit an exact terminal timestamp.
        terminal = (candidates[position+1].presentation_start_s if position+1 < len(candidates)
                    else self.get_object_metadata(object_id)["duration_seconds"])
        if position < 0 or time_s >= terminal:
            raise KeyError("Presentation time outside object lifecycle")
        return candidates[position]

    def get_payload_location(self, payload_id: str) -> PayloadLocation:
        self.get_content_index()
        return self._locations[payload_id]

    def get_quality_profile_reference(self, object_id: str) -> ProfileReference | None:
        self.get_quality_levels(object_id)
        return self._references.get((object_id, "quality_profile"))

    def get_density_profile_reference(self, object_id: str) -> ProfileReference | None:
        self.get_quality_levels(object_id)
        return self._references.get((object_id, "density_profile"))

    def get_quality_profile(self, object_id: str, *, density=False) -> QualityProfileMetadata | None:
        self.get_quality_levels(object_id)
        return self._profiles.get((object_id, "density_profile" if density else "quality_profile"))

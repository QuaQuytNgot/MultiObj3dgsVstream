"""Bind the production StreamSTGS catalog to verified offline PSNR profiles.

The existing MPD parser, immutable metadata types, publication format and asset
cache remain authoritative. This adapter adds measured quality rather than a
second content catalog. Interactive cameras use the arithmetic mean of PSNR in
dB over all measured frames/views at scale 1.0. No nearest-view lookup, CRF-rank
utility, quality interpolation or monotonicity assumption is made.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

# Reuse the independently validated metadata reader migrated into this package.
# Its strict four-content coverage describes this measured bundle; scene instance
# count and the production catalog are unrestricted by that coverage.
from .quality_metadata import MetadataError, load_dataset

QUALITY_POLICY = (
    "arithmetic mean measured PSNR in dB over every GOP frame and all 24 reference "
    "views at scale 1.0, including training and held-out views; interactive camera "
    "uses this aggregate with current-pose visibility, without view correspondence"
)


@dataclass(frozen=True, slots=True)
class MeasuredProfile:
    content_id: str
    gop_index: int
    crf: int
    psnr_db: float
    duration_s: float
    original_transfer_bytes: int
    original_cost_bps: int
    quality_profile_sha256: str
    representation_manifest_sha256: str
    sample_count: int


def metadata_directory(location):
    """Accept either the bundle itself, its parent data_stream, or its index."""
    path = Path(location).expanduser().resolve()
    if path.is_file():
        if path.name != "quality-index.json":
            raise MetadataError("quality metadata file must be quality-index.json")
        path = path.parent
    nested = path / "streamstgs_8i_quality_metadata"
    if not (path / "quality-index.json").is_file() and nested.is_dir():
        path = nested
    return path


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class QualityCatalog:
    """Measured object × GOP × CRF table attached to an existing ContentCatalog.

    ``original_cost_bps`` is audit/parity data only. An HTTP publication wraps
    original members into CPSEG plus hash-shared assets, so executable planners
    must use the production segment's actual missing resource union instead.
    """
    policy = QUALITY_POLICY

    def __init__(self, catalog, metadata_root):
        self.catalog = catalog
        self.metadata_root = metadata_directory(metadata_root)
        normalized, audit = load_dataset(self.metadata_root)
        profiles = {}
        manifests = {}
        reconstructions = {}
        for row in normalized:
            key = (row["object"], row["gop"], row["crf"])
            profiles[key] = MeasuredProfile(
                *key, row["psnr_db"], row["duration_seconds"],
                row["transfer_bytes"], row["cost_bps"], row["quality_profile_sha256"],
                row["representation_manifest_sha256"], row["sample_count"])
            directory = self.metadata_root / key[0] / "content" / f"gop_{key[1]:03d}_crf_{key[2]}"
            # All bytes and cross-links were hashed by the reused loader above.
            manifests[key] = json.loads((directory / "manifest.json").read_bytes())
            reconstructions[key] = json.loads((directory / "reconstruction.json").read_bytes())
        self.profiles = MappingProxyType(profiles)
        self.audit = MappingProxyType(audit)
        self._manifests = manifests
        self._reconstructions = reconstructions
        self._validate_catalog()

    def profile(self, content_id, gop_index, crf):
        try:
            return self.profiles[content_id, gop_index, crf]
        except KeyError as error:
            raise MetadataError(f"No measured PSNR for {content_id}/gop={gop_index}/crf={crf}") from error

    def psnr(self, content_id, gop_index, crf):
        return self.profile(content_id, gop_index, crf).psnr_db

    def _validate_catalog(self):
        for content in self.catalog.contents.values():
            for representation in content.representations.values():
                for segment in representation.segments:
                    key = (content.id, segment.gop_index, representation.crf)
                    profile = self.profile(*key)
                    original = self._manifests[key]
                    reconstruction = self._reconstructions[key]
                    mapping = reconstruction["segment"]
                    if not math.isclose(segment.duration, profile.duration_s, abs_tol=1e-10, rel_tol=0):
                        raise MetadataError(f"Publication/measured GOP duration mismatch: {key}")
                    actual = tuple((f.source_frame, f.seconds, f.model_time) for f in segment.frames)
                    expected = tuple((f["frame"], f["seconds"], f["model_time"]) for f in mapping["timestamps"])
                    if actual != expected or content.fps != mapping["fps"]:
                        raise MetadataError(f"Publication/measured source frame or model-time mapping mismatch: {key}")
                    if set(segment.members) != set(original["files"]) | {"manifest.json"}:
                        raise MetadataError(f"Publication/measured codec member coverage mismatch: {key}")
                    shared = {a.id: a for a in segment.shared_assets}
                    for name, asset_id in segment.members.items():
                        if asset_id:
                            record = original["files"].get(name)
                            asset = shared[asset_id]
                            if record is None or (asset.bytes, asset.sha256) != (record["bytes"], record["sha256"]):
                                raise MetadataError(f"Publication/measured shared asset mismatch: {key}/{name}")

    def verify_materialized(self, content_id, gop_index, crf, package_path):
        """Check a downloaded, materialized GOP before its decoder is invoked.

        HTTP media still enters through Request API and ResourceCache. Reading
        their verified package here does not bypass transport. The measured
        manifest binds every compressed member, including shared assets, to the
        exact representation used offline. This catches an unrelated publication
        reusing the same content/CRF labels before claiming measured quality.
        """
        profile = self.profile(content_id, gop_index, crf)
        root = Path(package_path).resolve()
        manifest = root / "manifest.json"
        if not manifest.is_file() or _sha(manifest) != profile.representation_manifest_sha256:
            raise MetadataError(f"Downloaded representation differs from measured manifest: {content_id}/{gop_index}/{crf}")
        original = self._manifests[content_id, gop_index, crf]
        for name, record in original["files"].items():
            # Names originate in the already verified bundle, nevertheless keep
            # this boundary strict if another measured bundle is supported later.
            path = (root / name).resolve()
            if path.parent != root or not path.is_file() or path.stat().st_size != record["bytes"] or _sha(path) != record["sha256"]:
                raise MetadataError(f"Downloaded member differs from measured representation: {content_id}/{gop_index}/{crf}/{name}")
        return profile

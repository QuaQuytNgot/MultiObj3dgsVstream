"""Strict offline SSE/contribution metadata and physical resource closures.

The existing publication catalog remains authoritative when an MPD is supplied.
The isolated schema additionally describes progressive research fixtures; it
does not alter the production standalone StreamSTGS publication format.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType

from src_final.multiobj3dgs.shared.metadata import AssetRef
from src_final.multiobj3dgs.client.metadata import MPDParser

from .common import finite, integer, read_json, require

SCHEMA = "visibility-weighted-sse.profiles.v1"


def _identity(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label} is required")
    return value


def _sampling(ids, weights):
    require(isinstance(ids, list) and ids, "offline_sample_ids must be nonempty and unique")
    for value in ids:
        _identity(value, "offline sample identity")
    require(len(set(ids)) == len(ids), "offline_sample_ids must be nonempty and unique")
    require(isinstance(weights, list) and len(weights) == len(ids),
            "offline_sample_weights must match sample IDs")
    weights = tuple(finite(v, "offline sample weight") for v in weights)
    require(all(v >= 0 for v in weights) and math.isclose(math.fsum(weights), 1., abs_tol=1e-10, rel_tol=0),
            "offline sample weights must be nonnegative and sum to one")
    return MappingProxyType({"sample_ids": tuple(ids), "sample_weights": weights})


def _compatibility(value):
    require(isinstance(value, dict), "profile compatibility metadata required")
    resolution = value.get("resolution")
    require(isinstance(resolution, list) and len(resolution) == 2, "profile resolution required")
    for item in resolution:
        integer(item, "resolution", 1)
    require(value.get("coordinate_convention") == "opencv_positive_z_column_vectors",
            "unsupported profile coordinate convention")
    require(value.get("contribution_convention") == "continuous_alpha_weighted_pixels",
            "unsupported alpha/contribution semantics")
    require(value.get("time_aggregation") in (
        "normalized_camera_time_mean", "SSE_camera_mean_timestamp_sum_alpha_camera_time_mean"),
        "unsupported declared offline SSE/contribution aggregation")
    _identity(value.get("distribution_id"), "offline distribution ID")
    _identity(value.get("proxy_representation"), "offline proxy representation")
    require(value.get("transform_convention") == "positive_similarity_object_to_world",
            "explicit object-to-world transform convention required")
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class QualityProfile:
    quality_id: str
    representation_id: str
    average_solo_sse: float
    resource_ids: tuple[str, ...]
    dependencies: tuple[str, ...]
    own_resource_ids: tuple[str, ...]


@dataclass(frozen=True)
class ObjectProfile:
    object_id: str
    instance_id: str
    gop_id: str
    offline_mean_solo_contribution: float
    qualities: tuple[QualityProfile, ...]
    sampling: object
    source_gop_index: int | None = None


@dataclass(frozen=True)
class OfflineProfiles:
    content_revision: str
    profile_revision: str
    representation_family: str
    compatibility: object
    resources: object
    objects: object
    provenance: object

    def for_gop(self, gop_id):
        rows = tuple(row for (instance, gop), row in sorted(self.objects.items()) if gop == gop_id)
        require(bool(rows), f"Missing offline profiles for GOP {gop_id}")
        return rows


def profiles_from_dict(data):
    require(data.get("schema") == SCHEMA, "unsupported offline profile schema")
    family = data.get("representation_family")
    require(family in ("standalone", "progressive"), "declare one representation family per run")
    compatibility = _compatibility(data.get("compatibility"))
    provenance = data.get("provenance", {})
    require(provenance.get("information_role") == "OFFLINE_AVERAGE_SOLO_SSE" and
            provenance.get("joint_quality_used") is False,
            "profile must declare offline solo SSE without joint-quality input")
    resources = {}
    sizes = {}
    require(isinstance(data.get("resources"), dict) and data["resources"], "physical resources required")
    for identifier, record in data["resources"].items():
        asset = AssetRef(identifier, record.get("url", "metadata:" + identifier),
                         record["bytes"], record["sha256"])
        require(asset.sha256 not in sizes or sizes[asset.sha256] == asset.bytes,
                "same physical SHA256 cannot have conflicting byte sizes")
        sizes[asset.sha256] = asset.bytes
        resources[identifier] = asset
    objects = {}
    seen_instances = set()
    require(isinstance(data.get("objects"), list) and data["objects"], "object profiles required")
    for obj in data["objects"]:
        object_id = _identity(obj.get("object_id"), "object ID")
        instance_id = _identity(obj.get("instance_id"), "instance ID")
        require(instance_id not in seen_instances, "duplicate instance profile")
        seen_instances.add(instance_id)
        require(isinstance(obj.get("gops"), list) and obj["gops"], "GOP profiles required")
        for gop in obj["gops"]:
            gop_id = _identity(gop.get("gop_id"), "GOP ID")
            denominator = finite(gop.get("offline_mean_solo_contribution"), "offline mean solo contribution")
            require(denominator > 0, "offline mean solo contribution must be positive")
            require(denominator <= math.prod(compatibility["resolution"]) + 1e-8,
                    "offline alpha contribution exceeds viewport pixel count")
            sampling = _sampling(gop.get("offline_sample_ids"), gop.get("offline_sample_weights"))
            for name in ("denominator_metadata", "sse_metadata"):
                if name in gop:
                    require(gop[name] == dict(compatibility), f"incompatible {name}")
            require(("denominator_metadata" in gop) == ("sse_metadata" in gop),
                    "both denominator and SSE metadata must be declared together")
            records = gop.get("offline_sse")
            require(isinstance(records, list) and 1 <= len(records) <= 4,
                    "existing native solver supports one to four quality candidates")
            ids = [_identity(q.get("quality_id"), "quality ID") for q in records]
            require(len(set(ids)) == len(ids), "duplicate quality ID")
            base = gop.get("base_quality_id", ids[0])
            require(base == ids[0], "first quality must be the mandatory base/Q0")
            by_id = dict(zip(ids, records))
            rep_ids = [_identity(q.get("representation_id"), "representation ID") for q in records]
            require(len(set(rep_ids)) == len(rep_ids), "duplicate representation ID")
            closures = {}
            dependencies = {}
            visiting = set()

            def expand(identifier):
                if identifier in closures:
                    return closures[identifier]
                require(identifier in by_id, f"unknown quality dependency {identifier}")
                require(identifier not in visiting, "cyclic progressive quality dependencies")
                visiting.add(identifier)
                q = by_id[identifier]
                own = q.get("resource_ids")
                require(isinstance(own, list) and own and len(set(own)) == len(own),
                        "quality requires unique physical resource IDs")
                require(all(v in resources for v in own), "unknown physical resource")
                deps = q.get("dependencies", [])
                require(isinstance(deps, list) and len(set(deps)) == len(deps), "invalid dependencies")
                require(family == "progressive" or not deps,
                        "standalone representations cannot have quality-layer dependencies")
                closure = set(own)
                transitive = set()
                for dep in deps:
                    closure.update(expand(dep))
                    transitive.update((dep, *dependencies[dep]))
                visiting.remove(identifier)
                dependencies[identifier] = tuple(qid for qid in ids if qid in transitive)
                closures[identifier] = tuple(sorted(closure))
                return closures[identifier]

            qualities = []
            for qid, rep_id, q in zip(ids, rep_ids, records):
                closure = expand(qid)
                if "offline_sample_ids" in q or "offline_sample_weights" in q:
                    require(q.get("offline_sample_ids") == list(sampling["sample_ids"]) and
                            q.get("offline_sample_weights") == list(sampling["sample_weights"]),
                            "quality SSE sampling differs from shared offline contribution distribution")
                if family == "progressive":
                    require((qid == base and not dependencies[qid]) or base in dependencies[qid],
                            "each progressive enhancement must depend on mandatory base")
                sse = finite(q.get("average_solo_sse"), "offline Average Solo SSE")
                require(sse >= 0, "offline SSE must be nonnegative")
                qualities.append(QualityProfile(qid, rep_id, sse, closure,
                                                dependencies[qid], tuple(q["resource_ids"])))
            key = instance_id, gop_id
            require(key not in objects, "duplicate instance/GOP profile")
            source_gop = gop.get("source_gop_index")
            if source_gop is not None:
                integer(source_gop, "source GOP index")
            objects[key] = ObjectProfile(object_id, instance_id, gop_id, denominator,
                                         tuple(qualities), sampling, source_gop)
    return OfflineProfiles(_identity(data.get("content_revision"), "content revision"),
                           _identity(data.get("profile_revision"), "profile revision"), family,
                           compatibility, MappingProxyType(resources), MappingProxyType(objects),
                           MappingProxyType(dict(provenance)))


def validate_publication(profiles, catalog):
    """Compare actual production CPSEG/shared-asset closures, never MPD average rate."""
    require(profiles.representation_family == "standalone", "production publication is standalone StreamSTGS")
    require(profiles.content_revision == catalog.revision, "publication/content revision mismatch")
    instances = {instance.id: instance for instance in catalog.instances}
    for profile in profiles.objects.values():
        require(profile.instance_id in instances and instances[profile.instance_id].content_id == profile.object_id,
                "publication object/instance identity mismatch")
        require(profile.source_gop_index is not None, "publication validation needs source_gop_index")
        content = catalog.contents[profile.object_id]
        for quality in profile.qualities:
            require(quality.representation_id in content.representations, "publication representation missing")
            segments = content.representations[quality.representation_id].segments
            require(profile.source_gop_index < len(segments), "publication GOP missing")
            expected = {r.sha256: r.bytes for r in segments[profile.source_gop_index].resources}
            actual = {profiles.resources[key].sha256: profiles.resources[key].bytes for key in quality.resource_ids}
            require(actual == expected, "offline profile physical closure differs from publication")


def load_profiles(path):
    data = read_json(path)
    profiles = profiles_from_dict(data)
    if data.get("publication_mpd"):
        mpd = (Path(path).resolve().parent / data["publication_mpd"]).resolve()
        validate_publication(profiles, MPDParser.from_local(mpd))
    return profiles


def mean_contribution_from_artifact(path, expected_compatibility, *, max_samples=100000,
                                    max_bytes=32 * 1024 * 1024):
    """Bounded CPU metadata construction from already validated contribution rows.

    Each record explicitly binds a camera/time identity and normalized weight.
    This helper never estimates area or generates a new visibility measurement.
    """
    integer(max_samples, "max_samples", 1)
    integer(max_bytes, "max_bytes", 1)
    require(Path(path).stat().st_size <= max_bytes, "offline contribution artifact exceeds byte bound")
    data = read_json(path)
    require(data.get("schema") == "visibility-weighted-sse.offline-contribution.v1",
            "unsupported offline contribution artifact")
    require(data.get("compatibility") == dict(expected_compatibility), "incompatible offline contribution artifact")
    require(data.get("provenance", {}).get("validated") is True, "existing contribution artifact must be validated")
    require(data.get("provenance", {}).get("joint_quality_used") is False,
            "offline contribution construction cannot consume joint-quality inputs")
    samples = data.get("samples")
    require(isinstance(samples, list) and 0 < len(samples) <= max_samples, "offline contribution sample bound")
    sampling = _sampling([v.get("sample_id") for v in samples], [v.get("weight") for v in samples])
    contributions = [finite(v.get("solo_contribution"), "offline solo contribution") for v in samples]
    require(all(v >= 0 for v in contributions), "offline contributions must be nonnegative")
    mean = math.fsum(w * v for w, v in zip(sampling["sample_weights"], contributions))
    require(mean > 0, "constructed offline mean contribution must be positive")
    return {"offline_mean_solo_contribution": mean,
            "offline_sample_ids": list(sampling["sample_ids"]),
            "offline_sample_weights": list(sampling["sample_weights"]),
            "denominator_metadata": dict(expected_compatibility), "provenance": data["provenance"]}


def build_utilities(profile, visible_contribution):
    """The sole policy: (visible/offline-mean) * (base-SSE - quality-SSE)."""
    visible = finite(visible_contribution, "GOP visible contribution")
    require(visible >= 0, "GOP visible contribution must be nonnegative")
    denominator = finite(profile.offline_mean_solo_contribution, "offline mean contribution")
    require(denominator > 0, "offline mean contribution must be positive")
    weight = visible / denominator
    require(math.isfinite(weight), "visibility weight overflow")
    baseline = profile.qualities[0].average_solo_sse
    rows = []
    for q in profile.qualities:
        gain = baseline - q.average_solo_sse
        utility = weight * gain
        require(math.isfinite(utility), "weighted SSE utility overflow")
        rows.append({"quality_id": q.quality_id, "representation_id": q.representation_id,
                     "offline_sse": q.average_solo_sse, "sse_improvement": gain,
                     "weighted_utility": utility, "dependencies": list(q.dependencies),
                     "resource_ids": list(q.resource_ids)})
    return {"object_id": profile.object_id, "instance_id": profile.instance_id,
            "gop_id": profile.gop_id, "offline_mean_solo_contribution": denominator,
            "visible_contribution": visible, "weight": weight, "qualities": rows}

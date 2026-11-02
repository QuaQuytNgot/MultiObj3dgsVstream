"""Bind source quality/cost metadata to global scene GOPs and fresh requests."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
import time

from experiments.adaptation_baselines.interface import AdaptationPlanner
from experiments.adaptation_baselines.runner import load_experiment
from experiments.visibility_weighted_sse.common import digest, read_json, require
from experiments.visibility_weighted_sse.trace import TraceCameraProvider
from experiments.visibility_weighted_sse.visibility import CachedVisibilityAdapter
from src_final.multiobj3dgs.shared.metadata import AssetRef


def independent_instance_transfer_profiles(profiles):
    """Namespace transfer hashes per scene instance while retaining source SHA.

    The optimizer's physical union is hash-based. A cloned source object is an
    independent stream instance, so identical package bytes across instances
    must still incur a separate transfer unless the experiment explicitly
    enables inter-instance payload sharing. Sharing inside one instance remains
    content-hash based.
    """
    resources, objects = {}, {}
    aliases = {}
    for key, row in profiles.objects.items():
        per_instance = {}

        def transfer_resource(resource_id):
            source = profiles.resources[resource_id]
            if source.sha256 not in per_instance:
                transfer_sha = digest({
                    "policy": "independent_scene_instance_payload_v1",
                    "instance_id": row.instance_id,
                    "source_sha256": source.sha256,
                })
                transfer_id = "instance_transfer_" + transfer_sha
                existing = resources.get(transfer_id)
                asset = AssetRef(transfer_id, source.url, source.bytes, transfer_sha)
                if existing is not None and existing.bytes != asset.bytes:
                    raise ValueError("Equal source SHA256 has conflicting byte lengths")
                resources.setdefault(transfer_id, asset)
                aliases[transfer_sha] = {"instance_id": row.instance_id,
                                         "source_sha256": source.sha256,
                                         "bytes": source.bytes}
                per_instance[source.sha256] = transfer_id
            return per_instance[source.sha256]

        qualities = []
        for quality in row.qualities:
            qualities.append(replace(quality,
                resource_ids=tuple(transfer_resource(resource_id) for resource_id in quality.resource_ids),
                own_resource_ids=tuple(transfer_resource(resource_id) for resource_id in quality.own_resource_ids)))
        objects[key] = replace(row, qualities=tuple(qualities))
    provenance = {**dict(profiles.provenance),
        "resource_transfer_policy": "independent_instance_no_inter_instance_payload_sharing",
        "resource_transfer_aliases": aliases,
    }
    return replace(profiles, resources=MappingProxyType(resources),
                   objects=MappingProxyType(objects), provenance=MappingProxyType(provenance))


def build_planner(config, config_path, timeline, scenario):
    """Only metadata is repeated; video deliveries are never made resident."""
    records = config.get("inputs", {})
    require("adaptation_config" in records and "visibility_cache" in records,
            "Prepared source adaptation_config and global visibility_cache are required; geometry is not visibility")
    base = Path(config_path).resolve().parent
    source = load_experiment(base / records["adaptation_config"]["path"])["planner"]
    require(not source.cached_resource_ids, "Source configuration must not grant resident video cache credit")
    require(not (set(config["adaptation"]["methods"]) - {"SOLO_SSE", "VISIBILITY_WEIGHTED_SSE"})
            or source.metadata is not None, "Selected baselines require independent source quality metadata")
    profiles = source.profiles
    expected_kind = "synthetic_fixture" if config["input_kind"] == "synthetic_fixture" else "validated_cached_artifacts"
    require(read_json(base / records["adaptation_config"]["path"])["input_kind"] == expected_kind,
            "Source adaptation metadata input kind mismatch")
    objects, metadata_objects = {}, {}
    for segment in timeline:
        source_gid = config["source_gop_ids"][segment.source_gop_index]
        source_rows = profiles.for_gop(source_gid)
        for instance in scenario["instances"]:
            matches = [p for p in source_rows if p.object_id == instance["object_id"]]
            require(len(matches) == 1, f"Source object/GOP profile missing or ambiguous: {instance['object_id']}/{source_gid}")
            original = matches[0]
            row = replace(original, instance_id=instance["instance_id"], gop_id=segment.gop_id)
            objects[row.instance_id, row.gop_id] = row
            if source.metadata is not None:
                metadata_objects[row.instance_id, row.gop_id] = deepcopy(
                    source.metadata["objects"][original.instance_id, source_gid])
        source_gop = source.trace.get_gop(source_gid)
        require(abs((source_gop["end_s"] - source_gop["start_s"]) - segment.source_segment_duration_s) < 1e-9,
                "Physical source GOP duration disagrees with source timeline")
    global_profiles = replace(profiles, objects=MappingProxyType(objects))
    sharing = config.get("adaptation", {}).get("resource_sharing", "global_sha256")
    if sharing == "per_instance":
        global_profiles = independent_instance_transfer_profiles(global_profiles)
    metadata = {**source.metadata, "objects": metadata_objects} if source.metadata is not None else None
    trace = TraceCameraProvider(scenario["trace"])
    started = time.perf_counter()
    cache = read_json(base / records["visibility_cache"]["path"])
    require(cache.get("input_kind") == expected_kind, "Visibility cache input kind mismatch")
    visibility = CachedVisibilityAdapter(cache, trace, content_revision=profiles.content_revision)
    acquisition_ms = (time.perf_counter() - started)*1000
    inherited = "quality_levels" not in config["adaptation"]
    expected = config["adaptation"].get("quality_levels", source.allowed_quality_ids)
    # Source instance IDs and scenario instance IDs can differ. Explicit quality
    # configuration uses scenario IDs; inherited source masks are rebound.
    if inherited and expected is not None:
        expected = {i["instance_id"]: expected[next(p.instance_id for p in source_rows if p.object_id == i["object_id"])]
                    for i in scenario["instances"]}
    if expected is not None:
        require(isinstance(expected, dict), "Eligible quality levels must map scenario instance IDs")
        for segment in timeline:
            rows = [row for (iid, gid), row in objects.items() if gid == segment.gop_id]
            require(set(expected) == {r.instance_id for r in rows}, "Eligible quality instance coverage mismatch")
            for row in rows:
                values = expected[row.instance_id]
                require(isinstance(values, list) and values and all(isinstance(v, str) for v in values)
                        and len(set(values)) == len(values) and set(values) <= {q.quality_id for q in row.qualities},
                        "Unknown or duplicate eligible quality")
                require(row.qualities[0].quality_id in values, "Eligible qualities must include mandatory base")
    planner = AdaptationPlanner(global_profiles, trace, visibility, baseline_metadata=metadata,
        cached_resource_ids=(), allowed_quality_ids=expected,
        solver_limits=config["adaptation"].get("solver_limits", source.solver_limits),
        expected_object_count=scenario["object_count"])
    return planner, {"cached_visibility_acquisition_ms": acquisition_ms,
                     "visibility_analysis_ms": None,
                     "visibility_analysis_scope": "not_executed_cached_outputs_only"}

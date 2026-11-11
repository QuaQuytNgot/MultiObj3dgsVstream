"""CPU-only adapter for frozen Visibility Analyzer contribution outputs.

The production analyzer requires CUDA contribution rasterization. This adapter
only retrieves its alpha/transmittance measurements and reuses its output
contracts and numerical validation; it never instantiates a renderer.
"""
from __future__ import annotations

import json
import math
from typing import Mapping

from src_final.multiobj3dgs.client.visibility.analyzer import VisibilityResult, _object_visibility

from .trace import COORDINATE_CONVENTION, TraceCameraProvider, finite, nonempty


def validate_compatibility(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("Explicit visibility/profile compatibility metadata required")
    row = dict(value)
    resolution = row.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2 or any(type(v) is not int or v <= 0 for v in resolution):
        raise ValueError("Compatibility resolution must be positive integer [width,height]")
    required = {"coordinate_convention": COORDINATE_CONVENTION,
                "contribution_convention": "continuous_alpha_weighted_pixels"}
    for key, expected in required.items():
        if row.get(key) != expected:
            raise ValueError(f"Unsupported or missing compatibility field {key}")
    nonempty(row.get("time_aggregation"), "time_aggregation")
    nonempty(row.get("proxy_representation"), "proxy_representation")
    if row.get("transform_convention") != "positive_similarity_object_to_world":
        raise ValueError("Explicit positive-similarity object-to-world transform convention required")
    # Further identities (proxy representation, renderer, content state,
    # distribution, transform convention) are compared without inventing them.
    return row


class CachedVisibilityAdapter:
    def __init__(self, data: Mapping, trace_provider: TraceCameraProvider,
                 expected_compatibility: Mapping | None = None,
                 content_revision: str | None = None):
        self.data = json.loads(json.dumps(data, allow_nan=False))
        data, self.trace_provider = self.data, trace_provider
        if data.get("schema") != "visibility-weighted-sse.visibility.v1":
            raise ValueError("Unsupported cached visibility schema")
        if data.get("trace_id") != trace_provider.trace_id:
            raise ValueError("Visibility trace identity mismatch")
        nonempty(data.get("content_revision"), "visibility content_revision")
        if content_revision is not None and data["content_revision"] != content_revision:
            raise ValueError("Visibility content revision mismatch")
        self.compatibility = validate_compatibility(data.get("compatibility"))
        if expected_compatibility is not None and self.compatibility != validate_compatibility(expected_compatibility):
            raise ValueError("Visibility/offline profile compatibility metadata mismatch")
        self.provenance = data.get("provenance", {})
        if not isinstance(self.provenance, dict) or self.provenance.get("information_role") != "VISIBILITY_PROXY":
            raise ValueError("Independent VISIBILITY_PROXY provenance required")
        if self.provenance.get("candidate_independent") is not True or self.provenance.get("joint_quality_used") is not False:
            raise ValueError("Candidate joint RGB or quality evaluation cannot enter the policy")
        nonempty(self.provenance.get("source"), "visibility provenance source")
        self._results = {}
        gops = data.get("gops")
        if not isinstance(gops, list) or not gops:
            raise ValueError("Cached visibility GOP data required")
        for gop in gops:
            key = str(gop["gop_id"])
            if key in self._results:
                raise ValueError("Duplicate cached visibility GOP")
            poses = trace_provider.get_trajectory(gop["gop_id"])
            samples = gop.get("samples")
            if not isinstance(samples, list) or len(samples) != len(poses):
                raise ValueError("Visibility must cover the complete upcoming trace")
            results = []
            for sample, pose in zip(samples, poses):
                if finite(sample["timestamp"], "visibility timestamp") != pose.timestamp or sample.get("pose_identity") != pose.identity:
                    raise ValueError("Cached visibility timestamp/camera/transform/content-state identity mismatch")
                resolution = [pose.camera.width, pose.camera.height]
                if resolution != self.compatibility["resolution"]:
                    raise ValueError("Trace and cached visibility viewport resolution mismatch")
                per_object = sample.get("per_object")
                if not isinstance(per_object, dict) or set(per_object) != {row["instance_id"] for row in pose.instances}:
                    raise ValueError("Visibility must include every declared object instance")
                objects = {}
                pixels = math.prod(resolution)
                tolerance = 1e-6 + 1e-5 * pixels
                for instance_id, measured in per_object.items():
                    # epsilon=0 classifies only exact zero as outside viewport;
                    # numerical contribution correction is production behavior.
                    value = _object_visibility(instance_id,
                        finite(measured["A_visible"], "A_visible"),
                        finite(measured["A_solo"], "A_solo"), 0.)
                    if value.A_solo > pixels + tolerance:
                        raise ValueError("Solo alpha contribution cannot exceed viewport pixel count")
                    objects[instance_id] = value
                if math.fsum(row.A_visible for row in objects.values()) > pixels + tolerance:
                    raise ValueError("Joint alpha contribution cannot exceed viewport pixel count")
                results.append(VisibilityResult(pose.timestamp, tuple(resolution), objects,
                    {"backend": "cached_visibility_analyzer_outputs", "area_units": "alpha_weighted_pixels",
                     "revision": data["content_revision"], "pose_identity": pose.identity,
                     "provenance": self.provenance}))
            self._results[key] = tuple(results)

    def get_results(self, gop_id) -> tuple[VisibilityResult, ...]:
        try:
            return self._results[str(gop_id)]
        except KeyError:
            raise ValueError(f"Missing cached visibility GOP data: {gop_id}") from None

    def aggregate(self, gop_id) -> dict:
        poses = self.trace_provider.get_trajectory(gop_id)
        results, weights = self.get_results(gop_id), self.trace_provider.temporal_weights(gop_id)
        per_object = {}
        for instance in poses[0].instances:
            key, series = instance["instance_id"], []
            for pose, result, weight in zip(poses, results, weights):
                source = next(row for row in pose.instances if row["instance_id"] == key)
                row = result.per_object[key]
                series.append({"timestamp": pose.timestamp, "temporal_weight": weight,
                    "A_solo": row.A_solo, "A_visible": row.A_visible,
                    "occlusion_ratio": None if row.A_solo == 0 else 1-row.A_visible/row.A_solo,
                    "pose_identity": pose.identity, "source_camera_id": source["source_camera_id"],
                    "source_state_id": source["source_state_id"],
                    "source_local_camera_identity": source["source_local_camera_identity"],
                    "source_local_camera": source["source_local_camera"]})
            solo = math.fsum(weight*row.per_object[key].A_solo for weight, row in zip(weights, results))
            visible = math.fsum(weight*row.per_object[key].A_visible for weight, row in zip(weights, results))
            per_object[key] = {"object_id": instance["object_id"], "instance_id": key,
                "gop_id": gop_id, "trace_id": self.trace_provider.trace_id,
                "A_solo": solo, "A_visible": visible,
                "occlusion_ratio": None if solo == 0 else 1-visible/solo,
                "time_series": series, "provenance": dict(self.provenance)}
        return {"gop_id": gop_id, "trace_id": self.trace_provider.trace_id,
                "timestamps": [pose.timestamp for pose in poses], "temporal_weights": list(weights),
                "temporal_integration": "left_hold_over_half_open_GOP",
                "per_object": per_object, "compatibility": dict(self.compatibility),
                "execution_mode": "cached_input_CPU_only"}

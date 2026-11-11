"""Frozen future camera geometry under the explicit oracle-trace assumption.

No interpolation or predictor is run. Every recorded sample in the requested
half-open GOP is returned. Left-hold integration gives sample t the interval
from t to the next sample (or GOP end), divided by the GOP duration.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from src_final.multiobj3dgs.client.visibility.prediction import PosePrediction
from src_final.multiobj3dgs.shared.geometry import Camera, validate_transform

from .common import digest as identity, finite


COORDINATE_CONVENTION = "opencv_positive_z_column_vectors"
ORACLE_CAMERA_TRACE = "ORACLE_CAMERA_TRACE"


def nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def camera_to_dict(camera: Camera) -> dict:
    return {"width": camera.width, "height": camera.height,
            "K": camera.K.tolist(), "world_to_camera": camera.world_to_camera.tolist(),
            "near": camera.near, "far": camera.far}


def camera_from_dict(row: Mapping) -> Camera:
    if not isinstance(row, Mapping):
        raise ValueError("Calibrated camera metadata must be an object")
    for name in ("width", "height"):
        if type(row[name]) is not int or row[name] <= 0:
            raise ValueError("Camera dimensions must be positive integers")
    return Camera(row["width"], row["height"], row["world_to_camera"], row["K"],
                  finite(row.get("near", 0.01), "near"), finite(row.get("far", 100.0), "far"))


def source_local_camera(camera: Camera, object_to_world) -> Camera:
    """Express a calibrated world camera in positive-similarity object space.

    Dividing camera coordinates by the object scale preserves projected pixels
    while restoring a proper rigid extrinsic; clipping distances scale too.
    """
    transform = validate_transform(object_to_world)
    scale = float(np.linalg.norm(transform[:3, 0]))
    local = camera.world_to_camera @ transform
    local[:3] /= scale
    return Camera(camera.width, camera.height, local, camera.K,
                  camera.near / scale, camera.far / scale)


@dataclass(frozen=True)
class TracePose:
    timestamp: float
    camera: Camera
    instances: tuple[Mapping, ...]
    identity: str
    gop_id: Any
    prediction: PosePrediction


class TraceCameraProvider:
    """Camera Predictor-compatible geometry adapter, with no prediction model."""

    def __init__(self, trace: Mapping):
        # A deep copy prevents callers from mutating frozen identities later.
        self.trace = json.loads(json.dumps(trace, allow_nan=False))
        trace = self.trace
        if trace.get("schema") != "visibility-weighted-sse.trace.v1":
            raise ValueError("Unsupported trace schema")
        self.trace_id = nonempty(trace.get("trace_id"), "trace_id")
        if trace.get("coordinate_convention") != COORDINATE_CONVENTION:
            raise ValueError("Explicit positive-Z column-vector camera convention required")
        if trace.get("prediction_assumption") != ORACLE_CAMERA_TRACE or trace.get("predictor_enabled") is not False:
            raise ValueError("ORACLE_CAMERA_TRACE with predictor_enabled=false is required")
        raw_gops = trace.get("gops")
        if not isinstance(raw_gops, list) or not raw_gops:
            raise ValueError("Trace needs one or more GOPs")
        self.gops = tuple(raw_gops)
        self._trajectories, self._gops = {}, {}
        previous_end = None
        for gop in self.gops:
            gop_id = gop.get("gop_id")
            if isinstance(gop_id, bool) or not isinstance(gop_id, (str, int)) or str(gop_id) == "":
                raise ValueError("GOP identity must be a string or integer")
            key = str(gop_id)
            if key in self._gops:
                raise ValueError("Duplicate GOP identity")
            start, end = finite(gop["start_s"], "GOP start"), finite(gop["end_s"], "GOP end")
            if end <= start or (previous_end is not None and start < previous_end):
                raise ValueError("GOPs must be chronologically ordered and nonoverlapping")
            previous_end = end
            samples = gop.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"Missing samples for GOP {gop_id}")
            poses, last, expected_instances = [], None, None
            for sample in samples:
                timestamp = finite(sample["timestamp"], "Trace timestamp")
                if not start <= timestamp < end or (last is not None and timestamp <= last):
                    raise ValueError("Trace timestamps must increase strictly within the half-open GOP")
                if last is None and timestamp != start:
                    raise ValueError("First trace sample must align exactly to GOP start")
                last = timestamp
                camera = camera_from_dict(sample["camera"])
                instances = sample.get("instances")
                if not isinstance(instances, list) or not instances:
                    raise ValueError("Every pose requires explicit object instance transforms and content states")
                canonical_instances, seen = [], set()
                for instance in instances:
                    row = dict(instance)
                    instance_id = nonempty(row.get("instance_id"), "instance_id")
                    nonempty(row.get("object_id"), "object_id")
                    nonempty(row.get("source_camera_id"), "source_camera_id")
                    nonempty(row.get("source_state_id"), "source_state_id")
                    if instance_id in seen:
                        raise ValueError("Duplicate instance within camera sample")
                    seen.add(instance_id)
                    transform = validate_transform(row["object_to_world"])
                    row["object_to_world"] = transform.tolist()
                    row["source_local_camera"] = camera_to_dict(source_local_camera(camera, transform))
                    row["source_local_camera_identity"] = identity(row["source_local_camera"])
                    canonical_instances.append(row)
                instance_pairs = {(row["instance_id"], row["object_id"]) for row in canonical_instances}
                if expected_instances is not None and instance_pairs != expected_instances:
                    raise ValueError("Every trace pose must cover the same declared instances")
                expected_instances = instance_pairs
                canonical_instances.sort(key=lambda row: row["instance_id"])
                pose_identity = identity({"trace_id": self.trace_id, "gop_id": gop_id,
                                          "timestamp": timestamp, "camera": camera_to_dict(camera),
                                          "instances": canonical_instances})
                prediction = PosePrediction(camera, start, timestamp, ORACLE_CAMERA_TRACE)
                poses.append(TracePose(timestamp, camera,
                                       tuple(MappingProxyType(row) for row in canonical_instances),
                                       pose_identity, gop_id, prediction))
            self._gops[key], self._trajectories[key] = gop, tuple(poses)

    def get_gop(self, gop_id) -> Mapping:
        try:
            return self._gops[str(gop_id)]
        except KeyError:
            raise ValueError(f"Missing GOP data: {gop_id}") from None

    def get_trajectory(self, gop_id) -> tuple[TracePose, ...]:
        self.get_gop(gop_id)
        return self._trajectories[str(gop_id)]

    def temporal_weights(self, gop_id) -> tuple[float, ...]:
        gop, poses = self.get_gop(gop_id), self.get_trajectory(gop_id)
        duration = gop["end_s"] - gop["start_s"]
        boundaries = [pose.timestamp for pose in poses] + [gop["end_s"]]
        weights = tuple((right-left)/duration for left, right in zip(boundaries, boundaries[1:]))
        if any(weight <= 0 for weight in weights) or not math.isclose(math.fsum(weights), 1., abs_tol=1e-12):
            raise ValueError("Invalid GOP temporal integration weights")
        return weights

"""Configurable CPU geometry for the three paper experiment scenario families.

Generated poses are the ground truth and the oracle viewport input. These
generators do not render, measure visibility, or certify source-camera support.
Camera calibration and positive-similarity transforms reuse ``src_final``;
the output trace reuses the existing Weighted SSE trace schema and provider.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np

from src_final.multiobj3dgs.shared.geometry import Camera, similarity_matrix, validate_transform
from experiments.visibility_weighted_sse.trace import (
    COORDINATE_CONVENTION, ORACLE_CAMERA_TRACE, camera_from_dict, camera_to_dict,
)
from .timeline import Timeline, build_timeline, exact_time


SCENARIO_KINDS = ("rotation_orbit", "long_line", "random_spatial")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}")
    return result


def _vector(value, name):
    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a finite XYZ vector") from None
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite XYZ vector")
    return array


def _name(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _layout(config, count):
    positions = config.get("positions")
    if positions is not None:
        if not isinstance(positions, list) or len(positions) != count:
            raise ValueError("positions must contain exactly object_count XYZ vectors")
        return [_vector(value, "position") for value in positions]
    layout = config.get("layout", {})
    origin = _vector(layout.get("origin", [0., 0., 0.]), "layout.origin")
    if config["kind"] == "long_line":
        direction = _vector(layout.get("direction", [1., 0., 0.]), "layout.direction")
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("layout.direction must be nonzero")
        spacing = _number(layout.get("spacing", 2.), "layout.spacing", positive=True)
        return [origin+(i-(count-1)/2)*spacing*direction/norm for i in range(count)]
    if config["kind"] == "rotation_orbit":
        radius = _number(layout.get("radius", 2.), "layout.radius", positive=True)
        return [origin+radius*np.array([math.cos(2*math.pi*i/count), 0.,
                                      math.sin(2*math.pi*i/count)]) for i in range(count)]
    seed = _integer(config.get("seed", 0), "seed")
    try:
        bounds = np.asarray(layout.get("bounds", [[-4., 0., -4.], [4., 3., 4.]]), dtype=float)
    except (ValueError, TypeError):
        raise ValueError("layout.bounds must be finite increasing 2x3 XYZ bounds") from None
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("layout.bounds must be finite increasing 2x3 XYZ bounds")
    # The generator is local: scenario creation never changes global RNG state.
    rng = np.random.Generator(np.random.PCG64(seed))
    return list(origin+rng.uniform(bounds[0], bounds[1], size=(count, 3)))


def _instance_specs(config):
    sources = config.get("source_objects", [])
    if not isinstance(sources, list):
        raise ValueError("source_objects must be a list")
    source_rows = [{"object_id": row} if isinstance(row, str) else deepcopy(row) for row in sources]
    if any(not isinstance(row, Mapping) for row in source_rows):
        raise ValueError("source_objects entries must be names or metadata objects")
    by_object = {}
    for row in source_rows:
        oid = _name(row.get("object_id"), "object_id")
        if oid in by_object:
            raise ValueError("source_objects must not contain duplicate object IDs")
        by_object[oid] = row
    explicit = config.get("instances")
    if explicit is not None and (not isinstance(explicit, list) or not explicit):
        raise ValueError("instances must be a nonempty list")
    default_count = len(explicit) if explicit is not None else len(source_rows)
    count = _integer(config.get("object_count", default_count), "object_count", 1)
    if explicit is not None and len(explicit) != count:
        raise ValueError("instances must contain exactly object_count entries")
    if explicit is None and not source_rows:
        raise ValueError("source_objects or explicit instances are required")
    positions = _layout(config, count)
    specs, seen = [], set()
    for i in range(count):
        given = deepcopy(explicit[i]) if explicit is not None else deepcopy(source_rows[i % len(source_rows)])
        if not isinstance(given, Mapping):
            raise ValueError("instances entries must be metadata objects")
        oid = _name(given.get("object_id"), "object_id")
        row = {**deepcopy(by_object.get(oid, {})), **given}
        iid = _name(row.get("instance_id", f"object-{i:04d}"), "instance_id")
        if iid in seen:
            raise ValueError("instance_id must be unique")
        seen.add(iid)
        if "object_to_world" in row:
            if "positions" in config or any(key in row for key in ("position", "translation", "rotation", "scale")):
                raise ValueError("object_to_world cannot be combined with position/rotation/scale overrides")
            transform = validate_transform(row["object_to_world"])
        else:
            transform = similarity_matrix(
                row.get("position", row.get("translation", positions[i])),
                row.get("rotation", [1., 0., 0., 0.]),
                _number(row.get("scale", 1.), "scale", positive=True))
            validate_transform(transform)
        specs.append({"instance_id": iid, "object_id": oid,
                      "object_to_world": transform.tolist(),
                      "source_camera_id": _name(row.get("source_camera_id", "UNBOUND_SOURCE_CAMERA"), "source_camera_id"),
                      "source_states": row.get("source_states", {}),
                      "source_state_id_template": row.get("source_state_id_template"),
                      "source_revision": _name(row.get("source_revision", config.get("content_revision", "UNBOUND_CONTENT_REVISION")), "source_revision")})
    return specs


def _calibration(config):
    row = config.get("camera", {})
    width, height = _integer(row.get("width", 640), "camera.width", 1), _integer(row.get("height", 480), "camera.height", 1)
    if "K" in row:
        intrinsic = row["K"]
    else:
        fov = _number(row.get("horizontal_fov_degrees", 60.), "horizontal_fov_degrees", positive=True)
        if fov >= 180:
            raise ValueError("horizontal_fov_degrees must be < 180")
        focal = width/(2*math.tan(math.radians(fov)/2))
        intrinsic = [[focal, 0., (width-1)/2], [0., focal, (height-1)/2], [0., 0., 1.]]
    return Camera(width, height, np.eye(4), intrinsic,
                  _number(row.get("near", .01), "camera.near", positive=True),
                  _number(row.get("far", 1000.), "camera.far", positive=True))


def _look_at(base, center, target, up):
    forward = _vector(target, "camera.target")-_vector(center, "camera.center")
    norm = np.linalg.norm(forward)
    if norm == 0:
        raise ValueError("Camera center and target must differ")
    forward /= norm
    right = np.cross(forward, _vector(up, "camera.up"))
    norm = np.linalg.norm(right)
    if norm <= 1e-12:
        raise ValueError("Camera up vector must not be parallel to its viewing direction")
    right /= norm
    # Same OpenCV +Z forward / +Y image-down construction as the prior scene study.
    rotation = np.array([right, np.cross(forward, right), forward])
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, -rotation@center
    return Camera(base.width, base.height, matrix, base.K, base.near, base.far)


def _trajectory(config, timeline, base):
    camera = config.get("camera", {})
    default_kind = "linear" if config["kind"] == "long_line" else "orbit"
    trajectory = camera.get("trajectory", {"kind": default_kind})
    if not isinstance(trajectory, Mapping):
        raise ValueError("camera.trajectory must be a metadata object")
    kind = trajectory.get("kind", default_kind)
    if kind not in ("orbit", "pan", "linear", "trace"):
        raise ValueError("camera.trajectory.kind must be orbit, pan, linear, or trace; zoom is unsupported")
    up = _vector(trajectory.get("up", [0., 1., 0.]), "camera.up")
    period = _number(trajectory.get("period_s", timeline.total_duration_s), "camera.period_s", positive=True)
    if kind == "orbit":
        target = _vector(trajectory.get("target", [0., 0., 0.]), "camera.target")
        radius = _number(trajectory.get("radius", 8.), "camera.radius", positive=True)
        height = _number(trajectory.get("height", 2.), "camera.height")
        initial = math.radians(_number(trajectory.get("start_degrees", -90.), "start_degrees"))
        sweep = math.radians(_number(trajectory.get("sweep_degrees", 360.), "sweep_degrees"))
        def at(timestamp):
            theta = initial+sweep*float(timestamp)/period
            center = target+np.array([radius*math.cos(theta), height, radius*math.sin(theta)])
            return _look_at(base, center, target, up)
    elif kind == "pan":
        center = _vector(trajectory.get("center", [0., 1., -8.]), "camera.center")
        initial = math.radians(_number(trajectory.get("start_degrees", -60.), "start_degrees"))
        final = math.radians(_number(trajectory.get("end_degrees", 60.), "end_degrees"))
        pitch = math.radians(_number(trajectory.get("pitch_degrees", 0.), "pitch_degrees"))
        def at(timestamp):
            theta = initial+(final-initial)*min(float(timestamp)/period, 1.)
            direction = np.array([math.cos(pitch)*math.sin(theta), math.sin(pitch), math.cos(pitch)*math.cos(theta)])
            return _look_at(base, center, center+direction, up)
    elif kind == "linear":
        start = _vector(trajectory.get("start", [-5., 1., -8.]), "camera.start")
        end = _vector(trajectory.get("end", [5., 1., -8.]), "camera.end")
        direction = _vector(trajectory.get("look_direction", [0., 0., 1.]), "camera.look_direction")
        if np.linalg.norm(direction) == 0:
            raise ValueError("camera.look_direction must be nonzero")
        if not math.isclose(float((end-start)@direction), 0., abs_tol=1e-10):
            raise ValueError("Linear motion must be lateral to the viewing direction; dolly/zoom is unsupported")
        def at(timestamp):
            center = start+(end-start)*min(float(timestamp)/period, 1.)
            return _look_at(base, center, center+direction, up)
    else:
        rows = trajectory.get("samples")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Ground-truth camera trace requires timestamp_s/camera samples")
        records, previous = {}, None
        for row in rows:
            timestamp = exact_time(row["timestamp_s"], "camera trace timestamp")
            if previous is not None and timestamp <= previous:
                raise ValueError("Ground-truth camera timestamps must increase strictly")
            calibrated = camera_from_dict(row["camera"])
            if (calibrated.width != base.width or calibrated.height != base.height or
                    not np.array_equal(calibrated.K, base.K) or calibrated.near != base.near or calibrated.far != base.far):
                raise ValueError("Camera trace must use one fixed calibration; zoom is unsupported")
            records[timestamp], previous = calibrated, timestamp
        def at(timestamp):
            try:
                return records[timestamp]
            except KeyError:
                raise ValueError(f"Missing exact ground-truth camera sample at {timestamp}s") from None
        return at, records
    return at, None


def _state(instance, timestamp):
    states = instance["source_states"]
    if not isinstance(states, Mapping):
        raise ValueError("source_states must map exact content seconds to source_state_id")
    matches = [_name(identifier, "source_state_id") for key, identifier in states.items()
               if exact_time(key, "source_states content time") == timestamp]
    if len(matches) > 1:
        raise ValueError("source_states contains duplicated exact content time")
    if matches:
        return matches[0], "EXPLICIT_SOURCE_STATE_ID"
    if states:
        raise ValueError(f"No exact source_state_id supplied for content time {timestamp}")
    template = instance.get("source_state_id_template")
    if template is not None:
        _name(template, "source_state_id_template")
        try:
            state = template.format(object_id=instance["object_id"], source_revision=instance["source_revision"],
                                    content_time_fraction=str(timestamp), content_time_s=float(timestamp))
        except (ValueError, KeyError, IndexError, AttributeError):
            raise ValueError("Invalid source_state_id_template") from None
        return _name(state, "source_state_id"), "DECLARED_SOURCE_STATE_TEMPLATE"
    return f"{instance['source_revision']}/{instance['object_id']}/content/{timestamp}", "UNBOUND_CONTENT_TIME_DESCRIPTOR"


def generate_scenario(config, timeline=None):
    """Return static instances, global timeline, and a perfect future pose trace.

    ``source_objects`` can cycle to any positive N. ``instances`` or ``positions``
    override generated layouts; transforms use src_final's WXYZ quaternion and
    positive uniform-scale convention. Fixed intrinsics forbid optical zoom.
    Scene validity stays unmeasured until actual source-domain/quality data are
    supplied; deterministic source-time descriptors are never visibility data.
    """
    if not isinstance(config, Mapping) or config.get("kind") not in SCENARIO_KINDS:
        raise ValueError(f"scenario.kind must be one of {SCENARIO_KINDS}")
    timeline = timeline if timeline is not None else build_timeline(config)
    if not isinstance(timeline, Timeline):
        raise ValueError("timeline must be a Timeline")
    specs = _instance_specs(config)
    samples_per_gop = _integer(config.get("samples_per_gop", 4), "samples_per_gop", 1)
    base = _calibration(config)
    camera_at, recorded = _trajectory(config, timeline, base)
    declared_instances = [{key: value for key, value in row.items() if key not in ("source_states", "source_state_id_template")}
                          for row in specs]
    gops = []
    for segment in timeline:
        times = timeline.sample_times_exact(segment, samples_per_gop)
        if recorded is not None:
            start, end = timeline.segment_bounds_exact(segment)
            times = tuple(timestamp for timestamp in recorded if start <= timestamp < end)
            if not times or times[0] != start:
                raise ValueError(f"Ground-truth camera trace must include exact GOP start {start}s")
        samples = []
        for timestamp in times:
            content_time = timeline.content_time_exact(timestamp)
            instances = []
            for spec in specs:
                state_id, binding = _state(spec, content_time)
                row = {key: value for key, value in spec.items() if key not in ("source_states", "source_state_id_template")}
                row.update(source_state_id=state_id, source_state_binding=binding,
                           content_time_s=float(content_time), content_time_fraction=str(content_time))
                instances.append(row)
            samples.append({"timestamp": float(timestamp), "camera": camera_to_dict(camera_at(timestamp)),
                            "instances": instances})
        gops.append({"gop_id": segment.gop_id, "start_s": segment.global_start_s,
                     "end_s": segment.global_end_s, "samples": samples,
                     "request_id": segment.request_id, "loop_index": segment.loop_index,
                     "source_gop_index": segment.source_gop_index,
                     "content_start_s": segment.content_start_s, "content_end_s": segment.content_end_s})
    identity_payload = {"config": dict(config), "timeline": timeline.to_dict()}
    fingerprint = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    scene_id = f"{config['kind']}-{fingerprint[:20]}"
    trace = {"schema": "visibility-weighted-sse.trace.v1",
             "trace_id": _name(config.get("trace_id", scene_id), "trace_id"),
             "coordinate_convention": COORDINATE_CONVENTION,
             "prediction_assumption": ORACLE_CAMERA_TRACE, "predictor_enabled": False,
             "gops": gops}
    return {"schema": "streaming-evaluation.scenario.v1", "scene_id": scene_id,
            "scenario_sha256": fingerprint, "kind": config["kind"], "object_count": len(specs),
            "seed": config.get("seed", 0) if config["kind"] == "random_spatial" else None,
            "instances": declared_instances, "timeline": timeline.to_dict(), "trace": trace,
            "validity": {"geometry_only": True, "visibility_measurements_generated": False,
                         "source_domain_audit": "NOT_PERFORMED", "primary_quality_eligible": False,
                         "quality_measurements": "UNAVAILABLE"}}


def camera_rotation_orbit(config, timeline=None):
    return generate_scenario({**config, "kind": "rotation_orbit"}, timeline)


def long_line_objects(config, timeline=None):
    return generate_scenario({**config, "kind": "long_line"}, timeline)


def random_spatial_objects(config, timeline=None):
    return generate_scenario({**config, "kind": "random_spatial"}, timeline)

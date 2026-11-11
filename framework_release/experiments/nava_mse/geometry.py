"""CPU camera validation and NAVA geometry, preserving source-world +Y up.

The production Camera/positive-similarity implementation is reused. OpenCV's
camera basis does not imply an Rx(pi) rotation of the source geometry. The joint
construction keeps each decoder's *source-local camera center* at its selected
recorded center. The adapter conditions on that center, not the camera rotation.
The noncentral instances may therefore have different local raster orientations;
these exact local poses are recorded and used for their solo distortion images.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from src_final.multiobj3dgs.shared.geometry import Camera, validate_transform
from experiments.abr_network_evaluation.nava_visibility import projected_area


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def camera_metadata(camera):
    return dict(width=camera.width, height=camera.height,
                world_to_camera=camera.world_to_camera.tolist(), K=camera.K.tolist(),
                near=camera.near, far=camera.far)


def camera_identity(camera):
    """An exact lookup key; deliberately no rounding or nearest-pose matching."""
    return _digest(camera_metadata(camera))


def load_camera_records(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Recorded camera file must contain a nonempty list")
    ids = []
    for row in rows:
        if not isinstance(row, dict) or "id" not in row or row.get("split") not in ("train", "test"):
            raise ValueError("Recorded camera requires an identity and train/test split")
        camera_from_record(row)
        ids.append(row["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate recorded camera identity")
    return rows


def camera_from_record(row, width=None, height=None):
    """Read actual K/w2c; 512 is the pinned source recipe's recorded resolution."""
    if "K" not in row or "w2c" not in row:
        raise ValueError("Recorded camera requires K and w2c")
    original_width = int(row.get("width", row.get("image_width", 512)))
    original_height = int(row.get("height", row.get("image_height", 512)))
    camera = Camera(original_width, original_height, np.asarray(row["w2c"]), np.asarray(row["K"]))
    if width is not None or height is not None:
        camera = camera.resized(width if width is not None else original_width,
                                height if height is not None else original_height)
    return camera


def select_camera_record(records, camera_id):
    rows = [row for row in records if row.get("id") == camera_id]
    if len(rows) != 1:
        raise ValueError(f"Unavailable or duplicate exact recorded camera ID {camera_id!r}")
    camera_from_record(rows[0])
    return rows[0]


def recorded_camera(path, camera_id, width=None, height=None):
    return camera_from_record(select_camera_record(load_camera_records(path), camera_id), width, height)


def relative_camera(camera, object_to_world):
    """C_local=(sR)^-1(C_world-t), R_local=R_camera R; never add Rx(pi)."""
    matrix = validate_transform(object_to_world)
    scale = float(np.linalg.norm(matrix[:3, 0]))
    center = np.linalg.solve(matrix[:3, :3], camera.center - matrix[:3, 3])
    rotation = camera.world_to_camera[:3, :3] @ (matrix[:3, :3] / scale)
    view = np.eye(4)
    view[:3, :3], view[:3, 3] = rotation, -rotation @ center
    # Clipping distances are expressed in the local unit system.
    return Camera(camera.width, camera.height, view, camera.K,
                  camera.near / scale, camera.far / scale)


def validate_source_conditioning(camera, object_to_world, source_record, *,
                                 mode="exact_pose", atol=1e-8):
    """Require exact recorded conditioning support, never just a similar radius.

    `recorded_center` permits the documented joint-view rotation/crop: StreamSTGS
    opacity/color MLPs consume source-local anchor direction and distance only.
    `exact_pose` additionally requires identical local orientation. Intrinsics
    may be calibrated resizes or a centered vertical crop at the same HFOV.
    """
    local = relative_camera(camera, object_to_world)
    source = camera_from_record(source_record)
    if mode not in ("exact_pose", "recorded_center"):
        raise ValueError("Unsupported source-camera conditioning mode")
    if not np.allclose(local.center, source.center, rtol=0., atol=atol):
        raise ValueError("Unsupported source-local camera center; exact recorded support required")
    if mode == "exact_pose" and not np.allclose(local.world_to_camera[:3, :3],
                                                source.world_to_camera[:3, :3], rtol=0., atol=atol):
        raise ValueError("Source-local camera orientation differs from exact recorded pose")
    expected = source.resized(local.width, local.height)
    isotropic_k = source.K.copy() * (local.width / source.width)
    isotropic_k[2] = [0., 0., 1.]
    isotropic_k[0, 2] = (local.width - 1) / 2
    isotropic_k[1, 2] = (local.height - 1) / 2
    if not (np.allclose(local.K, expected.K, rtol=0., atol=atol)
            or np.allclose(local.K, isotropic_k, rtol=0., atol=atol)):
        raise ValueError("Camera intrinsics do not match recorded calibration/resize/crop")
    return dict(source_camera_id=source_record["id"], source_camera_split=source_record["split"],
                conditioning_mode=mode, source_camera_pose_sha256=camera_identity(source),
                source_local_camera=camera_metadata(local),
                source_local_camera_pose_sha256=camera_identity(local),
                source_local_camera_center=local.center.tolist(),
                camera_to_object_distance=float(np.linalg.norm(local.center)),
                camera_relative_position=local.center.tolist(),
                camera_relative_orientation=local.world_to_camera[:3, :3].tolist(),
                source_center_max_error=float(np.max(np.abs(local.center-source.center))))


def supported_scene(camera_records, camera_id, *, yaw_offsets_degrees=(-15., 0., 15.),
                    width=640, height=480):
    """Deterministic supported-center scene, with one exact source center each.

    All objects and their identities are supplied explicitly. Central placement
    equals the recorded pose; other objects rotate about the common camera
    center, preserving decoder conditioning. This is a synthetic controlled
    arrangement, not a claim of physical capture or disjoint object envelopes.
    """
    if not camera_records or len(camera_records) != len(yaw_offsets_degrees):
        raise ValueError("One declared yaw offset is required per object")
    selected = {oid: select_camera_record(rows, camera_id) for oid, rows in camera_records.items()}
    base = camera_from_record(next(iter(selected.values())))
    # Isotropic calibration scaling plus centered crop, as in the supported control.
    k = base.K.copy() * (width / base.width)
    k[2] = [0., 0., 1.]
    k[0, 2], k[1, 2] = (width-1)/2, (height-1)/2
    camera = Camera(width, height, base.world_to_camera, k, .01, 1000.)
    instances = []
    for oid, angle in zip(selected, yaw_offsets_degrees):
        source = camera_from_record(selected[oid])
        yaw = math.radians(float(angle))
        if not math.isfinite(yaw):
            raise ValueError("Finite yaw offset required")
        rotation = np.array([[math.cos(yaw), 0., math.sin(yaw)], [0., 1., 0.],
                             [-math.sin(yaw), 0., math.cos(yaw)]])
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, camera.center - rotation @ source.center
        validate_source_conditioning(camera, matrix, selected[oid], mode="recorded_center")
        instances.append(dict(instance_id=oid, object_id=oid, object_to_world=matrix.tolist(),
                              source_camera=selected[oid], conditioning_mode="recorded_center"))
    return camera, instances


def camera_clip(camera):
    """CV +Z, camera-Y down, depth[0,w] -> OpenGL Y up, depth[-w,w]."""
    conversion = np.diag([1., -1., 2., 1.])
    conversion[2, 3] = -1.
    return conversion @ camera.projection @ camera.world_to_camera


def aabb_area(bounds, camera, object_to_world, *, method="paper_discard"):
    """NAVA III-E positive projected triangle area in pixels.

    The paper discards triangles with any exterior vertex. `homogeneous_clip`
    reuses the earlier repository's clipped-polygon improvement and is an
    explicitly different geometry method. Neither method uses Gaussian alpha.
    """
    matrix = validate_transform(object_to_world)
    mvp = camera_clip(camera) @ matrix
    if method == "homogeneous_clip":
        return projected_area(bounds, mvp.tolist(), camera.width, camera.height)
    if method != "paper_discard":
        raise ValueError("Unknown AABB projection method")
    box = np.asarray(bounds, dtype=np.float64)
    if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] >= box[1]):
        raise ValueError("Bounds must be finite strictly ordered min/max XYZ")
    vertices = np.array(list(itertools.product(*zip(*box))))
    clips = np.column_stack([vertices, np.ones(8)]) @ mvp.T
    area = 0.
    for axis, side in itertools.product(range(3), (0, 1)):
        others = [d for d in range(3) if d != axis]
        corners = []
        for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
            bits = [0, 0, 0]
            bits[axis], bits[others[0]], bits[others[1]] = side, a, b
            corners.append(bits[0]*4 + bits[1]*2 + bits[2])
        if (-1 if axis % 2 else 1) != (1 if side else -1):
            corners.reverse()
        for indices in ((0, 1, 2), (0, 2, 3)):
            triangle = clips[[corners[j] for j in indices]]
            if np.any(triangle[:, 3] <= 0.) or np.any(np.abs(triangle[:, :3]) > triangle[:, 3:]):
                continue
            xy = triangle[:, :2] / triangle[:, 3:]
            signed = np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)-xy[:, 1]*np.roll(xy[:, 0], -1))/2
            area += max(0., float(signed)) * camera.width * camera.height / 4
    return area

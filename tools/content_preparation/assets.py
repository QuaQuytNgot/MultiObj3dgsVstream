"""CPU-only Gaussian assets with deterministic files and explicit identity.

The internal feature layout is N x RGB x SH coefficient. Scale and opacity
remain in the upstream model's log/logit domains; rotation is raw wxyz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

ATTRIBUTES = ("xyz", "rotation", "scale", "opacity", "sh")
STATE_VERSION = "gaussian-state-v1"


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False).encode("utf-8")


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class GaussianState:
    arrays: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)
    ids: np.ndarray | None = None

    def __post_init__(self):
        if set(self.arrays) != set(ATTRIBUTES):
            raise ValueError(f"Gaussian attributes must be exactly {ATTRIBUTES}")
        self.metadata = dict(self.metadata)
        self.arrays = {name: np.ascontiguousarray(self.arrays[name], dtype="<f4")
                       for name in ATTRIBUTES}
        n = len(self.arrays["xyz"])
        explicit_ids = self.ids is not None
        if self.ids is None:
            self.ids = np.arange(n, dtype="<i8")
        else:
            original_ids = np.asarray(self.ids)
            if not np.issubdtype(original_ids.dtype, np.integer):
                raise ValueError("Gaussian IDs must be integers")
            if original_ids.size and np.issubdtype(original_ids.dtype, np.unsignedinteger) and original_ids.max() > np.iinfo(np.int64).max:
                raise ValueError("Gaussian IDs must fit signed int64")
            self.ids = np.ascontiguousarray(original_ids, dtype="<i8")
        self.metadata.setdefault("stable_ids", explicit_ids)
        self.validate()

    def validate(self):
        n = len(self.ids)
        expected = {"xyz": (n, 3), "rotation": (n, 4), "scale": (n, 3),
                    "opacity": (n, 1)}
        for name, shape in expected.items():
            if self.arrays[name].shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {self.arrays[name].shape}")
        sh = self.arrays["sh"]
        if sh.ndim != 3 or sh.shape[:2] != (n, 3):
            raise ValueError("sh must have shape (N,3,(degree+1)^2)")
        k = sh.shape[2]
        if k < 1 or int(np.sqrt(k)) ** 2 != k:
            raise ValueError("SH coefficient count must be a positive perfect square")
        if self.ids.ndim != 1 or len(np.unique(self.ids)) != n:
            raise ValueError("Gaussian IDs must be a unique one-dimensional array")
        for name, value in self.arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains nonfinite values")
        if n and (np.linalg.norm(self.arrays["rotation"].astype(np.float64), axis=1) == 0).any():
            raise ValueError("A raw wxyz quaternion may not be zero")
        canonical_json(self.metadata)
        return self

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def sh_degree(self) -> int:
        return int(np.sqrt(self.arrays["sh"].shape[2])) - 1

    def subset(self, indices) -> "GaussianState":
        return GaussianState({k: v[indices].copy() for k, v in self.arrays.items()},
                             dict(self.metadata), self.ids[indices].copy())


def state_hash(state: GaussianState) -> str:
    """Hash numeric decoder state, including row order and IDs, not provenance."""
    state.validate()
    digest = hashlib.sha256(STATE_VERSION.encode("ascii"))
    for name, array in [("ids", state.ids)] + list(state.arrays.items()):
        digest.update(canonical_json([name, array.dtype.str, list(array.shape)]))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _npy(array) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def save_state(path, state: GaussianState) -> dict:
    """Deterministic NPZ: fixed ZIP timestamps, no pickle, uncompressed members."""
    state.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = {f"{k}.npy": _npy(v) for k, v in state.arrays.items()}
    contents["ids.npy"] = _npy(state.ids)
    contents["metadata.npy"] = _npy(np.frombuffer(canonical_json({
        "version": STATE_VERSION, "metadata": state.metadata,
        "state_sha256": state_hash(state)}), dtype=np.uint8))
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(contents):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, contents[name])
    temporary.replace(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": file_hash(path), "state_sha256": state_hash(state),
            "version": STATE_VERSION}


def load_state(path) -> GaussianState:
    path = Path(path)
    if path.suffix.lower() == ".ply":
        return read_ply(path)
    with np.load(path, allow_pickle=False) as document:
        meta = json.loads(document["metadata"].tobytes().decode("utf-8"))
        if meta.get("version") != STATE_VERSION:
            raise ValueError(f"Unsupported Gaussian state version: {meta.get('version')}")
        state = GaussianState({k: document[k].copy() for k in ATTRIBUTES},
                              meta["metadata"], document["ids"].copy())
    if state_hash(state) != meta["state_sha256"]:
        raise ValueError(f"Gaussian state checksum mismatch: {path}")
    return state


_PLY_TYPES = {"char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
              "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
              "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply(path) -> GaussianState:
    """Read upstream Gaussian PLY, preserving its SH field ordering on CPU.

    Original PLY row indices are unverified identities. A caller must verify
    checkpoint lineage before using those indices for a progressive correction.
    """
    path = Path(path)
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        elements, fmt, metadata = [], None, {}
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("Truncated PLY header")
            line = raw.decode("ascii").strip()
            if line == "end_header":
                break
            fields = line.split()
            if fields[:1] == ["format"]:
                fmt = fields[1]
            elif fields[:1] == ["element"]:
                elements.append({"name": fields[1], "count": int(fields[2]), "properties": []})
            elif fields[:1] == ["property"]:
                if len(fields) != 3 or fields[1] not in _PLY_TYPES:
                    raise ValueError("Gaussian PLY supports scalar properties only")
                elements[-1]["properties"].append((fields[2], _PLY_TYPES[fields[1]]))
            elif line.startswith("comment content_preparation_metadata "):
                metadata = json.loads(line[len("comment content_preparation_metadata "):])
        if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY format {fmt}")
        vertex = None
        for element in elements:
            endian = ">" if fmt == "binary_big_endian" else "<"
            dtype = np.dtype([(name, endian + typ) for name, typ in element["properties"]])
            n = element["count"]
            if fmt == "ascii":
                rows = [handle.readline().decode("ascii").split() for _ in range(n)]
                if any(len(row) != len(dtype.names) for row in rows):
                    raise ValueError("Truncated or malformed ASCII PLY rows")
                values = np.empty(n, dtype=dtype)
                for j, name in enumerate(dtype.names):
                    values[name] = [row[j] for row in rows]
            else:
                data = handle.read(n * dtype.itemsize)
                if len(data) != n * dtype.itemsize:
                    raise ValueError("Truncated binary PLY")
                values = np.frombuffer(data, dtype=dtype).copy()
            if element["name"] == "vertex":
                vertex = values
                break
    if vertex is None:
        raise ValueError("Gaussian PLY has no vertex element")
    names = set(vertex.dtype.names)
    required = {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"}
    if not required <= names:
        raise ValueError(f"Not an exported Gaussian PLY; missing {sorted(required - names)}")
    collect = lambda prefix: sorted((p for p in names if p.startswith(prefix)),
                                    key=lambda p: int(p.rsplit("_", 1)[1]))
    rest_names = collect("f_rest_")
    if len(rest_names) % 3:
        raise ValueError("Malformed SH fields")
    n = len(vertex)
    dc = np.stack([vertex[f"f_dc_{c}"] for c in range(3)], axis=1)[:, :, None]
    rest = (np.stack([vertex[p] for p in rest_names], axis=1).reshape(n, 3, len(rest_names) // 3)
            if rest_names else np.empty((n, 3, 0), dtype=np.float32))
    arrays = {
        "xyz": np.stack([vertex[p] for p in ("x", "y", "z")], axis=1),
        "opacity": vertex["opacity"][:, None],
        "rotation": np.stack([vertex[p] for p in collect("rot_")], axis=1),
        "scale": np.stack([vertex[p] for p in collect("scale_")], axis=1),
        "sh": np.concatenate((dc, rest), axis=2),
    }
    ids = vertex["gaussian_id"].astype("<i8") if "gaussian_id" in names else None
    metadata.setdefault("stable_ids", ids is not None)
    metadata.setdefault("source_sha256", file_hash(path))
    metadata.setdefault("source_format", "upstream-gaussian-ply")
    return GaussianState(arrays, metadata, ids)


def write_ply(path, state: GaussianState) -> dict:
    """Write standard upstream fields, plus an ignored uint Gaussian ID field."""
    state.validate()
    if state.count and (state.ids.min() < 0 or state.ids.max() > np.iinfo(np.uint32).max):
        raise ValueError("PLY Gaussian IDs must fit uint32; use NPZ for other IDs")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = state.arrays
    columns = [(name, arrays["xyz"][:, j]) for j, name in enumerate(("x", "y", "z"))]
    columns += [(name, np.zeros(state.count, dtype=np.float32)) for name in ("nx", "ny", "nz")]
    columns += [(f"f_dc_{j}", arrays["sh"][:, j, 0]) for j in range(3)]
    rest = arrays["sh"][:, :, 1:].reshape(state.count, 3 * (arrays["sh"].shape[2] - 1))
    columns += [(f"f_rest_{j}", rest[:, j]) for j in range(rest.shape[1])]
    columns += [("opacity", arrays["opacity"][:, 0])]
    for attr, prefix in (("scale", "scale"), ("rotation", "rot")):
        columns += [(f"{prefix}_{j}", arrays[attr][:, j]) for j in range(arrays[attr].shape[1])]
    dtype = np.dtype([(name, "<f4") for name, _ in columns] + [("gaussian_id", "<u4")])
    vertex = np.empty(state.count, dtype=dtype)
    for name, values in columns:
        vertex[name] = values
    vertex["gaussian_id"] = state.ids
    header = ["ply", "format binary_little_endian 1.0",
              "comment content_preparation_metadata " + json.dumps(state.metadata, sort_keys=True, ensure_ascii=True),
              f"element vertex {state.count}"]
    header += [f"property float {name}" for name, _ in columns]
    header += ["property uint gaussian_id", "end_header"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(vertex.tobytes())
    temporary.replace(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def synthetic_state(frame=0, quality_count=16, seed=17) -> GaussianState:
    """Tiny stable-ID fixture; prefix attributes also vary with quality."""
    n = int(quality_count)
    if n < 1:
        raise ValueError("quality_count must be positive")
    rng = np.random.default_rng(seed)
    # Generate a common sequence before truncating so qualities share identities.
    xyz = rng.uniform(-0.45, 0.45, size=(max(256, n), 3)).astype(np.float32)[:n]
    xyz[:, 0] += np.float32(0.02 * frame)
    sh = np.zeros((n, 3, 1), dtype=np.float32)
    sh[:, :, 0] = (rng.uniform(0.15, 0.85, size=(max(256, n), 3))[:n] - 0.5) / 0.28209479177387814
    opacity = np.full((n, 1), 1.5 + n / 100, dtype=np.float32)
    rotation = np.zeros((n, 4), dtype=np.float32)
    rotation[:, 0] = 1
    return GaussianState({"xyz": xyz, "rotation": rotation,
                          "scale": np.full((n, 3), -2.5, dtype=np.float32),
                          "opacity": opacity, "sh": sh},
                         {"synthetic": True, "frame": int(frame), "stable_ids": True,
                          "seed": int(seed)}, np.arange(n, dtype="<i8"))

"""Validated YAML configuration, resolved relative to the repository root."""
from .paths import PROJECT_ROOT
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import os
import re
import sys
import yaml

ROOT = PROJECT_ROOT
STAGES = ("preprocess", "train", "export", "encode", "package", "decode", "profile", "proxy", "manifest")
VALIDATION_STAGES = ("environment", "dataset", "preprocess", "checkpoints", "export",
                     "encoding", "package", "decode", "profile", "final")


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resolve_path(value, root=ROOT):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def frame_records(obj):
    frames = obj["frames"]
    if isinstance(frames, dict):
        if any(not isinstance(frames.get(k),int) or isinstance(frames.get(k),bool) for k in ("start","end")) or not isinstance(frames.get("step",1),int) or frames.get("step",1)<=0:
            raise ValueError("Frame range requires integer start/end and positive integer step")
        frames = list(range(frames["start"], frames["end"] + 1, frames.get("step", 1)))
    if not isinstance(frames,list) or not frames:
        raise ValueError("Frames must be a nonempty list or inclusive range")
    fps = obj.get("fps", 30.)
    if not isinstance(fps,(int,float)) or not math.isfinite(fps) or fps<=0:
        raise ValueError("fps must be finite and positive")
    origin = frames[0]
    times = obj.get("timestamps")
    if times is not None and (not isinstance(times,list) or len(times)!=len(frames)):
        raise ValueError("timestamps must match frame count")
    return [{"frame": f, "timestamp": float(times[i] if times else (f-origin)/fps)} for i, f in enumerate(frames)]


RUNTIME_OVERRIDE_FIELDS = {"runtime": {"gpu_batch_size", "extension_path", "cpu_threads"},
                           "metrics": {"device", "batch_size"}}


def apply_runtime_override(raw, override):
    """Merge execution settings only; research configuration cannot be replaced."""
    if not isinstance(override, dict) or set(override) - set(RUNTIME_OVERRIDE_FIELDS):
        raise ValueError("Runtime configuration accepts only runtime and metrics execution settings")
    cfg = deepcopy(raw)
    for section, values in override.items():
        if not isinstance(values, dict) or set(values) - RUNTIME_OVERRIDE_FIELDS[section]:
            raise ValueError(f"Unsupported runtime override fields in {section}; research parameters cannot be overridden")
        cfg.setdefault(section, {})
        if section == "metrics" and isinstance(cfg[section].get("lpips"), dict):
            nested = cfg[section]["lpips"]
            for key in ("device", "batch_size"):
                if key in values and key in nested and values[key] != nested[key]:
                    raise ValueError(f"Runtime metrics.{key} conflicts with metrics.lpips.{key}")
        cfg[section].update(deepcopy(values))
    return cfg


def configure_runtime(config):
    """Set CPU thread limits before numeric imports, including loaded PyTorch."""
    threads = config.get("runtime", {}).get("cpu_threads", 2)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads <= 0:
        raise ValueError("runtime.cpu_threads must be a positive integer")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(threads)


def read_config(path, runtime_path=None):
    """Read and merge YAML without importing numeric libraries or changing state."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if runtime_path is not None:
        raw = apply_runtime_override(raw, yaml.safe_load(Path(runtime_path).read_text()))
    return raw


def load_config(path, runtime_path=None):
    return validate_config(read_config(path, runtime_path))


def validate_config(raw):
    cfg = deepcopy(raw)
    allowed = {"version", "output", "objects", "delivery_modes", "training", "encoding", "packaging", "sampling", "renderer", "metrics", "proxy", "runtime"}
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if cfg.get("version") != 1 or not cfg.get("output") or not cfg.get("objects"):
        raise ValueError("Require version: 1, output, and nonempty objects")
    cfg.setdefault("delivery_modes", ["independent", "progressive"])
    if not cfg["delivery_modes"] or len(set(cfg["delivery_modes"])) != len(cfg["delivery_modes"]) or set(cfg["delivery_modes"]) - {"independent", "progressive"}:
        raise ValueError("delivery_modes must contain independent and/or progressive")
    defaults = {"training": {"backend": "upstream", "initial_iterations": 30000, "dynamic_iterations": 30000, "lambda_dssim": .8, "sh_degree": 3, "extra_args": []},
                "encoding": {"precision": "f32", "compression": "zlib", "level": 6},
                "packaging": {"gof_frames": 30, "segment_frames": 4, "refresh_frames": 8},
                "sampling": {"azimuth": [0, 120, 240], "elevation": [0], "scales": [1.], "distance": 3., "times": "all"},
                "renderer": {"backend": "upstream_cuda", "width": 256, "height": 256, "fov_degrees": 45., "background": [0., 0., 0.], "center": [0., 0., 0.], "up_axis": "y"},
                "metrics": {"lpips": True, "lpips_net": "vgg", "device": "cpu", "batch_size": 1, "distortion": "mse"},
                "proxy": {"enabled": True, "external_descriptor": None, "backend": "fixed_gaussian_subset", "max_gaussians": 256, "alpha_threshold": .01},
                "runtime": {"seed": 0, "gpu_batch_size": 1, "extension_path": None, "cpu_threads": 2}}
    for name, default in defaults.items():
        cfg[name] = dict(default, **cfg.get(name, {}))
    from .codec_adapter import get_codec
    get_codec(cfg["encoding"])
    threads=cfg["runtime"]["cpu_threads"]
    if not isinstance(threads,int) or isinstance(threads,bool) or threads<=0:
        raise ValueError("runtime.cpu_threads must be a positive integer")
    extension=cfg["runtime"].get("extension_path")
    if extension is not None and (not isinstance(extension,str) or not extension):
        raise ValueError("runtime.extension_path must be a directory path or null")
    for key in ("initial_iterations","dynamic_iterations"):
        value=cfg["training"][key]
        if not isinstance(value,int) or isinstance(value,bool) or value<=0:
            raise ValueError(f"training.{key} must be a positive integer")
    lpips_config=cfg["metrics"]["lpips"] if isinstance(cfg["metrics"]["lpips"],dict) else {}
    if cfg["metrics"]["device"] not in {"cpu","cuda"} or lpips_config.get("device",cfg["metrics"]["device"]) not in {"cpu","cuda"}:
        raise ValueError("metrics.device must be cpu or cuda")
    if cfg["metrics"]["batch_size"] != 1 or lpips_config.get("batch_size",1)!=1 or cfg["runtime"]["gpu_batch_size"] != 1:
        raise ValueError("Sequential preparation requires GPU and LPIPS batch_size = 1")
    if cfg["metrics"]["distortion"] not in {"mse", "lpips"}:
        raise ValueError("distortion must be mse or lpips; PSNR is a reporting transform of MSE")
    if not cfg["metrics"]["lpips"] or lpips_config.get("enabled",True) is not True:
        raise ValueError("Preparation requires actual LPIPS; no surrogate metric is accepted")
    if cfg["training"]["backend"] not in {"upstream", "existing", "synthetic_fixture"} or cfg["renderer"]["backend"] not in {"upstream_cuda", "cpu_fixture"}:
        raise ValueError("Unknown training or renderer backend")
    if cfg["renderer"]["up_axis"] not in ("y", "z"):
        raise ValueError("renderer.up_axis must be 'y' or 'z'")
    for name in ("gof_frames", "segment_frames"):
        if not isinstance(cfg["packaging"][name], int) or cfg["packaging"][name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    refresh = cfg["packaging"]["refresh_frames"]
    if isinstance(refresh, dict):
        if not refresh or any(not isinstance(v, int) or v <= 0 for v in refresh.values()):
            raise ValueError("refresh_frames values must be positive integers")
    elif not isinstance(refresh, int) or refresh <= 0:
        raise ValueError("refresh_frames must be a positive integer or layer mapping")
    for name in ("width", "height"):
        if not isinstance(cfg["renderer"][name], int) or cfg["renderer"][name] < 32:
            raise ValueError("Render resolution must be at least 32 for LPIPS VGG")
    if not 0 < cfg["renderer"]["fov_degrees"] < 179:
        raise ValueError("fov_degrees must lie in (0,179)")
    for name in ("azimuth", "elevation", "scales"):
        values = cfg["sampling"][name]
        if not values or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"sampling.{name} must be a nonempty finite numeric list")
    if any(s <= 0 for s in cfg["sampling"]["scales"]) or cfg["sampling"]["distance"] <= 0:
        raise ValueError("scales and distance must be positive")
    if not isinstance(cfg["proxy"]["enabled"], bool):
        raise ValueError("proxy.enabled must be a boolean")
    if cfg["proxy"].get("external_descriptor") is not None:
        from .validation import safe_relative_path
        safe_relative_path(cfg["proxy"]["external_descriptor"])
        if cfg["proxy"]["enabled"]:
            raise ValueError("An external proxy descriptor requires proxy.enabled: false")
    if cfg["proxy"]["enabled"] and (cfg["proxy"]["backend"] not in {"fixed_gaussian_subset", "gaussian_subset"} or cfg["proxy"]["max_gaussians"] < 1):
        raise ValueError("Baseline proxy backend is fixed_gaussian_subset with positive max_gaussians")
    object_ids = set()
    for obj in cfg["objects"]:
        name = obj.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in object_ids:
            raise ValueError("Object IDs must be unique safe path components")
        object_ids.add(name)
        obj.setdefault("fps", 30.)
        obj.setdefault("dataset", {"kind": "synthetic"})
        if obj["dataset"]["kind"] not in {"synthetic", "prepared", "raw", "checkpoints"}:
            raise ValueError("dataset.kind must be synthetic, prepared, raw or checkpoints")
        required={"prepared":"source_template","raw":"raw_template","checkpoints":"checkpoint_manifest"}.get(obj["dataset"]["kind"])
        if required and not obj["dataset"].get(required):
            raise ValueError(f"dataset.kind {obj['dataset']['kind']} requires {required}")
        if obj["dataset"]["kind"] != "synthetic" and cfg["renderer"]["backend"] == "cpu_fixture":
            raise ValueError("cpu_fixture is only valid for synthetic data, not production profiling")
        records = frame_records(obj)
        if not records or any(not isinstance(r["frame"], int) for r in records) or obj["fps"] <= 0:
            raise ValueError("Frames must be nonempty integer IDs and fps positive")
        if [r["frame"] for r in records] != sorted(set(r["frame"] for r in records)):
            raise ValueError("Frames must be strictly increasing and unique")
        if obj.get("timestamps") and len(obj["timestamps"]) != len(records):
            raise ValueError("timestamps must match frames")
        if any(not math.isfinite(r["timestamp"]) for r in records) or any(b["timestamp"] <= a["timestamp"] for a,b in zip(records,records[1:])):
            raise ValueError("Timestamps must be finite and strictly increasing")
        qualities = obj.get("qualities", [])
        names = [q.get("id", "") for q in qualities]
        if not names or len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", n) for n in names):
            raise ValueError("Qualities require unique safe IDs in increasing nominal quality order")
        if obj["dataset"]["kind"] == "synthetic" and cfg["training"]["backend"] != "synthetic_fixture":
            raise ValueError("Synthetic fixture uses synthetic_fixture training backend")
        if cfg["training"]["backend"] == "existing" and not obj["dataset"].get("checkpoint_manifest"):
            raise ValueError("existing backend requires dataset.checkpoint_manifest")
        if (obj["dataset"]["kind"]=="checkpoints") != (cfg["training"]["backend"]=="existing"):
            raise ValueError("dataset.kind checkpoints and training.backend existing must be used together")
        for q in qualities:
            get_codec(dict(cfg["encoding"],**q.get("encoding",{})))
            q.setdefault("resolution_scale", 1)
            if not isinstance(q["resolution_scale"],int) or isinstance(q["resolution_scale"],bool) or q["resolution_scale"] < 1 or (q["resolution_scale"] & (q["resolution_scale"]-1)):
                raise ValueError("resolution_scale must be a power of two")
        if isinstance(refresh, dict):
            layers = ["Base"] + [f"E{i}" for i in range(1, len(qualities))]
            for layer, q in zip(layers, names):
                if layer not in refresh and q not in refresh and "default" not in refresh:
                    raise ValueError(f"Missing refresh schedule for {layer}/{q}")
        samples = cfg["sampling"]["times"]
        if samples != "all" and (not isinstance(samples, list) or not samples or set(samples) - {r["frame"] for r in records}):
            raise ValueError("sampling.times is 'all' or existing frame IDs (no temporal state interpolation)")
    return cfg

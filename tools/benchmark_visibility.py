#!/usr/bin/env python3
"""Measure RGB, contribution passes, reductions, and complete visibility analysis.

No GPU dependencies are imported by ``--help``. Use scripts/run_gpu0.sh with
the project's CUDA environment. Outputs are measurements, not predictions.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene", type=Path, action="append", default=[],
                        help="existing-proxy scene JSON; may be repeated")
    result.add_argument("--no-synthetic", action="store_true",
                        help="benchmark supplied proxy scenes only")
    result.add_argument("--objects", nargs="+", type=positive_int, default=[1, 3, 6])
    result.add_argument("--resolutions", nargs="+", type=positive_int, default=[128, 256, 512],
                        help="square analysis resolutions; FOV is held fixed")
    result.add_argument("--gaussians", type=positive_int, default=256,
                        help="Gaussians per synthetic object")
    result.add_argument("--warmup", type=positive_int, default=10)
    result.add_argument("--samples", type=positive_int, default=100)
    result.add_argument("--diagnostics", action="store_true",
                        help="also time full analysis with CPU diagnostic map export")
    result.add_argument("--extension-path", default=os.environ.get("VISIBILITY_EXTENSION_PATH"),
                        help="directory containing the installed pinned rasterizer package")
    result.add_argument("--output", type=Path, default=ROOT / "output/visibility_benchmark.json")
    return result


def distribution(values):
    """Milliseconds; p95 uses the nearest-rank definition."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("A timing distribution requires samples")
    return {"mean": statistics.fmean(ordered), "median": statistics.median(ordered),
            "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)]}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scene(path):
    """Resolve existing proxies once; frame/timestamp mapping belongs to the JSON."""
    import numpy as np
    from src.renderer.analysis import VisibilityObject
    from tools.content_preparation.proxy import load_proxy

    path = Path(path).expanduser().resolve()
    document = json.loads(path.read_text())
    timestamp = float(document["timestamp"])
    if not math.isfinite(timestamp):
        raise ValueError("Scene timestamp must be finite")
    objects, assets = [], []
    for record in document["objects"]:
        descriptor = Path(record["proxy_path"]).expanduser()
        if not descriptor.is_absolute():
            descriptor = path.parent / descriptor
        descriptor = descriptor.resolve()
        if descriptor.suffix.lower() != ".json":
            raise ValueError("Benchmark proxy_path must be a proxy JSON descriptor with explicit frames")
        frame = record["frame"]
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError("Each existing proxy needs an explicit integer frame")
        snapshot_timestamp = float(record["timestamp"])
        proxy = load_proxy(descriptor, frame=frame)
        transform = np.asarray(record.get("object_to_world", np.eye(4)), dtype=np.float64)
        objects.append(VisibilityObject(record["object_id"], proxy, snapshot_timestamp, transform))
        assets.append({"object_id": record["object_id"], "proxy_path": str(descriptor),
                       "proxy_sha256": _sha256(descriptor), "frame": frame,
                       "timestamp": snapshot_timestamp, "gaussians": proxy.count})
    if not objects:
        raise ValueError("A benchmark scene must contain at least one object")
    return {"name": document.get("name", path.stem), "objects": objects,
            "timestamp": timestamp, "camera": document["camera"],
            "provenance": {"scene_path": str(path), "scene_sha256": _sha256(path),
                           "assets": assets}}


def synthetic_scene(object_count, gaussian_count):
    import numpy as np
    from src.renderer.analysis import VisibilityObject
    from tools.content_preparation.assets import GaussianState

    rng = np.random.default_rng(20261012)
    objects = []
    for slot in range(object_count):
        xyz = rng.uniform(-.3, .3, (gaussian_count, 3)).astype(np.float32)
        xyz[:, 0] += .25 * ((slot % 3) - 1)
        xyz[:, 2] += 2.0 + .4 * (slot // 3)
        color = np.asarray([.2 + .1 * (slot % 3), .4, .6], dtype=np.float32)
        state = GaussianState({"xyz": xyz,
            "rotation": np.tile(np.asarray([1., 0., 0., 0.], dtype=np.float32), (gaussian_count, 1)),
            "scale": np.full((gaussian_count, 3), math.log(.025), dtype=np.float32),
            "opacity": np.full((gaussian_count, 1), math.log(.6 / .4), dtype=np.float32),
            "sh": np.tile(((color - .5) / .28209479177387814)[None, :, None], (gaussian_count, 1, 1))},
            {"asset_role": "quality_independent_visibility_proxy", "fixture": "visibility_benchmark"})
        objects.append(VisibilityObject(f"synthetic_{slot}", state, 0.0, np.eye(4)))
    camera = {"world_view": np.eye(4).tolist(), "width": 256, "height": 256,
              "fovx": math.pi / 2, "fovy": math.pi / 2, "near": .01, "far": 100.}
    return {"name": "synthetic", "objects": objects, "camera": camera, "timestamp": 0.,
            "provenance": {"seed": 20261012, "gaussians_per_object": gaussian_count}}


def _git(path, *arguments):
    try:
        return subprocess.check_output(["git", "-C", str(path), *arguments],
                                       text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(torch):
    import diff_gaussian_rasterization as rasterizer
    from tools.content_preparation.paths import UPSTREAM_ROOT

    modules = {"python": getattr(rasterizer, "__file__", None),
               "binary": getattr(getattr(rasterizer, "_C", None), "__file__", None)}
    modules = {name: {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
               for name, path in modules.items() if path}
    rasterizer_root = UPSTREAM_ROOT / "submodules/diff-gaussian-rasterization"
    return {"python": platform.python_version(), "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "project_commit": _git(ROOT, "rev-parse", "HEAD"),
            "upstream_commit": _git(UPSTREAM_ROOT, "rev-parse", "HEAD"),
            "rasterizer_commit": _git(rasterizer_root, "rev-parse", "HEAD"),
            "rasterizer_worktree_status": _git(rasterizer_root, "status", "--short"),
            "rasterizer_modules": modules,
            "forward_source_sha256": _sha256(rasterizer_root / "cuda_rasterizer/forward.cu")}


def memory_baseline(torch):
    torch.cuda.synchronize()
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved()}


def measure_phase(torch, operation, warmup, samples, *, events=True):
    """Operation must release outputs before returning; no allocator flush in loops."""
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    baseline = memory_baseline(torch)
    torch.cuda.reset_peak_memory_stats()
    gpu_ms, wall_ms = [], []
    start = torch.cuda.Event(enable_timing=True) if events else None
    end = torch.cuda.Event(enable_timing=True) if events else None
    for _ in range(samples):
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        if events:
            start.record()
        operation()
        if events:
            end.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000)
        if events:
            gpu_ms.append(start.elapsed_time(end))
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    return {"samples": samples, "wall_ms": distribution(wall_ms),
            "gpu_ms": distribution(gpu_ms) if events else None,
            "memory": {"baseline": baseline, "peak_allocated_bytes": allocated,
                       "peak_reserved_bytes": reserved,
                       "incremental_allocated_bytes": max(0, allocated - baseline["allocated_bytes"]),
                       "incremental_reserved_bytes": max(0, reserved - baseline["reserved_bytes"])}}


def benchmark_case(source, object_count, resolution, args):
    import torch
    from src.client.visibility import analyze_visibility
    from src.renderer.upstream import prepare_scene, validate_and_snapshot
    from tools.content_preparation.renderer_adapter import GPUExecutionGuard

    objects = source["objects"][:object_count]
    if not sum(obj.proxy.count for obj in objects):
        raise ValueError("A timed benchmark scene must contain at least one Gaussian")
    camera = dict(source["camera"], width=resolution, height=resolution)
    timestamp = source["timestamp"]
    settings = {"extension_path": args.extension_path} if args.extension_path else {}
    snapshots, validated_camera, settings = validate_and_snapshot(objects, camera, timestamp, settings)
    phases = {}
    with GPUExecutionGuard(), torch.no_grad():
        scene = prepare_scene(snapshots, validated_camera, settings)
        torch.cuda.synchronize()
        prepared_baseline = memory_baseline(torch)
        scene_metadata = dict(scene.metadata)
        environment = provenance(torch)
        groups = list(range(0, object_count, 3))

        def rgb():
            scene.render(scene.joint_model)

        def joint(start):
            scene.render(scene.joint_model, colors=scene.joint_colors(start))

        def solo(index):
            scene.render(scene.object_models[index], colors=scene.solo_colors(index))

        def all_joint():
            for start in groups:
                joint(start)

        def all_solo():
            for index in range(object_count):
                if objects[index].proxy.count:
                    solo(index)

        phases["rgb"] = measure_phase(torch, rgb, args.warmup, args.samples)
        for start in groups:
            phases[f"joint_group_{start // 3}"] = measure_phase(
                torch, lambda start=start: joint(start), args.warmup, args.samples)
        phases["joint_total"] = measure_phase(torch, all_joint, args.warmup, args.samples)
        for index, object_id in enumerate(scene.object_ids):
            if objects[index].proxy.count:
                phases[f"solo_{index}"] = dict(measure_phase(
                    torch, lambda index=index: solo(index), args.warmup, args.samples), object_id=object_id)
        phases["solo_total"] = measure_phase(torch, all_solo, args.warmup, args.samples)

        # Staged maps exist only in this isolated reduction benchmark. Its memory
        # baseline includes these maps; the normal analyzer reduces one pass at a time.
        maps = [(scene.render(scene.joint_model, colors=scene.joint_colors(start))["render"],
                 min(3, object_count - start)) for start in groups]
        maps.extend((scene.render(model, colors=scene.solo_colors(index))["render"], 1)
                    for index, model in enumerate(scene.object_models) if objects[index].proxy.count)

        def reduce_and_transfer():
            areas = [image[:channels].sum(dim=(1, 2), dtype=torch.float64)
                     for image, channels in maps]
            torch.cat(areas).cpu().numpy()

        phases["reduction_and_scalar_transfer"] = measure_phase(
            torch, reduce_and_transfer, args.warmup, args.samples)
        del maps, scene
    # Drop prepared GPU input before timing the public API; it owns its own guard.
    gc.collect()
    torch.cuda.synchronize()

    def complete(diagnostics=False):
        analyze_visibility(objects, camera, timestamp, diagnostics=diagnostics,
                           renderer_settings=settings)

    with torch.no_grad():
        phases["complete_analysis"] = measure_phase(
            torch, complete, args.warmup, args.samples, events=False)
        if args.diagnostics:
            phases["complete_analysis_with_diagnostics"] = measure_phase(
                torch, lambda: complete(True), args.warmup, args.samples, events=False)
    return {"scene": source["name"], "objects": object_count,
            "gaussians_per_object": {obj.object_id: obj.proxy.count for obj in objects},
            "total_gaussians": sum(obj.proxy.count for obj in objects),
            "resolution": [resolution, resolution], "timestamp": timestamp,
            "joint_passes": len(groups), "solo_passes": sum(bool(obj.proxy.count) for obj in objects),
            "prepared_input_baseline": prepared_baseline, "renderer_metadata": scene_metadata,
            "input_provenance": source["provenance"], "environment": environment, "phases": phases}


def print_table(cases):
    print("scene objects Gaussians resolution phase mean_ms median_ms p95_ms peak_alloc_MiB peak_reserved_MiB")
    for case in cases:
        for name, phase in case["phases"].items():
            timing = phase["gpu_ms"] or phase["wall_ms"]
            memory = phase["memory"]
            print(f"{case['scene']} {case['objects']} {case['total_gaussians']} "
                  f"{case['resolution'][0]}x{case['resolution'][1]} {name} "
                  f"{timing['mean']:.3f} {timing['median']:.3f} {timing['p95']:.3f} "
                  f"{memory['peak_allocated_bytes'] / 2**20:.3f} "
                  f"{memory['peak_reserved_bytes'] / 2**20:.3f}")
    print("Pass phases use CUDA-event ms; complete_analysis phases use synchronized wall ms.")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.no_synthetic and not args.scene:
        raise ValueError("--no-synthetic requires at least one --scene")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Benchmark requires the project's CUDA PyTorch environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no performance measurements were produced")
    sources = [] if args.no_synthetic else [synthetic_scene(max(args.objects), args.gaussians)]
    sources.extend(load_scene(path) for path in args.scene)
    cases, skipped = [], []
    for source in sources:
        for count in dict.fromkeys(args.objects):
            if count > len(source["objects"]):
                skipped.append({"scene": source["name"], "objects": count,
                                "reason": "fewer objects in supplied scene"})
                continue
            for resolution in dict.fromkeys(args.resolutions):
                cases.append(benchmark_case(source, count, resolution, args))
    if not cases:
        raise ValueError("No supplied scene contains a requested object count")
    report = {"schema": "visibility-benchmark-v1", "warmup": args.warmup,
              "samples": args.samples, "diagnostics_default": False,
              "measurement_notes": [
                  "RGB and analysis use identical prepared proxy geometry, transforms, camera, and timestamp.",
                  "CUDA-event pass timings exclude initial model preparation/upload; complete API wall timings include them.",
                  "Reduction phase stages maps only for measurement; its baseline includes those maps.",
                  "Reserved memory includes allocator caches from earlier phases; no empty_cache inside timing loops.",
                  "Statistics use nearest-rank p95; no full display-quality RGB comparison is implied."],
              "cases": cases, "skipped_cases": skipped}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print_table(cases)
    print(f"JSON: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        print(f"Visibility benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Evaluate delivered decoded states against original prepared GT camera images.

This diagnostic is separate from adaptation profiles, whose reference remains
the highest decoded quality. It never trains, encodes or substitutes a renderer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.content_preparation.assets import load_state, state_hash
from tools.content_preparation.checkpoint import Journal, file_lock, input_hashes, sha256, write_json
from tools.content_preparation.config import configure_runtime, resolve_path
from tools.content_preparation.metrics import MetricEvaluator
from tools.content_preparation.renderer_adapter import camera_parameters, render
from tools.content_preparation.validation import safe_relative_path
from tools.content_preparation.upstream import runtime_provenance, source_snapshot


def sample_from_pose(pose, settings):
    """Recover an orbit only when its matrix agrees with the actual GT camera."""
    c2w = np.asarray(pose, dtype=np.float64).copy()
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
        raise ValueError("GT pose must be a finite 4x4 camera-to-world matrix")
    if settings.get("up_axis") != "z":
        raise ValueError("This GT orbit diagnostic requires explicit Z-up")
    eye = c2w[:3, 3]
    distance = float(np.linalg.norm(eye))
    if distance <= 0:
        raise ValueError("GT camera cannot coincide with the origin")
    sample = {"azimuth": math.degrees(math.atan2(eye[0], eye[1])),
              "elevation": math.degrees(math.asin(float(np.clip(eye[2]/distance, -1, 1)))),
              "distance": distance, "scale": 1., "target": [0., 0., 0.]}
    c2w[:3, 1:3] *= -1  # Same OpenGL-to-COLMAP convention as upstream.
    expected = np.linalg.inv(c2w)
    error = float(np.max(np.abs(camera_parameters(sample, settings).world_view-expected)))
    if error > 2e-5:
        raise ValueError(f"GT pose has roll/target/orientation incompatible with orbit renderer (matrix error {error}); refusing a mismatched comparison")
    return sample, error


def read_gt(path, background, width, height):
    with Image.open(path) as im:
        if im.size != (width, height):
            raise ValueError(f"GT dimensions {im.size} != configured native render {(width,height)}; no automatic resize: {path}")
        rgba = np.asarray(im.convert("RGBA"), dtype=np.float32)/255
    rgb = rgba[..., :3]*rgba[..., 3:4] + np.asarray(background, dtype=np.float32)*(1-rgba[..., 3:4])
    return np.clip(rgb, 0, 1).astype(np.float32)


def evaluate(object_root, source_template, poses, frames, views, output, resume=False):
    object_root, output = Path(object_root).resolve(), Path(output).resolve()
    if output == object_root or output.is_relative_to(object_root):
        raise ValueError("GT diagnostic output must be outside the prepared object directory")
    cfg = json.loads((object_root.parent/".preparation/resolved_config.json").read_text())
    configure_runtime(cfg)
    settings = dict(cfg["renderer"])
    settings.update(up_axis="z", fov_degrees=math.degrees(json.loads(Path(poses).read_text())["camera_angle_x"]))
    extension = resolve_path(cfg["runtime"]["extension_path"]) if cfg["runtime"].get("extension_path") else None
    if extension is not None:
        if not extension.is_dir():
            raise FileNotFoundError(f"Configured extension_path does not exist: {extension}")
        sys.path.insert(0, str(extension))
    camera_list = json.loads(Path(poses).read_text())["frames"]
    if any(i < 0 or i >= len(camera_list) for i in views):
        raise ValueError("GT view index is outside the provided pose list")
    manifest = json.loads((object_root/"manifest.json").read_text())
    decoder = json.loads((object_root/"decoding.json").read_text())
    signatures = {"schema": "decoded-gt-diagnostic-v1", "object_root": str(object_root),
                  "prepared_manifest_sha256": sha256(object_root/"manifest.json"),
                  "decoding_index_sha256": sha256(object_root/"decoding.json"),
                  "source_template": str(source_template), "poses": str(Path(poses).resolve()),
                  "frames": frames, "views": views, "renderer": settings,
                  "metrics": cfg["metrics"], "upstream_sources": source_snapshot(),
                  "runtime": {k: v for k, v in runtime_provenance(cfg).items()
                              if k not in {"gpu", "cuda_runtime"}},
                  "implementation_hashes": input_hashes(sorted(
                      (ROOT/"tools/content_preparation").glob("*.py"))),
                  "tool_sha256": sha256(__file__)}
    marker = output/"owner.json"
    if output.exists() and any(output.iterdir()) and not marker.exists():
        raise FileExistsError("Refusing unowned nonempty GT diagnostic output")
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output/"run.lock"), file_lock(Path(tempfile.gettempdir())/f"content-preparation-gpu-{os.getuid()}.lock"):
        if marker.exists() and json.loads(marker.read_text()) != signatures:
            raise FileExistsError("GT diagnostic settings changed; choose a new output")
        write_json(marker, signatures)
        journal = Journal(output, resume=resume)
        rows = []
        with MetricEvaluator(cfg["metrics"]) as metrics:
            for quality in manifest["quality_order"]:
                for frame in frames:
                    decoded = next((r for r in decoder["states"] if r["quality"] == quality and r["frame"] == frame and r["mode"] == "independent"), None)
                    if not decoded:
                        raise ValueError(f"Missing independent decoded state {quality}/{frame}")
                    safe_relative_path(decoded["path"])
                    state_path = object_root/decoded["path"]
                    for view in views:
                        camera = camera_list[view]
                        relative = Path(camera["file_path"])
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError("Unsafe GT image path")
                        gt = Path(str(source_template).format(frame=frame))/relative
                        if gt.suffix != ".png":
                            gt = gt.with_suffix(".png")
                        sample, matrix_error = sample_from_pose(camera["transform_matrix"], settings)
                        key = f"{quality}/{frame}/view{view}"
                        target = output/quality/f"{frame}_view{view}.json"
                        image_path = target.with_suffix(".png")

                        def action():
                            state = load_state(state_path)
                            if state_hash(state) != decoded["state_hash"]:
                                raise ValueError("Decoded state differs from packaged decode provenance")
                            reference = read_gt(gt, settings["background"], settings["width"], settings["height"])
                            result = render(state, sample, settings)
                            value = {"quality": quality, "frame": frame, "gt_view_index": view,
                                     "reference": "original prepared GT at verified matching camera",
                                     "metrics": metrics.compare(result["rgb"], reference).to_dict(),
                                     "metric_method": metrics.metadata(),
                                     "camera_world_view_max_absolute_error": matrix_error,
                                     "sample": sample, "gt_sha256": sha256(gt),
                                     "decoded_state_hash": decoded["state_hash"],
                                     "render_metadata": result["metadata"],
                                     "image_path": image_path.relative_to(output).as_posix()}
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(np.rint(np.clip(result["rgb"],0,1)*255).astype(np.uint8)).save(image_path)
                            write_json(target, value)
                            return value, [target, image_path]

                        row = journal.run(key, signatures, [state_path, gt, Path(poses)], action)
                        rows.append(row)
                        print(f"GT {key}: MSE={row['metrics']['mse']:.6g}, PSNR={row['metrics']['psnr']}, LPIPS={row['metrics']['lpips']}", flush=True)
            metric_metadata = metrics.metadata()
            if metric_metadata.get("lpips_backend") is None:
                # Resume may load only CPU JSON rows, never the LPIPS criterion.
                # Retain the method of the measurements instead of replacing it
                # with the lazy evaluator's not-yet-loaded backend.
                metric_metadata = next((r["metric_method"] for r in rows
                    if r.get("metric_method", {}).get("lpips_backend") is not None), metric_metadata)
        aggregates = [{"quality": q, "sample_count": sum(r["quality"]==q for r in rows),
                       **{f"mean_{m}": float(np.mean([float(r["metrics"][m]) for r in rows if r["quality"]==q])) for m in ("mse","psnr","lpips")}}
                      for q in manifest["quality_order"]]
        report = {"schema": "decoded-gt-diagnostic-v1", "comparison_scope": "finite sampled GT cameras; not a full-sequence convergence claim",
                  "adaptation_profile_reference_unchanged": "highest-quality decoded representation",
                  "source_provenance": signatures, "metric_method": metric_metadata,
                  "samples": rows, "aggregates": aggregates,
                  "tasks_executed": journal.executed, "tasks_skipped": journal.skipped}
        # Infinite PSNR is JSON-safe and explicit, including aggregate values.
        for row in aggregates:
            if math.isinf(row["mean_psnr"]):
                row["mean_psnr"] = "inf"
        write_json(output/"report.json", report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-root",type=Path,required=True)
    parser.add_argument("--source-template",required=True)
    parser.add_argument("--poses",type=Path,required=True)
    parser.add_argument("--frames",type=int,nargs="+",default=[1051,1055])
    parser.add_argument("--views",type=int,nargs="+",default=[5,10])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    report=evaluate(args.object_root,args.source_template,args.poses,args.frames,args.views,args.output,args.resume)
    print(json.dumps({"samples":len(report["samples"]),"aggregates":report["aggregates"]},indent=2))


if __name__=="__main__":
    main()

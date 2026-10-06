#!/usr/bin/env python3
"""Extend verified Dynamic-LapisGS checkpoints using the unchanged trainer.

Existing qualities are imported. Only missing finer qualities are trained, one
quality/frame per subprocess. This wrapper does not encode or change upstream.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.content_preparation import __version__
from tools.content_preparation.checkpoint import Journal, file_lock, input_hashes, sha256, write_json
from tools.content_preparation.config import config_hash, frame_records, load_config, resolve_path
from tools.content_preparation.upstream import (build_training_command,
    initialize_prepared_points, run_command, source_snapshot, verify_checkpoint_lineage)

EVALUATION_NOTE = ("Upstream first-frame foundation training_report reads the stale Scene.gaussians "
                   "object; its printed PSNR is not used for quality selection. Evaluate exported "
                   "decoded checkpoints through the preparation renderer/profile pipeline.")


def _image_path(frame):
    path = Path(frame["file_path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Pose image paths must be safe relative paths")
    return path if path.suffix.lower() == ".png" else Path(str(path) + ".png")


class LadderExtension:
    def __init__(self, config, config_path, source_manifest, work_root,
                 fullres_template, poses_root, resume=False, overwrite=False):
        self.cfg, self.config_path = config, Path(config_path).resolve()
        if len(config["objects"]) != 1 or config["training"]["backend"] != "existing":
            raise ValueError("Extension requires one object and a preparation config with training.backend: existing")
        self.obj = config["objects"][0]
        self.frames = frame_records(self.obj)
        self.qualities = self.obj["qualities"]
        self.source_path = resolve_path(source_manifest)
        self.work = resolve_path(work_root)
        self.manifest_path = resolve_path(self.obj["dataset"]["checkpoint_manifest"])
        if not self.manifest_path.is_relative_to(self.work) or self.manifest_path == self.work:
            raise ValueError("Configured checkpoint_manifest must be inside work-root")
        self.fullres_template, self.poses_root = fullres_template, resolve_path(poses_root)
        self.resume, self.overwrite = resume, overwrite
        self.snapshot = source_snapshot()
        self.wrapper_sha256 = sha256(__file__)
        self.source = json.loads(self.source_path.read_text())
        for row in self.source["frames"]:
            for q in self.qualities:
                if q["id"] in row:
                    row[q["id"]] = str(resolve_path(row[q["id"]], self.source_path.parent))
        frame_ids = [r["frame"] for r in self.frames]
        if frame_ids != [r["frame"] for r in self.source["frames"][:len(frame_ids)]]:
            raise ValueError("Requested frames must retain the original initial frame and consecutive inherited lineage")
        self.source_rows = {r["frame"]: r for r in self.source["frames"]}
        self.source_commands = {(c["frame"], c["level"]): c for c in self.source.get("commands", [])}
        existing = [q["id"] for q in self.qualities if all(q["id"] in self.source_rows[f] for f in frame_ids)]
        if not existing or existing != [q["id"] for q in self.qualities[:len(existing)]]:
            raise ValueError("Existing qualities must form the inherited prefix of the requested quality ladder")
        self.existing, self.missing = existing, self.qualities[len(existing):]
        for f in frame_ids:
            for q in existing:
                command = self.source_commands.get((f, q), {})
                if not command.get("sha256") or not command.get("argv") or sha256(self.source_rows[f][q]) != command["sha256"]:
                    raise ValueError(f"Original hash/training argv missing or changed for {q}/{f}")
        verify_checkpoint_lineage(self.source, existing)
        if self.source.get("data_spec"):
            self.data_spec = resolve_path(self.source["data_spec"], self.source_path.parent)
            if self.source.get("data_spec_sha256") != sha256(self.data_spec):
                raise ValueError("Original raw/preprocessing data_spec provenance mismatch")
        protected = [self.source_path, self.poses_root,
                     *[self.source_frame(f) for f in frame_ids],
                     *[Path(self.source_rows[f][q]) for f in frame_ids for q in existing]]
        if self.source.get("data_spec"):
            protected.append(self.data_spec)
        if any(self.work.is_relative_to(p) or p.is_relative_to(self.work) for p in protected):
            raise ValueError("work-root must not overlap original images, poses, checkpoints, or provenance")

    def source_frame(self, frame):
        return resolve_path(self.fullres_template.format(frame=frame, object=self.obj["id"]))

    def plan(self, stage):
        return {"dry_run": True, "stage": stage, "work_root": str(self.work),
                "manifest": str(self.manifest_path), "frames": [r["frame"] for r in self.frames],
                "reused_qualities": self.existing,
                "new_qualities": [{"id": q["id"], "resolution_scale": q["resolution_scale"],
                                   "training_resolution": 1024 // q["resolution_scale"]}
                                  for q in self.missing],
                "initial_iterations": self.cfg["training"]["initial_iterations"],
                "dynamic_iterations": self.cfg["training"]["dynamic_iterations"],
                "trainer": "unchanged train.py; one quality and frame per subprocess",
                "training_evaluation_note": EVALUATION_NOTE}

    def task(self, stage, key, inputs, action, extra=None, command=None):
        return self.journal.run(f"{stage}/{key}", {"object": self.obj,
            "training": self.cfg["training"], "runtime": self.cfg["runtime"],
            "wrapper_sha256": self.wrapper_sha256, "extra": extra},
            inputs, action, command=command, version=__version__)

    def prepare_frame(self, quality, frame, target):
        from PIL import Image
        # Import only the original image-resizing helper; no Open3D rendering.
        from tools.content_preparation.compatibility import rescale_image
        source, scale = self.source_frame(frame), quality["resolution_scale"]
        if scale not in (1, 2):
            raise ValueError("This extension prepares missing native res2/res1 qualities only")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        counts = {}
        for split in ("train", "test"):
            pose_path = self.poses_root / f"transforms_{split}.json"
            poses = json.loads(pose_path.read_text())
            if not poses.get("frames") or not np.isfinite(float(poses["camera_angle_x"])):
                raise ValueError(f"Invalid {split} camera metadata")
            if len(list((source / split).glob("*.png"))) != len(poses["frames"]):
                raise ValueError(f"{split} image count does not match original poses: {source}")
            for camera in poses["frames"]:
                matrix = np.asarray(camera["transform_matrix"], dtype=float)
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError("Camera transforms must be finite 4x4 matrices")
                relative = _image_path(camera)
                if relative.parts[0] != split:
                    raise ValueError(f"Camera image path must belong to {split}")
                original, output = source / relative, target / relative
                with Image.open(original) as image:
                    if image.size != (1024, 1024) or image.mode != "RGBA":
                        raise ValueError(f"Expected original 1024x1024 RGBA image: {original}")
                output.parent.mkdir(parents=True, exist_ok=True)
                if scale == 1:
                    shutil.copyfile(original, output)
                else:
                    with rescale_image(str(original), scale) as resized:
                        resized.save(output)
                with Image.open(output) as image:
                    if image.size != (1024 // scale, 1024 // scale):
                        raise ValueError("Prepared image resolution differs from configured quality")
            shutil.copyfile(pose_path, target / pose_path.name)
            counts[split] = len(poses["frames"])
        initialize_prepared_points(target / "points3d.ply", self.cfg["runtime"]["seed"])
        return {"quality": quality["id"], "frame": frame, "source": str(target),
                "full_resolution_source": str(source), "resolution": 1024 // scale,
                "image_counts": counts}, [p for p in sorted(target.rglob("*")) if p.is_file()]

    def stage_prepare(self):
        rows = []
        for quality in self.missing:
            for sample in self.frames:
                frame, q = sample["frame"], quality["id"]
                target = self.work / "source" / q / str(frame)
                rows.append(self.task("prepare", f"{q}/{frame}",
                    [self.source_frame(frame), self.poses_root / "transforms_train.json", self.poses_root / "transforms_test.json"],
                    lambda quality=quality, frame=frame, target=target: self.prepare_frame(quality, frame, target)))
        path = self.work / "prepared.json"
        def action():
            value = {"sources": rows, "original_data_spec": self.source.get("data_spec"),
                     "original_data_spec_sha256": self.source.get("data_spec_sha256")}
            write_json(path, value)
            return value, [path]
        self.task("prepare", "index", [r["source"] for r in rows], action)

    def stage_train(self):
        prepared = json.loads((self.work / "prepared.json").read_text())
        prepared_map = {(r["quality"], r["frame"]): r for r in prepared["sources"]}
        rows = {r["frame"]: deepcopy(self.source_rows[r["frame"]]) for r in self.frames}
        commands = []
        for qi, quality in enumerate(self.qualities):
            q, previous = quality["id"], None
            for fi, sample in enumerate(self.frames):
                frame = sample["frame"]
                if q in self.existing:
                    original = deepcopy(self.source_commands[(frame, q)])
                    path = self.work / "imports" / q / f"{frame}.json"
                    def action(original=original, path=path, frame=frame, q=q):
                        value = dict(original, path=rows[frame][q], reused=True)
                        write_json(path, value)
                        return value, [path]
                    record = self.task("train", f"{q}/{frame}", [self.source_path, rows[frame][q]], action)
                else:
                    source = Path(prepared_map[(q, frame)]["source"])
                    model = self.work / "models" / q / str(frame)
                    steps = self.cfg["training"]["initial_iterations" if fi == 0 else "dynamic_iterations"]
                    foundation = rows[frame][self.qualities[qi - 1]["id"]] if fi == 0 else None
                    command = build_training_command(source, model, quality, self.cfg["training"], steps, previous, foundation)
                    path = model / "point_cloud" / f"iteration_{steps}" / "point_cloud.ply"
                    log = self.work / "logs" / f"train_{q}_{frame}.log"
                    def action(command=command, path=path, model=model, log=log, frame=frame, q=q, source=source):
                        if model.exists():
                            shutil.rmtree(model)
                        run_command(command, log, self.cfg)
                        if not path.is_file():
                            raise RuntimeError(f"Original trainer produced no checkpoint: {path}")
                        return {"frame": frame, "level": q, "path": str(path), "sha256": sha256(path),
                                "argv": command, "log": str(log), "source": str(source), "reused": False}, [path, log]
                    record = self.task("train", f"{q}/{frame}",
                        [source, *([previous] if previous else []), *([foundation] if foundation else [])], action,
                        extra={"steps": steps, "foundation": foundation, "previous": previous}, command=command)
                rows[frame][q] = record["path"]
                previous = record["path"]
                commands.append(record)
        value = deepcopy(self.source)
        prepared_path = self.work / "prepared.json"
        prepared_sources = [dict(r, content_sha256=input_hashes([r["source"]])[r["source"]])
                            for r in prepared["sources"]]
        value.update(frames=list(rows.values()), commands=commands, lineage_verified=True,
                     source_manifest=str(self.source_path), source_manifest_sha256=sha256(self.source_path),
                     extended_quality_ladder=[q["id"] for q in self.qualities], upstream_sources=self.snapshot,
                     extension_wrapper_sha256=self.wrapper_sha256, preparation_config_sha256=config_hash(self.cfg),
                     extension_prepared_index=str(prepared_path), extension_prepared_index_sha256=sha256(prepared_path),
                     extension_prepared_sources=prepared_sources,
                     training_evaluation_note=EVALUATION_NOTE)
        def action():
            verify_checkpoint_lineage(value, [q["id"] for q in self.qualities])
            write_json(self.manifest_path, value)
            return {"manifest": str(self.manifest_path), "lineage_verified": True}, [self.manifest_path]
        self.task("train", "manifest", [self.source_path, prepared_path,
                  *[r["source"] for r in prepared_sources], *[c["path"] for c in commands]], action)

    def run(self, stage="all", dry_run=False):
        if dry_run:
            return self.plan(stage)
        marker = self.work / ".extension" / "owner.json"
        if self.work.exists() and any(self.work.iterdir()) and not marker.exists():
            raise FileExistsError(f"Refusing unowned work directory: {self.work}")
        self.work.mkdir(parents=True, exist_ok=True)
        with file_lock(self.work / ".extension" / "run.lock"):
            if marker.exists() and json.loads(marker.read_text()).get("purpose") != "extend-quality-ladder-v1":
                raise FileExistsError("Work directory belongs to a different pipeline")
            write_json(marker, {"purpose": "extend-quality-ladder-v1", "object": self.obj["id"]})
            self.journal = Journal(self.work, self.resume, self.overwrite)
            selected = ("prepare", "train") if stage == "all" else (stage,)
            try:
                for name in selected:
                    print(f"[{self.obj['id']}] extend {name}", flush=True)
                    with self.journal.stage(f"extension/{name}"):
                        if name == "train":
                            with file_lock(Path(tempfile.gettempdir()) / f"content-preparation-gpu-{os.getuid()}.lock"):
                                self.stage_train()
                        else:
                            self.stage_prepare()
            finally:
                after = source_snapshot()
                write_json(self.work / "validation_upstream.json", {"unchanged": after == self.snapshot,
                           "before": self.snapshot, "after": after})
                if after != self.snapshot:
                    raise RuntimeError("Protected upstream sources changed during quality extension")
        return {"work_root": str(self.work), "manifest": str(self.manifest_path),
                "tasks_executed": self.journal.executed, "tasks_skipped": self.journal.skipped,
                "reused_qualities": self.existing, "new_qualities": [q["id"] for q in self.missing],
                "upstream_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--fullres-template", required=True)
    parser.add_argument("--poses-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = LadderExtension(load_config(args.config), args.config, args.source_manifest,
            args.work_root, args.fullres_template, args.poses_root, args.resume, args.overwrite).run(args.stage, args.dry_run)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError, KeyError, ImportError) as error:
        print(f"quality extension: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

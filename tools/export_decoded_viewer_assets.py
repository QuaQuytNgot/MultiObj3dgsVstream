#!/usr/bin/env python3
"""Export already decoded packaged Gaussian states into the existing web viewer.

This CPU-only tool never trains, encodes, decodes or renders a Gaussian model.
Its PLY assets are diagnostics, separately accounted from the encoded media.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.content_preparation.assets import load_state, read_ply, write_ply, state_hash
from tools.content_preparation.checkpoint import Journal, file_lock, sha256, write_json
from tools.content_preparation.run_report import report_prepared_run, _object_locations, _encode_timings
from tools.content_preparation.validation import safe_relative_path
from tools.summarize_8i_trial import _identifier, _quality_metadata, _renderer_settings, _thumbnail

CATALOG_VERSION = "content-preparation.8i-trial-catalog.v2"
VIEWER = ROOT / "tools/content_trial_viewer"


def _safe(root, value):
    safe_relative_path(value)
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Viewer input escapes its root: {value}")
    return path


def _relative(root, path):
    return Path(path).resolve().relative_to(root).as_posix()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def export_viewer(prepared_root, mode="progressive", frames="all", resume=False, overwrite=False):
    root = Path(prepared_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if mode not in {"progressive", "independent"}:
        raise ValueError("Unsupported delivery mode")
    requested = None if frames == "all" else set(int(frame) for frame in frames)
    from tools.content_trial_viewer.vendor_dependencies import vendor
    vendor(verify=True)
    state_dir = root / ".viewer_export"
    with file_lock(root / ".preparation/run.lock"), file_lock(state_dir / "process.lock"):
        owner = state_dir / "owner.json"
        if owner.exists():
            if _read(owner) != {"schema": "content-preparation.viewer-owner.v1", "root": str(root)}:
                raise ValueError("Viewer export owner differs from prepared root")
        else:
            collisions = [root / name for name in ("catalog.json", "index.html", "viewer_assets/index.json") if (root / name).exists()]
            if collisions and not overwrite:
                raise FileExistsError(f"Unowned viewer artifacts exist: {collisions}; use a new output or --overwrite")
            write_json(owner, {"schema": "content-preparation.viewer-owner.v1", "root": str(root)})
        journal = Journal(root, resume, overwrite)
        journal.path = state_dir / "journal.json"
        journal.data = _read(journal.path) if journal.path.exists() else {"version": 1, "tasks": {}, "stages": {}}
        implementation = {"exporter": sha256(Path(__file__)), "assets": sha256(ROOT / "tools/content_preparation/assets.py"),
                          "viewer": sha256(VIEWER / "index.html"), "viewport": sha256(VIEWER / "gaussian_viewport.js"),
                          "dependencies": sha256(VIEWER / "dependencies.lock.json")}
        config_path = root / ".preparation/resolved_config.json"
        config = _read(config_path) if config_path.is_file() else {}
        fps_values = {obj.get("fps", 30) for obj in config.get("objects", [])}
        # report_prepared_run uses the explicit FPS only for each final sample.
        # Objects with a different FPS are reported separately below.
        locations = _object_locations(root)
        entries, object_rows = [], []
        pipeline_journal_path = root / ".preparation/journal.json"
        pipeline_journal = _read(pipeline_journal_path) if pipeline_journal_path.is_file() else {}
        for object_root, manifest in locations:
            object_id = _identifier((manifest or {}).get("object", {}).get("id", object_root.name))
            obj_config = next((obj for obj in config.get("objects", []) if obj["id"] == object_id), {})
            fps = obj_config.get("fps", next(iter(fps_values), 30))
            report = report_prepared_run(object_root, fps=fps, variant_id="prepared")
            obj = report["objects"][0]
            # Report paths are relative to object_root; catalog paths are root-relative.
            for rep in obj["representations"]:
                for segment in rep["segments"]:
                    segment["path"] = _relative(root, _safe(object_root, segment["path"]))
                for access in rep["cold_access"]:
                    access["segment_paths"] = [_relative(root, _safe(object_root, path)) for path in access["segment_paths"]]
                for count in rep["gaussian_counts"]:
                    if count.get("state_asset"):
                        count["state_asset"]["path"] = _relative(root, _safe(object_root, count["state_asset"]["path"]))
            decoding_path = object_root / "decoding.json"
            decoding = _read(decoding_path)
            if decoding.get("source") != "requestable segment containers only":
                raise ValueError("Viewer requires states decoded from requestable segment containers")
            package_path = object_root / "package.json"
            package = _read(package_path)
            obj["encode_timings"] = _encode_timings(pipeline_journal, object_id, package["payloads"])
            expected_frames = {row["frame"] for row in package["frames"]}
            if requested is not None and requested - expected_frames:
                raise ValueError(f"Requested frames unavailable for {object_id}: {sorted(requested - expected_frames)}")
            states = [row for row in decoding["states"] if row["mode"] == mode and (requested is None or row["frame"] in requested)]
            if not states:
                raise ValueError(f"No decoded {mode} states for {object_id}")
            rows = []
            for record in states:
                quality = _identifier(record["quality"])
                frame = int(record["frame"])
                source = _safe(object_root, record["path"])
                payload = next(item for item in package["payloads"] if item["mode"] == mode and item["quality"] == quality and item["frame"] == frame)
                rep = next(item for item in obj["representations"] if item["mode"] == mode and item["quality"] == quality)
                access = next(item for item in rep["cold_access"] if item["frame"] == frame)
                segment_paths = [_safe(root, path) for path in access["segment_paths"]]
                path = root / "viewer_assets" / object_id / quality / f"{frame}.ply"
                def action(source=source, path=path, record=record, payload=payload, access=access):
                    state = load_state(source)
                    if state.sh_degree > 3:
                        raise ValueError("Pinned web viewer supports SH degree 0–3; use upstream PNGs for higher degrees")
                    numeric_hash = state_hash(state)
                    if numeric_hash != record["state_hash"] or numeric_hash != payload["decoded_state_hash"]:
                        raise ValueError("Decoded viewer source differs from its packaged reconstruction hash")
                    result = write_ply(path, state)
                    if state_hash(read_ply(path)) != numeric_hash:
                        raise ValueError("Viewer PLY roundtrip changed the decoded numeric state")
                    result.update(path=_relative(root, path), object=object_id, quality=record["quality"], frame=record["frame"],
                                  mode=record["mode"], layer=record["layer"], decoded_state_hash=numeric_hash,
                                  gaussian_count=state.count, sh_degree=state.sh_degree,
                                  source_decoded_path=_relative(root, source), source_decoded_sha256=sha256(source),
                                  decoding_index_sha256=sha256(decoding_path), package_index_sha256=sha256(package_path),
                                  dependencies_in_decode_order=access["dependencies_in_decode_order"],
                                  source_segment_paths=access["segment_paths"], source_payload_id=payload["id"],
                                  format="binary_little_endian_gaussian_ply", attribute_domains="float32 xyz/log-scale/logit-opacity/raw-wxyz/SH",
                                  purpose="diagnostic web asset, excluded from encoded media bitrate")
                    return result, [path]
                rows.append(journal.run(f"viewer/{object_id}/{quality}/{frame}", {"mode": mode, "implementation": implementation},
                                        [source, decoding_path, package_path, *segment_paths], action))
            entries.extend(rows)
            profile_path = object_root / "profiles/profile.json"
            profile = _read(profile_path) if profile_path.is_file() else {}
            previews = []
            for row in profile.get("rows", []):
                if row["mode"] != mode or (requested is not None and row["frame"] not in requested) or not row.get("image_path"):
                    continue
                source = _safe(object_root, row["image_path"])
                def action(row=row):
                    preview = _thumbnail(root, "prepared", object_id, row, object_root)
                    if preview is None:
                        raise FileNotFoundError("Missing decoded-render RGB cache")
                    return preview, [root / preview["path"]]
                previews.append(journal.run(f"preview/{object_id}/{row['sample_id']}/{row['quality']}", implementation, [source, profile_path], action))
            qualities = [rep["quality"] for rep in obj["representations"] if rep["mode"] == mode]
            quality_meta = _quality_metadata(root, object_id, object_root, qualities)
            reps = []
            for rep in obj["representations"]:
                if rep["mode"] != mode:
                    continue
                rep["thumbnails"] = [preview for preview in previews if preview["quality"] == rep["quality"]]
                rep["viewer_assets"] = [row for row in rows if row["quality"] == rep["quality"]]
                rep["quality_metadata"] = quality_meta[rep["quality"]]
                timings = [row["wall_seconds"] for row in obj["encode_timings"]["tasks"] if row["quality"] == rep["quality"] and row["mode"] == mode and row["wall_seconds"] is not None]
                rep["recorded_successful_encode_task_wall_seconds"] = sum(timings) if timings else None
                reps.append(rep)
            metadata = (manifest or {}).get("object", {})
            object_rows.append({**obj, "representations": reps, "renderer_settings": _renderer_settings(root, object_id, profile),
                               "manifest_path": _relative(root, object_root / "manifest.json") if manifest else None,
                               "quality_profile_path": _relative(root, profile_path) if profile_path.is_file() else None,
                               "proxy_path": _relative(root, _safe(object_root, metadata["proxy_path"])) if metadata.get("proxy_path") else None,
                               "sampled_frame_ids": sorted({row["frame"] for row in previews}), "thumbnails": previews})
        catalog = {"schema_version": CATALOG_VERSION, "path_base": "directory containing catalog.json", "viewer_path": "index.html",
                   "viewer_asset_index": "viewer_assets/index.json", "default_display": "gaussians",
                   "raw_source": {"source_raw_samples_duration_seconds": None, "source_archive_frame_count": None,
                                  "capture_fps": None, "inventoried_frame_count": None, "recorded_complete_for_selection": False},
                   "variants": [{"id": "prepared", "root_path": ".", "run_report_path": None, "objects": object_rows}],
                   "viewer_backend": {"name": "SparkJS", "version": "2.2.0", "three_version": "0.180.0", "extSplats": True, "lod": False,
                                      "limitations": "Browser packs opacity/scale/quaternion/SH for qualitative inspection; objective metrics use original upstream renderer."}}
        index = {"schema": "content-preparation.decoded-viewer-assets.v1", "source": "offline project decoder from requestable segment containers",
                 "assets": entries, "asset_bytes": sum(row["bytes"] for row in entries), "included_in_encoded_media_bytes": False,
                 "implementation": implementation, "backend": catalog["viewer_backend"]}
        sources = [Path(__file__), VIEWER / "index.html", VIEWER / "gaussian_viewport.js", VIEWER / "dependencies.lock.json",
                   *sorted((VIEWER / "vendor").rglob("*"))]
        sources = [path for path in sources if path.is_file()]
        def finalize():
            write_json(root / "catalog.json", catalog)
            write_json(root / "viewer_assets/index.json", index)
            outputs = [root / "catalog.json", root / "viewer_assets/index.json"]
            for source in sources[1:]:
                relative = source.relative_to(VIEWER)
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                outputs.append(target)
            return {"catalog": "catalog.json", "asset_count": len(entries)}, outputs
        descriptive_inputs = [config_path] if config_path.is_file() else []
        descriptive_inputs += [object_root / name for object_root, _ in locations for name in
                               ("manifest.json", "package.json", "decoding.json", "profiles/profile.json", "training.json")
                               if (object_root / name).is_file()]
        journal.run("viewer/catalog", {"mode": mode, "frames": sorted(requested) if requested else "all", "implementation": implementation},
                    [*sources, *[root / row["path"] for row in entries], *[root / row["path"] for obj in object_rows for row in obj["thumbnails"]],
                     *descriptive_inputs], finalize)
        return {"asset_count": len(entries), "executed": journal.executed, "skipped": journal.skipped, "catalog": str(root / "catalog.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("progressive", "independent"), default="progressive")
    parser.add_argument("--frames", nargs="+", default=["all"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    frames = "all" if args.frames == ["all"] else args.frames
    print(json.dumps(export_viewer(args.prepared_root, args.mode, frames, args.resume, args.overwrite), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

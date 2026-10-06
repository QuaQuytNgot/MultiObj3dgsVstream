"""Audit actual preparation artifacts without training or producing proxy assets.

Decoding is independently repeated from CPSEG members in a temporary directory.
No checkpoint or encoder reconstruction cache is used as a decode input.
"""
from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
import tempfile

import numpy as np

from .assets import GaussianState, load_state, read_ply, save_state, state_hash
from .checkpoint import sha256
from .codec_adapter import _quantize, get_codec, get_decoder, inspect_payload
from .config import VALIDATION_STAGES, config_hash, frame_records, resolve_path
from .quality_profile import QualityProfile
from .upstream import runtime_provenance, source_snapshot, verify_checkpoint_lineage
from .validation import safe_relative_path, validate_asset, validate_dependency_graph, validate_manifest, validate_package_index

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def within(root, relative):
    safe_relative_path(relative)
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Asset escapes prepared directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing prerequisite {path}; run its preparation stage first")
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


class PreparedValidator:
    """Config-aware, CPU sequential artifact validation with explicit GPU preflight."""
    def __init__(self, config):
        self.config = config
        self.root = resolve_path(config["output"])

    def selected_checkpoint_manifest(self, obj):
        path = resolve_path(obj["dataset"]["checkpoint_manifest"])
        manifest = read_json(path)
        frames = {r["frame"] for r in frame_records(obj)}
        qualities = [q["id"] for q in obj["qualities"]]
        rows = []
        for row in manifest["frames"]:
            if row["frame"] in frames:
                rows.append({"frame": row["frame"], **{
                    q: str(resolve_path(row[q], path.parent)) for q in qualities}})
        commands = [c for c in manifest.get("commands", [])
                    if c["frame"] in frames and c["level"] in qualities]
        require({r["frame"] for r in rows} == frames and len(rows) == len(frames),
                "Checkpoint manifest lacks unique configured frame coverage")
        return {"frames": sorted(rows, key=lambda r: r["frame"]), "commands": commands}, path

    def inputs(self, stage):
        """Hash actual source/media/profile files for safe validator resume."""
        paths = []
        if stage in ("environment", "all"):
            executable = self.config["encoding"].get("executable")
            if executable:
                paths.append(resolve_path(executable))
        if stage in ("dataset", "all"):
            for obj in self.config["objects"]:
                dataset = obj["dataset"]
                if dataset["kind"] == "checkpoints":
                    manifest, path = self.selected_checkpoint_manifest(obj)
                    paths += [path] + [Path(row[q["id"]]) for row in manifest["frames"] for q in obj["qualities"]]
                elif dataset["kind"] in ("raw", "prepared"):
                    template = dataset["raw_template"] if dataset["kind"] == "raw" else dataset["source_template"]
                    for frame in frame_records(obj):
                        for q in obj["qualities"]:
                            paths.append(resolve_path(template.format(frame=frame["frame"], object=obj["id"],
                                                                      quality=q["id"], scale=q["resolution_scale"])))
        # These are immutable preparation inputs, excluding validator reports,
        # journals and scratch caches. Include every cached image for profiling.
        names = {"preprocess": ["preprocess.json", "input"],
                 "checkpoints": ["training.json", "checkpoints", "models"],
                 "export": ["export.json", "qualities", "training.json"],
                 "encoding": ["encoding.json", "encoded", "export.json", "qualities"],
                 "package": ["package.json", "package_index.json", "media", "encoded"],
                 "decode": ["package.json", "media", "decoding.json", "decoded", "export.json", "qualities"],
                 "profile": ["profiles", "decoding.json", "decoded"],
                 "final": ["manifest.json", "validation.json", "package.json", "media", "encoded", "decoded", "profiles", "training.json", "export.json", "qualities"]}
        stages = list(names) if stage == "all" else [stage]
        for obj in self.config["objects"]:
            for selected in stages:
                for name in names.get(selected, []):
                    path = self.root / obj["id"] / name
                    if path.exists():
                        paths.append(path)
                if selected == "checkpoints":
                    training = self.root / obj["id"] / "training.json"
                    if training.exists():
                        paths += [Path(c["path"]) for c in read_json(training)["commands"]]
        if stage in ("final", "all"):
            for name in ("manifest.json", "manifest.mpd", "content_index.json", "catalog.json", "viewer_assets", "validation_upstream.json",
                         "index.html", "gaussian_viewport.js", "dependencies.lock.json", "vendor", "viewer_validation.json"):
                if (self.root / name).exists():
                    paths.append(self.root / name)
        return sorted(set(paths))

    def validate(self, stage):
        if stage not in VALIDATION_STAGES and stage != "all":
            raise ValueError(f"Unknown validation stage {stage}")
        results = {}
        for name in VALIDATION_STAGES if stage == "all" else (stage,):
            if name == "environment":
                results[name] = self.environment()
            else:
                results[name] = {obj["id"]: getattr(self, name)(obj, self.root / obj["id"])
                                 for obj in self.config["objects"]}
        return {"schema": "content-preparation.validation.v1", "passed": True,
                "stage": stage, "config_sha256": config_hash(self.config),
                "upstream_sources": source_snapshot(), "checks": results,
                "scope": "artifact audit; no training, proxy generation or network runtime"}

    def environment(self):
        from .metrics import MetricEvaluator
        from .paths import configure_upstream_imports
        import torch
        runtime = runtime_provenance(self.config)
        codec = get_codec(self.config["encoding"])
        codec_runtime = None
        if codec.name == "gaussian_attribute_draco_byteplanes":
            from .draco_adapter import runtime_info
            codec_runtime = runtime_info(self.config["encoding"])
        if self.config["renderer"]["backend"] == "upstream_cuda":
            require(torch.cuda.is_available(), "Configured upstream_cuda requires accessible CUDA; run preflight with GPU access")
            extension = self.config["runtime"].get("extension_path")
            configure_upstream_imports(resolve_path(extension) if extension else None)
            from gaussian_renderer import render as upstream_render
            require(callable(upstream_render), "Original renderer import failed")
        from .assets import synthetic_state
        from .renderer_adapter import render
        renderer_settings = dict(self.config["renderer"], width=32, height=32, center=[0, 0, 0])
        extension = self.config["runtime"].get("extension_path")
        if extension:
            renderer_settings["vendor_path"] = str(resolve_path(extension))
        tiny = render(synthetic_state(0, 8), {"azimuth": 0, "elevation": 0, "scale": 1, "distance": 3}, renderer_settings)
        require(np.isfinite(tiny["rgb"]).all() and float(tiny["alpha"].max()) > 0, "Tiny configured-renderer preflight failed")
        # Real pretrained LPIPS sanity at one 32px image, not a surrogate.
        evaluator = MetricEvaluator(self.config["metrics"])
        try:
            image = np.zeros((max(32, 64 if evaluator.net == "alex" else 32),) * 2 + (3,), dtype=np.float32)
            metric = evaluator.compare(image, image)
            require(metric.mse == 0 and math.isinf(metric.psnr) and abs(metric.lpips) < 1e-7,
                    "Same-image MSE/PSNR/pretrained LPIPS sanity failed")
        finally:
            evaluator.close()
        return {"runtime": runtime, "codec": {"name": codec.name, "version": codec.version, "native": codec_runtime},
                "same_image_metrics": metric.to_dict(), "lpips_batch_size": 1,
                "renderer_backend": self.config["renderer"]["backend"], "tiny_render": tiny.get("metadata", {})}

    def dataset(self, obj, root):
        dataset = obj["dataset"]
        count = len(frame_records(obj))
        if dataset["kind"] == "checkpoints":
            manifest, path = self.selected_checkpoint_manifest(obj)
            commands = {(c["frame"], c["level"]): c for c in manifest["commands"]}
            for row in manifest["frames"]:
                for q in obj["qualities"]:
                    c = commands.get((row["frame"], q["id"]))
                    require(c is not None and c.get("sha256") == sha256(row[q["id"]]),
                            f"Original checkpoint provenance missing or changed: {q['id']}/{row['frame']}")
                    require(bool(c.get("argv")), "Checkpoint import requires ORIGINAL training argv")
            if "progressive" in self.config["delivery_modes"]:
                verify_checkpoint_lineage(manifest, [q["id"] for q in obj["qualities"]])
            return {"kind": dataset["kind"], "source_sha256": sha256(path), "frames": count,
                    "checkpoints_verified": count * len(obj["qualities"]), "training_started": False}
        if dataset["kind"] != "synthetic":
            for path in self.inputs("dataset"):
                require(path.exists(), f"Dataset input missing: {path}")
        return {"kind": dataset["kind"], "frames": count}

    def preprocess(self, obj, root):
        prepared = read_json(root / "preprocess.json")
        require(prepared["kind"] == obj["dataset"]["kind"], "Preprocessing dataset kind disagrees with config")
        require([r["frame"] for r in prepared["frames"]] == [r["frame"] for r in frame_records(obj)],
                "Preprocessing frame coverage mismatch")
        if prepared["kind"] == "checkpoints":
            require(prepared["source_sha256"] == sha256(resolve_path(obj["dataset"]["checkpoint_manifest"])),
                    "Checkpoint manifest changed after import")
            require(all(r.get("checkpoint_import") for r in prepared["frames"]), "Missing checkpoint-import flag")
        elif prepared["kind"] == "synthetic":
            for row in prepared["frames"]:
                load_state(within(root, row["input"]))
        else:
            for row in prepared["frames"]:
                for source in row["sources"].values():
                    source = Path(source).resolve()
                    require(source.is_relative_to(root.resolve()), "Prepared source is not an owned copy")
                    require((source / "points3d.ply").is_file(), "Prepared Gaussian initialization missing")
                    for split in ("train", "test"):
                        cameras = read_json(source / f"transforms_{split}.json")
                        require(bool(cameras.get("frames")), "Prepared camera split is empty")
                        for camera in cameras["frames"]:
                            image = source / camera["file_path"]
                            require(image.is_file() or image.with_suffix(".png").is_file(), f"Prepared camera image missing: {image}")
        return {"frames": len(prepared["frames"]), "kind": prepared["kind"]}

    def checkpoints(self, obj, root):
        training = read_json(root / "training.json")
        qualities = [q["id"] for q in obj["qualities"]]
        require(training["backend"] == self.config["training"]["backend"], "Training backend mismatch")
        expected = {(r["frame"], q) for r in frame_records(obj) for q in qualities}
        commands = {(c["frame"], c["level"]): c for c in training["commands"]}
        require(set(commands) == expected and len(commands) == len(training["commands"]), "Training checkpoint coverage mismatch")
        for key, command in commands.items():
            require(sha256(command["path"]) == command["sha256"], f"Training checkpoint hash mismatch: {key}")
        require(training["lineage_verified"] is True, "Stable Gaussian lineage is unverified")
        if training["backend"] != "synthetic_fixture":
            verify_checkpoint_lineage(training, qualities)
        return {"checkpoints": len(commands), "backend": training["backend"], "lineage_verified": True}

    def export(self, obj, root):
        rows = read_json(root / "export.json")["states"]
        training = read_json(root / "training.json")
        expected = {(q["id"], r["frame"]) for q in obj["qualities"] for r in frame_records(obj)}
        require({(r["quality"], r["frame"]) for r in rows} == expected and len(rows) == len(expected), "Export state coverage mismatch")
        for row in rows:
            state = load_state(within(root, row["path"]))
            checkpoint = Path(next(r[row["quality"]] for r in training["frames"] if r["frame"] == row["frame"]))
            require(sha256(checkpoint) == row["checkpoint_sha256"], "Export checkpoint provenance mismatch")
            original = read_ply(checkpoint) if checkpoint.suffix == ".ply" else load_state(checkpoint)
            if training["lineage_verified"]:
                original.ids = np.arange(original.count, dtype="int64")
            require(state_hash(state) == row["state_hash"] == state_hash(original), "Export numeric state differs from its checkpoint")
            require(state.metadata.get("checkpoint_sha256") == row["checkpoint_sha256"], "Export embedded provenance mismatch")
        return {"states": len(rows), "numeric_checkpoint_equality": True}

    def encoding(self, obj, root):
        payloads = read_json(root / "encoding.json")["payloads"]
        validate_dependency_graph(payloads)
        expected = {(m, q["id"], r["frame"]) for m in self.config["delivery_modes"] for q in obj["qualities"] for r in frame_records(obj)}
        require({(p["mode"], p["quality"], p["frame"]) for p in payloads} == expected and len(payloads) == len(expected), "Encoded payload coverage mismatch")
        for payload in payloads:
            validate_asset(payload, root)
            header = inspect_payload(within(root, payload["path"]))
            for key in ("codec", "decoded_state_hash", "input_state_hash", "parent_state_hash", "self_contained"):
                require(header[key] == payload[key], f"Codec header disagrees with encoding index: {key}")
            require(header["temporal_prediction"] is False, "Unexpected temporal prediction")
        return {"payloads": len(payloads), "actual_payload_bytes": sum(p["bytes"] for p in payloads), "temporal_prediction": False}

    def package(self, obj, root):
        from .packaging import extract_segment_member
        index = read_json(root / "package.json")
        require(index == read_json(root / "package_index.json"), "Package indices differ")
        validate_package_index(index, root)
        for payload in index["payloads"]:
            data = extract_segment_member(within(root, payload["segment_path"]), payload["id"])
            import hashlib
            require(len(data) == payload["bytes"] and hashlib.sha256(data).hexdigest() == payload["sha256"], "Extracted encoded segment member mismatch")
        return {"segments": len(index["segments"]), "payloads": len(index["payloads"]),
                "media_bytes": index["media_bytes"], "container_overhead_bytes": index["media_bytes"] - sum(p["bytes"] for p in index["payloads"]),
                "all_frames_are_access_points": True, "temporal_prediction": False}

    def decode(self, obj, root):
        from .packaging import extract_segment_member
        index = read_json(root / "package.json")
        # Segment files are authoritative, including when standalone packets have
        # been archived. Structural validation below avoids requiring the latter.
        validate_package_index(index)
        for segment in index["segments"]:
            validate_asset(segment, root)
        decoded = read_json(root / "decoding.json")
        require(decoded["source"] == "requestable segment containers only", "Decoded state has an invalid source")
        rows = {(r["mode"], r["quality"], r["frame"]): r for r in decoded["states"]}
        exported = {(r["quality"], r["frame"]): r for r in read_json(root / "export.json")["states"]}
        require(len(rows) == len(decoded["states"]) == len(index["payloads"]), "Decoded state coverage mismatch")
        with tempfile.TemporaryDirectory(prefix="prepared-decode-audit-") as temporary:
            temporary = Path(temporary)
            rebuilt = {}
            pending = list(index["payloads"])
            while pending:
                progress = False
                for payload in list(pending):
                    if any(parent not in rebuilt for parent in payload["dependencies"]):
                        continue
                    parent = load_state(rebuilt[payload["dependencies"][0]]) if payload["dependencies"] else None
                    data = extract_segment_member(within(root, payload["segment_path"]), payload["id"])
                    packet = temporary / "packet.cpgs"
                    packet.write_bytes(data)
                    quality = next(q for q in obj["qualities"] if q["id"] == payload["quality"])
                    runtime_config = dict(self.config["encoding"], **quality.get("encoding", {}))
                    state = get_decoder(payload["codec"], runtime_config).decode(packet, parent)
                    row = rows[(payload["mode"], payload["quality"], payload["frame"])]
                    numeric_hash = state_hash(state)
                    require(numeric_hash == payload["decoded_state_hash"] == row["state_hash"] == state_hash(load_state(within(root, row["path"]))),
                            "Decoded cached state differs from actual delivered bytes")
                    original = load_state(within(root, exported[(payload["quality"], payload["frame"])]["path"]))
                    codec_config = payload["codec"]["config"]
                    quantized = GaussianState({name: _quantize(value, codec_config.get("attribute_precision", {}).get(name, codec_config["precision"]))[2]
                                               for name, value in original.arrays.items()}, original.metadata, original.ids)
                    require(numeric_hash == state_hash(quantized), "Decoder differs from declared quantized export state")
                    out = temporary / f"state_{len(rebuilt)}.npz"
                    save_state(out, state)
                    rebuilt[payload["id"]] = out
                    pending.remove(payload)
                    progress = True
                    del parent, state, original, quantized
                require(progress, "Decode dependency graph cannot be resolved")
        return {"states": len(rows), "independent_redecode": True,
                "decode_input": "actual requestable segment members", "declared_precision_verified": True}

    def profile(self, obj, root):
        profile = QualityProfile.load(root / "profiles/profile.json")
        qualities = [q["id"] for q in obj["qualities"]]
        cfg = self.config["sampling"]
        frames = [r for r in frame_records(obj) if cfg["times"] == "all" or r["frame"] in cfg["times"]]
        coordinates = {(m, q, r["frame"], f"a{a}_e{e}", f"s{s}")
                       for m, q, r, a, e, s in product(self.config["delivery_modes"], qualities, frames,
                           range(len(cfg["azimuth"])), range(len(cfg["elevation"])), range(len(cfg["scales"])))}
        require({(r["mode"], r["quality"], r["frame"], r["view_bin"], r["scale_bin"]) for r in profile.rows} == coordinates
                and len(profile.rows) == len(coordinates), "View/scale/time-conditioned profile coverage mismatch")
        require(profile.metadata["reference_quality"] == qualities[-1], "Profile reference is not highest decoded quality")
        require(profile.metadata["renderer_backend"] == self.config["renderer"]["backend"], "Profile renderer backend mismatch")
        require(profile.metadata["renderer_settings"] == self.config["renderer"] and profile.metadata["metrics"] == self.config["metrics"],
                "Profile renderer/metric settings disagree with config")
        sample_coordinates = {(r["frame"], f"a{a}_e{e}", f"s{s}"):
                              {"azimuth": cfg["azimuth"][a], "elevation": cfg["elevation"][e], "scale": cfg["scales"][s],
                               "distance": cfg["distance"], "timestamp": r["timestamp"], "time_bin": str(r["frame"]),
                               "sample_id": f"{r['frame']}_a{a}_e{e}_s{s}"}
                              for r, a, e, s in product(frames, range(len(cfg["azimuth"])), range(len(cfg["elevation"])), range(len(cfg["scales"])))}
        by_key = {(r["mode"], r["quality"], r["view_bin"], r["scale_bin"], r["time_bin"]): r for r in profile.rows}
        for row in profile.rows:
            require(row["object"] == obj["id"], "Profile object identity mismatch")
            expected_sample = sample_coordinates[(row["frame"], row["view_bin"], row["scale_bin"])]
            require(all(row[k] == v for k, v in expected_sample.items()), "Profile sample coordinates disagree with config")
            require(row["reference_state"] == "decoded" and row["reference_quality"] == qualities[-1], "Profile reference must be highest decoded representation")
            image = self._image(within(root, row["image_path"]))
            reference_row = by_key[(row["mode"], qualities[-1], row["view_bin"], row["scale_bin"], row["time_bin"])]
            reference = self._image(within(root, reference_row["image_path"]))
            mse = float(np.mean((image.astype(np.float64) - reference.astype(np.float64)) ** 2))
            metric = row["metrics"]
            require(math.isfinite(float(metric["mse"])) and math.isclose(float(metric["mse"]), mse, rel_tol=1e-9, abs_tol=1e-12), "MSE differs from cached decoded render")
            psnr = float(metric["psnr"])
            require((mse == 0 and psnr == math.inf) or (mse > 0 and math.isclose(psnr, -10 * math.log10(mse), abs_tol=1e-8)), "PSNR is inconsistent with MSE")
            require(metric.get("lpips") is not None and math.isfinite(float(metric["lpips"])) and float(metric["lpips"]) >= -1e-7,
                    "Missing or invalid actual LPIPS metric")
            if row["quality"] == qualities[-1]:
                require(mse == 0 and abs(float(metric["lpips"])) < 1e-7, "Highest decoded reference must have zero self-distortion")
            require(profile.indexed(obj["id"], row["quality"], row["view_bin"], row["scale_bin"], row["time_bin"], row["mode"]) == row,
                    "Raw quality profile index lookup failed")
        require(read_json(root / "profiles/gains.json") == profile.transition_gains(qualities), "Transition gain metadata mismatch")
        require(read_json(root / "profiles/aggregates.json") == profile.summarize(), "Profile aggregation mismatch")
        return {"raw_samples": len(profile.rows), "transition_samples": len(profile.transition_gains(qualities)),
                "reference": "highest quality decoded payload", "objective_renderer": self.config["renderer"]["backend"],
                "metrics": ["mse", "psnr", "lpips"], "planner_distortion": self.config["metrics"]["distortion"]}

    def _image(self, path):
        with np.load(path, allow_pickle=False) as image:
            rgb, alpha, depth = image["rgb"], image["alpha"], image["depth"]
        expected = (self.config["renderer"]["height"], self.config["renderer"]["width"])
        require(rgb.shape == expected + (3,) and alpha.shape == expected and depth.shape == expected, "Render shape mismatch")
        require(np.isfinite(rgb).all() and np.isfinite(alpha).all() and np.isfinite(depth).all(), "Nonfinite decoded render")
        require(rgb.min() >= 0 and rgb.max() <= 1 and alpha.min() >= 0 and alpha.max() <= 1, "Render range mismatch")
        return rgb

    def final(self, obj, root):
        manifest = read_json(root / "manifest.json")
        validate_manifest(manifest, root)
        require(manifest["provenance"]["config_sha256"] == config_hash(self.config), "Manifest config provenance mismatch")
        require(manifest["provenance"]["upstream_sources"] == source_snapshot(), "Upstream source provenance changed")
        if not self.config["proxy"]["enabled"]:
            expected = "external" if self.config["proxy"].get("external_descriptor") else "disabled"
            require(manifest["object"]["proxy_preparation"] == expected, "Proxy-off manifest is inconsistent")
            require(manifest["object"]["proxy_path"] == self.config["proxy"].get("external_descriptor"), "Proxy-off reference mismatch")
        validation = read_json(root / "validation.json")
        require(validation["passed"] and validation["decoded_from_segments"] and validation["lpips_batch_size"] == 1, "Pipeline did not pass its final validation")
        require(validation["gpu_guard"]["active"] == 0 and validation["gpu_guard"]["peak_active"] <= 1, "GPU sequential execution violated")
        upstream = read_json(self.root / "validation_upstream.json")
        require(upstream["unchanged"] and upstream["before"] == upstream["after"] == source_snapshot(), "Upstream training/renderer sources were changed")
        server = read_json(self.root / "manifest.json")
        record = next(r for r in server["objects"] if r["id"] == obj["id"])
        validate_asset({"path": record["manifest_path"], "bytes": record["bytes"], "sha256": record["sha256"]}, self.root)
        from .mpd import validate_mpd
        mpd_result = validate_mpd(self.root / "manifest.mpd", self.root)
        # Exporter must prove PLY numeric equality against decoder NPZ. Re-check
        # here to make the final gate independent of the exporter's done marker.
        catalog = read_json(self.root / "catalog.json")
        viewer_index = read_json(self.root / "viewer_assets/index.json")
        require(catalog.get("schema_version") == "content-preparation.8i-trial-catalog.v2", "Final Gaussian viewer requires catalog v2")
        require(viewer_index.get("included_in_encoded_media_bytes") is False, "Viewer PLY files must not enter codec bitrate accounting")
        assets = [a for a in viewer_index["assets"] if a["object"] == obj["id"]]
        expected = {(q["id"], r["frame"]) for q in obj["qualities"] for r in frame_records(obj)}
        require({(a["quality"], a["frame"]) for a in assets} == expected and len(assets) == len(expected), "Viewer decoded quality/frame coverage mismatch")
        decoding = read_json(root / "decoding.json")
        package = read_json(root / "package.json")
        from .packaging import cold_access
        for asset in assets:
            validate_asset(asset, self.root)
            state = load_state(within(self.root, asset["source_decoded_path"]))
            require(sha256(within(self.root, asset["source_decoded_path"])) == asset["source_decoded_sha256"], "Viewer decoded source file changed")
            require(state_hash(read_ply(within(self.root, asset["path"]))) == state_hash(state) == asset["decoded_state_hash"],
                    "Viewer PLY numeric state differs from decoder output")
            require(asset["decoding_index_sha256"] == sha256(root / "decoding.json") and asset["package_index_sha256"] == sha256(root / "package.json"),
                    "Viewer decoder/package provenance mismatch")
            row = next(r for r in decoding["states"] if (r["mode"], r["quality"], r["frame"]) == (asset["mode"], asset["quality"], asset["frame"]))
            require(row["state_hash"] == asset["decoded_state_hash"], "Viewer decoding row provenance mismatch")
            access = cold_access(package, asset["source_payload_id"])
            require(access["dependencies_in_decode_order"] == asset["dependencies_in_decode_order"], "Viewer payload dependency closure mismatch")
            require([f"{obj['id']}/{p}" for p in access["segment_paths"]] == asset["source_segment_paths"], "Viewer segment source paths mismatch")
        catalog_assets = [a for variant in catalog["variants"] for item in variant["objects"] if item["id"] == obj["id"]
                          for rep in item["representations"] for a in rep.get("viewer_assets", [])]
        require(sorted(catalog_assets, key=lambda a: a["path"]) == sorted(assets, key=lambda a: a["path"]), "Viewer catalog differs from decoded asset index")
        for file, label in (("index.html", "viewer"), ("gaussian_viewport.js", "viewport"), ("dependencies.lock.json", "dependencies")):
            require(sha256(within(self.root, file)) == viewer_index["implementation"][label], "Published viewer implementation hash mismatch")
        dependencies = read_json(self.root / "dependencies.lock.json")
        for dependency in dependencies["packages"]:
            for file in dependency["files"]:
                require(sha256(within(self.root, "vendor/" + file["path"])) == file["sha256"], "Published viewer dependency hash mismatch")
        browser = read_json(self.root / "viewer_validation.json")
        require(browser.get("schema") == "content-preparation.gaussian-viewer-validation.v1" and browser.get("status") == "passed",
                "Final validation requires a passed Gaussian web browser check")
        require(browser["catalog_sha256"] == sha256(self.root / "catalog.json"), "Browser evidence belongs to a different catalog")
        require(browser["conditions_checked"] == browser["conditions_expected"] == len(viewer_index["assets"]), "Browser did not inspect every decoded quality/frame")
        require(all(browser["navigation"].get(k) for k in ("rotation", "pan", "zoom")), "Required free camera navigation was not validated")
        require("WebGL 2" in browser.get("webgl", {}).get("version", "") and not browser.get("page_errors") and not browser.get("http_errors"),
                "Gaussian browser check must have WebGL2 and no JS/HTTP errors")
        expected_checks = {(a["object"], a["quality"], a["frame"], a["decoded_state_hash"], a["gaussian_count"]) for a in viewer_index["assets"]}
        checks = browser.get("checks", [])
        require({(r["object"], r["quality"], r["frame"], r["state_hash"], r["gaussian_count"]) for r in checks} == expected_checks
                and len(checks) == len(expected_checks) and all(r.get("nonblack_pixels", 0) > 0 and r.get("camera_preserved") for r in checks),
                "Browser evidence does not cover the actual decoded assets")
        require(browser.get("rapid_switch", {}).get("passed") is True or len(obj["qualities"]) == 1,
                "Browser rapid quality switching was not validated")
        deployed_files = ["index.html", "gaussian_viewport.js", "dependencies.lock.json"]
        deployed_files += ["vendor/" + f["path"] for dependency in dependencies["packages"] for f in dependency["files"]]
        require({r["path"] for r in browser.get("viewer_files", [])} == set(deployed_files), "Browser evidence lacks deployed implementation hashes")
        for file in browser["viewer_files"]:
            validate_asset(file, self.root)
        return {"manifest_valid": True, "upstream_unchanged": True,
                "mpd_valid": bool(mpd_result is not False), "viewer_schema": catalog.get("schema", catalog.get("schema_version")),
                "viewer_index_schema": viewer_index.get("schema", viewer_index.get("schema_version")),
                "viewer_assets_verified": len(assets), "viewer_numeric_equality": True,
                "web_visual_validation": "automated Gaussian WebGL checks passed; user acceptance remains a visual review",
                "proxy_preparation": manifest["object"].get("proxy_preparation"),
                "peak_cuda_allocated_bytes": validation["render_peak_cuda_allocated_bytes"],
                "gpu_peak_active_models": validation["gpu_guard"]["peak_active"]}

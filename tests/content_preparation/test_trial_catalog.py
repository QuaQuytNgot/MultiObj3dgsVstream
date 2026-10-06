"""Small CPU catalog/PNG tests; no real raw download or pipeline invocation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.content_preparation.manifest import build_manifest, write_manifest
from tools.content_preparation.packaging import package_object
from tools.summarize_8i_trial import summarize_trial, _quality_metadata, _preview_max_size


class TrialCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "trial"
        self.root.mkdir()
        self.inventory = Path(self.temp.name) / "inventory.json"
        self.inventory.write_text(json.dumps({"schema": "official-8i-download-v1", "object": "longdress",
            "source_url": "https://example.invalid/longdress.zip", "archive": {"frame_ids": list(range(1051, 1057)), "frame_count": 6, "bytes": 1000},
            "capture": {"fps": 30, "samples_duration_seconds": .2, "timestamp_span_seconds": 5 / 30},
            "frames": [{"frame": frame, "path": f"Ply/{frame}.ply", "sha256": "a" * 64} for frame in range(1051, 1057)],
            "complete_for_selection": True, "selection": {"all_archive_frames": True}}))
        frames = [{"frame": 1051, "timestamp": 0.0}, {"frame": 1052, "timestamp": 1 / 30}]
        for variant in ("f32_lossless", "f16"):
            variant_root = self.root / variant
            object_root = variant_root / "longdress"
            object_root.mkdir(parents=True)
            records, tasks, rows = [], {}, []
            for mode in ("independent", "progressive"):
                for qi, quality in enumerate(("Q0", "Q1")):
                    layer = quality if mode == "independent" else ("Base" if qi == 0 else "E1")
                    for frame in frames:
                        data = f"{mode},{quality},{frame['frame']},{variant}".encode()
                        encoded = object_root / "encoded" / mode / layer / f"{frame['frame']}.cpgs"
                        encoded.parent.mkdir(parents=True, exist_ok=True)
                        encoded.write_bytes(data)
                        records.append({"id": f"{mode}:{layer}:{frame['frame']}", "mode": mode, "quality": quality,
                            "layer": layer, "frame": frame["frame"], "path": encoded.relative_to(object_root).as_posix(),
                            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                            "decoded_state_hash": hashlib.sha256(data).hexdigest(), "codec": {"name": "cpu_test_fixture", "version": "1"}})
                        tasks[f"longdress/encode/{mode}/{layer}/{frame['frame']}"] = {"status": "complete",
                            "started_at": "2026-10-05T00:00:00Z", "completed_at": "2026-10-05T00:00:03Z"}
                        for view in (0, 1):
                            for scale in (0, 1):
                                sample_id = f"{frame['frame']}_a{view}_e0_s{scale}"
                                cache = object_root / "profiles" / "images" / mode / quality / f"{sample_id}.npz"
                                cache.parent.mkdir(parents=True, exist_ok=True)
                                image = np.zeros((16, 20, 3), dtype=np.float32)
                                image[..., qi] = .25 + .25 * view
                                np.savez(cache, rgb=image)
                                rows.append({"frame": frame["frame"], "timestamp": frame["timestamp"], "quality": quality,
                                    "mode": mode, "azimuth": view * 90, "elevation": 0, "scale": .75 + scale * .25,
                                    "view_bin": f"a{view}_e0", "scale_bin": f"s{scale}", "time_bin": str(frame["frame"]),
                                    "sample_id": sample_id, "image_path": cache.relative_to(object_root).as_posix()})
            index = package_object("longdress", frames, ["Q0", "Q1"], records,
                {"gof_frames": 2, "segment_frames": 2, "refresh_frames": {"Base": 2, "E1": 1}}, object_root)
            (object_root / "package.json").write_text(json.dumps(index))
            (object_root / "training.json").write_text(json.dumps({"backend": "existing", "lineage_verified": True, "commands": []}))
            profile = object_root / "profiles" / "profile.json"
            profile.write_text(json.dumps({"rows": rows, "aggregates": [], "reference": "highest decoded"}))
            proxy = object_root / "proxy" / "index.json"
            proxy.parent.mkdir()
            proxy.write_text('{"backend":"cpu_test_fixture"}')
            manifest = build_manifest("longdress", frames, ["Q0", "Q1"], index,
                proxy_path="proxy/index.json", quality_profile_path="profiles/profile.json", output_dir=object_root)
            write_manifest(object_root / "manifest.json", manifest)
            contents = (object_root / "manifest.json").read_bytes()
            (variant_root / "manifest.json").write_text(json.dumps({"schema_version": "content-preparation.server.v1",
                "objects": [{"id": "longdress", "manifest_path": "longdress/manifest.json",
                             "bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}]}))
            preparation = variant_root / ".preparation"
            preparation.mkdir()
            (preparation / "resolved_config.json").write_text(json.dumps({"renderer": {"backend": "upstream_cuda", "width": 20, "height": 16}}))
            (preparation / "journal.json").write_text(json.dumps({"tasks": tasks, "stages": {"longdress/encode": {
                "status": "complete", "started_at": "2026-10-05T01:00:00Z", "completed_at": "2026-10-05T01:00:00.2Z"}}}))

    def test_raw_encoded_and_encoder_durations_remain_distinct(self):
        catalog = summarize_trial(self.root, self.inventory, thumbnails=False)
        self.assertAlmostEqual(catalog["raw_source"]["source_raw_samples_duration_seconds"], .2)
        self.assertEqual(catalog["raw_source"]["source_archive_frame_count"], 6)
        self.assertEqual(catalog["raw_source"]["original_inventory_sha256"], hashlib.sha256(self.inventory.read_bytes()).hexdigest())
        obj = catalog["variants"][0]["objects"][0]
        self.assertAlmostEqual(obj["timeline"]["media_duration_seconds"], 2 / 30)
        self.assertAlmostEqual(obj["timeline"]["time_span_seconds"], 1 / 30)
        self.assertEqual(obj["encode_timings"]["recorded_successful_encode_task_wall_seconds"], 24)
        self.assertEqual(obj["encode_timings"]["latest_recorded_encode_stage_wall_seconds"], .2)
        self.assertEqual(obj["renderer_settings"]["width"], 20)
        self.assertEqual(obj["manifest_path"], "f32_lossless/longdress/manifest.json")
        self.assertEqual(obj["proxy_path"], "f32_lossless/longdress/proxy/index.json")
        self.assertEqual(obj["quality_profile_path"], "f32_lossless/longdress/profiles/profile.json")
        self.assertEqual(json.loads((self.root / "catalog.json").read_text()), catalog)
        self.assertTrue((self.root / "trial_report.md").is_file())
        self.assertTrue((self.root / "index.html").is_file())

    def test_all_sampled_views_become_real_pngs_and_paths_are_catalog_relative(self):
        catalog = summarize_trial(self.root, self.inventory)
        self.assertEqual(len(list((self.root / "previews").rglob("*.png"))), 64)
        for variant in catalog["variants"]:
            obj = variant["objects"][0]
            self.assertEqual(len(obj["thumbnails"]), 32)
            self.assertEqual(obj["sampled_frame_ids"], [1051, 1052])
            for preview in obj["thumbnails"]:
                path = self.root / preview["path"]
                self.assertEqual(path.stat().st_size, preview["bytes"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), preview["sha256"])
                with Image.open(path) as image:
                    self.assertEqual(image.size, (20, 16))
                    self.assertEqual(image.mode, "RGB")
                self.assertTrue((self.root / preview["source_render_cache"]["path"]).is_file())
            reps = {r["id"]: r for r in obj["representations"]}
            self.assertEqual(reps["progressive:E1"]["cumulative_quality_segment_bytes"],
                             reps["progressive:Base"]["stored_layer_segment_bytes"] + reps["progressive:E1"]["stored_layer_segment_bytes"])
            for rep in obj["representations"]:
                for segment in rep["segments"]:
                    self.assertTrue(segment["path"].startswith(variant["id"] + "/longdress/media/"))
                for access in rep["cold_access"]:
                    self.assertTrue(all((self.root / path).is_file() for path in access["segment_paths"]))
        self.assertEqual(summarize_trial(self.root, self.inventory), catalog)

    def test_optional_limit_and_partial_inventory_status(self):
        raw = json.loads(self.inventory.read_text())
        raw["complete_for_selection"] = False
        self.inventory.write_text(json.dumps(raw))
        catalog = summarize_trial(self.root, self.inventory, max_thumbnails_per_quality=1)
        self.assertFalse(catalog["raw_source"]["recorded_complete_for_selection"])
        self.assertEqual(len(catalog["variants"][0]["objects"][0]["thumbnails"]), 4)
        self.assertIn("False", (self.root / "trial_report.md").read_text())

    def test_native_preview_preserves_pixels_and_optional_limit_never_upscales(self):
        object_root = self.root/"f32_lossless"/"longdress"
        profile = json.loads((object_root/"profiles/profile.json").read_text())
        cache = object_root/profile["rows"][0]["image_path"]
        yy, xx = np.indices((768, 1024))
        pixels = np.zeros((768, 1024, 3), dtype=np.uint8)
        pixels[..., 0] = (xx % 2)*255
        pixels[..., 1] = (yy % 2)*255
        np.savez(cache, rgb=pixels)
        catalog = summarize_trial(self.root, self.inventory)
        previews = catalog["variants"][0]["objects"][0]["thumbnails"]
        preview = next(row for row in previews if row["source_render_cache"]["path"].endswith(profile["rows"][0]["image_path"]))
        self.assertEqual(preview["preview_size"], [1024, 768])
        self.assertFalse(preview["was_downsampled"])
        self.assertEqual(catalog["preview_settings"]["size_mode"], "native")
        with Image.open(self.root/preview["path"]) as image:
            np.testing.assert_array_equal(np.asarray(image), pixels)
        catalog = summarize_trial(self.root, self.inventory, preview_max_size=256)
        previews = catalog["variants"][0]["objects"][0]["thumbnails"]
        resized = next(row for row in previews if row["source_render_cache"]["path"].endswith(profile["rows"][0]["image_path"]))
        self.assertEqual(resized["preview_size"], [256, 192])
        self.assertTrue(resized["was_downsampled"])
        self.assertEqual(next(row for row in previews if row is not resized)["preview_size"], [20, 16])
        self.assertEqual(_preview_max_size("native"), None)
        self.assertEqual(_preview_max_size("1024"), 1024)
        with self.assertRaises(ValueError):
            summarize_trial(self.root, self.inventory, preview_max_size=0)

    def test_four_quality_labels_use_config_and_verified_training_image_dimensions(self):
        variant_root = self.root/"f32_lossless"
        object_root = variant_root/"longdress"
        scales = [8, 4, 2, 1]
        qualities = [f"Q{i}" for i in range(4)]
        configuration = json.loads((variant_root/".preparation/resolved_config.json").read_text())
        configuration["objects"] = [{"id": "longdress", "qualities": [{"id": quality, "resolution_scale": scale} for quality, scale in zip(qualities, scales)]}]
        (variant_root/".preparation/resolved_config.json").write_text(json.dumps(configuration))
        commands = []
        for quality, scale in zip(qualities, scales):
            source = Path(self.temp.name)/f"source_{quality}"
            source.mkdir()
            Image.new("RGB", (1024//scale, 1024//scale)).save(source/"training.png")
            (source/"transforms_train.json").write_text('{"frames":[{"file_path":"./training"}]}')
            commands.append({"frame": 1051, "level": quality, "argv": ["python", "train.py", "-s", str(source), "--iterations", "6000"]})
            commands.append({"frame": 1052, "level": quality, "argv": ["python", "train.py", "-s", str(source), "--iterations", "1500"]})
        (object_root/"training.json").write_text(json.dumps({"backend": "existing", "commands": commands}))
        metadata = _quality_metadata(variant_root, "longdress", object_root, qualities)
        for quality, scale in zip(qualities, scales):
            self.assertEqual(metadata[quality]["label"], f"{quality} · res{scale} · {1024//scale}×{1024//scale}")
            self.assertEqual(metadata[quality]["training_image_size"], [1024//scale]*2)
            self.assertEqual(metadata[quality]["resolution_scale"], scale)
            self.assertEqual(len(metadata[quality]["training_image_provenance"]["source_image_sha256"]), 64)
            self.assertEqual(metadata[quality]["training_iterations"]["first_frame_iterations"], 6000)
            self.assertEqual(metadata[quality]["training_iterations"]["later_frame_iteration_values"], [1500])
        catalog = summarize_trial(self.root, self.inventory, thumbnails=False)
        for rep in catalog["variants"][0]["objects"][0]["representations"]:
            self.assertEqual(rep["quality_metadata"], metadata[rep["quality"]])

    def test_training_camera_resolution_overrides_are_reported_and_unknowns_are_not_invented(self):
        variant_root = self.root/"f32_lossless"
        object_root = variant_root/"longdress"
        missing = _quality_metadata(variant_root, "longdress", object_root, ["Q3"])["Q3"]
        self.assertIsNone(missing["training_image_size"])
        self.assertIsNone(missing["resolution_scale"])
        source = Path(self.temp.name)/"camera_source"
        source.mkdir()
        Image.new("RGB", (1024, 768)).save(source/"input.png")
        (source/"transforms_train.json").write_text('{"frames":[{"file_path":"input.png"}]}')
        (object_root/"training.json").write_text(json.dumps({"commands": [{"frame": 1051, "quality": "Q0", "argv": ["python", "train.py", "--source_path="+str(source), "--resolution", "2"]}]}))
        actual = _quality_metadata(variant_root, "longdress", object_root, ["Q0"])["Q0"]
        self.assertEqual(actual["training_image_size"], [512, 384])
        self.assertEqual(actual["training_image_provenance"]["source_image_size"], [1024, 768])

    def test_missing_variant_and_bad_inventory_have_clear_errors(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing variant"):
            summarize_trial(self.root, self.inventory, variants=("absent",))
        self.assertFalse((self.root / "catalog.json").exists())
        raw = json.loads(self.inventory.read_text())
        raw["capture"]["samples_duration_seconds"] = 5
        self.inventory.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            summarize_trial(self.root, self.inventory)
        raw["capture"]["samples_duration_seconds"] = .2
        self.inventory.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            summarize_trial(self.root, self.inventory, variants=("../escape",))


@unittest.skipUnless(importlib.util.find_spec("playwright"), "Optional browser verification requires installed Playwright; catalog has no browser dependency")
class TrialViewerTests(unittest.TestCase):
    def test_native_pixels_full_png_link_and_all_existing_condition_controls(self):
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        import threading
        from playwright.sync_api import sync_playwright
        fixture = TrialCatalogTests("test_raw_encoded_and_encoder_durations_remain_distinct")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for variant in ("f32_lossless", "f16"):
            variant_root = fixture.root/variant
            config_path = variant_root/".preparation/resolved_config.json"
            config = json.loads(config_path.read_text())
            config["objects"] = [{"id": "longdress", "qualities": [{"id": "Q0", "resolution_scale": 8}, {"id": "Q1", "resolution_scale": 4}]}]
            config_path.write_text(json.dumps(config))
        object_root = fixture.root/"f32_lossless/longdress"
        profile_path = object_root/"profiles/profile.json"
        profile = json.loads(profile_path.read_text())
        row = profile["rows"][0]
        row.update({"metrics": {"mse": .1, "psnr": 10., "lpips": .02}, "reference_quality": "Q1"})
        profile_path.write_text(json.dumps(profile))
        np.savez(object_root/row["image_path"], rgb=np.full((768, 1024, 3), .5, dtype=np.float32))
        summarize_trial(fixture.root, fixture.inventory)
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(fixture.root)))
        except PermissionError:
            self.skipTest("Sandbox forbids localhost sockets; run optional browser verification outside that sandbox")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 1000}, device_scale_factor=1)
                errors = []
                page.on("pageerror", lambda error: errors.append(error))
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                page.wait_for_function("document.getElementById('preview').naturalWidth === 1024")
                self.assertIn("res8", page.locator("#quality option:checked").inner_text())
                initial = page.locator("#preview").evaluate("img => ({natural: img.naturalWidth, shown: img.getBoundingClientRect().width})")
                self.assertLessEqual(initial["shown"], initial["natural"])
                page.locator("#nativePixels").check()
                native = page.locator("#preview").evaluate("img => img.getBoundingClientRect().width")
                self.assertEqual(native, 1024)
                self.assertTrue(page.locator("#imageContainer").evaluate("el => el.scrollWidth > el.clientWidth"))
                self.assertEqual(page.locator("#originalPNG").get_attribute("href"), page.locator("#preview").get_attribute("src"))
                self.assertIn("MSE", page.locator("#viewMetrics").inner_text())
                page.select_option("#quality", "Q1")
                page.wait_for_function("document.getElementById('preview').naturalWidth === 20")
                page.locator("#nativePixels").uncheck()
                self.assertEqual(page.locator("#preview").evaluate("img => img.getBoundingClientRect().width"), 20)
                for mode in ("independent", "progressive"):
                    page.select_option("#mode", mode)
                    for view in ("a0_e0", "a1_e0"):
                        page.select_option("#view", view)
                        for scale in ("0.75", "1"):
                            page.select_option("#scale", scale)
                            page.wait_for_function("document.getElementById('preview').complete && document.getElementById('preview').naturalWidth > 0")
                self.assertFalse(errors)
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Validate a served local trial catalog in optional, CPU-only Chromium.

This checks the actual viewer and every catalog thumbnail. It never imports the
training/renderer modules or modifies media. Install Playwright separately;
``--help`` and importing this module do not require it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import urlopen


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _local_url(value):
    parsed = urlparse(value)
    _require(parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
             and not parsed.username and not parsed.password,
             "--url must be an HTTP URL on localhost, 127.0.0.1 or ::1")
    return value


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    os.replace(temporary, path)


def _metric_text(actual, metrics, reference):
    for label, key in (("MSE", "mse"), ("PSNR", "psnr"), ("LPIPS", "lpips")):
        match = re.search(r"\b" + label + r"\s+(∞|[+\-\d.eE]+|chưa đo)", actual)
        _require(match is not None, f"Missing {label} in per-view metrics: {actual}")
        value, expected = match.group(1), metrics.get(key)
        if expected == "inf":
            _require(value == "∞", f"Expected infinite {label}, got {value}")
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            _require(math.isclose(float(value), expected, rel_tol=6e-6, abs_tol=1e-12),
                     f"Wrong per-view {label}: {value}, expected {expected}")
        else:
            _require(value == "chưa đo", f"Unexpected measured {label}: {value}")
    if reference:
        _require(f"reference {reference}" in actual, f"Wrong reference quality: {actual}")


def _image_geometry(page):
    return page.locator("#preview").evaluate("""img => {
        const box = img.getBoundingClientRect(), container = document.getElementById('imageContainer');
        const style = getComputedStyle(container);
        return {natural: [img.naturalWidth, img.naturalHeight], shown: [box.width, box.height],
            client: [container.clientWidth, container.clientHeight],
            scroll: [container.scrollWidth, container.scrollHeight],
            padding: [parseFloat(style.paddingLeft) + parseFloat(style.paddingRight),
                parseFloat(style.paddingTop) + parseFloat(style.paddingBottom)],
            overflow: [style.overflowX, style.overflowY]};
    }""")


def _gaussian_expected_keys(catalog):
    return {
        (obj["id"], rep["quality"], int(asset["frame"]), asset["decoded_state_hash"],
         int(asset["gaussian_count"]))
        for variant in catalog["variants"]
        for obj in variant["objects"]
        for rep in obj["representations"]
        for asset in rep.get("viewer_assets", [])
    }


def reusable_gaussian_evidence(previous, catalog_sha256, viewer_files, catalog):
    """Return whether a partial browser run already proves every costly check.

    The Gaussian renderer audit reads pixels repeatedly; on a CPU WebGL backend,
    one redundant readback after camera reset can time out after every asset and
    interaction was already verified. Reuse is safe only when the complete
    catalog, deployed viewer files, and every required browser assertion match.
    """
    if not isinstance(previous, dict) or previous.get("schema") != "content-preparation.gaussian-viewer-validation.v1":
        return False
    if previous.get("catalog_sha256") != catalog_sha256:
        return False
    if sorted(previous.get("viewer_files", []), key=lambda row: row.get("path", "")) != sorted(viewer_files, key=lambda row: row.get("path", "")):
        return False
    expected = _gaussian_expected_keys(catalog)
    checks = previous.get("checks", [])
    observed = {(row.get("object"), row.get("quality"), int(row.get("frame", -1)),
                 row.get("state_hash"), int(row.get("gaussian_count", -1))) for row in checks}
    if (not expected or previous.get("conditions_expected") != len(expected)
            or previous.get("conditions_checked") != len(expected) or len(checks) != len(expected)
            or observed != expected
            or not all(row.get("nonblack_pixels", 0) > 0 and row.get("camera_preserved") for row in checks)):
        return False
    if not all(previous.get("navigation", {}).get(key) for key in ("rotation", "pan", "zoom")):
        return False
    if "WebGL 2" not in previous.get("webgl", {}).get("version", ""):
        return False
    if previous.get("page_errors") or previous.get("http_errors"):
        return False
    rapid = previous.get("rapid_switch", {})
    if not rapid.get("passed") or not rapid.get("single_mesh"):
        return False
    if rapid.get("final_quality") not in {key[1] for key in expected}:
        return False
    return True


def _served_gaussian_snapshot(url, timeout):
    """Fetch the small, hashed browser deployment metadata from a local server."""
    parsed = urlparse(_local_url(url))
    catalog_path = parse_qs(parsed.query).get("catalog", ["./catalog.json"])[0]
    catalog_url = _local_url(urljoin(url, catalog_path))

    def get(address):
        _local_url(address)
        with urlopen(address, timeout=timeout) as response:
            return response.read()

    catalog_bytes = get(catalog_url)
    catalog = json.loads(catalog_bytes)
    if catalog.get("schema_version") != "content-preparation.8i-trial-catalog.v2":
        raise ValueError("Gaussian viewport requires catalog v2")
    lock = json.loads(get(urljoin(catalog_url, "dependencies.lock.json")))
    files = ["index.html", "gaussian_viewport.js", "dependencies.lock.json"]
    expected_dependencies = {}
    for package in lock["packages"]:
        for entry in package["files"]:
            relative = "vendor/" + entry["path"]
            _require(not Path(relative).is_absolute() and ".." not in Path(relative).parts and "\\" not in relative,
                     "Unsafe deployed viewer dependency path")
            files.append(relative)
            expected_dependencies[relative] = entry["sha256"]
    viewer_files = []
    for relative in files:
        data = get(urljoin(catalog_url, relative))
        digest = hashlib.sha256(data).hexdigest()
        if relative in expected_dependencies:
            _require(digest == expected_dependencies[relative], f"Deployed dependency checksum differs: {relative}")
        viewer_files.append({"path": relative, "bytes": len(data), "sha256": digest})
    return catalog, hashlib.sha256(catalog_bytes).hexdigest(), viewer_files


def resume_gaussian_validation(url, output, timeout_ms):
    """Complete a timed-out browser report when all expensive assertions passed."""
    if not output.is_file():
        raise ValueError("--resume needs an existing Gaussian viewer validation report")
    previous = json.loads(output.read_text(encoding="utf8"))
    catalog, catalog_digest, files = _served_gaussian_snapshot(url, max(timeout_ms / 1000, 30))
    if not reusable_gaussian_evidence(previous, catalog_digest, files, catalog):
        raise ValueError("Existing browser evidence is incomplete or no longer matches the served catalog/viewer")
    resumed = dict(previous)
    resumed.update(status="passed", completed_at=datetime.now(timezone.utc).isoformat(), url=url,
                   catalog_url=urljoin(url, parse_qs(urlparse(url).query).get("catalog", ["./catalog.json"])[0]),
                   catalog_sha256=catalog_digest, viewer_files=files, error=None,
                   resume={"reused_complete_browser_checks": len(previous["checks"]),
                           "revalidated_catalog_and_viewer_files": True,
                           "prior_failure": previous.get("error")})
    _write_report(output, resumed)
    return resumed


def _select(page, variant, representation, thumbnail, catalog_url, object_id=None):
    view = thumbnail.get("view_bin") or f"{thumbnail['azimuth']}/{thumbnail['elevation']}"
    values = (variant["id"], representation["mode"], representation["quality"],
              thumbnail["frame"], view, thumbnail["scale"])
    # Number.toString semantics matter for scale 1.0, which HTML selects as "1".
    for name, value in zip(("variant", "mode", "quality", "frame", "view", "scale"), values):
        if name == "mode" and object_id is not None:
            page.select_option("#object", object_id)
        selected = page.evaluate("value => String(value)", value)
        page.select_option("#" + name, selected)
    expected_url = urljoin(catalog_url, thumbnail["path"])
    _local_url(expected_url)
    page.wait_for_function("""url => {
        const img = document.getElementById('preview');
        return img.src === url && img.currentSrc === url && img.complete && img.naturalWidth > 0;
    }""", arg=expected_url)
    _require(page.locator("#originalPNG").get_attribute("href") == expected_url,
             f"Original PNG link differs from selected thumbnail: {expected_url}")
    _require(page.locator("#originalPNG").is_visible(), "Original PNG link is hidden")
    return expected_url


def validate(url, output, timeout_ms=30000, display="png", require_webgl=False, resume=False):
    if display == "gaussians":
        return validate_gaussians(url, output, timeout_ms, require_webgl, resume)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError("Optional dependency missing. Install with: python -m pip install playwright; "
                           "then: python -m playwright install chromium. Use the same Python environment.") from error
    report = {"schema": "content-preparation.trial-viewer-validation.v1", "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat(), "url": url,
              "browser": "headless Chromium", "gpu_enabled": False, "device_scale_factor": 1,
              "conditions_checked": 0, "checks": [], "page_errors": [], "http_errors": []}
    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
            except Exception as error:
                raise RuntimeError("Cannot launch optional Chromium. Run python -m playwright install chromium "
                                   f"in this Python environment; check PLAYWRIGHT_BROWSERS_PATH. Details: {error}") from error
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 1050}, device_scale_factor=1)
                page.set_default_timeout(timeout_ms)
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on("response", lambda response: report["http_errors"].append(
                    {"url": response.url, "status": response.status})
                    if response.status >= 400 and not urlparse(response.url).path.endswith("/favicon.ico") else None)
                response = page.goto(url, wait_until="domcontentloaded")
                _require(response is not None and response.ok, f"Viewer HTTP request failed: {url}")
                page.wait_for_function("!document.getElementById('content').classList.contains('hidden')")
                catalog_url = page.evaluate("""() => new URL(new URLSearchParams(location.search).get('catalog')
                    || './catalog.json', location.href).href""")
                _local_url(catalog_url)
                catalog_response = page.request.get(catalog_url)
                _require(catalog_response.ok, f"Catalog HTTP {catalog_response.status}")
                catalog_bytes = catalog_response.body()
                catalog = json.loads(catalog_bytes)
                _require(catalog.get("schema_version") in {"content-preparation.8i-trial-catalog.v1", "content-preparation.8i-trial-catalog.v2"},
                         "Unsupported trial catalog schema")
                if catalog.get("schema_version").endswith(".v2"):
                    page.select_option("#display", "png")
                report.update({"catalog_url": catalog_url, "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
                               "variants": [variant["id"] for variant in catalog["variants"]]})
                conditions = []
                for variant in catalog["variants"]:
                    for obj in variant["objects"]:
                        for representation in obj["representations"]:
                            conditions.extend((variant, obj, representation, row) for row in representation.get("thumbnails", []))
                _require(conditions, "Catalog has no thumbnails to validate")
                report["conditions_expected"] = len(conditions)
                for variant, obj, representation, thumbnail in conditions:
                    png_url = _select(page, variant, representation, thumbnail, catalog_url, obj["id"])
                    metadata = representation.get("quality_metadata", {})
                    _require(page.locator("#quality option:checked").inner_text() == metadata.get("label", representation["quality"]),
                             "Quality selector label differs from catalog training metadata")
                    training_text = page.locator("#trainingResolution").inner_text()
                    for text in (metadata.get("resolution_label"), "×".join(map(str, metadata.get("training_image_size") or []))):
                        if text:
                            _require(text in training_text, f"Training resolution label missing: {text}")
                    metrics = thumbnail.get("metrics")
                    _require(metrics is not None, "Selected thumbnail has no per-view metrics")
                    _metric_text(page.locator("#viewMetrics").inner_text(), metrics, thumbnail.get("reference_quality"))
                    page.locator("#nativePixels").uncheck()
                    fit = _image_geometry(page)
                    _require(fit["natural"] == thumbnail["preview_size"], f"PNG dimensions differ: {png_url}")
                    _require(all(shown <= natural + .1 for shown, natural in zip(fit["shown"], fit["natural"])),
                             "Fit mode upscales the PNG")
                    page.locator("#nativePixels").check()
                    native = _image_geometry(page)
                    _require(all(abs(shown - natural) <= .1 for shown, natural in zip(native["shown"], native["natural"])),
                             "1:1 mode does not display native PNG pixels")
                    _require(all(value in {"auto", "scroll"} for value in native["overflow"]),
                             "1:1 preview lacks a scroll container")
                    for natural, client, padding, scroll in zip(native["natural"], native["client"], native["padding"], native["scroll"]):
                        if natural + padding > client + 1:
                            _require(scroll > client, "Oversized 1:1 preview does not scroll")
                    report["checks"].append({"variant": variant["id"], "mode": representation["mode"],
                        "quality": representation["quality"], "frame": thumbnail["frame"],
                        "view_bin": thumbnail.get("view_bin"), "azimuth": thumbnail["azimuth"],
                        "elevation": thumbnail["elevation"], "scale": thumbnail["scale"],
                        "png_url": png_url, "native_size": native["natural"], "fit_size": fit["shown"],
                        "training_resolution": training_text, "view_metrics": page.locator("#viewMetrics").inner_text()})
                    report["conditions_checked"] += 1
                output.parent.mkdir(parents=True, exist_ok=True)
                screenshots = {}
                for kind, width, height in (("desktop", 1280, 1050), ("mobile", 390, 844)):
                    page.set_viewport_size({"width": width, "height": height})
                    for one_to_one in (False, True):
                        page.locator("#nativePixels").set_checked(one_to_one)
                        geometry = page.evaluate("() => ({scroll: document.documentElement.scrollWidth, width: innerWidth})")
                        _require(geometry["scroll"] <= geometry["width"] + 1,
                                 f"{kind} page overflows horizontally (1:1={one_to_one})")
                        if kind == "mobile" and one_to_one:
                            mobile_native = _image_geometry(page)
                            _require(all(abs(shown - natural) <= .1 for shown, natural in zip(mobile_native["shown"], mobile_native["natural"])),
                                     "Mobile 1:1 preview changes native PNG dimensions")
                    page.locator("#nativePixels").uncheck()
                    screenshot = output.with_name(output.stem + f"_{kind}.png")
                    page.screenshot(path=str(screenshot), full_page=True)
                    screenshots[kind] = screenshot.name
                report["screenshots"] = screenshots
                _require(not report["page_errors"], f"Viewer JavaScript errors: {report['page_errors']}")
                _require(not report["http_errors"], f"Viewer HTTP errors: {report['http_errors']}")
                report["status"] = "passed"
            finally:
                browser.close()
    except Exception as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(output, report)
    return report


def validate_gaussians(url, output, timeout_ms=30000, require_webgl=True, resume=False):
    """Exercise the actual locally served decoded Gaussian viewport in WebGL2.

    SwiftShader is used in headless validation, so CUDA training and a hardware
    display are unnecessary. This is a visual renderer check, not a metric pass.
    """
    if resume:
        return resume_gaussian_validation(url, output, timeout_ms)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError("Install optional Playwright and its Chromium browser") from error
    report = {"schema": "content-preparation.gaussian-viewer-validation.v1", "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat(), "url": url,
              "browser": "headless Chromium / ANGLE SwiftShader", "hardware_gpu_required": False,
              "conditions_checked": 0, "checks": [], "page_errors": [], "http_errors": []}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage",
                "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 1050}, device_scale_factor=1)
                page.set_default_timeout(timeout_ms)
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on("response", lambda response: report["http_errors"].append({"url": response.url, "status": response.status})
                    if response.status >= 400 and not urlparse(response.url).path.endswith("/favicon.ico") else None)
                response = page.goto(url, wait_until="domcontentloaded")
                _require(response is not None and response.ok, "Viewer HTTP request failed")
                page.wait_for_function("!document.getElementById('content').classList.contains('hidden')")
                catalog_url = page.evaluate("() => new URL(new URLSearchParams(location.search).get('catalog') || './catalog.json', location.href).href")
                _local_url(catalog_url)
                response = page.request.get(catalog_url)
                _require(response.ok, "Cannot read viewer catalog")
                catalog_bytes = response.body()
                catalog = json.loads(catalog_bytes)
                _require(catalog.get("schema_version") == "content-preparation.8i-trial-catalog.v2", "Gaussian viewport requires catalog v2")
                report.update(catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(), catalog_url=catalog_url)
                # Bind evidence to the deployed implementation, not only the
                # asset catalog. A UI update must require a new browser check.
                lock_response = page.request.get(urljoin(catalog_url, "dependencies.lock.json"))
                _require(lock_response.ok, "Cannot read deployed dependency lock")
                dependency_lock = json.loads(lock_response.body())
                viewer_files = ["index.html", "gaussian_viewport.js", "dependencies.lock.json"]
                expected_dependencies = {}
                for package in dependency_lock["packages"]:
                    for file in package["files"]:
                        relative = "vendor/" + file["path"]
                        _require(not Path(relative).is_absolute() and ".." not in Path(relative).parts and "\\" not in relative,
                                 "Unsafe deployed viewer dependency path")
                        viewer_files.append(relative)
                        expected_dependencies[relative] = file["sha256"]
                report["viewer_files"] = []
                for relative in viewer_files:
                    served = page.request.get(urljoin(catalog_url, relative))
                    _require(served.ok, f"Cannot read deployed viewer file {relative}")
                    data = served.body()
                    digest = hashlib.sha256(data).hexdigest()
                    if relative in expected_dependencies:
                        _require(digest == expected_dependencies[relative], f"Deployed dependency checksum differs: {relative}")
                    report["viewer_files"].append({"path": relative, "bytes": len(data), "sha256": digest})
                conditions = [(variant, obj, rep, row) for variant in catalog["variants"] for obj in variant["objects"]
                              for rep in obj["representations"] for row in rep.get("viewer_assets", [])]
                _require(conditions, "Catalog contains no decoded Gaussian assets")
                report["conditions_expected"] = len(conditions)
                page.select_option("#display", "gaussians")

                def await_loaded(asset):
                    page.wait_for_function("""row => {
                        const v = window.contentGaussianViewport, loaded = v?.snapshot().loaded;
                        return loaded && loaded.object === row.object && loaded.quality === row.quality && loaded.frame === row.frame
                            && loaded.decoded_state_hash === row.decoded_state_hash && v.snapshot().activeLoads === 0
                            && v.snapshot().meshCount === 1 && !v.snapshot().pending
                            && document.getElementById('loadedRepresentation').dataset.stateHash === row.decoded_state_hash;
                    }""", arg=asset)
                    return page.wait_for_function("window.contentGaussianViewport.nonblackPixels() || false").json_value()

                def select(variant, obj, rep, asset):
                    for name, value in (("variant", variant["id"]), ("object", obj["id"]), ("mode", rep["mode"]),
                                        ("quality", rep["quality"]), ("frame", str(asset["frame"]))):
                        page.select_option("#" + name, value)
                    return await_loaded(asset)

                for variant, obj, rep, asset in conditions:
                    previous = page.evaluate("window.contentGaussianViewport?.snapshot() || null")
                    nonblack_pixels = select(variant, obj, rep, asset)
                    current = page.evaluate("window.contentGaussianViewport.snapshot()")
                    _require(current["extSplats"] and not current["lod"], "Unexpected Gaussian packing/LoD policy")
                    if previous and previous["loaded"] and previous["loaded"]["object"] == obj["id"]:
                        _require(current["cameraPosition"] == previous["cameraPosition"] and current["target"] == previous["target"],
                                 "Quality/frame switch changed the free camera")
                    label = page.locator("#loadedRepresentation").inner_text()
                    for value in (obj["id"], rep["quality"], str(asset["frame"]), asset["decoded_state_hash"]):
                        _require(value in label, f"Loaded representation label lacks {value}")
                    ply_url = urljoin(catalog_url, asset["path"])
                    _require(page.locator("#decodedPLY").get_attribute("href") == ply_url, "PLY download link differs from selected state")
                    asset_response = page.request.get(ply_url)
                    _require(asset_response.ok and hashlib.sha256(asset_response.body()).hexdigest() == asset["sha256"], "Served PLY checksum differs")
                    _require(current["loaded"]["gaussian_count"] == asset["gaussian_count"], "Gaussian count differs")
                    report["checks"].append({"object": obj["id"], "quality": rep["quality"], "frame": asset["frame"],
                        "state_hash": asset["decoded_state_hash"], "gaussian_count": asset["gaussian_count"],
                        "nonblack_pixels": nonblack_pixels, "camera_preserved": True})
                    report["conditions_checked"] += 1
                context = page.evaluate("""() => {
                    const gl = window.contentGaussianViewport.renderer.getContext();
                    return {version: gl.getParameter(gl.VERSION), renderer: gl.getParameter(gl.RENDERER)};
                }""")
                _require("WebGL 2" in context["version"], "Decoded viewport requires WebGL2")
                report["webgl"] = context
                box = page.locator("#gaussianCanvas").bounding_box()
                x, y = box["x"] + box["width"] * .5, box["y"] + box["height"] * .5
                before = page.evaluate("window.contentGaussianViewport.snapshot()")
                camera_baseline = {"cameraPosition": before["cameraPosition"], "target": before["target"]}
                page.mouse.move(x, y); page.mouse.down(); page.mouse.move(x + 45, y + 15, steps=8); page.mouse.up()
                rotated = page.evaluate("window.contentGaussianViewport.snapshot()")
                _require(rotated["cameraPosition"] != before["cameraPosition"], "Free camera rotation did not change camera")
                page.mouse.move(x, y); page.mouse.down(button="right"); page.mouse.move(x + 25, y, steps=6); page.mouse.up(button="right")
                panned = page.evaluate("window.contentGaussianViewport.snapshot()")
                _require(panned["target"] != rotated["target"], "Free camera pan did not change target")
                page.mouse.move(x, y); page.mouse.wheel(0, -120)
                page.wait_for_function("old => JSON.stringify(window.contentGaussianViewport.snapshot().cameraPosition) !== JSON.stringify(old)", arg=panned["cameraPosition"])
                report["navigation"] = {"rotation": True, "pan": True, "zoom": True}
                # Stress a burst of quality changes without awaiting intermediate loads.
                variant, obj, rep, asset = conditions[-1]
                select(variant, obj, rep, asset)
                candidates = [condition for condition in conditions if condition[1]["id"] == obj["id"] and condition[3]["frame"] == asset["frame"]]
                if len(candidates) > 1:
                    sequence = [candidates[0], candidates[-1], candidates[0], candidates[-1]]
                    page.evaluate("""qualities => {
                        const element = document.getElementById('quality');
                        for (const quality of qualities) { element.value = quality; element.dispatchEvent(new Event('change')); }
                    }""", [condition[2]["quality"] for condition in sequence])
                    await_loaded(sequence[-1][3])
                    _require(page.evaluate("window.contentGaussianViewport.snapshot().activeLoads") == 0, "Gaussian loads remained active")
                    report["rapid_switch"] = {"passed": True, "final_quality": sequence[-1][2]["quality"], "single_mesh": True}
                page.locator("#resetCamera").click()
                page.wait_for_function("""expected => {
                    const state = window.contentGaussianViewport.snapshot();
                    return JSON.stringify(state.cameraPosition) === JSON.stringify(expected.cameraPosition)
                        && JSON.stringify(state.target) === JSON.stringify(expected.target);
                }""", arg=camera_baseline)
                screenshot = output.with_name(output.stem + "_gaussians.png")
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot), full_page=True)
                report["screenshot"] = screenshot.name
                _require(not report["page_errors"], f"Viewer JavaScript errors: {report['page_errors']}")
                _require(not report["http_errors"], f"Viewer HTTP errors: {report['http_errors']}")
                report["status"] = "passed"
            finally:
                browser.close()
    except Exception as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, type=_local_url)
    parser.add_argument("--output", required=True, type=Path, help="Local JSON report; screenshots are saved beside it")
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--display", choices=("png", "gaussians"), default="png")
    parser.add_argument("--require-webgl", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse complete browser evidence when only final report writing failed")
    args = parser.parse_args(argv)
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be positive")
    try:
        report = validate(args.url, args.output.resolve(), args.timeout_ms, args.display, args.require_webgl, args.resume)
    except Exception as error:
        print(f"Viewer validation failed: {error}", file=sys.stderr)
        return 1
    print(f"Passed {report['conditions_checked']} catalog conditions; report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

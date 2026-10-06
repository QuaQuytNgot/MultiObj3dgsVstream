"""Exercise CUDA training, layer inheritance, dynamic updates, render and metrics.

Run from the repository with: python scripts/smoke_test.py
Uses generated data in a fresh temporary directory; keeps logs for inspection.
"""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.content_preparation.paths import UPSTREAM_ROOT, configure_upstream_imports
from tools.content_preparation.compatibility import render_2d_image, rescale_image, store_point_cloud


def test_preprocessing():
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    import open3d as o3d
    sys.path.insert(0, str(ROOT))
    configure_upstream_imports()
    work = Path(tempfile.mkdtemp(prefix="dynamic-lapis-open3d-"))
    rng = np.random.default_rng(3)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(rng.uniform(-150, 150, (500, 3)))
    cloud.colors = o3d.utility.Vector3dVector(rng.uniform(0.1, 1, (500, 3)))
    assert o3d.io.write_point_cloud(str(work / "input.ply"), cloud)
    pose = np.eye(4)
    pose[2, 3] = 3
    poses = work / "transforms.json"
    poses.write_text(json.dumps({"camera_angle_x": 0.8,
                                "frames": [{"transform_matrix": pose.tolist()}]}))
    render_2d_image(str(work / "input.ply"), str(work / "render"), str(poses),
                    pt_size=3, width=64, height=64)
    image = Image.open(work / "render/r_0.png")
    assert image.size == (64, 64) and image.mode == "RGBA"
    pixels = np.asarray(image)
    assert pixels[:, :, :3].max() > 0
    assert pixels[:, :, 3].min() == 0 and pixels[:, :, 3].max() == 255
    assert rescale_image(work / "render/r_0.png", 2).size == (32, 32)
    print(f"PASS Open3D render, RGBA alpha and downsampling: {work}", flush=True)


def run(label, command, work):
    log = work / f"{label}.log"
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", WANDB_MODE="offline",
               PYTHONPATH=os.pathsep.join([str(ROOT), str(UPSTREAM_ROOT), os.environ.get("PYTHONPATH", "")]))
    with log.open("w") as output:
        result = subprocess.run(command, cwd=UPSTREAM_ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}):\n{log.read_text()[-8000:]}")
    print(f"PASS {label}: {log}", flush=True)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def main():
    assert torch.cuda.is_available(), "A working NVIDIA GPU is required."
    work = Path(tempfile.mkdtemp(prefix="dynamic-lapis-hoang-"))
    print(f"Artifacts: {work}", flush=True)
    print(f"Python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}", flush=True)
    run("prepare_dataset", [sys.executable, str(Path(__file__).resolve()),
                             "--preprocess-only"], work)
    for entry in ["train.py", "train_full_pipeline.py", "render.py", "metrics.py",
                  "dataset_prepare.py"]:
        run(f"help_{Path(entry).stem}", [sys.executable, str(UPSTREAM_ROOT / entry), "--help"], work)
    source = work / "source"
    source.mkdir()
    rng = np.random.default_rng(42)
    sys.path.insert(0, str(ROOT))
    storePly = store_point_cloud
    storePly(str(source / "points3d.ply"), rng.uniform(-0.5, 0.5, (128, 3)),
             rng.uniform(60, 230, (128, 3)))
    yy, xx = np.mgrid[:64, :64]
    image = np.zeros((64, 64, 4), dtype=np.uint8)
    mask = (xx - 32) ** 2 + (yy - 32) ** 2 < 18 ** 2
    image[mask] = [190, 100, 60, 255]
    frames = []
    for index, x in enumerate([-0.3, 0.3, 0.0]):
        name = f"view_{index}"
        Image.fromarray(image).save(source / f"{name}.png")
        pose = np.eye(4)
        pose[:3, 3] = [x, 0.0, 3.0]
        frames.append({"file_path": name, "transform_matrix": pose.tolist()})
    for split, selected in [("train", frames[:2]), ("test", frames[2:])]:
        (source / f"transforms_{split}.json").write_text(json.dumps(
            {"camera_angle_x": 0.8, "frames": selected}))

    base = work / "base"
    enhancement = work / "enhancement"
    dynamic = work / "dynamic"
    models = [("base", base, []),
              ("enhancement", enhancement, ["--dynamic_opacity", "--foundation_gs_path",
               str(base / "point_cloud/iteration_12/point_cloud.ply")]),
              ("dynamic", dynamic, ["--dynamic_lapis", "--initial_gs_path",
               str(enhancement / "point_cloud/iteration_12/point_cloud.ply")])]
    for name, model, extra in models:
        command = [sys.executable, str(ROOT / "tools/content_preparation/backend.py"), "--entry", str(UPSTREAM_ROOT / "train.py"), "-s", str(source), "-m", str(model),
                   "--iterations", "12", "--eval", "--test_iterations", "12",
                   "--save_iterations", "12", "--port", free_port(),
                   "--opacity_reset_interval", "100", "--densify_until_iter", "0"]
        if name == "base":
            command += ["--densify_until_iter", "10", "--densify_from_iter", "2",
                        "--densification_interval", "4"]
        run(f"train_{name}", command + extra, work)
        checkpoint = model / "point_cloud/iteration_12/point_cloud.ply"
        vertices = PlyData.read(checkpoint)["vertex"]
        assert len(vertices) > 0
        for field in vertices.data.dtype.names:
            assert np.isfinite(vertices[field]).all(), f"Nonfinite {name}.{field}"

    before = PlyData.read(enhancement / "point_cloud/iteration_12/point_cloud.ply")["vertex"]
    after = PlyData.read(dynamic / "point_cloud/iteration_12/point_cloud.ply")["vertex"]
    assert len(before) == len(after), "Dynamic training must preserve Gaussian count."
    for field in before.data.dtype.names:
        if field.startswith(("f_", "scale_")) or field == "opacity":
            np.testing.assert_array_equal(before[field], after[field])
    assert any(np.any(before[field] != after[field]) for field in ["x", "y", "z"])
    print("PASS dynamic updates positions and preserves color/scaling/opacity", flush=True)
    run("render", [sys.executable, str(ROOT / "tools/content_preparation/backend.py"), "--entry", str(UPSTREAM_ROOT / "render.py"), "-m", str(dynamic), "--iteration", "12"], work)
    for split, expected in [("train", 2), ("test", 1)]:
        images = list((dynamic / split / "ours_12/renders").glob("*.png"))
        assert len(images) == expected
        assert np.asarray(Image.open(images[0])).max() > 0
    run("metrics", [sys.executable, str(ROOT / "tools/content_preparation/backend.py"), "--entry", str(UPSTREAM_ROOT / "metrics.py"), "-m", str(dynamic)], work)
    # metrics.py catches errors internally, so validate artifacts as well as exit code.
    for filename in ["results.json", "train_results.json"]:
        metrics = json.loads((dynamic / filename).read_text())["ours_12"]
        assert set(metrics) == {"SSIM", "PSNR", "LPIPS"}
        assert all(np.isfinite(value) for value in metrics.values()), metrics
        print(f"PASS {filename}: {metrics}", flush=True)
    print(f"ALL CHECKS PASSED. Logs and generated checkpoints: {work}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] in (["--help"], ["-h"]):
        print("Usage: python scripts/smoke_test.py [--preprocess-only]. Explicit CUDA/Open3D smoke run; bounded CPU validation uses tools/content_preparation/self_test.py.")
    elif sys.argv[1:] == ["--preprocess-only"]:
        test_preprocessing()
    elif sys.argv[1:]:
        raise SystemExit("Usage: python scripts/smoke_test.py [--preprocess-only]")
    else:
        main()

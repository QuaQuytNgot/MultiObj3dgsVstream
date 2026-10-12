"""Benchmark input and reporting checks that require no CUDA runtime."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import benchmark_visibility as benchmark
from tools.content_preparation.assets import GaussianState, state_hash
from tools.content_preparation.proxy import generate_proxy


def test_help_remains_available_without_importing_torch():
    code = """\
import sys
from tools.benchmark_visibility import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'torch' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=benchmark.ROOT,
                               text=True, capture_output=True, check=True)
    assert "--scene" in completed.stdout


@pytest.fixture
def proxy_scene(tmp_path):
    state = GaussianState({
        "xyz": np.asarray([[0., 0., 2.]], dtype=np.float32),
        "rotation": np.asarray([[1., 0., 0., 0.]], dtype=np.float32),
        "scale": np.full((1, 3), -3., dtype=np.float32),
        "opacity": np.zeros((1, 1), dtype=np.float32),
        "sh": np.zeros((1, 3, 1), dtype=np.float32),
    }, ids=np.asarray([17]))
    generate_proxy(state, {"count": 1}, tmp_path / "proxy", frame=7)
    transform = np.eye(4)
    transform[0, 3] = 3.0
    document = {"name": "existing", "timestamp": 0.25,
                "camera": {"world_view": np.eye(4).tolist(), "width": 256, "height": 256,
                           "fovx": np.pi / 2, "fovy": np.pi / 2},
                "objects": [{"object_id": "object", "proxy_path": "proxy/proxy.json",
                             "frame": 7, "timestamp": 0.25,
                             "object_to_world": transform.tolist()}]}
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(document))
    return path, document, state


def test_existing_proxy_loader_preserves_explicit_frame_timestamp_and_transform(proxy_scene):
    path, document, state = proxy_scene
    source = benchmark.load_scene(path)
    obj = source["objects"][0]
    assert obj.object_id == "object" and obj.timestamp == 0.25
    assert obj.proxy.ids.tolist() == [17]
    # Loading preserves numeric snapshot; transforms are deferred to the renderer.
    assert state_hash(obj.proxy) == state_hash(state)
    np.testing.assert_array_equal(obj.object_to_world, document["objects"][0]["object_to_world"])
    assert source["provenance"]["assets"][0]["frame"] == 7
    assert Path(source["provenance"]["assets"][0]["proxy_path"]).is_absolute()
    assert len(source["provenance"]["assets"][0]["proxy_sha256"]) == 64


def test_missing_requested_proxy_frame_is_rejected(proxy_scene):
    path, document, _ = proxy_scene
    document["objects"][0]["frame"] = 8
    path.write_text(json.dumps(document))
    with pytest.raises(KeyError, match="frame 8"):
        benchmark.load_scene(path)


def test_existing_proxy_payload_checksum_is_validated(proxy_scene):
    path, _, _ = proxy_scene
    payload = path.parent / "proxy/frame_000007.npz"
    with payload.open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        benchmark.load_scene(path)


@pytest.mark.parametrize("frame", [True, 7.0, "7"])
def test_frame_must_be_explicit_integer(proxy_scene, frame):
    path, document, _ = proxy_scene
    document["objects"][0]["frame"] = frame
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="integer frame"):
        benchmark.load_scene(path)


def test_cuda_unavailability_never_produces_a_measurement_report(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    output = tmp_path / "measurement.json"
    with pytest.raises(RuntimeError, match="no performance measurements"):
        benchmark.main(["--output", str(output)])
    assert not output.exists()


def test_synthetic_fixture_is_reproducible_and_matches_requested_counts():
    first = benchmark.synthetic_scene(6, 11)
    second = benchmark.synthetic_scene(6, 11)
    assert len(first["objects"]) == 6
    assert all(obj.proxy.count == 11 for obj in first["objects"])
    assert [state_hash(obj.proxy) for obj in first["objects"]] == [
        state_hash(obj.proxy) for obj in second["objects"]]
    assert len({obj.object_id for obj in first["objects"]}) == 6


def test_timing_distribution_uses_nearest_rank_p95_and_milliseconds():
    result = benchmark.distribution(list(range(1, 101)))
    assert result == {"mean": 50.5, "median": 50.5, "p95": 95}
    with pytest.raises(ValueError, match="requires samples"):
        benchmark.distribution([])

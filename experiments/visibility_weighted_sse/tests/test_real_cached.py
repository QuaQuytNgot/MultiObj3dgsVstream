"""Real cached inputs stay distinct from the synthetic mathematics fixtures."""
import json
from pathlib import Path
import shutil

import pytest

from experiments.visibility_weighted_sse.common import read_json
from experiments.visibility_weighted_sse.real_cached import convert
from experiments.visibility_weighted_sse.runner import validate
from experiments.visibility_weighted_sse.trace import TraceCameraProvider
from experiments.visibility_weighted_sse.visibility import CachedVisibilityAdapter


SOURCE = Path(__file__).resolve().parents[2] / "visibility_guided_sse" / "runs" / "pilot_v1"


@pytest.fixture
def cached_source():
    if not (SOURCE / "artifacts" / "proxy").is_dir():
        pytest.skip("Validated historical contribution cache is not present in this checkout")
    return SOURCE


def test_real_cache_conversion_uses_only_frozen_independent_inputs(cached_source, tmp_path):
    result = convert(cached_source, tmp_path)
    assert result["input_kind"] == "validated_cached_artifacts"
    assert result["new_rendering_calls"] == result["RGB_payloads_opened"] == 0
    assert result["source_files_verified"] > 200
    config = tmp_path / "config.json"
    assert validate(config)["status"] == "ready"
    profiles = read_json(tmp_path / "profiles.json")
    assert profiles["provenance"]["source_sse_values_modified"] is False
    assert profiles["compatibility"]["time_aggregation"] == "SSE_camera_mean_timestamp_sum_alpha_camera_time_mean"
    offline = read_json(cached_source / "prepared" / "offline_sse.json")
    for index, obj in enumerate(profiles["objects"]):
        assert [q["average_solo_sse"] for q in obj["gops"][0]["offline_sse"]] == list(reversed(offline["distortions"][index]))
        for quality, cost in zip(obj["gops"][0]["offline_sse"], reversed(offline["costs"][index])):
            assert sum(profiles["resources"][key]["bytes"] for key in quality["resource_ids"]) == cost
    trace = TraceCameraProvider(read_json(tmp_path / "trace.json"))
    visibility = CachedVisibilityAdapter(read_json(tmp_path / "visibility.json"), trace)
    measured = visibility.aggregate("0")["per_object"]
    assert measured["longdress"]["A_visible"] == pytest.approx(22041.341252912185)
    assert measured["loot"]["A_visible"] == pytest.approx(31649.034376292257)
    assert measured["soldier"]["A_visible"] == pytest.approx(28848.387601247174)


def test_converter_rejects_missing_inputs(tmp_path):
    with pytest.raises(ValueError, match="Missing"):
        convert(tmp_path / "missing", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_converter_rejects_changed_freeze(cached_source, tmp_path):
    freeze = read_json(cached_source / "freeze.json")
    freeze["prediction_assumption"] = "changed"
    (tmp_path / "freeze.json").write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="freeze hash mismatch"):
        convert(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_converter_rejects_changed_prepared_metadata(cached_source, tmp_path):
    shutil.copy(cached_source / "freeze.json", tmp_path / "freeze.json")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "offline_sse.json").write_text("{}")
    with pytest.raises(ValueError, match="prepared input hash mismatch"):
        convert(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()

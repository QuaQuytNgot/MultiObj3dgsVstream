"""One CPU-only historical import checks provenance and quality conventions."""
from pathlib import Path

import pytest

from experiments.adaptation_baselines import real_cached
from experiments.visibility_weighted_sse.common import digest, file_digest, read_json, write_json


PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "experiments/visibility_guided_sse/runs/pilot_v1"


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    if not (SOURCE / "artifacts/proxy").is_dir():
        pytest.skip("Validated historical contribution cache is not present in this checkout")
    output = tmp_path_factory.mktemp("baseline_cached")
    calls, opened = [], []
    original_convert = real_cached.proposed_cached.convert
    original_open = Path.open

    def existing_converter(source, destination):
        calls.append((source, destination))
        return original_convert(source, destination)

    def metadata_or_encoded_only(path, *args, **kwargs):
        parts = Path(path).parts
        assert "images" not in parts and "evaluation" not in parts and "joint" not in parts
        opened.append(str(path))
        return original_open(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(real_cached.proposed_cached, "convert", existing_converter)
        patch.setattr(Path, "open", metadata_or_encoded_only)
        # An accidental dependency on rendering must fail before any renderer
        # or CUDA backend can be initialized.
        patch.setitem(__import__("sys").modules, "experiments.nava_mse.rendering", None)
        result = real_cached.convert(SOURCE, output)
    return output, result, calls, opened


def test_existing_proposed_converter_is_reused_and_preserved(imported):
    output, result, calls, opened = imported
    assert calls == [(SOURCE.resolve(), output / "weighted_sse")]
    assert result["status"] == "ready"
    assert result["input_kind"] == "validated_cached_artifacts"
    assert result["new_rendering_calls"] == result["RGB_payloads_opened"] == 0
    assert result["source_files_verified"] > 250
    assert read_json(output / "weighted_sse/config.json")["schema"] == "visibility-weighted-sse.experiment.v1"
    assert not any("/images/" in path or "/evaluation/" in path for path in opened)


def test_original_psnr_and_pooled_mse_remain_different_references(imported):
    output, *_ = imported
    metadata = read_json(output / "baseline_metadata.json")
    frozen = read_json(PROJECT / real_cached.NAVA_FREEZE)
    average = read_json(PROJECT / frozen["config"]["output_dir"] / "profiles/average.json")
    entries = {(row["object_id"], row["crf"]): row["metrics"] for row in average["entries"]}
    profiles = read_json(output / "weighted_sse/profiles.json")
    assert metadata["profile_binding_sha256"] == file_digest(output / "weighted_sse/profiles.json")
    assert metadata["content_revision"] == profiles["content_revision"]
    assert metadata["profile_revision"] == digest({k: v for k, v in metadata.items() if k != "profile_revision"})
    assert metadata["quality_references"]["psnr"]["reference"] == "original_ply_synthetic_rgb"
    assert metadata["quality_references"]["mse"]["distribution_id"] == profiles["compatibility"]["distribution_id"]
    assert metadata["provenance"]["joint_quality_used"] is False
    for obj in metadata["objects"]:
        index = frozen["config"]["objects"].index(obj["object_id"])
        for quality in obj["gops"][0]["qualities"]:
            crf = int(quality["quality_id"][3:])
            assert quality["representation_id"] == f'{obj["object_id"]}/0/{crf}'
            assert quality["psnr_db"] == frozen["inputs"]["nava_psnr"][index][frozen["config"]["crfs"].index(crf)]
            assert quality["mse"] == entries[(obj["object_id"], crf)]["mse"]
            if crf == 20:
                assert quality["mse"] == 0
                assert 0 < quality["psnr_db"] < 100


def test_source_local_bounds_cover_every_presentation_frame(imported):
    output, *_ = imported
    metadata = read_json(output / "baseline_metadata.json")
    frozen = read_json(PROJECT / real_cached.NAVA_FREEZE)
    bounds = read_json(PROJECT / frozen["config"]["bounds_source"])
    for obj in metadata["objects"]:
        original = bounds["contents"][obj["object_id"]]["gops"]["0"]
        assert len(original["frames"]) == 60
        assert obj["gops"][0]["bounds"] == original["bounds"]
    assert metadata["provenance"]["source_baseline_values_modified"] is False
    assert all(file_digest(path) == expected for path, expected in metadata["provenance"]["source_dependencies_sha256"].items())


def test_all_methods_share_frozen_inputs_and_budgets(imported):
    from experiments.adaptation_baselines.runner import validate

    output, result, *_ = imported
    config = read_json(result["config"])
    assert config["methods"] == list(real_cached.METHODS)
    assert config["configuration_hash"] == digest({k: v for k, v in config.items() if k != "configuration_hash"})
    assert config["prediction_assumption"] == "ORACLE_CAMERA_TRACE"
    assert [row["bytes"] for row in config["budgets"]] == [10_000_000, 20_000_000, 30_000_000]
    assert set(config["inputs"]) == {"trace", "profiles", "visibility", "baseline_metadata"}
    for record in config["inputs"].values():
        assert record["sha256"] == file_digest(output / record["path"])
    assert validate(result["config"])["status"] == "ready"


def test_historical_average_tampering_fails_before_conversion(tmp_path, monkeypatch):
    if not (SOURCE / "artifacts/proxy").is_dir():
        pytest.skip("Validated historical contribution cache is not present in this checkout")
    original_digest = real_cached.file_digest

    def corrupt_average_hash(path):
        if str(path).endswith("nava_mse/runs/gop0_v1/profiles/average.json"):
            return "0" * 64
        return original_digest(path)

    def conversion_must_not_start(*args, **kwargs):
        pytest.fail("Proposed conversion started before audited baseline inputs were checked")

    monkeypatch.setattr(real_cached, "file_digest", corrupt_average_hash)
    monkeypatch.setattr(real_cached.proposed_cached, "convert", conversion_must_not_start)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Historical baseline input hash mismatch.*average.json"):
        real_cached.convert(SOURCE, output)
    assert not output.exists()


def test_converter_rejects_a_different_historical_source_freeze(tmp_path, monkeypatch):
    if not (SOURCE / "artifacts/proxy").is_dir():
        pytest.skip("Validated historical contribution cache is not present in this checkout")
    source = tmp_path / "source"
    freeze = read_json(SOURCE / "freeze.json")
    freeze["source_freeze_sha256"] = "0" * 64
    write_json(source / "freeze.json", freeze)
    monkeypatch.setattr(real_cached.proposed_cached, "convert", lambda *a: pytest.fail("Unexpected conversion"))
    with pytest.raises(ValueError, match="different frozen source inputs"):
        real_cached.convert(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_converter_rejects_missing_source_without_writing_output(tmp_path):
    with pytest.raises(ValueError, match="Missing"):
        real_cached.convert(tmp_path / "missing", tmp_path / "output")
    assert not (tmp_path / "output").exists()

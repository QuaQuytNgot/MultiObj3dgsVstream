"""Fail-closed reporting contracts; these tests do not execute a GPU renderer."""
import json
import sys

import pytest

from tools import validate_gpu_playback as validation


def test_unavailable_cuda_records_blocked_without_decoding_or_rendering(tmp_path, monkeypatch):
    report = tmp_path / "gpu.json"
    monkeypatch.setattr(sys, "argv", ["validate_gpu_playback", "--output", str(report)])
    monkeypatch.setattr(validation, "probe_cuda", lambda device: ({"cuda_available": False}, "No visible CUDA device"))

    def unexpected(*args, **kwargs):
        pytest.fail("A blocked CUDA probe must not trigger media decoding, compilation or a substitute renderer")

    monkeypatch.setattr(validation, "extension_load", unexpected)
    monkeypatch.setattr(validation, "isolated", unexpected)
    monkeypatch.setattr(validation, "playback_scenarios", unexpected)
    assert validation.main() == 2
    result = json.loads(report.read_text())
    assert result["status"] == "blocked"
    assert result["environment"]["cuda_available"] is False
    assert "isolated" not in result and "playback" not in result


def test_isolated_media_state_window_cannot_grow_unbounded(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validate_gpu_playback", "--frames", "100000"])
    with pytest.raises(SystemExit) as exc:
        validation.arguments()
    assert exc.value.code == 2

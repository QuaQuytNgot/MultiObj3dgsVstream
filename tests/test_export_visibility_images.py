"""Offline image packaging preserves measurement sources and needs no GPU."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

from PIL import Image
import pytest

from tools.export_visibility_images import export_presentation


def _writer(title, calls):
    def write(*args):
        directory = Path(args[-1])
        calls.append(args)
        directory.mkdir(parents=True, exist_ok=True)
        filename = "sample.png"
        Image.new("RGB", (1920, 1080), "white").save(directory / filename)
        return [{"file": filename, "title": title, "kind": title,
                 "description": "Known static image; no measurements changed."}]
    return write


def test_packages_both_kinds_and_preserves_original_report(tmp_path, monkeypatch):
    report = tmp_path / "report"
    report.mkdir()
    source = b'{"status":"PASSED","value":0.957247}'
    (report / "report.json").write_bytes(source)
    output = tmp_path / "images"
    calls = []
    monkeypatch.setitem(sys.modules, "tools.visibility_concept_diagrams",
                        SimpleNamespace(write_concept_diagrams=_writer("concept", calls)))
    monkeypatch.setitem(sys.modules, "tools.visibility_result_figures",
                        SimpleNamespace(write_result_figures=_writer("measurement", calls)))
    manifest = export_presentation(report, output)
    assert len(manifest["figures"]) == 2
    assert (report / "report.json").read_bytes() == source
    assert (output / "visibility_slides.pdf").read_bytes().startswith(b"%PDF")
    with zipfile.ZipFile(output / "visibility_images.zip") as archive:
        assert set(archive.namelist()) == {"README.md", "manifest.json", "concepts/sample.png", "results/sample.png"}
    exported = json.loads((output / "manifest.json").read_text())
    assert exported["size"] == [1920, 1080]
    assert "concepts/sample.png" in (output / "index.html").read_text()
    assert "results/sample.png" in (output / "index.html").read_text()
    assert len(calls) == 2


def test_concepts_only_does_not_load_measurements(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "tools.visibility_concept_diagrams",
                        SimpleNamespace(write_concept_diagrams=_writer("concept", calls)))
    monkeypatch.setitem(sys.modules, "tools.visibility_result_figures", None)
    manifest = export_presentation(tmp_path / "absent_report", tmp_path / "images", only="concepts")
    assert manifest["report_directory"] is None
    assert len(manifest["figures"]) == 1


def test_rejects_unknown_export_mode(tmp_path):
    with pytest.raises(ValueError, match="only"):
        export_presentation(tmp_path, tmp_path / "images", only="wrong")

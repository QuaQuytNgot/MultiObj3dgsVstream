"""Metadata-only MPD, transport and planner integration; no decoder or GPU."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from src.client.adaptation_inputs import (
    AvailabilitySnapshot, DecodedState, DensityCapability, ObjectViewContext,
    PlanningContext, ProfileCoordinates, QualityTarget, RequirementsResolver,
    SegmentViewSample,
)
from src.client.adaptation_planner import AdaptationPlanner
from src.client.bandwidth import BandwidthEstimate
from src.client.mpd_parser import MPDParser, MetadataError
from src.content_metadata import digest, json_bytes, relative_path, resolve_uri
from src.content_mpd import NAMESPACE, PROPERTY, build_mpd
from tools.content_preparation.manifest import build_manifest
from tools.content_preparation.mpd import export_mpd, load_prepared
from tools.content_preparation.packaging import package_object


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_sources(root, objects=("actor",), *, density=True, fps=2):
    """Tiny synthetic packaging fixture; no real media/checkpoints are touched."""
    server, density_paths = [], {}
    for obj in objects:
        object_root = root / obj
        frames = [{"frame": 10+i, "timestamp": i/fps} for i in range(7)]
        qualities = [f"Q{i}" for i in range(4)]
        records, state_hashes = [], {}
        for q in range(4):
            layer = "Base" if q == 0 else f"E{q}"
            for frame in frames:
                f = frame["frame"]
                raw = f"fixture:{obj}:{f}:{q}".encode()
                relative = f"encoded/{layer}/{f}.bin"
                path = object_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                state_hashes[q, f] = digest(f"state:{obj}:{f}:{q}".encode())
                records.append({"id": f"progressive:{layer}:{f}", "mode": "progressive",
                                "quality": qualities[q], "layer": layer, "frame": f,
                                "path": relative, "bytes": len(raw), "sha256": digest(raw),
                                "codec": {"name": "metadata-fixture", "version": "1.0.0"},
                                "decoded_state_hash": state_hashes[q, f],
                                "parent_state_hash": state_hashes[q-1, f] if q else None})
        package = package_object(obj, frames, qualities, records,
                                 {"gof_frames": 5, "segment_frames": 3,
                                  "refresh_frames": {"Base": 5, "E1": 2, "E2": 2, "E3": 1}}, object_root)
        settings = {"backend": "metadata-fixture", "background": [0, 0, 0], "width": 8, "height": 8}
        profile = {"schema": "dynamic-lapisgs.quality-profile.v1",
                   "reference": "highest_quality_decoded_delivered_representation",
                   "metadata": {"reference_quality": "Q3", "renderer_settings": settings}, "rows": []}
        for frame in (frames[0], frames[-1]):
            for q in range(4):
                profile["rows"].append({"object": obj, "mode": "progressive", "quality": qualities[q],
                                        **frame, "azimuth": 0, "elevation": 0, "scale": 1,
                                        "view_bin": "v0", "scale_bin": "s0", "time_bin": str(frame["frame"]),
                                        "reference_quality": "Q3", "reference_state": "decoded",
                                        "metrics": {"mse": (3-q)/10}})
        profile_path = object_root / "profiles/profile.json"
        write_json(profile_path, profile)
        manifest = build_manifest(obj, frames, qualities, package,
                                  provenance={"implementation_sha256": "b"*64},
                                  quality_profile_path="profiles/profile.json", output_dir=object_root)
        manifest["object"].update(fps=fps, duration_seconds=7/fps)
        manifest_path = object_root / "manifest.json"
        write_json(manifest_path, manifest)
        write_json(object_root / "package_index.json", package)
        server.append({"id": obj, "manifest_path": f"{obj}/manifest.json",
                       "bytes": manifest_path.stat().st_size, "sha256": digest(manifest_path.read_bytes())})
        if density:
            enriched = json.loads(json.dumps(profile))
            enriched["rows"] = enriched["rows"][:4]
            enriched["metadata"]["error_density"] = {
                "schema": "dynamic-lapisgs.error-density.v1", "version": "1.0.0",
                "area_quality": "Q0", "area_reference_state": "decoded",
                "reference_quality": "Q3", "reference_state": "decoded",
                "mse_convention": "full-image RGB [0,1] sum squared error / (3*width*height); unmasked",
                "source_profile": {"sha256": digest(profile_path.read_bytes())},
                "canonical_transform": manifest["object"]["canonical_transform"],
                "renderer_settings": settings, "source_implementation_sha256": "b"*64,
                "epsilon_fraction": 1e-8}
            for row in enriched["rows"]:
                row.update(width=8, height=8, pixel_count=64, solo_area=32, solo_area_fraction=.5,
                           area_quality="Q0", area_reference_state="decoded", area_epsilon_fraction=1e-8,
                           alpha_convention="one_minus_transmittance", area_status="valid",
                           solo_area_source={"decoded_state_hash": state_hashes[0, 10]},
                           error_density=row["metrics"]["mse"]*2)
            relative = f"{obj}/profiles/density.json"
            write_json(root / relative, enriched)
            density_paths[obj] = relative
        # These are fixture intermediates only. Tests below have metadata alone.
        shutil.rmtree(object_root / "encoded")
        shutil.rmtree(object_root / "media")
    write_json(root / "manifest.json", {"schema_version": "content-preparation.server.v1", "objects": server})
    return density_paths


def build_bundle(root, density_paths=None):
    records, _ = load_prepared(root, metadata_only=True, density_profiles=density_paths)
    xml, index = build_mpd(records)
    (root / "manifest.mpd").write_bytes(xml)
    index["mpd"] = {"path": "manifest.mpd", "bytes": len(xml), "sha256": digest(xml)}
    write_json(root / "content_index.json", index)
    return index


@pytest.fixture
def bundle(tmp_path):
    densities = write_sources(tmp_path)
    build_bundle(tmp_path, densities)
    return tmp_path


def local(root):
    return MPDParser.from_local(root / "manifest.mpd", media_base_url="https://media.test/run/")


def change_xml(root, edit):
    path = root / "manifest.mpd"
    tree = ET.fromstring(path.read_bytes())
    edit(tree)
    raw = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
    path.write_bytes(raw)
    index = json_bytes((root / "content_index.json").read_bytes())
    index["mpd"].update(bytes=len(raw), sha256=digest(raw))
    write_json(root / "content_index.json", index)


def test_typed_index_progressive_costs_offsets_and_nonaligned_boundaries(bundle):
    parser = local(bundle)
    index = parser.get_content_index()
    assert parser.get_objects() == ("actor",)
    assert parser.get_quality_levels("actor") == (0, 1, 2, 3)
    assert len(index.payloads) == 28
    base = parser.get_segment_info("actor", 0, 1.1)
    enhancement = parser.get_segment_info("actor", 3, 1.1)
    assert base.frames != enhancement.frames
    target = QualityTarget("actor", (12,), 3)
    cold = AvailabilitySnapshot(index.content_revision, "cold")
    result = RequirementsResolver(index).required_units(target, cold)
    assert result.valid and len(result.missing_units) == 4
    assert result.required_bytes == sum(u.size_bytes for u in result.missing_units)
    q0 = index.payload_for("actor", 12, 0)
    retained = replace(cold, decoded_states={q0.payload_id: DecodedState.from_payload(q0)})
    upgraded = RequirementsResolver(index).required_units(target, retained)
    assert upgraded.required_bytes == result.required_bytes - base.request_unit.size_bytes
    payload = index.payload_for("actor", 12, 3)
    location = parser.get_payload_location(payload.payload_id)
    assert location.request_unit_id == payload.request_unit_id
    assert location.offset + location.length <= index.request_units[payload.request_unit_id].size_bytes
    assert index.request_units[payload.request_unit_id].request.byte_range is None
    assert base.request_unit.request.metadata["traffic_kind"] == "base"
    assert enhancement.request_unit.request.metadata["traffic_kind"] == "enhancement"
    assert parser.get_segment_info("actor", 3, 3.49).frames == (16,)
    with pytest.raises(KeyError):
        parser.get_segment_info("actor", 3, 3.5)
    with pytest.raises(ValueError):
        parser.get_segment_info("actor", True, 0)


def test_raw_mse_and_partial_density_are_distinct(bundle):
    parser = local(bundle)
    mse = parser.get_quality_profile("actor")
    density = parser.get_quality_profile("actor", density=True)
    coordinates = dict(azimuth=0, elevation=0, scale=1, timestamp=0)
    assert mse.reference.metric == "full-image-rgb-mse"
    assert mse.lookup_mse(0, **coordinates) == .3
    with pytest.raises(MetadataError, match="Raw MSE"):
        mse.lookup_density(0, **coordinates)
    assert density.lookup_density(0, **coordinates).value == .6
    missing = density.lookup_density(0, **dict(coordinates, timestamp=3))
    assert missing.value is None and missing.reason == "density_context_unavailable"
    assert density.reference.metadata["coverage"]["partial"] is True
    assert density.reference.metadata["producer_parity"] == "not_verified_by_metadata_export"
    assert density.reference.metadata["alpha_area_epsilon"] == 64e-8
    with pytest.raises(TypeError):
        density.reference.metadata["coverage"]["partial"] = False


def test_absent_density_is_explicit(tmp_path):
    write_sources(tmp_path, density=False)
    build_bundle(tmp_path)
    parser = local(tmp_path)
    assert parser.get_quality_profile_reference("actor") is not None
    assert parser.get_density_profile_reference("actor") is None
    assert parser.get_quality_profile("actor", density=True) is None


def test_fractional_terminal_time_is_excluded(tmp_path):
    write_sources(tmp_path, density=False, fps=30)
    build_bundle(tmp_path)
    parser = local(tmp_path)
    with pytest.raises(KeyError):
        parser.get_segment_info("actor", 0, 7/30)
    assert parser.get_segment_info("actor", 0, 7/30-1e-10).frames[-1] == 16


def test_multiobject_order_and_namespaced_identities(tmp_path):
    densities = write_sources(tmp_path, objects=("soldier", "loot"))
    build_bundle(tmp_path, densities)
    parser = local(tmp_path)
    assert parser.get_objects() == ("soldier", "loot")
    index = parser.get_content_index()
    assert len(index.payloads) == 56
    assert index.payload_for("soldier", 10, 0).payload_id != index.payload_for("loot", 10, 0).payload_id


def test_root_base_url_and_xml_namespace_prefixes(bundle):
    destination = bundle / "cdn"
    destination.mkdir()
    shutil.move(str(bundle / "actor"), str(destination / "actor"))
    def edit(root):
        ET.SubElement(root, f"{{{NAMESPACE}}}BaseURL").text = "cdn/"
    change_xml(bundle, edit)
    shutil.move(str(bundle / "content_index.json"), str(destination / "content_index.json"))
    parser = local(bundle)
    assert "/run/cdn/actor/" in parser.get_segment_info("actor", 0, 0).request_unit.request.url


def test_alternate_xml_prefix_is_supported(bundle):
    try:
        ET.register_namespace("dash", NAMESPACE)
        change_xml(bundle, lambda root: None)
    finally:
        ET.register_namespace("", NAMESPACE)
    assert b"dash:MPD" in (bundle / "manifest.mpd").read_bytes()
    assert local(bundle).get_objects() == ("actor",)


@pytest.mark.parametrize("bad", [b"<bad", b'<MPD xmlns="wrong"/>',
                                   b'<!DOCTYPE x [<!ENTITY z "x">]><MPD/>'])
def test_malformed_or_unsupported_xml(bad):
    with pytest.raises(MetadataError):
        MPDParser().parse_mpd(bad, source_uri="https://media.test/manifest.mpd", documents={})


@pytest.mark.parametrize("field,value", [("content_revision", "wrong"), ("schema_version", "future.v99")])
def test_incompatible_content_index(bundle, field, value):
    path = bundle / "content_index.json"
    index = json_bytes(path.read_bytes())
    index[field] = value
    write_json(path, index)
    with pytest.raises(MetadataError):
        local(bundle)


@pytest.mark.parametrize("field", ["bytes", "sha256", "partial", "metric", "timeline"])
def test_reject_index_disagreement(bundle, field):
    path = bundle / "content_index.json"
    index = json_bytes(path.read_bytes())
    obj = index["objects"][0]
    if field in ("bytes", "sha256"):
        obj["representations"][0]["segments"][0][field] = None
    elif field == "partial":
        obj["density_profile"]["coverage"]["partial"] = False
    elif field == "metric":
        obj["density_profile"]["metric"] = "full-image-rgb-mse"
    else:
        obj["representations"][0]["segments"][0]["duration_ticks"] += 1
    write_json(path, index)
    with pytest.raises(MetadataError):
        local(bundle)


@pytest.mark.parametrize("mode", ["dependency", "timeline", "unknown_extension", "missing_object", "bandwidth"])
def test_reject_schema_valid_but_inconsistent_xml(bundle, mode):
    def edit(root):
        adaptation = root.find(f".//{{{NAMESPACE}}}AdaptationSet")
        rep = adaptation.findall(f"{{{NAMESPACE}}}Representation")[-1]
        if mode == "dependency":
            rep.set("dependencyId", "actor:nonexistent")
        elif mode == "timeline":
            rep.find(f".//{{{NAMESPACE}}}S").set("d", "999")
        elif mode == "unknown_extension":
            ET.SubElement(rep, f"{{{NAMESPACE}}}EssentialProperty", schemeIdUri="urn:unknown:99", value="x")
        elif mode == "missing_object":
            adaptation.remove(next(n for n in adaptation if n.get("schemeIdUri") == PROPERTY + "object-id"))
        else:
            rep.set("bandwidth", "1")
    change_xml(bundle, edit)
    with pytest.raises(MetadataError):
        local(bundle)


def test_missing_and_corrupt_referenced_json(bundle):
    profile = bundle / "actor/profiles/profile.json"
    profile.write_bytes(b'{"rows": []}')
    with pytest.raises(MetadataError, match="hash"):
        local(bundle)
    profile.unlink()
    with pytest.raises(MetadataError):
        local(bundle)


@pytest.mark.parametrize("path", ["../x.json", "a/../x", "/absolute", "a\\b", "a/%2e%2e/b", "//evil/x", "a?x=1"])
def test_unsafe_references(path):
    with pytest.raises(ValueError):
        relative_path(path)
    with pytest.raises(ValueError):
        resolve_uri("https://media.test/run/manifest.mpd", path)


def test_json_duplicate_and_nonfinite_values():
    for raw in (b'{"a":1,"a":2}', b'{"mse":NaN}', b'{"mse":Infinity}', b'{"mse":1e999}'):
        with pytest.raises(ValueError):
            json_bytes(raw)


@pytest.mark.parametrize("mode", ["cycle", "missing_parent", "size", "parent_hash", "coverage"])
def test_invalid_authoritative_package_is_rejected(tmp_path, mode):
    write_sources(tmp_path, density=False)
    manifest_path = tmp_path / "actor/manifest.json"
    manifest = json_bytes(manifest_path.read_bytes())
    package = manifest["package"]
    if mode == "cycle":
        base = package["payloads"][0]
        base["dependencies"] = [package["payloads"][7]["id"]]
    elif mode == "missing_parent":
        package["payloads"][7]["dependencies"] = ["missing"]
    elif mode == "size":
        package["segments"][0]["bytes"] = None
    elif mode == "parent_hash":
        package["payloads"][7]["parent_state_hash"] = "f" * 64
    else:
        package["segments"][0]["frames"] = [999]
    write_json(manifest_path, manifest)
    write_json(tmp_path / "actor/package_index.json", package)
    server = json_bytes((tmp_path / "manifest.json").read_bytes())
    server["objects"][0].update(bytes=manifest_path.stat().st_size, sha256=digest(manifest_path.read_bytes()))
    write_json(tmp_path / "manifest.json", server)
    with pytest.raises((ValueError, TypeError)):
        build_bundle(tmp_path)


@pytest.mark.parametrize("mode", ["normalization", "version", "mse", "area", "source_hash"])
def test_density_source_validation(tmp_path, mode):
    densities = write_sources(tmp_path)
    path = tmp_path / densities["actor"]
    doc = json_bytes(path.read_bytes())
    meta = doc["metadata"]["error_density"]
    if mode == "normalization":
        meta["area_quality"] = "Q3"
    elif mode == "version":
        meta["version"] = "99.0.0"
    elif mode == "mse":
        doc["rows"][0]["metrics"]["mse"] += .1
    elif mode == "area":
        doc["rows"][0]["solo_area"] = 0
    else:
        meta["source_profile"]["sha256"] = "c" * 64
    write_json(path, doc)
    with pytest.raises(ValueError):
        build_bundle(tmp_path, densities)


def test_utf16_entity_xml_is_rejected():
    raw = '<!DOCTYPE MPD [<!ENTITY x "bad">]><MPD/>'.encode("utf-16")
    with pytest.raises(MetadataError):
        MPDParser().parse_mpd(raw, source_uri="https://media.test/m.mpd", documents={})


def test_local_symlink_escape_is_rejected(bundle, tmp_path_factory):
    external = tmp_path_factory.mktemp("external") / "profile.json"
    target = bundle / "actor/profiles/profile.json"
    external.write_bytes(target.read_bytes())
    target.unlink()
    try:
        target.symlink_to(external)
    except OSError:
        pytest.skip("Symlinks unavailable")
    with pytest.raises(MetadataError, match="escapes"):
        local(bundle)


def test_parser_has_no_gpu_yaml_or_transport_import_side_effects(bundle):
    code = '''
import sys
from src.client.mpd_parser import MPDParser
p = MPDParser.from_local(sys.argv[1], media_base_url="https://media.test/")
assert p.get_objects() == ("actor",)
assert not any(name in sys.modules for name in ("torch", "numpy", "yaml", "curl_cffi"))
assert not any(name.startswith("src.renderer") for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(bundle / "manifest.mpd")],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_oversize_reference_rejected_before_source_fetch(bundle):
    index = json_bytes((bundle / "content_index.json").read_bytes())
    index["objects"][0]["manifest_bytes"] = 33 * 1024 * 1024
    reads = []
    def loader(uri):
        reads.append(uri)
        assert uri.endswith("content_index.json")
        return json.dumps(index).encode()
    with pytest.raises(MetadataError, match="size budget"):
        MPDParser(metadata_loader=loader).parse_mpd((bundle / "manifest.mpd").read_bytes(),
                                                    source_uri="https://media.test/manifest.mpd")
    assert len(reads) == 1


def test_reparse_failure_clears_available_snapshot(bundle):
    parser = local(bundle)
    with pytest.raises(MetadataError):
        parser.parse_mpd(b"bad", source_uri="https://media.test/manifest.mpd")
    with pytest.raises(MetadataError, match="No successfully"):
        parser.get_content_index()


@pytest.mark.asyncio
async def test_http_transport_reuses_transfer_client_and_tags_metadata(bundle):
    requested = []
    class Client:
        async def fetch(self, spec):
            requested.append(spec)
            path = bundle / spec.url.removeprefix("https://media.test/run/")
            return SimpleNamespace(success=True, body=path.read_bytes())
    parser = await MPDParser.from_http("https://media.test/run/manifest.mpd", Client())
    assert parser.get_objects() == ("actor",)
    assert {s.metadata["traffic_kind"] for s in requested} == {"metadata", "profile"}
    assert all(s.metadata["bandwidth_eligible"] is False for s in requested)
    assert not any(s.url.endswith((".cpseg", ".bin", ".npz")) for s in requested)


def test_parser_derived_index_drives_existing_planner(bundle):
    parser = local(bundle)
    index = parser.get_content_index()
    base = index.payload_for("actor", 10, 0)
    density = parser.get_quality_profile("actor", density=True)
    capability = DensityCapability(index.content_revision, "synthetic-calibration", (8, 8),
                                   "metadata-fixture", "RGB_0_1_black",
                                   alpha_area_epsilon=density.reference.metadata["alpha_area_epsilon"])
    # This test explicitly supplies synthetic calibration; real producer parity
    # and camera calibration are not inferred by MPDParser.
    provider = SimpleNamespace(capability=capability,
                               get_density=lambda obj, q, c: density.lookup_density(q,
                                   azimuth=c.azimuth_deg, elevation=c.elevation_deg,
                                   scale=c.projected_scale, timestamp=c.media_time_s))
    coordinates = ProfileCoordinates(0, 0, 1, 0, capability.calibration_id)
    vis = SimpleNamespace(timestamp=0., resolution=(8, 8),
                          per_object={"actor": SimpleNamespace(A_visible=32., A_solo=32.)},
                          metadata={"sources": {"actor": {"source_type": "decoded_quality",
                              "quality_id": "Q0", "frame": 10, "timestamp": 0.,
                              "provenance": {"state_sha256": base.decoded_state_hash}}}})
    sample = SegmentViewSample(0., {"fixture": True}, vis,
                               {"actor": ObjectViewContext(10, coordinates, base.decoded_state_hash)})
    availability = AvailabilitySnapshot(index.content_revision, "synthetic-post-base",
                                        decoded_states={base.payload_id: DecodedState.from_payload(base)})
    context = PlanningContext(index, availability, {"actor": (10,)}, (sample,), provider, 1.)
    planner = AdaptationPlanner()
    inputs = planner.collect_inputs(context)
    cost = planner.calculate_cost(inputs)
    assert cost.values[0][0] == 0
    assert all(cost.valid[0])
    assert cost.values[0][3] > cost.values[0][1]
    plan = planner.plan(context, BandwidthEstimate(1e9, 5, "ready", 0))
    assert plan.status == "ok" and plan.qualities == (3,)
    assert plan.actual_bytes == cost.values[0][3]


@pytest.mark.skipif(shutil.which("xmllint") is None, reason="Exporter XSD gate requires xmllint")
def test_metadata_export_is_deterministic_and_never_reads_payloads(tmp_path, monkeypatch):
    densities = write_sources(tmp_path)
    original_open = Path.open
    def metadata_open(path, *args, **kwargs):
        assert path.suffix not in (".cpseg", ".bin", ".npz", ".ply")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", metadata_open)
    source_before = (tmp_path / "actor/manifest.json").read_bytes()
    first = export_mpd(tmp_path, metadata_only=True, density_profiles=densities)
    before = (tmp_path / "manifest.mpd").read_bytes()
    second = export_mpd(tmp_path, resume=True, metadata_only=True, density_profiles=densities)
    assert first["tasks_executed"] == 1 and second["tasks_skipped"] == 1
    assert (tmp_path / "manifest.mpd").read_bytes() == before
    assert (tmp_path / "actor/manifest.json").read_bytes() == source_before
    report = json_bytes((tmp_path / "mpd_validation.json").read_bytes())
    assert report["validated"] and report["exact_files"] is False
    assert report["payload_integrity"] == "NOT_RUN"


@pytest.mark.skipif(shutil.which("xmllint") is None, reason="Exporter XSD gate requires xmllint")
def test_density_can_be_added_only_with_explicit_owned_overwrite(tmp_path):
    densities = write_sources(tmp_path)
    export_mpd(tmp_path, metadata_only=True)
    assert local(tmp_path).get_density_profile_reference("actor") is None
    with pytest.raises(FileExistsError, match="ownership/options"):
        export_mpd(tmp_path, metadata_only=True, density_profiles=densities)
    export_mpd(tmp_path, metadata_only=True, density_profiles=densities, overwrite=True)
    assert local(tmp_path).get_density_profile_reference("actor") is not None


def test_actual_longdress_metadata_only():
    root = Path(__file__).resolve().parents[1] / "dataset"
    density_path = root / "longdress/profiles/profile.density-subset.json"
    if not density_path.is_file():
        pytest.skip("Optional real Longdress metadata not installed")
    records, _ = load_prepared(root, metadata_only=True,
                               density_profiles={"longdress": "longdress/profiles/profile.density-subset.json"})
    xml, index = build_mpd(records)
    index["mpd"] = {"path": "manifest.mpd", "bytes": len(xml), "sha256": digest(xml)}
    base_uri = "https://media.test/dataset/"
    documents = {base_uri + "content_index.json": json.dumps(index).encode()}
    for record in records:
        for path in (record["manifest_path"], record["package_path"],
                     record["quality_profile"]["path"], record["density_profile"]["path"]):
            documents[base_uri + path] = (root / path).read_bytes()
    parser = MPDParser().parse_mpd(xml, source_uri=base_uri + "manifest.mpd", documents=documents)
    snapshot = parser.get_content_index()
    assert len(snapshot.payloads) == 1200 and len(snapshot.request_units) == 80
    assert sum(u.size_bytes for u in snapshot.request_units.values()) == 20_282_105_152
    assert parser.get_density_profile_reference("longdress").metadata["coverage"]["contexts"] == 12
    assert parser.get_quality_profile_reference("longdress").metadata["coverage"]["rows"] == 6048
    assert parser.get_representation("longdress", 3).required_bandwidth_bps > 2**32-1
    assert parser.get_segment_info("longdress", 0, 9.99).frames[-1] == 1350

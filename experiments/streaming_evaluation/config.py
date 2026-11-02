"""Strict experiment configuration and frozen artifact references."""
from __future__ import annotations

import os
from pathlib import Path

from experiments.adaptation_baselines.methods import SUPPORTED_METHODS
from experiments.visibility_weighted_sse.common import digest, finite, file_digest, integer, read_json, require, write_json
from .network import load_trace, CausalBandwidthEstimator, NetworkLink
from .scenarios import generate_scenario
from .timeline import Timeline

SCHEMA = "streaming-evaluation.config.v1"


def load_config(path, *, require_frozen=True):
    path = Path(path).resolve()
    config = read_json(path)
    require(config.get("schema") == SCHEMA, "Unsupported streaming evaluation configuration")
    require(isinstance(config.get("experiment_id"), str) and bool(config["experiment_id"]), "Experiment ID required")
    require(config.get("input_kind") in ("synthetic_fixture", "prepared_experiment"), "Declare input_kind")
    require(config.get("prediction_assumption") == "ORACLE_CAMERA_TRACE", "Perfect viewport assumption must be explicit")
    if require_frozen:
        require(config.get("configuration_hash") == digest({k: v for k, v in config.items() if k != "configuration_hash"}),
                "Configuration is not frozen or hash changed")
    network = config["network"]
    require(not ("path" in network and "trace_file" in network), "Use one network trace path")
    if "trace_file" in network:
        # Normalize the existing loader alias after checking the frozen logical
        # identity. Newly frozen files always store and bind the canonical path.
        network["path"] = network.pop("trace_file")
    timeline = Timeline(**config["timeline"])
    scenario = generate_scenario(config["scenario"], timeline)
    adaptation = config.get("adaptation", {})
    methods = adaptation.get("methods", [])
    require(isinstance(methods, list) and methods and all(m in SUPPORTED_METHODS for m in methods)
            and len(set(methods)) == len(methods), "Choose unique methods from the common interface")
    safety = finite(adaptation.get("safety_factor", .9), "safety_factor")
    require(0 < safety <= 1, "safety_factor must be in (0,1]")
    require(adaptation.get("resource_sharing", "global_sha256") in ("global_sha256", "per_instance"),
            "resource_sharing must be global_sha256 or per_instance")
    require(not adaptation.get("cached_resource_ids"), "Video resource caching is disabled for this evaluation")
    gop_ids = config.get("source_gop_ids")
    require(isinstance(gop_ids, list) and gop_ids and all(isinstance(g, str) and g for g in gop_ids),
            "source_gop_ids must map content GOP indices to source profile IDs")
    count = max(s.source_gop_index for s in Timeline(timeline.source_duration_s, timeline.source_duration_s,
                                                  timeline.gop_duration_s)) + 1
    require(len(gop_ids) == count and len(set(gop_ids)) == count, "Source GOP mapping/duration mismatch")
    playback = config.get("playback", {})
    startup = finite(playback.get("startup_buffer_s", timeline.gop_duration_s), "startup_buffer_s")
    maximum = finite(playback.get("max_buffer_s", 3*timeline.gop_duration_s), "max_buffer_s")
    require(startup >= 0 and maximum >= min(startup, timeline.total_duration_s)
            and maximum >= max(s.duration_s for s in timeline), "Buffer limits cannot admit startup/whole scene GOPs")
    prefix = 0.
    for segment in timeline:
        prefix += segment.duration_s
        if prefix >= min(startup, timeline.total_duration_s) - 1e-10:
            break
    require(prefix <= maximum + 1e-10, "Whole-GOP startup prefix cannot fit within max_buffer_s")
    wall_limit = finite(config.get("max_wall_time_s", 10*timeline.total_duration_s+60), "max_wall_time_s")
    require(wall_limit > 0, "max_wall_time_s must be positive")
    trace = load_trace(config["network"], path.parent)
    require("duration_s" in config["network"] or trace.duration_s is not None,
            "Declare network duration independently of media looping")
    latency = finite(network.get("request_latency_s", 0.), "request_latency_s")
    require(latency >= 0, "request_latency_s must be nonnegative")
    require(network.get('transport','virtual') in ('virtual','real_http2'), 'Unknown transport mode')
    NetworkLink(trace, latency_s=latency, max_concurrent_streams=network.get("max_concurrent_streams", 16))
    CausalBandwidthEstimator(**network.get("estimator", {}))
    quality = config.get("quality", {})
    require(quality.get("mode", "trace_samples") in ("trace_samples", "frame_grid"), "Unsupported quality sampling mode")
    if quality.get("mode") == "frame_grid":
        require(finite(quality.get("fps"), "quality fps") > 0, "quality fps must be positive")
    for name, record in config.get("inputs", {}).items():
        require(isinstance(record, dict) and isinstance(record.get("path"), str), f"Input path required: {name}")
        source = (path.parent / record["path"]).resolve()
        if require_frozen:
            require(source.is_file(), f"Missing input {name}: {source}")
            require(record.get("sha256") == file_digest(source), f"Frozen input hash mismatch: {name}")
    if require_frozen and config["network"].get("path"):
        network_path = (path.parent / config["network"]["path"]).resolve()
        require(config["network"].get("sha256") == file_digest(network_path), "Frozen network trace hash mismatch")
    return config, timeline, scenario, trace


def freeze_config(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    config, _, _, _ = load_config(source, require_frozen=False)
    for name, record in config.get("inputs", {}).items():
        path = (source.parent / record["path"]).resolve()
        require(path.is_file(), f"Missing input {name}: {path}")
        record.update(path=os.path.relpath(path, destination.parent), sha256=file_digest(path))
    network = config["network"]
    if network.get("path"):
        path = (source.parent / network["path"]).resolve()
        network.update(path=os.path.relpath(path, destination.parent), sha256=file_digest(path))
    config.pop("configuration_hash", None)
    config["configuration_hash"] = digest(config)
    write_json(destination, config)
    return config

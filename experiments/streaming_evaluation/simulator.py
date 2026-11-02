"""Shared Phase A scheduling: reproducible virtual control or real HTTP/2 IO."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
import time
from pathlib import Path

from experiments.adaptation_baselines.methods import construct_utilities
from experiments.visibility_weighted_sse.allocation import allocate_utilities
from experiments.visibility_weighted_sse.common import digest, require, write_json
from .config import load_config
from .inputs import build_planner
from .metrics import summarize_metrics
from .network import NetworkLink, CausalBandwidthEstimator
from .playback import PlaybackBuffer
from .quality import CachedQualityProvider, collect_quality, contexts_for_playback


def validate(config_path):
    """Check preparation only; never run a bandwidth/playback evaluation."""
    config, timeline, scenario, _ = load_config(config_path, require_frozen=False)
    base = Path(config_path).resolve().parent
    required = ("adaptation_config", "visibility_cache")
    missing = [name for name in required if name not in config.get("inputs", {})]
    missing += [name for name, record in config.get("inputs", {}).items()
                if not (base / record["path"]).is_file()]
    if missing:
        return {"status": "geometry_ready_inputs_missing", "missing_inputs": sorted(set(missing)),
                "scene_id": scenario["scene_id"], "object_count": scenario["object_count"],
                "segment_count": len(timeline), "experimental_measurements_executed": False}
    config, timeline, scenario, _ = load_config(config_path)
    planner, _ = build_planner(config, config_path, timeline, scenario)
    placeholder = []
    for segment in timeline:
        poses, aggregate, _ = planner.prepare(segment.gop_id)
        for method in config["adaptation"]["methods"]:
            construct_utilities(method, planner.profiles, segment.gop_id, poses, aggregate, planner.metadata)
        placeholder.append({"global_segment_index": segment.global_segment_index,
                            "selected_quality": {row.instance_id: row.qualities[0].quality_id
                                                 for row in planner.profiles.for_gop(segment.gop_id)}})
    quality_config = config.get("quality", {})
    contexts_for_playback(scene_id=scenario["scene_id"], content_revision=planner.profiles.content_revision,
        trace_provider=planner.trace, timeline=timeline, decisions=placeholder,
        played_duration_s=timeline.total_duration_s, source_gop_ids=config["source_gop_ids"],
        mode=quality_config.get("mode", "trace_samples"), fps=quality_config.get("fps"))
    if "quality_measurements" in config.get("inputs", {}):
        CachedQualityProvider.from_json(base / config["inputs"]["quality_measurements"]["path"])
    return {"status": "ready_for_future_execution", "scene_id": scenario["scene_id"],
            "object_count": scenario["object_count"], "segment_count": len(timeline),
            "methods": config["adaptation"]["methods"], "experimental_measurements_executed": False}


def _q0_fallback(decision, planner, gop_id):
    """Build delivery resources with the existing closure/cost adapter.

    Keep the original infeasible solver result intact. The temporary base-sized
    allocation only recovers Q0 resource unions; it does not certify feasibility
    under the decision's estimated budget. No media residency is granted.
    """
    started = time.perf_counter()
    original = decision["allocation"]
    rows = planner.profiles.for_gop(gop_id)
    delivery = allocate_utilities(planner.profiles, gop_id, original["base_bytes"],
        decision["utilities"], base_only_instance_ids=tuple(row.instance_id for row in rows),
        allowed_quality_ids=planner.allowed_quality_ids, solver_limits=getattr(planner,'solver_limits',None))
    require(delivery["feasible"], "Q0 physical resource construction failed")
    native_ms = delivery["diagnostics"].pop("solver_elapsed_ms")
    delivery["diagnostics"].pop("allocation_adapter_elapsed_ms")
    fallback_solver_wall_ms = delivery["diagnostics"].pop("solver_wall_ms")
    fallback_peak_rss_kib = delivery["diagnostics"].pop("process_peak_rss_kib")
    delivery.update(status="q0_fallback", solver_status="not_a_solver_solution",
                    feasible=False, budget_feasible=False,
                    available_budget_bytes=decision["budget_bytes"])
    decision.update(delivery_policy="q0_fallback", delivery_allocation=delivery,
        budget_overrun_bytes=delivery["total_selected_bytes"]-decision["budget_bytes"],
        selected_quality={r["instance_id"]: r["selected_quality"] for r in delivery["allocations"]},
        estimated_utility_per_object={r["instance_id"]: r["weighted_utility"] for r in delivery["allocations"]},
        estimated_total_utility=delivery["selected_utility"],
        total_required_bytes=delivery["total_selected_bytes"])
    decision["timings"].update(q0_fallback_ms=(time.perf_counter()-started)*1000,
                              q0_fallback_native_solver_ms=native_ms,
                              q0_fallback_solver_wall_ms=fallback_solver_wall_ms,
                              q0_fallback_peak_rss_kib=fallback_peak_rss_kib)
    return delivery


def simulate(config, timeline, scenario, network_trace, planner, method, *, quality_provider=None,
             input_timings=None):
    """Use the same planner/buffer/request path for either explicit transport."""
    network = config["network"]
    mode = network.get("transport", "virtual")
    require(mode in ("virtual", "real_http2"), "Unknown transport mode")
    preparation_ms = 0.
    if network.get("prepare_visibility_before_clock", False):
        started = time.perf_counter()
        for segment in timeline:
            planner.prepare(segment.gop_id)
        preparation_ms = (time.perf_counter()-started)*1000
    if mode == "real_http2":
        from .real_http2 import RealHTTP2Link
        link = RealHTTP2Link(network_trace,
            resource_paths={r.sha256:Path(r.url) for r in planner.profiles.resources.values()},
            latency_s=network.get("request_latency_s", .02),
            max_concurrent_streams=network.get("max_concurrent_streams",16))
    else:
        link = NetworkLink(network_trace, latency_s=network.get("request_latency_s", 0.),
                           max_concurrent_streams=network.get("max_concurrent_streams", 16))
    try:
        simulation, performance = _simulate(config, timeline, scenario, network_trace, planner, method,
            link=link, quality_provider=quality_provider, input_timings=input_timings)
        performance["visibility_preparation_before_transport_ms"] = preparation_ms
        performance["online_clock_policy"] = ("actual_wall_clock_includes_online_planning_and_IO" if mode=="real_http2"
                                              else "virtual_control_excludes_host_computation")
        return simulation, performance
    finally:
        if mode == "real_http2":
            link.close()


def _simulate(config, timeline, scenario, network_trace, planner, method, *, link, quality_provider=None,
              input_timings=None):
    """Execute one policy against a fresh network/playback state, retaining loops.

    Scene GOPs are requested in order; resources within each GOP multiplex over
    one shared connection. Every selected resource is queued
    anew; only identical physical resources within this scene GOP are deduped.
    All objects must finish before the GOP becomes playable. Solver/CPU timings
    are recorded but do not alter virtual time or future network predictions.
    """
    require(method in config["adaptation"]["methods"], "Method not selected by configuration")
    require(not planner.cached_resource_ids, "Cross-segment video cache credit is forbidden")
    network = config["network"]
    real_clock = network.get("transport", "virtual") == "real_http2"
    estimator = CausalBandwidthEstimator(**network.get("estimator", {}))
    playback_config = config.get("playback", {})
    player = PlaybackBuffer(timeline.total_duration_s,
                            playback_config.get("startup_buffer_s", timeline.gop_duration_s))
    max_buffer = playback_config.get("max_buffer_s", 3*timeline.gop_duration_s)
    wall_limit = config.get("max_wall_time_s", 10*timeline.total_duration_s+60)
    safety = config["adaptation"].get("safety_factor", .9)
    by_hash = {r.sha256: r for r in planner.profiles.resources.values()}
    aliases = planner.profiles.provenance.get('resource_transfer_aliases', {})
    decisions, requests, downloads, completed_decisions = [], [], [], []
    status = "completed"
    for segment in timeline:
        # Rates passed to the production DP are bits per *actual source GOP*
        # duration, including a short final source GOP. Never assume two seconds
        # in the generic experiment interface.
        planner.solver_limits = {**(planner.solver_limits or {}),
                                 "gop_duration_seconds": segment.source_segment_duration_s}
        if real_clock:
            player.advance_to(link.now)
        # A bounded, one-GOP-at-a-time FIFO scheduler. Waiting consumes playback
        # and advances network wall time; it never resets at a content loop.
        wait = max(0., player.buffer_s + segment.duration_s - max_buffer)
        if wait:
            require(player.started, "Startup GOP prefix cannot fit within max_buffer_s")
            target = min(link.now + wait, wall_limit)
            player.advance_to(target)
            link.advance_to(target)
        if link.now >= wall_limit:
            status = "time_limit"
            break
        snapshot = estimator.snapshot(link.now)
        # Upper-bound request count across eligible representations without
        # changing utilities/solver. Reserve connection idle latency once, then
        # spend the remaining safe deadline on shared body capacity.
        max_requests = sum(max(len({planner.profiles.resources[r].sha256 for r in q.resource_ids})
                               for q in row.qualities)
                           for row in planner.profiles.for_gop(segment.gop_id))
        latency_reserve = link.latency_reserve(max_requests)
        safe_duration = segment.source_segment_duration_s * safety
        budget = max(0, math.floor(snapshot["body_bps"] * max(0., safe_duration - latency_reserve) / 8))
        decision = planner.plan(method, segment.gop_id, budget)
        decision.update(global_segment_index=segment.global_segment_index, loop_index=segment.loop_index,
                        global_start_s=segment.global_start_s, global_end_s=segment.global_end_s,
                        source_gop_id=config["source_gop_ids"][segment.source_gop_index],
                        content_start_s=segment.content_start_s, duration_s=segment.duration_s,
                        request_group_id=segment.request_id, decision_wall_s=link.now,
                        estimated_bandwidth_bps=snapshot["body_bps"], bandwidth_estimate=deepcopy(snapshot),
                        budget_model="body_capacity_after_concurrent_latency_reserve",
                        request_count_upper_bound=max_requests, latency_reserve_s=latency_reserve,
                        safe_download_duration_s=safe_duration)
        decision["timings"].update(input_timings or {})
        decisions.append(decision)
        if decision["allocation"]["status"] == "infeasible_base":
            allocation = _q0_fallback(decision, planner, segment.gop_id)
        elif decision["selected_quality"] is None:
            status = "adaptation_" + decision["allocation"]["status"]
            break
        else:
            allocation = decision["allocation"]
            decision.update(delivery_policy="solver_selection", budget_overrun_bytes=0)
        selected = allocation["selected_resource_sha256"]
        require(sum(by_hash[key].bytes for key in selected) == decision["total_required_bytes"],
                "Physical requests disagree with selected allocation")
        require(decision["total_required_bytes"] <= budget or
                (decision["delivery_policy"] == "q0_fallback" and not allocation["budget_feasible"]
                 and decision["budget_overrun_bytes"] == decision["total_required_bytes"]-budget > 0),
                "Unrecorded physical budget overrun")
        base_hashes = {planner.profiles.resources[r].sha256
                       for row in planner.profiles.for_gop(segment.gop_id)
                       for r in row.qualities[0].resource_ids}
        pending = []
        for key in sorted(selected, key=lambda key: (key not in base_hashes, key)):
            pending.append({"request_id": segment.request_id + "/resource-" + key,
                "global_segment_index": segment.global_segment_index, "loop_index": segment.loop_index,
                "source_gop_id": decision["source_gop_id"], "resource_sha256": key,
                **({"instance_id": aliases[key]['instance_id'],
                    "source_resource_sha256": aliases[key]['source_sha256'],
                    "transfer_identity_sha256": key} if key in aliases else {}),
                "requested_bytes": by_hash[key].bytes, "delivered_bytes": 0.,
                "queued_s": link.now, "start_s": None, "ready_s": None, "end_s": None,
                "completed_at_s": None, "status": "queued",
                "kind": ("base" if key in base_hashes else "enhancement")
                        if planner.profiles.representation_family == "progressive" else "media"})
        requests.extend(pending)
        expected_duration = link.expected_duration([r["requested_bytes"] for r in pending], snapshot["body_bps"])
        decision.update(expected_download_duration_s=expected_duration,
            expected_download_completed_at_s=link.now + expected_duration,
            expected_deadline_feasible=expected_duration <= safe_duration + 1e-9)
        require(decision["delivery_policy"] == "q0_fallback" or decision["expected_deadline_feasible"],
                "Byte budget admitted a selection beyond the scheduling model's safe deadline")
        batch_start = link.now
        group_results = link.download_batch(pending, deadline_s=wall_limit)
        complete = all(result.status == "complete" for result in group_results)
        for request, result in zip(pending, group_results):
            request.update(result.to_dict())
        # Input order differs from completion order under multiplexing.
        for result in sorted(group_results, key=lambda r: (r.end_s, r.request_id)):
            player.advance_to(result.end_s)
        player.advance_to(link.now)
        if not complete:
            status = "network_" + link.failure_status
        downloads.append({"global_segment_index": segment.global_segment_index,
            "loop_index": segment.loop_index, "request_group_id": segment.request_id,
            "requested_bytes": decision["total_required_bytes"],
            "delivered_bytes": math.fsum(row["delivered_bytes"] for row in pending),
            "base_bytes": allocation["base_bytes"], "enhancement_bytes": allocation["enhancement_bytes"],
            "completed": complete, "duration_s": segment.duration_s,
            "download_start_s": batch_start, "download_end_s": link.now,
            "download_duration_s": link.now - batch_start,
            "expected_download_duration_s": expected_duration,
            "full_source_segment_duration_s": segment.source_segment_duration_s,
            "global_start_s": segment.global_start_s})
        if not complete:
            break
        estimator.observe_batch(group_results, link.now)
        player.append(segment.global_start_s, segment.duration_s, link.now)
        completed_decisions.append(decision)
    # Drain only delivered data, and respect the explicit simulation wall bound.
    if player.started and player.buffer_s > 0:
        drain_target = min(player.now + player.buffer_s, wall_limit)
        if real_clock:
            link.advance_to(drain_target)
            player.advance_to(min(link.now, wall_limit))
        else:
            player.advance_to(drain_target)
    if status == "completed" and player.completion_wall_s is None:
        status = "time_limit" if player.now >= wall_limit else "incomplete_playback"
    quality_config = config.get("quality", {})
    contexts = contexts_for_playback(scene_id=scenario["scene_id"],
        content_revision=planner.profiles.content_revision, trace_provider=planner.trace,
        timeline=timeline, decisions=completed_decisions, played_duration_s=player.position_s,
        source_gop_ids=config["source_gop_ids"], mode=quality_config.get("mode", "trace_samples"),
        fps=quality_config.get("fps"))
    quality = collect_quality(contexts, quality_provider)
    quality["summary"].update(sampling_mode=quality_config.get("mode", "trace_samples"),
                              full_frame_grid=quality_config.get("mode") == "frame_grid")
    transfer_aliases = dict(planner.profiles.provenance.get("resource_transfer_aliases", {}))
    simulation = {"schema": "streaming-evaluation.simulation.v1", "experiment_id": config["experiment_id"],
        "configuration_hash": config.get("configuration_hash"), "input_kind": config["input_kind"],
        "method": method, "status": status, "scene_id": scenario["scene_id"],
        "prediction_assumption": "ORACLE_CAMERA_TRACE", "future_bandwidth_available_to_policy": False,
        "resource_sharing_policy": planner.profiles.provenance.get("resource_transfer_policy", "global_sha256"),
        "transfer_resource_count": len(transfer_aliases),
        "transfer_map_sha256": digest(transfer_aliases) if transfer_aliases else None,
        "cache_policy": "no_video_residency_fresh_resource_requests_each_global_segment",
        "scheduling_policy": ("real_http2_bounded_streams_scene_GOP_barrier" if real_clock else
                              "virtual_http2_multiplexed_FIFO_stream_admission_scene_GOP_barrier"),
        "transport_model": (link.transport_metadata() if real_clock else
                            {"mode":"virtual", "protocol_negotiated": False, "max_concurrent_streams": link.max_concurrent_streams,
                            "request_latency_s": link.latency_s, "body_sharing": "equal_ready_streams",
                            "flow_control": "nonbinding_windows", "connection_count": 1}),
        "request_accounting": "all_resources_queued_at_GOP_decision_including_cancelled_before_service",
        "total_duration_s": timeline.total_duration_s, "source_duration_s": timeline.source_duration_s,
        "timeline": timeline.to_dict(), "requests": requests, "segment_downloads": downloads,
        "decisions": decisions, "bandwidth_estimator_samples": estimator.samples,
        "network_service_ledger": link.ledger,
        "playback": player.summary(), "buffer_samples": player.snapshots,
        "buffer_intervals": player.intervals, "playback_events": player.events,
        "stalls": player.stalls, "end_wall_s": player.now,
        "available_capacity_bytes": link.capacity_between(0., player.now), "quality_frames": quality["frames"]}
    metrics = summarize_metrics(simulation, quality)
    performance = metrics.pop("performance")
    performance["input_timings"] = input_timings
    performance["records"] = [{"global_segment_index": row["global_segment_index"], **row["timings"]}
                              for row in decisions]
    for index, decision in enumerate(decisions):
        decision.pop("timings")
        decision["performance_reference"] = {"file": "performance.json", "record": index}
    simulation["metrics"] = metrics
    return simulation, performance


def run(config_path, output, *, method=None):
    """Future experiment entry point. Do not invoke for Task-2 measurements."""
    config, timeline, scenario, trace = load_config(config_path)
    selected = [method] if method is not None else config["adaptation"]["methods"]
    require(all(m in config["adaptation"]["methods"] for m in selected), "Method not enabled by configuration")
    base = Path(config_path).resolve().parent
    quality_provider = None
    if "quality_measurements" in config.get("inputs", {}):
        quality_provider = CachedQualityProvider.from_json(base / config["inputs"]["quality_measurements"]["path"])
    summaries = []
    transfer_map_written = False
    for policy in selected:
        # Recreate each policy's planner and clocks from identical inputs; no
        # state leaks between policies or policy-dependent trace scaling.
        planner, timing = build_planner(config, config_path, timeline, scenario)
        if config.get("quality", {}).get("mode") == "frame_grid":
            # Fail before any virtual transfer if a claimed full-frame protocol
            # lacks exact camera samples. The real allocation remains unknown
            # until the causal decisions; only geometry coverage is checked.
            placeholders = [{"global_segment_index": s.global_segment_index,
                             "selected_quality": {r.instance_id: r.qualities[0].quality_id
                                                  for r in planner.profiles.for_gop(s.gop_id)}}
                            for s in timeline]
            contexts_for_playback(scene_id=scenario["scene_id"], content_revision=planner.profiles.content_revision,
                trace_provider=planner.trace, timeline=timeline, decisions=placeholders,
                played_duration_s=timeline.total_duration_s, source_gop_ids=config["source_gop_ids"],
                mode="frame_grid", fps=config["quality"]["fps"])
        solver_init_ms = planner.initialize_solver()
        simulation, performance = simulate(config, timeline, scenario, trace, planner, policy,
                                           quality_provider=quality_provider, input_timings=timing)
        aliases = planner.profiles.provenance.get("resource_transfer_aliases", {})
        if aliases and not transfer_map_written:
            write_json(Path(output) / "resource_transfer_map.json", aliases)
            transfer_map_written = True
        performance["shared_solver_initialization_ms"] = solver_init_ms
        destination = Path(output) / policy
        write_json(destination / "simulation.json", simulation)
        write_json(destination / "metrics.json", simulation["metrics"])
        write_json(destination / "performance.json", performance)
        for name, records in (("requests", simulation["requests"]),
                              ("buffer", simulation["buffer_samples"]),
                              ("decisions", simulation["metrics"]["adaptation"]["selected_quality"]),
                              ("frames", simulation["quality_frames"])):
            _csv(destination / (name + ".csv"), records)
        summaries.append({"method": policy, "status": simulation["status"], "metrics": simulation["metrics"]})
    write_json(Path(output) / "comparison.json", {"configuration_hash": config["configuration_hash"],
        "methods": summaries, "timings_are_observations_not_simulated_delay": True})
    return summaries


def _csv(path, records):
    if not records:
        path.write_text("")
        return
    fields = sorted(set().union(*(row.keys() for row in records)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})

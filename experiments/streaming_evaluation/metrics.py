"""Pure simulator accounting and JSON/CSV-ready comparison metrics.

Timing observations are returned in ``performance`` for a separate sidecar;
they never change simulated wall time or deterministic adaptation artifacts.
No utility or offline object PSNR enters the image-quality measurements here.
"""
from __future__ import annotations

import math
import statistics
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from experiments.abr_network_evaluation.predictor import busy_interval_seconds


def _number(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _get(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    return next((mapping[name] for name in names if name in mapping), default)


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    values = list(values)
    return {"count": len(values), "total_ms": math.fsum(values) if values else None,
            "mean_ms": statistics.fmean(values) if values else None,
            "median_ms": statistics.median(values) if values else None,
            "minimum_ms": min(values) if values else None,
            "maximum_ms": max(values) if values else None}


def _switches(decisions: list[Mapping[str, Any]]) -> dict[str, Any]:
    previous: dict[str, Any] = {}
    per_object: dict[str, int] = {}
    transitions = 0
    for decision in sorted(decisions, key=lambda row: row["global_segment_index"]):
        selected = decision.get("selected_quality")
        if selected is None:
            continue
        transitions += bool(previous)
        for instance, quality in sorted(selected.items()):
            per_object.setdefault(instance, 0)
            if instance in previous and previous[instance] != quality:
                per_object[instance] += 1
        previous = dict(selected)
    return {"switch_count": sum(per_object.values()), "switch_count_per_object": per_object,
            "segment_transition_count": transitions,
            "definition": "quality_identifier_change_per_object_between_adjacent_available_segments"}


def _buffer_summary(simulation: Mapping[str, Any]) -> dict[str, Any]:
    samples = simulation.get("buffer_samples", [])
    occupancies = [_number(row["buffer_s"], "buffer_s") for row in samples]
    intervals = simulation.get("buffer_intervals", [])
    duration, integral = 0.0, 0.0
    for interval in intervals:
        start, end = _number(interval["start_s"], "start_s"), _number(interval["end_s"], "end_s")
        if end < start:
            raise ValueError("Buffer interval end precedes start")
        before = _number(interval["buffer_start_s"], "buffer_start_s")
        after = _number(interval["buffer_end_s"], "buffer_end_s")
        duration += end - start
        integral += (before + after) * .5 * (end - start)
    return {"sample_count": len(samples), "minimum_s": min(occupancies) if occupancies else None,
            "maximum_s": max(occupancies) if occupancies else None,
            "time_weighted_mean_s": integral / duration if duration else None,
            "time_weighted_duration_s": duration,
            "mean_definition": "exact_integral_of_piecewise_linear_event_intervals" if duration else None}


def summarize_metrics(simulation: Mapping[str, Any], quality: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Summarize one deterministic simulation result, with optional real quality.

    Request rows are the physical bytes, including repeated video loops.  A
    standalone representation's Q0 reservation is an algebraic cost baseline,
    not a separately transmitted layer. Its requested/delivered decomposition
    is labeled accordingly; physical layer bytes stay null unless all request
    rows explicitly declare base/enhancement packet kinds.
    """
    requests = simulation.get("requests", [])
    segments = simulation.get("segment_downloads", [])
    decisions = simulation.get("decisions", [])
    playback = simulation.get("playback", {})
    request_ids: set[str] = set()
    requested = delivered = 0.0
    physical_base = physical_enhancement = 0.0
    kinds_known = True
    for row in requests:
        request_id = str(row["request_id"])
        if request_id in request_ids:
            raise ValueError("Duplicate network request_id; repeated loops need fresh requests")
        request_ids.add(request_id)
        cost = _number(row["requested_bytes"], "requested_bytes")
        received = _number(row["delivered_bytes"], "delivered_bytes")
        if received > cost + max(1e-9, cost * 1e-12):
            raise ValueError("Delivered bytes exceed the request size")
        requested += cost
        delivered += received
        if row.get("kind") == "base":
            physical_base += received
        elif row.get("kind") == "enhancement":
            physical_enhancement += received
        else:
            kinds_known = False
    wall = _get(simulation, "end_wall_s", default=_get(playback, "wall_time_s", "completion_time_s", "completion_wall_time_s", default=0.))
    wall = _number(wall, "end_wall_s")
    capacity = simulation.get("available_capacity_bytes")
    if capacity is not None:
        capacity = _number(capacity, "available_capacity_bytes")
        if delivered > capacity + max(1e-7, capacity * 1e-10):
            raise ValueError("Delivered bytes exceed integrated network capacity")
    base_requested = enhancement_requested = base_delivered = enhancement_delivered = 0.0
    decomposition_available = bool(segments)
    completed_segment_indices: set[int] = set()
    for row in segments:
        total = _number(row["requested_bytes"], "segment requested_bytes")
        received = _number(row["delivered_bytes"], "segment delivered_bytes")
        if received > total + max(1e-9, total * 1e-12):
            raise ValueError("Segment delivered bytes exceed the selected size")
        if row.get("completed", False):
            completed_segment_indices.add(row["global_segment_index"])
        if row.get("base_bytes") is None or row.get("enhancement_bytes") is None:
            decomposition_available = False
            continue
        base = _number(row["base_bytes"], "base_bytes")
        increment = _number(row["enhancement_bytes"], "enhancement_bytes", nonnegative=False)
        if not math.isclose(base + increment, total, rel_tol=1e-12, abs_tol=1e-7):
            raise ValueError("Base reservation plus signed enhancement increment must match selected bytes")
        fraction = received / total if total else float(bool(row.get("completed")))
        base_requested += base
        enhancement_requested += increment
        base_delivered += base * fraction
        enhancement_delivered += increment * fraction
    if segments and (not math.isclose(math.fsum(float(s["requested_bytes"]) for s in segments), requested, rel_tol=1e-12, abs_tol=1e-7)
                     or not math.isclose(math.fsum(float(s["delivered_bytes"]) for s in segments), delivered, rel_tol=1e-12, abs_tol=1e-7)):
        raise ValueError("Physical requests and segment download accounting disagree")
    completed_decisions = [row for row in decisions if row["global_segment_index"] in completed_segment_indices]
    played_position = _number(_get(playback, "played_duration_s", "position_s", default=0.), "played_duration_s")
    have_media_starts = all("global_start_s" in row or "global_media_start_s" in row for row in completed_decisions)
    played_decisions = [row for row in completed_decisions
                        if _get(row, "global_start_s", "global_media_start_s", default=math.inf) < played_position]
    selected_rows, budget_violations, infeasible_count = [], 0, 0
    fallback_count, overrun_bytes = 0, 0
    for row in decisions:
        selected = row.get("selected_quality")
        used, budget = row.get("total_required_bytes"), row.get("budget_bytes")
        feasible = selected is not None and used is not None
        if feasible:
            if budget is None:
                raise ValueError("Feasible decision requires its byte budget")
            used, budget = _number(used, "total_required_bytes"), _number(budget, "budget_bytes")
            budget_violations += used > budget
        if not row.get("allocation", {}).get("feasible", feasible):
            infeasible_count += 1
        fallback_count += row.get("delivery_policy") == "q0_fallback"
        overrun = max(0, used-budget) if feasible else 0
        overrun_bytes += overrun
        selected_rows.append({"global_segment_index": row["global_segment_index"],
                              "loop_index": row.get("loop_index"), "source_gop_id": row.get("source_gop_id"),
                              "selected_quality": selected, "total_required_bytes": used,
                              "budget_bytes": budget, "budget_feasible": feasible and used <= budget,
                              "solver_feasible": row.get("allocation", {}).get("feasible", feasible),
                              "solver_status": row.get("allocation", {}).get("status"),
                              "native_dp_cells_max": row.get("allocation", {}).get("diagnostics", {}).get("native_dp_cells_max"),
                              "delta_bps": row.get("allocation", {}).get("diagnostics", {}).get("delta_bps"),
                              "delivery_policy": row.get("delivery_policy"), "budget_overrun_bytes": overrun,
                              "download_completed": row["global_segment_index"] in completed_segment_indices})
    timing_names = ("decision_call_ms", "visibility_analysis_ms", "native_solver_ms", "utility_construction_ms",
                    "visibility_native_pipeline_ms", "visibility_cache_lookup_ms",
                    "visibility_original_analysis_ms", "decision_compute_excluding_preparation_ms",
                    "allocation_ms", "allocation_adapter_ms", "shared_GOP_preparation_ms",
                    "q0_fallback_ms", "q0_fallback_native_solver_ms", "solver_wall_ms", "q0_fallback_solver_wall_ms")
    timing_rows, timing_values = [], {name: [] for name in timing_names}
    for decision in decisions:
        timing = decision.get("timings", {})
        record = {"global_segment_index": decision["global_segment_index"]}
        for name in timing_names:
            value = timing.get(name)
            record[name] = None if value is None else _number(value, name)
            if value is not None:
                timing_values[name].append(record[name])
        record["shared_GOP_preparation_reused"] = timing.get("shared_GOP_preparation_reused")
        timing_rows.append(record)
    quality_summary = dict(quality.get("summary", quality)) if quality is not None else {
        "quality_source": "actual_joint_viewport_measurements", "average_frame_psnr_db": None,
        "pooled_psnr_db": None, "measurement_coverage": None,
        "unavailable_reason": "No actual joint viewport frame measurements supplied"}
    network_timing = {}
    if "network_service_ledger" in simulation:
        ledger = simulation["network_service_ledger"]
        real_transport = simulation.get("transport_model", {}).get("mode") == "real_http2"
        body_time = (busy_interval_seconds([SimpleNamespace(start_s=r["ready_s"], end_s=r["end_s"])
                    for r in requests if r.get("ready_s") is not None and r["ready_s"] <= r["end_s"]])
                    if real_transport else
                    math.fsum(r["end_s"] - r["start_s"] for r in ledger if r["active_bodies"] > 0))
        request_time = busy_interval_seconds([SimpleNamespace(start_s=r["start_s"], end_s=r["end_s"])
                                             for r in requests if r.get("start_s") is not None])
        network_timing = dict(body_busy_duration_s=body_time, request_busy_duration_s=request_time,
            body_throughput_bps=delivered * 8 / body_time if body_time else None,
            application_goodput_bps=delivered * 8 / request_time if request_time else None,
            timing_definition="shared body/request interval unions; excludes buffer waits and final drain")
    return {
        "status": simulation.get("status"),
        "bandwidth": {**network_timing, "request_count": len(requests), "total_requested_bytes": requested,
                      "total_delivered_bytes": delivered, "observation_wall_duration_s": wall,
                      "actual_consumption_bps": delivered * 8 / wall if wall else None,
                      "available_capacity_bytes": capacity,
                      "bandwidth_utilization": delivered / capacity if capacity else None,
                      "base_requested_bytes": base_requested if decomposition_available else None,
                      "enhancement_requested_bytes": enhancement_requested if decomposition_available else None,
                      "base_delivered_accounting_bytes": base_delivered if decomposition_available else None,
                      "enhancement_delivered_accounting_bytes": enhancement_delivered if decomposition_available else None,
                      "base_enhancement_accounting": "algebraic_Q0_reservation_plus_signed_selected_increment_standalone_not_separate_layer_packets",
                      "partial_delivery_accounting": "prorated_by_segment_delivered_fraction",
                      "physical_base_delivered_bytes": physical_base if kinds_known and requests else None,
                      "physical_enhancement_delivered_bytes": physical_enhancement if kinds_known and requests else None},
        "playback": {"startup_delay_s": playback.get("startup_delay_s"),
                     "stall_count": playback.get("stall_count", 0),
                     "total_rebuffering_s": _get(playback, "total_rebuffering_s", "rebuffer_duration_s"),
                     "completion_wall_time_s": _get(playback, "completion_wall_time_s", "completion_time_s"),
                     "played_duration_s": played_position,
                     "media_duration_s": _get(playback, "media_duration_s", default=simulation.get("total_duration_s")),
                     "final_buffer_s": playback.get("buffer_s"), "buffer": _buffer_summary(simulation)},
        "adaptation": {"decision_count": len(decisions), "infeasible_decision_count": infeasible_count,
                       "q0_fallback_count": fallback_count, "total_budget_overrun_bytes": overrun_bytes,
                       "budget_violation_count": budget_violations, "all_selected_budgets_feasible": budget_violations == 0,
                       "selected_quality": selected_rows, "decision_switching": _switches(decisions),
                       "completed_download_switching": _switches(completed_decisions),
                       "played_quality_switching": _switches(played_decisions) if have_media_starts else None},
        "quality": quality_summary,
        "performance": {"information_role": ("REAL_WALL_CLOCK_ONLINE_COMPUTATION_AND_HTTP_IO" if
                        simulation.get("transport_model",{}).get("mode")=="real_http2" else
                        "OBSERVED_RUNTIME_ONLY_NOT_SIMULATED_TIME"),
                        "timings": timing_rows,
                        "summary": {name: _distribution(values) for name, values in timing_values.items()},
                        "solver_resources": [{"global_segment_index": d["global_segment_index"],
                            "solver_status": d.get("allocation",{}).get("status"),
                            "solver_wall_ms": d.get("timings",{}).get("solver_wall_ms"),
                            "process_peak_rss_kib": d.get("timings",{}).get("process_peak_rss_kib"),
                            **{key:d.get("allocation",{}).get("diagnostics",{}).get(key) for key in
                               ("discretization", "delta_bps", "native_dp_cells_max", "native_dp_cells_total",
                                "native_requested_dp_cells_max", "native_estimated_working_set_bytes_max", "exact_byte_feasible",
                                "process_peak_rss_scope")}}
                            for d in decisions],
                        "visibility_latency_note": "shared_GOP_preparation_ms includes trace/profile/visibility work; it is not a pure visibility timing"},
    }

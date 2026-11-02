"""Synthetic correctness fixtures only; these are not experimental measurements."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest

from experiments.nava_mse.metrics import MetricStats
from experiments.streaming_evaluation.quality import (
    CachedQualityProvider, FrameQualityContext, FrameQualityMeasurement,
    collect_quality, contexts_for_playback,
)
from experiments.streaming_evaluation.metrics import summarize_metrics
from experiments.streaming_evaluation.timeline import Timeline


def context(**kwargs):
    defaults = dict(scene_id="synthetic-scene", content_revision="synthetic-v1",
                    trace_id="exact-trace-v1", camera_sample_id="frame0",
                    global_segment_id="s0", content_gop_id="g0", loop_index=0,
                    global_time_s=0, content_time_s=0, selected_quality={"obj0": 40},
                    frame_duration_s=.1)
    return FrameQualityContext(**{**defaults, **kwargs})


def cache(frames):
    return dict(schema_version=1, information_role="ACTUAL_VIEWPORT_FRAME_MEASUREMENTS",
                measurement_kind="joint_viewport_rgb", reference_identity="synthetic-reference",
                protocol=dict(mask_mode="full_viewport", channels=3, data_range=1., normalization="raw"),
                frames=[dict(context=c.to_dict(), metrics=m.to_dict(),
                             provenance=dict(reference_frame_id="reference:"+c.camera_sample_id,
                                             candidate_frame_id="candidate:"+c.camera_sample_id,
                                             measurement_source="synthetic_test_fixture")) for c, m in frames])


def test_absent_measurements_are_null_and_not_policy_utility():
    result = collect_quality([context()])
    assert result["frames"][0]["psnr_db"] is None
    summary = result["summary"]
    assert summary["measured_frame_count"] == 0
    assert summary["unavailable_frame_count"] == 1
    assert summary["measurement_coverage"] == 0
    assert summary["pooled_psnr_db"] is None
    assert summary["pooled_sse"] is None
    assert not summary["pooled_psnr_infinite"]
    json.dumps(result, allow_nan=False)


def test_empty_quality_has_no_invented_zero_psnr():
    summary = collect_quality([])["summary"]
    assert summary["average_frame_psnr_db"] is None
    assert summary["measurement_coverage"] is None
    assert summary["duration_coverage"] is None


def test_average_pooled_and_duration_mean_are_different_and_correct():
    first = context()
    second = context(camera_sample_id="frame1", global_time_s=.1, content_time_s=.1, frame_duration_s=.3)
    # RGB scalar counts 3 and 9. MSE .01 and .1 give PSNR 20 and 10.
    provider = CachedQualityProvider(cache([(first, MetricStats(.03, 1)), (second, MetricStats(.9, 3))]))
    result = collect_quality([first, second], provider)
    summary = result["summary"]
    assert summary["average_frame_psnr_db"] == pytest.approx(15)
    assert summary["duration_weighted_psnr_db"] == pytest.approx(12.5)
    assert summary["pooled_sse"] == pytest.approx(.93)
    assert summary["pooled_rgb_sample_count"] == 12
    assert summary["pooled_psnr_db"] == pytest.approx(10 * math.log10(12/.93))
    assert summary["measurement_coverage"] == 1
    assert result["frames"][1]["global_time_s"] == .1
    assert result["frames"][1]["rgb_sample_count"] == 9


def test_perfect_frames_preserve_infinity_as_json_safe_flags():
    first, second = context(), context(camera_sample_id="f1", global_time_s=.1, content_time_s=.1)
    provider = CachedQualityProvider(cache([(first, MetricStats(0, 1)), (second, MetricStats(.3, 1))]))
    result = collect_quality([first, second], provider)
    assert result["frames"][0]["psnr_db"] is None
    assert result["frames"][0]["psnr_infinite"]
    assert result["summary"]["average_frame_psnr_infinite"]
    assert result["summary"]["duration_weighted_psnr_infinite"]
    assert not result["summary"]["pooled_psnr_infinite"]
    json.dumps(result, allow_nan=False)
    perfect = collect_quality([first], provider)["summary"]
    assert perfect["pooled_psnr_infinite"]
    assert perfect["pooled_psnr_db"] is None


@pytest.mark.parametrize("changes", [
    {"scene_id": "different-scene"}, {"content_revision": "different-content"},
    {"trace_id": "different-trace"}, {"camera_sample_id": "different-pose"},
    {"global_segment_id": "loop1:g0"}, {"content_gop_id": "g1"},
    {"loop_index": 1}, {"global_time_s": 10}, {"content_time_s": 1},
    {"selected_quality": {"obj0": 20}}, {"frame_duration_s": .2},
])
def test_cache_requires_exact_frame_allocation_loop_and_provenance(changes):
    original = context()
    provider = CachedQualityProvider(cache([(original, MetricStats(.3, 1))]))
    assert provider.measure(original) is not None
    assert provider.measure(replace(original, **changes)) is None


def test_partial_coverage_labels_available_only_averages():
    first = context()
    second = context(camera_sample_id="frame1", global_time_s=.1, content_time_s=.1, frame_duration_s=.3)
    provider = CachedQualityProvider(cache([(first, MetricStats(.03, 1))]))
    summary = collect_quality([first, second], provider)["summary"]
    assert summary["measurement_coverage"] == .5
    assert summary["duration_coverage"] == pytest.approx(.25)
    assert summary["aggregation_scope"] == "available_measured_frames_only"
    assert summary["average_frame_psnr_db"] == 20


@pytest.mark.parametrize("field,value", [("information_role", "OFFLINE_BASELINE_METADATA"),
                                         ("measurement_kind", "object_profile_psnr"),
                                         ("reference_identity", ""), ("schema_version", 2)])
def test_profiles_and_utility_documents_cannot_be_quality_caches(field, value):
    document = cache([(context(), MetricStats(.3, 1))])
    document[field] = value
    with pytest.raises(ValueError):
        CachedQualityProvider(document)


def test_duplicate_cache_entries_rejected():
    document = cache([(context(), MetricStats(.3, 1))])
    document["frames"].append(document["frames"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        CachedQualityProvider(document)


@pytest.mark.parametrize("mutation", ["missing_provenance", "inconsistent_mse", "wrong_protocol", "invalid_mask"])
def test_cache_must_have_valid_measurement_protocol_and_provenance(mutation):
    document = cache([(context(), MetricStats(.3, 1))])
    if mutation == "missing_provenance":
        document["frames"][0]["provenance"].pop("reference_frame_id")
    elif mutation == "inconsistent_mse":
        document["frames"][0]["metrics"]["mse"] = 99
    elif mutation == "wrong_protocol":
        document["frames"][0]["metrics"]["normalization"] = "clip"
    else:
        document["protocol"]["mask_mode"] = "candidate_mask"
    with pytest.raises(ValueError):
        CachedQualityProvider(document)


def test_json_cache_roundtrip(tmp_path):
    first = context()
    path = tmp_path / "quality.json"
    path.write_text(json.dumps(cache([(first, MetricStats(.3, 1))])), encoding="utf8")
    provider = CachedQualityProvider.from_json(path)
    assert provider.measure(first).metrics.sse == .3


def test_provider_cannot_return_measurements_for_different_pose():
    first = context()
    wrong = context(camera_sample_id="wrong")
    class IncorrectProvider:
        def measure(self, context):
            return CachedQualityProvider(cache([(wrong, MetricStats(.3, 1))])).measure(wrong)
    with pytest.raises(ValueError, match="different frame"):
        collect_quality([first], IncorrectProvider())


def test_mixed_reference_or_protocol_cannot_be_pooled():
    first = context()
    second = context(camera_sample_id="f1")
    class IncorrectProvider:
        def measure(self, query):
            document = cache([(query, MetricStats(.3, 1))])
            if query.camera_sample_id == "f1":
                document["reference_identity"] = "other-reference"
            return CachedQualityProvider(document).measure(query)
    with pytest.raises(ValueError, match="reference"):
        collect_quality([first, second], IncorrectProvider())


def test_duplicate_requested_frames_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        collect_quality([context(), context()])


@pytest.mark.parametrize("field,value", [("frame_duration_s", 0), ("global_time_s", -1),
                                         ("loop_index", True), ("trace_id", ""),
                                         ("content_time_s", math.nan), ("selected_quality", {})])
def test_frame_context_rejects_invalid_identity_or_timing(field, value):
    with pytest.raises(ValueError):
        context(**{field: value})


def simulation_fixture():
    return {
        "status": "completed", "end_wall_s": 4., "available_capacity_bytes": 400,
        "total_duration_s": 3,
        "requests": [
            {"request_id": "loop0:r", "requested_bytes": 100, "delivered_bytes": 100, "kind": "representation"},
            {"request_id": "loop1:r", "requested_bytes": 200, "delivered_bytes": 200, "kind": "representation"},
        ],
        "segment_downloads": [
            {"global_segment_index": 0, "requested_bytes": 100, "delivered_bytes": 100,
             "base_bytes": 80, "enhancement_bytes": 20, "completed": True},
            {"global_segment_index": 1, "requested_bytes": 200, "delivered_bytes": 200,
             "base_bytes": 80, "enhancement_bytes": 120, "completed": True},
        ],
        "decisions": [
            {"global_segment_index": 0, "global_start_s": 0, "loop_index": 0, "source_gop_id": "g0",
             "selected_quality": {"a": 40, "b": 30}, "total_required_bytes": 100, "budget_bytes": 100,
             "timings": {"decision_call_ms": 2., "native_solver_ms": .5}},
            {"global_segment_index": 1, "global_start_s": 1.5, "loop_index": 1, "source_gop_id": "g0",
             "selected_quality": {"a": 20, "b": 30}, "total_required_bytes": 200, "budget_bytes": 200,
             "timings": {"decision_call_ms": 4., "native_solver_ms": 1.5}},
        ],
        "playback": {"startup_delay_s": .5, "stall_count": 1, "total_rebuffering_s": .5,
                     "completion_wall_time_s": 4, "media_duration_s": 3, "played_duration_s": 3,
                     "wall_time_s": 4, "buffer_s": 0},
        "buffer_samples": [{"wall_time_s": 0, "buffer_s": 0}, {"wall_time_s": 1, "buffer_s": 2},
                           {"wall_time_s": 3, "buffer_s": 0}],
        "buffer_intervals": [{"start_s": 0, "end_s": 1, "buffer_start_s": 0, "buffer_end_s": 0},
                             {"start_s": 1, "end_s": 3, "buffer_start_s": 2, "buffer_end_s": 0},
                             {"start_s": 3, "end_s": 4, "buffer_start_s": 0, "buffer_end_s": 0}],
    }


def test_bandwidth_metrics_count_new_requests_across_repeated_content():
    result = summarize_metrics(simulation_fixture())
    bandwidth = result["bandwidth"]
    assert bandwidth["request_count"] == 2
    assert bandwidth["total_requested_bytes"] == 300
    assert bandwidth["total_delivered_bytes"] == 300
    assert bandwidth["actual_consumption_bps"] == 600
    assert bandwidth["bandwidth_utilization"] == .75
    assert bandwidth["base_requested_bytes"] == 160
    assert bandwidth["enhancement_requested_bytes"] == 140
    assert bandwidth["physical_base_delivered_bytes"] is None
    assert bandwidth["physical_enhancement_delivered_bytes"] is None
    assert "not_separate_layer_packets" in bandwidth["base_enhancement_accounting"]


def test_playback_buffer_metrics_are_time_weighted_not_sample_average():
    result = summarize_metrics(simulation_fixture())["playback"]
    assert result["startup_delay_s"] == .5
    assert result["stall_count"] == 1
    assert result["total_rebuffering_s"] == .5
    assert result["completion_wall_time_s"] == 4
    assert result["buffer"]["time_weighted_mean_s"] == .5
    assert result["buffer"]["maximum_s"] == 2
    assert result["buffer"]["time_weighted_duration_s"] == 4


def test_quality_switching_is_per_object_across_loop_boundaries():
    result = summarize_metrics(simulation_fixture())["adaptation"]
    assert result["decision_switching"]["switch_count"] == 1
    assert result["played_quality_switching"]["switch_count_per_object"] == {"a": 1, "b": 0}
    assert result["budget_violation_count"] == 0
    assert result["all_selected_budgets_feasible"]


def test_unplayed_prefetched_segment_not_counted_in_played_quality_switches():
    data = simulation_fixture()
    data["playback"]["played_duration_s"] = 1
    result = summarize_metrics(data)["adaptation"]
    assert result["completed_download_switching"]["switch_count"] == 1
    assert result["played_quality_switching"]["switch_count"] == 0


def test_partial_delivery_preserves_signed_increment_accounting():
    data = simulation_fixture()
    data["requests"][1]["delivered_bytes"] = 100
    data["segment_downloads"][1].update(delivered_bytes=100, completed=False,
                                        base_bytes=220, enhancement_bytes=-20)
    result = summarize_metrics(data)
    bandwidth = result["bandwidth"]
    assert bandwidth["base_requested_bytes"] == 300
    assert bandwidth["enhancement_requested_bytes"] == 0
    assert bandwidth["base_delivered_accounting_bytes"] == 190
    assert bandwidth["enhancement_delivered_accounting_bytes"] == 10
    assert bandwidth["total_delivered_bytes"] == 200
    assert result["adaptation"]["completed_download_switching"]["switch_count"] == 0


def test_physical_layer_cost_only_when_actual_requests_are_labeled_layers():
    data = simulation_fixture()
    data["requests"][0]["kind"] = "base"
    data["requests"][1]["kind"] = "enhancement"
    bandwidth = summarize_metrics(data)["bandwidth"]
    assert bandwidth["physical_base_delivered_bytes"] == 100
    assert bandwidth["physical_enhancement_delivered_bytes"] == 200


@pytest.mark.parametrize("mutation", ["duplicate_id", "overdelivery", "overcapacity", "segments_disagree", "wrong_increment"])
def test_invalid_network_accounting_rejected(mutation):
    data = simulation_fixture()
    if mutation == "duplicate_id":
        data["requests"][1]["request_id"] = "loop0:r"
    elif mutation == "overdelivery":
        data["requests"][0]["delivered_bytes"] = 101
    elif mutation == "overcapacity":
        data["available_capacity_bytes"] = 1
    elif mutation == "segments_disagree":
        data["segment_downloads"][0]["delivered_bytes"] = 1
    else:
        data["segment_downloads"][0]["enhancement_bytes"] = 9
    with pytest.raises(ValueError):
        summarize_metrics(data)


def test_latencies_observed_separately_without_fabricating_visibility_timing():
    result = summarize_metrics(simulation_fixture())
    performance = result["performance"]
    assert performance["summary"]["decision_call_ms"]["mean_ms"] == 3
    assert performance["summary"]["native_solver_ms"]["total_ms"] == 2
    assert performance["summary"]["visibility_analysis_ms"]["mean_ms"] is None
    assert performance["information_role"] == "OBSERVED_RUNTIME_ONLY_NOT_SIMULATED_TIME"
    assert result["quality"]["pooled_psnr_db"] is None
    assert result["playback"]["completion_wall_time_s"] == 4


def test_budget_infeasible_and_overbudget_reported_without_fake_quality():
    data = simulation_fixture()
    data["decisions"][0].update(selected_quality=None, total_required_bytes=None)
    data["decisions"][1]["budget_bytes"] = 199
    result = summarize_metrics(data)
    assert result["adaptation"]["infeasible_decision_count"] == 1
    assert result["adaptation"]["budget_violation_count"] == 1
    assert not result["adaptation"]["all_selected_budgets_feasible"]


def test_zero_capacity_and_zero_wall_metrics_are_explicit_null():
    result = summarize_metrics({"end_wall_s": 0, "available_capacity_bytes": 0})
    assert result["bandwidth"]["actual_consumption_bps"] is None
    assert result["bandwidth"]["bandwidth_utilization"] is None
    json.dumps(result, allow_nan=False)


def sampling_fixture(timeline, sample_times):
    class Trace:
        trace_id = "synthetic-global-trace"
        def get_trajectory(self, gop_id):
            index = int(gop_id)
            return tuple(SimpleNamespace(timestamp=t, identity=f"pose:{index}:{t}")
                         for t in sample_times[index])
    decisions = [{"global_segment_index": segment.global_segment_index,
                  "selected_quality": {"a": 40 if segment.loop_index == 0 else 20}}
                 for segment in timeline]
    return dict(scene_id="synthetic-scene", content_revision="synthetic-content",
                trace_provider=Trace(), timeline=timeline, decisions=decisions,
                played_duration_s=timeline.total_duration_s,
                source_gop_ids=["source-gop-0", "source-gop-1"])


def test_contexts_full_frame_grid_uses_exact_rational_end_boundaries():
    timeline = Timeline(1, 1, 1)
    inputs = sampling_fixture(timeline, [[float(k / 30) for k in range(30)]])
    frames = contexts_for_playback(**inputs, mode="frame_grid", fps=30)
    assert len(frames) == 30
    assert frames[-1].global_time_s == 29/30
    assert math.fsum(f.frame_duration_s for f in frames) == pytest.approx(1)
    assert all(f.frame_duration_s == pytest.approx(1/30) for f in frames)
    with pytest.raises(ValueError, match="frame-aligned"):
        contexts_for_playback(**sampling_fixture(Timeline(.9, .9, .9), [[0]]), mode="frame_grid", fps=3)


def test_contexts_full_frame_grid_rejects_missing_pose_and_invalid_fps():
    timeline = Timeline(1, 1, 1)
    inputs = sampling_fixture(timeline, [[0, .5]])
    with pytest.raises(ValueError, match="Missing exact camera pose"):
        contexts_for_playback(**inputs, mode="frame_grid", fps=4)
    with pytest.raises(ValueError, match="fps"):
        contexts_for_playback(**inputs, mode="frame_grid", fps=0)


def test_contexts_sparse_trace_preserves_samples_without_inventing_frames():
    timeline = Timeline(2, 2, 2)
    frames = contexts_for_playback(**sampling_fixture(timeline, [[0, .4, 1.5]]))
    assert [f.global_time_s for f in frames] == [0, .4, 1.5]
    assert [f.frame_duration_s for f in frames] == pytest.approx([.4, 1.1, .5])
    assert all(f.content_gop_id == "source-gop-0" for f in frames)
    assert [f.camera_sample_id for f in frames] == ["pose:0:0", "pose:0:0.4", "pose:0:1.5"]


def test_contexts_clip_temporal_support_to_partial_playback():
    timeline = Timeline(1, 2, 1)
    inputs = sampling_fixture(timeline, [[0, .5], [1, 1.5]])
    inputs["played_duration_s"] = 1.3
    frames = contexts_for_playback(**inputs, mode="frame_grid", fps=2)
    assert [f.global_time_s for f in frames] == [0, .5, 1]
    assert [f.frame_duration_s for f in frames] == pytest.approx([.5, .5, .3])
    inputs["decisions"][1]["selected_quality"] = None
    assert len(contexts_for_playback(**inputs)) == 2


def test_contexts_bind_repeated_content_to_fresh_global_segment_pose_allocation():
    timeline = Timeline(1, 2, 1)
    inputs = sampling_fixture(timeline, [[0, .5], [1, 1.5]])
    frames = contexts_for_playback(**inputs, mode="frame_grid", fps=2)
    original, repeated = frames[0], frames[2]
    assert original.content_time_s == repeated.content_time_s == 0
    assert original.content_gop_id == repeated.content_gop_id == "source-gop-0"
    assert original.global_segment_id != repeated.global_segment_id
    assert original.camera_sample_id != repeated.camera_sample_id
    assert (original.loop_index, repeated.loop_index) == (0, 1)
    assert original.selected_quality == {"a": 40}
    assert repeated.selected_quality == {"a": 20}
    assert original.key != repeated.key

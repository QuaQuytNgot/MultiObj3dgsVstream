"""Base-first and utility integration using synthetic contracts, never a GPU.

The fixture's retained states represent an external decoder's verified output;
it deliberately does not implement that decoder or a request scheduler.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.client.adaptation_inputs import (
    AvailabilitySnapshot, ContentIndexSnapshot, DecodedState, DensityCapability,
    DensityValue, ObjectViewContext, PayloadDescriptor, PlanningContext,
    ProfileCoordinates, RequestUnit, SegmentViewSample, density_from_mse,
)
from src.client.adaptation_planner import AdaptationPlanner
from src.client.bandwidth import BandwidthEstimator, TransferObserver
from src.client.request_api import BandwidthSample, RequestSpec, TransferResult


class Densities:
    def __init__(self, revision="revision", values=(.4, .3, .1, 0), area=50):
        self.capability = DensityCapability(revision, "calibrated", (10, 10), "test-renderer", "RGB_0_1_black")
        self.values, self.area = values, area
        self.calls = []

    def get_density(self, obj, quality, coordinates):
        self.calls.append((obj, quality, coordinates.media_time_s))
        value = self.values[quality]
        if isinstance(value, DensityValue):
            return value
        if value is None:
            raise KeyError("missing")
        return DensityValue(value, self.area)


def context(*, frames=(0,), objects=("a",), visible=25., provider=None, weights=None):
    payloads, units, states = {}, {}, {}
    for obj in objects:
        for quality in range(4):
            for frame in frames:
                key = f"{obj}:{frame}:Q{quality}"
                parent = f"{obj}:{frame}:Q{quality-1}"
                unit = f"{obj}:Base" if quality == 0 else key
                request = RequestSpec(unit, f"https://media.test/{unit}", object_id=obj)
                units[unit] = RequestUnit(unit, obj, 100 if quality == 0 else quality * 10, request, f"hash:{unit}")
                payloads[key] = PayloadDescriptor(key, obj, frame, quality, 0, float(frame), unit,
                    f"state:{key}", () if quality == 0 else (parent,),
                    None if quality == 0 else f"state:{parent}")
                if quality == 0:
                    states[key] = DecodedState.from_payload(payloads[key])
    index = ContentIndexSnapshot("revision", objects, payloads, units)
    availability = AvailabilitySnapshot("revision", "post-base", decoded_states=states)
    samples = []
    for i, frame in enumerate(frames):
        per_object = {obj: SimpleNamespace(A_visible=visible, A_solo=50.,
                                          occlusion_ratio=1-visible/50., out_of_view=False) for obj in objects}
        source = {obj: {"source_type": "decoded_quality", "quality_id": "Q0", "frame": frame,
                        "timestamp": float(frame), "provenance": {"state_sha256": f"state:{obj}:{frame}:Q0"}}
                  for obj in objects}
        result = SimpleNamespace(timestamp=float(frame), resolution=(10, 10), per_object=per_object,
                                 metadata={"sources": source})
        views = {obj: ObjectViewContext(frame, ProfileCoordinates(0, 0, 1, float(frame), "calibrated"),
                                        f"state:{obj}:{frame}:Q0") for obj in objects}
        samples.append(SegmentViewSample(float(frame), {"world_view": ((1, 0), (0, 1))}, result, views,
                                          1 if weights is None else weights[i]))
    return PlanningContext(index, availability, {obj: frames for obj in objects}, tuple(samples),
                           provider or Densities(), 1.)


def estimate(rate=8000):
    estimator = BandwidthEstimator()
    estimator.accept_transfer_sample(BandwidthSample(1., 1000, 0., 8000 / rate, 8000 / rate,
                                                       rate, rate, "h2", True))
    return estimator.estimate_bandwidth()


def result(kind, received=1000, *, success=True, representative=True):
    spec = RequestSpec(kind, "https://media.test/unit", metadata={"traffic_kind": kind,
                       "bandwidth_eligible": representative})
    return TransferResult(spec=spec, status_code=200 if success else 503, body=None, output_path=None,
        received_bytes=received, start_time=0., first_byte_time=.1, end_time=1.1,
        ttfb=.1, transfer_duration=1., total_duration=1.1, body_goodput_bps=8. * received,
        effective_goodput_bps=8. * received / 1.1, http_version="h2", response_headers={},
        success=success, error=None, range_honored=None, timestamp=1., timing_source="fixture")


def test_utility_is_visible_fraction_times_density_improvement():
    inputs = AdaptationPlanner().collect_inputs(context())
    utility = AdaptationPlanner().calculate_utility(inputs)
    assert utility.values[0] == pytest.approx((0, .025, .075, .1))
    assert utility.valid[0] == (True,) * 4


def test_temporal_weights_include_hidden_samples_and_are_normalized():
    ctx = context(frames=(0, 1), weights=(1, 3))
    hidden = replace(ctx.view_samples[1].visibility, per_object={"a": SimpleNamespace(A_visible=0, A_solo=50)})
    ctx = replace(ctx, view_samples=(ctx.view_samples[0], replace(ctx.view_samples[1], visibility=hidden)))
    planner = AdaptationPlanner()
    table = planner.calculate_utility(planner.collect_inputs(ctx))
    assert table.values[0][3] == pytest.approx(.025)


def test_matched_unoccluded_utility_recovers_mse_improvement():
    mse = (.2, .1, .05, 0)
    provider = Densities(values=tuple(density_from_mse(m, 100, 50) for m in mse))
    planner = AdaptationPlanner()
    table = planner.calculate_utility(planner.collect_inputs(context(visible=50, provider=provider)))
    assert table.values[0] == pytest.approx((0, .1, .15, .2))


def test_negative_gain_is_valid_and_never_clamped():
    planner = AdaptationPlanner()
    table = planner.calculate_utility(planner.collect_inputs(context(provider=Densities(values=(.1, .2, .05, 0)))))
    assert table.values[0][1] == pytest.approx(-.025)
    assert table.valid[0][1]


@pytest.mark.parametrize("area", [0, 1e-8, 1e-6])
def test_negligible_area_does_not_create_zero_density(area):
    planner = AdaptationPlanner()
    provider = Densities(area=area)
    table = planner.calculate_utility(planner.collect_inputs(context(provider=provider)))
    assert table.valid[0] == (True, False, False, False)
    assert table.reasons[0][1] == "q0_support_below_guard"


def test_zero_visible_area_needs_no_profile_even_if_q3_has_missing_geometry():
    provider = Densities(values=(None,) * 4, area=0)
    planner = AdaptationPlanner()
    table = planner.calculate_utility(planner.collect_inputs(context(visible=0, provider=provider)))
    assert table.values[0] == (0.,) * 4
    assert table.valid[0] == (True,) * 4
    assert provider.calls == []


def test_tiny_positive_visibility_does_not_short_circuit_missing_density():
    planner = AdaptationPlanner()
    ctx = context(visible=1e-9, provider=Densities(values=(None,) * 4))
    table = planner.calculate_utility(planner.collect_inputs(ctx))
    assert not table.valid[0][1]


def test_quality_specific_denominator_is_invalid():
    provider = Densities(values=(DensityValue(.4, 50), DensityValue(.3, 60), DensityValue(.1, 50), DensityValue(0, 50)))
    planner = AdaptationPlanner()
    table = planner.calculate_utility(planner.collect_inputs(context(provider=provider)))
    assert not table.valid[0][1]
    assert table.reasons[0][1] == "density_denominator_mismatch"


@pytest.mark.parametrize("mode", ["base", "revision", "quality", "frame", "timestamp", "state", "calibration", "resolution", "normalization", "legacy"])
def test_plan_rejects_missing_prerequisites_or_incompatible_provenance(mode):
    ctx = context()
    if mode == "base":
        ctx = replace(ctx, availability=replace(ctx.availability, decoded_states={}))
    elif mode == "revision":
        ctx = replace(ctx, availability=replace(ctx.availability, content_revision="old"))
    elif mode in ("quality", "frame", "state"):
        sample = ctx.view_samples[0]
        source = dict(sample.visibility.metadata["sources"]["a"])
        if mode == "quality":
            source["quality_id"] = "Q3"
        elif mode == "frame":
            source["frame"] = 100
        else:
            source["provenance"] = {"state_sha256": "old"}
        vis = replace(sample.visibility, metadata={"sources": {"a": source}})
        ctx = replace(ctx, view_samples=(replace(sample, visibility=vis),))
    elif mode == "timestamp":
        sample = ctx.view_samples[0]
        ctx = replace(ctx, view_samples=(replace(sample, timestamp_s=1.),))
    elif mode == "legacy":
        ctx = replace(ctx, density_provider=SimpleNamespace())
    else:
        capability = ctx.density_provider.capability
        updates = {"calibration_id": "wrong"} if mode == "calibration" else (
                  {"resolution": (20, 20)} if mode == "resolution" else {"normalization_basis": "decoded_Q3_solo_alpha_area"})
        ctx.density_provider.capability = replace(capability, **updates)
    plan = AdaptationPlanner().plan(ctx, estimate())
    assert plan.status == "inputs_unavailable"
    assert plan.qualities is None


def test_future_frame_base_must_exist_even_if_only_one_time_is_sampled():
    ctx = context(frames=(0, 1))
    states = {key: state for key, state in ctx.availability.decoded_states.items() if state.frame == 0}
    ctx = replace(ctx, view_samples=ctx.view_samples[:1], availability=replace(ctx.availability, decoded_states=states))
    assert AdaptationPlanner().plan(ctx, estimate()).status == "inputs_unavailable"


def test_base_first_accounting_and_post_base_budget_are_separate():
    ctx = context(objects=("a", "b"))
    estimator = BandwidthEstimator()
    observer = TransferObserver(estimator)
    observer(result("metadata", 120))
    observer(result("profile", 300))
    assert estimator.estimate_bandwidth().bps is None
    observer(result("base", 1000))
    observer(result("base", 1000))
    planner = AdaptationPlanner()
    plan = planner.plan(ctx, estimator.estimate_bandwidth())
    assert plan.status == "ok"
    assert plan.budget_bytes == 850  # .85 * 8000 bps * post-Base 1s / 8
    assert plan.qualities == (3, 3)
    assert plan.actual_bytes == 120
    assert len(plan.missing_unit_ids) == 6
    before = observer.snapshot()
    assert before.total_bytes == 2420
    assert before.base_bytes == 2000
    # Planned requests are not prematurely counted as actual traffic.
    assert before.enhancement_bytes == 0
    observer(result("enhancement", 12, success=False))
    assert observer.snapshot().total_bytes == 2432
    assert observer.snapshot().failed_bytes == 12


def test_cached_enhancement_is_not_charged_again():
    ctx = context()
    unit = ctx.index.request_units["a:0:Q1"]
    ctx = replace(ctx, availability=replace(ctx.availability, cached_units={unit.unit_id: unit.checksum}))
    plan = AdaptationPlanner().plan(ctx, estimate())
    assert plan.actual_bytes == 50
    assert unit.unit_id not in plan.missing_unit_ids


def test_zero_remaining_time_returns_available_base_without_new_requests():
    ctx = replace(context(), transfer_budget_seconds=0)
    plan = AdaptationPlanner().plan(ctx, estimate())
    assert plan.status == "ok" and plan.qualities == (0,)
    assert plan.actual_bytes == 0 and plan.budget_bytes == 0


def test_no_bandwidth_does_not_fabricate_a_budget():
    plan = AdaptationPlanner().plan(context(), BandwidthEstimator().estimate_bandwidth())
    assert plan.status == "bandwidth_unavailable" and plan.budget_bytes is None


def test_missing_option_size_is_reported_but_other_options_remain_usable():
    ctx = context()
    units = dict(ctx.index.request_units)
    units["a:0:Q3"] = replace(units["a:0:Q3"], size_bytes=None)
    ctx = replace(ctx, index=replace(ctx.index, request_units=units))
    plan = AdaptationPlanner().plan(ctx, estimate())
    assert plan.qualities == (2,)
    assert any("exact_size_unavailable" in reason for reason in plan.diagnostics)


def test_progressive_layers_must_cover_the_same_media_timestamp():
    ctx = context()
    payloads = dict(ctx.index.payloads)
    payloads["a:0:Q1"] = replace(payloads["a:0:Q1"], media_time_s=.5)
    ctx = replace(ctx, index=replace(ctx.index, payloads=payloads))
    plan = AdaptationPlanner().plan(ctx, estimate())
    assert plan.qualities == (0,)
    assert any("progressive_parent_timestamp_mismatch" in reason for reason in plan.diagnostics)


def test_profile_coordinates_cannot_be_mutable_external_objects():
    with pytest.raises(ValueError, match="ProfileCoordinates"):
        ObjectViewContext(0, SimpleNamespace(calibration_id="calibrated"), "hash")


@pytest.mark.parametrize("ranges", [(None, None), (None, (0, 9)), ((0, 9), (5, 14))])
def test_shared_missing_physical_bytes_across_objects_are_rejected(ranges):
    ctx = context(objects=("a", "b"))
    units = dict(ctx.index.request_units)
    for obj, extent in zip(("a", "b"), ranges):
        unit = units[f"{obj}:0:Q1"]
        spec = replace(unit.request, url="https://media.test/shared", byte_range=extent)
        units[unit.unit_id] = replace(unit, request=spec)
    ctx = replace(ctx, index=replace(ctx.index, request_units=units))
    plan = AdaptationPlanner().plan(ctx, estimate())
    assert plan.status == "inputs_unavailable"
    assert "shared_missing_request_unit_across_objects" in plan.diagnostics

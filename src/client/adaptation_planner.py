"""One-interval fixed-Q0 utility allocation, without delivery scheduling.

The DP operates on exact consumed bins. Only cost is discretized; utilities
remain floating point. A successful plan proves byte feasibility, not deadlines.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import math

from .adaptation_inputs import (
    CostTable, DensityCapability, DensityValue, PlanningContext, PlanningInputs,
    QualityTarget, RequirementsResolver, UtilityTable, finite, integer,
)
from .bandwidth import BandwidthEstimate


@dataclass(frozen=True, slots=True)
class SolverResult:
    status: str
    qualities: tuple[int, ...] | None
    total_utility: float | None
    actual_bytes: int | None
    budget_bytes: int


@dataclass(frozen=True, slots=True)
class DPBounds:
    """Evaluation-only numerical bounds; upper allocation is never executable."""
    lower: SolverResult
    upper_utility: float | None
    absolute_gap: float | None
    relative_gap: float | None


@dataclass(frozen=True, slots=True)
class PlanResult:
    status: str
    object_ids: tuple[str, ...]
    qualities: tuple[int, ...] | None
    total_utility: float | None
    actual_bytes: int | None
    budget_bytes: int | None
    missing_unit_ids: tuple[str, ...]
    snapshot_id: str
    diagnostics: tuple[str, ...] = ()


class AdaptationPlanner:
    def __init__(self, *, budget_bins: int = 1024, safety_factor: float = .85):
        self.budget_bins = integer(budget_bins, "budget_bins", 1)
        self.safety_factor = finite(safety_factor, "safety_factor", 0)
        if not 0 < self.safety_factor <= 1:
            raise ValueError("safety_factor must be in (0, 1]")

    def collect_inputs(self, context: PlanningContext) -> PlanningInputs:
        """Check Base-first readiness and supplied analysis provenance.

        Camera/transform-to-profile calibration is an adapter responsibility;
        visibility results do not themselves retain the renderer's camera.
        """
        index, snapshot = context.index, context.availability
        if index.content_revision != snapshot.content_revision:
            raise ValueError("content_revision_mismatch")
        if set(context.frames_by_object) != set(index.object_ids):
            raise ValueError("object_frame_coverage_mismatch")
        for obj in index.object_ids:
            frames = context.frames_by_object[obj]
            if not frames or len(set(frames)) != len(frames):
                raise ValueError("target_frames_must_be_nonempty_and_unique")
            for frame in frames:
                try:
                    base = index.payload_for(obj, frame, 0)
                except KeyError as exc:
                    raise ValueError(f"base_payload_unavailable:{obj}:{frame}") from exc
                decoded = snapshot.decoded_states.get(base.payload_id)
                if decoded is None or not decoded.matches(base):
                    raise ValueError(f"base_first_prerequisite_unmet:{obj}:{frame}")
        capability = getattr(context.density_provider, "capability", None)
        if not isinstance(capability, DensityCapability):
            raise ValueError("density_capability_unavailable")
        capability.validate()
        if capability.content_revision != index.content_revision:
            raise ValueError("density_content_revision_mismatch")
        if not context.view_samples:
            raise ValueError("view_samples_unavailable")
        weights = tuple(finite(sample.weight, "sample weight", 0) for sample in context.view_samples)
        scale = max(weights)
        if scale == 0:
            raise ValueError("temporal_weights_have_zero_total")
        scaled = tuple(weight / scale for weight in weights)
        total = math.fsum(scaled)
        weights = tuple(weight / total for weight in scaled)
        for sample in context.view_samples:
            visibility = sample.visibility
            if visibility.timestamp != sample.timestamp_s:
                raise ValueError("visibility_timestamp_mismatch")
            if visibility.resolution != capability.resolution:
                raise ValueError("profile_resolution_mismatch")
            if set(sample.objects) != set(index.object_ids) or set(visibility.per_object) != set(index.object_ids):
                raise ValueError("visibility_scene_coverage_mismatch")
            sources = visibility.metadata.get("sources", {})
            for obj in index.object_ids:
                view = sample.objects[obj]
                if view.frame not in context.frames_by_object[obj]:
                    raise ValueError("view_frame_outside_target")
                base = index.payload_for(obj, view.frame, 0)
                if not math.isclose(base.media_time_s, sample.timestamp_s, rel_tol=0, abs_tol=1e-9):
                    raise ValueError("base_timestamp_mismatch")
                if view.q0_state_hash != base.decoded_state_hash:
                    raise ValueError("visibility_state_hash_mismatch")
                coordinates = view.coordinates
                if coordinates.calibration_id != capability.calibration_id:
                    raise ValueError("profile_calibration_mismatch")
                if coordinates.media_time_s != sample.timestamp_s:
                    raise ValueError("profile_timestamp_mismatch")
                source = sources.get(obj, {})
                if source.get("source_type") != "decoded_quality" or source.get("quality_id") != "Q0":
                    raise ValueError("visibility_requires_decoded_Q0_for_all_objects")
                if source.get("frame") != view.frame or source.get("timestamp") != sample.timestamp_s:
                    raise ValueError("visibility_source_timestamp_or_frame_mismatch")
                if source.get("provenance", {}).get("state_sha256") != base.decoded_state_hash:
                    raise ValueError("visibility_source_state_hash_mismatch")
        return PlanningInputs(context, weights)

    @staticmethod
    def _density(provider, obj, quality, coordinates) -> DensityValue:
        try:
            result = provider.get_density(obj, quality, coordinates)
        except LookupError as exc:
            raise ValueError("density_sample_unavailable") from exc
        if not isinstance(result, DensityValue):
            raise ValueError("density_result_invalid")
        if result.value is None or result.q0_solo_alpha_area is None:
            raise ValueError(result.reason or "density_sample_unavailable")
        finite(result.value, "density", 0)
        area = finite(result.q0_solo_alpha_area, "q0_solo_alpha_area", 0)
        if area <= provider.capability.alpha_area_epsilon:
            raise ValueError("q0_support_below_guard")
        width, height = provider.capability.resolution
        if area > width * height * (1 + 1e-5):
            raise ValueError("density_area_exceeds_image")
        return result

    def calculate_utility(self, inputs: PlanningInputs) -> UtilityTable:
        context = inputs.context
        provider = context.density_provider
        rows, masks, reasons = [], [], []
        for obj in context.index.object_ids:
            row, mask, why = [0.0] * 4, [True] * 4, [None] * 4
            for quality in range(1, 4):
                terms = []
                try:
                    for sample, weight in zip(context.view_samples, inputs.weights):
                        visible = sample.visibility.per_object[obj].A_visible
                        if visible == 0 or weight == 0:
                            continue
                        coordinates = sample.objects[obj].coordinates
                        baseline = self._density(provider, obj, 0, coordinates)
                        target = self._density(provider, obj, quality, coordinates)
                        if not math.isclose(baseline.q0_solo_alpha_area, target.q0_solo_alpha_area,
                                            rel_tol=1e-12, abs_tol=0):
                            raise ValueError("density_denominator_mismatch")
                        width, height = sample.visibility.resolution
                        term = weight * (visible / (width * height)) * (baseline.value - target.value)
                        if not math.isfinite(term):
                            raise ValueError("nonfinite_utility")
                        terms.append(term)
                    row[quality] = math.fsum(terms)
                    if not math.isfinite(row[quality]):
                        raise ValueError("nonfinite_utility")
                except (ValueError, OverflowError) as exc:
                    mask[quality], why[quality] = False, str(exc)
            rows.append(tuple(row))
            masks.append(tuple(mask))
            reasons.append(tuple(why))
        return UtilityTable(tuple(rows), tuple(masks), tuple(reasons))

    def calculate_cost(self, inputs: PlanningInputs) -> CostTable:
        context = inputs.context
        resolver = RequirementsResolver(context.index)
        requirements, rows, masks, reasons = [], [], [], []
        physical_owners = {}
        for obj in context.index.object_ids:
            reqs = tuple(resolver.required_units(QualityTarget(obj, context.frames_by_object[obj], quality),
                                                context.availability) for quality in range(4))
            for result in reqs:
                for unit in result.missing_units:
                    url, extent = unit.request.url, unit.request.byte_range
                    for other_extent, owner in physical_owners.get(url, ()):
                        overlap = (extent is None or other_extent is None
                                   or max(extent[0], other_extent[0]) <= min(extent[1], other_extent[1]))
                        if owner != obj and overlap:
                            raise ValueError("shared_missing_request_unit_across_objects")
                    physical_owners.setdefault(url, set()).add((extent, obj))
            requirements.append(reqs)
            rows.append(tuple(result.required_bytes if result.valid else 0 for result in reqs))
            masks.append(tuple(result.valid for result in reqs))
            reasons.append(tuple(result.reason for result in reqs))
        return CostTable(tuple(rows), tuple(masks), tuple(reasons), tuple(requirements))

    @staticmethod
    def _tables(utility, cost, valid):
        try:
            utility, cost = tuple(tuple(row) for row in utility), tuple(tuple(row) for row in cost)
            if len(utility) != len(cost) or any(len(row) != 4 for row in utility + cost):
                raise ValueError("utility/cost must have matching N x 4 shape")
            mask = tuple(tuple(row) for row in valid) if valid is not None else tuple((True,) * 4 for _ in utility)
            if len(mask) != len(utility) or any(len(row) != 4 for row in mask):
                raise ValueError("valid must have matching N x 4 shape")
            if any(type(value) is not bool for row in mask for value in row):
                raise ValueError("valid entries must be bool")
            costs = tuple(tuple(integer(c, "cost") if enabled else 0
                                for c, enabled in zip(row, flags)) for row, flags in zip(cost, mask))
            utilities = tuple(tuple(finite(u, "utility") if enabled else 0.0
                                    for u, enabled in zip(row, flags)) for row, flags in zip(utility, mask))
            return utilities, costs, mask
        except TypeError as exc:
            raise ValueError("utility/cost/valid must be N x 4 sequences") from exc

    def discretize_cost(self, cost, budget_bytes: int, *, rounding: str = "up") -> tuple[tuple[int, ...], ...]:
        budget = integer(budget_bytes, "budget_bytes")
        if rounding not in ("up", "down"):
            raise ValueError("rounding must be up or down")
        result = []
        for row in cost:
            if len(row) != 4:
                raise ValueError("cost must have N x 4 shape")
            out = []
            for value in row:
                value = integer(value, "cost")
                if not budget:
                    out.append(0 if value == 0 else 1)
                else:
                    out.append((value * self.budget_bins + (budget - 1 if rounding == "up" else 0)) // budget)
            result.append(tuple(out))
        return tuple(result)

    @staticmethod
    def backtrack(choices: Sequence[Sequence[int]], final_bin: int, rounded_cost) -> tuple[int, ...]:
        b = integer(final_bin, "final_bin")
        selected = [0] * len(choices)
        for n in range(len(choices) - 1, -1, -1):
            if b >= len(choices[n]) or choices[n][b] not in range(4):
                raise ValueError("Unreachable backtracking state")
            quality = choices[n][b]
            selected[n] = quality
            b -= rounded_cost[n][quality]
            if b < 0:
                raise ValueError("Invalid predecessor budget")
        if b != 0:
            raise ValueError("Backtracking did not end at (0,0)")
        return tuple(selected)

    def _solve_bins(self, utility, cost, valid, rounded, budget):
        capacity = self.budget_bins if budget else 0
        prev, prev_bytes = [-math.inf] * (capacity + 1), [0] * (capacity + 1)
        prev[0] = 0.0
        choices = []
        for n, row in enumerate(utility):
            curr, actual = [-math.inf] * (capacity + 1), [0] * (capacity + 1)
            choice = bytearray([255]) * (capacity + 1)
            for b in range(capacity + 1):
                for quality in range(4):
                    c = rounded[n][quality]
                    if not valid[n][quality] or c > b or prev[b-c] == -math.inf:
                        continue
                    value, count = prev[b-c] + row[quality], prev_bytes[b-c] + cost[n][quality]
                    if not math.isfinite(value):
                        raise ValueError("Utility accumulation exceeds floating-point range")
                    if value > curr[b] or (value == curr[b] and count < actual[b]):
                        curr[b], actual[b], choice[b] = value, count, quality
            choices.append(choice)
            prev, prev_bytes = curr, actual
        final, best, count = None, -math.inf, 0
        for b, value in enumerate(prev):
            if value > best or (value == best and value != -math.inf and prev_bytes[b] < count):
                final, best, count = b, value, prev_bytes[b]
        if final is None:
            return None
        qualities = self.backtrack(choices, final, rounded)
        return SolverResult("ok", qualities, sum(utility[n][k] for n, k in enumerate(qualities)),
                            sum(cost[n][k] for n, k in enumerate(qualities)), budget)

    def solve_dp(self, utility, cost, budget_bytes: int, valid=None) -> SolverResult:
        budget = integer(budget_bytes, "budget_bytes")
        utility, cost, valid = self._tables(utility, cost, valid)
        failure = lambda status: SolverResult(status, None, None, None, budget)
        if any(not any(row) for row in valid):
            return failure("inputs_unavailable")
        if sum(min(c for c, enabled in zip(row, flags) if enabled)
               for row, flags in zip(cost, valid)) > budget:
            return failure("byte_budget_infeasible")
        rounded = self.discretize_cost(cost, budget)
        result = self._solve_bins(utility, cost, valid, rounded, budget)
        if result is None:
            return failure("discretization_infeasible")
        if result.actual_bytes > budget:
            raise RuntimeError("Conservative DP violated the actual byte budget")
        return result

    def evaluate_bounds(self, utility, cost, budget_bytes: int, valid=None) -> DPBounds:
        """Floor solve is evaluation-only; its overspending is NOT filtered."""
        budget = integer(budget_bytes, "budget_bytes")
        utility, cost, valid = self._tables(utility, cost, valid)
        lower = self.solve_dp(utility, cost, budget, valid)
        if any(not any(row) for row in valid):
            return DPBounds(lower, None, None, None)
        if lower.status == "byte_budget_infeasible":
            return DPBounds(lower, None, None, None)
        if lower.status != "ok":
            # Verified feasible fallback; after Base-first this is normally all Q0.
            vector = tuple(min((k for k in range(4) if valid[n][k]),
                               key=lambda k: (cost[n][k], -utility[n][k], k)) for n in range(len(cost)))
            count = sum(cost[n][k] for n, k in enumerate(vector))
            if count <= budget:
                lower = SolverResult("ok", vector, sum(utility[n][k] for n, k in enumerate(vector)), count, budget)
        upper = self._solve_bins(utility, cost, valid, self.discretize_cost(cost, budget, rounding="down"), budget)
        if upper is None or lower.status != "ok":
            return DPBounds(lower, None if upper is None else upper.total_utility, None, None)
        gap = upper.total_utility - lower.total_utility
        relative = gap / upper.total_utility if upper.total_utility > 0 and lower.total_utility >= 0 else None
        return DPBounds(lower, upper.total_utility, gap, relative)

    def plan(self, context: PlanningContext, bandwidth_estimate: BandwidthEstimate) -> PlanResult:
        objects = context.index.object_ids
        snapshot_id = context.availability.snapshot_id
        budget = None
        def failure(status, *diagnostics):
            return PlanResult(status, objects, None, None, None, budget, (), snapshot_id, tuple(diagnostics))
        if bandwidth_estimate.bps is None:
            return failure("bandwidth_unavailable")
        rate = finite(bandwidth_estimate.bps, "bandwidth estimate", 0)
        if rate == 0:
            raise ValueError("bandwidth estimate must be positive")
        # Interpret displayed decimal parameters directly: .85 * 8000 / 8
        # is 850 bytes, not 849 from expanding .85's binary representation.
        predicted_bytes = (Fraction(str(rate)) * Fraction(str(self.safety_factor))
                           * Fraction(str(context.transfer_budget_seconds)) / 8)
        budget = predicted_bytes.numerator // predicted_bytes.denominator
        try:
            inputs = self.collect_inputs(context)
            utilities = self.calculate_utility(inputs)
            costs = self.calculate_cost(inputs)
        except ValueError as exc:
            return failure("inputs_unavailable", str(exc))
        valid = tuple(tuple(u and c for u, c in zip(ur, cr))
                      for ur, cr in zip(utilities.valid, costs.valid))
        diagnostics = tuple(f"{obj}:Q{k}:{reason}" for n, obj in enumerate(objects) for k in range(4)
                            for reason in (utilities.reasons[n][k], costs.reasons[n][k]) if reason is not None)
        solved = self.solve_dp(utilities.values, costs.values, budget, valid)
        if solved.status != "ok":
            return failure(solved.status, *diagnostics)
        units = {}
        for n, quality in enumerate(solved.qualities):
            for unit in costs.requirements[n][quality].missing_units:
                units[unit.unit_id] = unit
        if sum(unit.size_bytes for unit in units.values()) != solved.actual_bytes:
            raise RuntimeError("Selected costs disagree with required-unit union")
        return PlanResult("ok", objects, solved.qualities, solved.total_utility,
                          solved.actual_bytes, budget, tuple(units), snapshot_id, diagnostics)

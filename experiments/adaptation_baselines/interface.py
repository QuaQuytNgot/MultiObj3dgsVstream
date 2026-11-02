"""One adaptation interface; method-specific objectives, shared physical constraints."""
from __future__ import annotations

import time

from experiments.visibility_weighted_sse import allocation as shared_allocation
from experiments.visibility_weighted_sse.common import require
from .methods import SUPPORTED_METHODS, SOLO_SSE, VISIBILITY_WEIGHTED_SSE, construct_utilities
from .metadata import validate_quality_references


class AdaptationPlanner:
    """No renderer, evaluator, network or playback object is an input."""

    def __init__(self, profiles, trace_provider, visibility_adapter, *, baseline_metadata=None,
                 cached_resource_ids=(), allowed_quality_ids=None, solver_limits=None,
                 expected_object_count=None):
        self.profiles, self.trace, self.visibility = profiles, trace_provider, visibility_adapter
        self.metadata = baseline_metadata
        self.cached_resource_ids, self.allowed_quality_ids = cached_resource_ids, allowed_quality_ids
        self.solver_limits, self.expected_object_count = solver_limits, expected_object_count
        require(visibility_adapter.trace_provider is trace_provider,
                "Use the same frozen trace provider for every method")
        require(visibility_adapter.data["content_revision"] == profiles.content_revision,
                "Visibility/content revision mismatch")
        require(all(visibility_adapter.compatibility.get(k) == v for k, v in profiles.compatibility.items()
                    if k != "distribution_id"), "Visibility/profile compatibility mismatch")
        self._prepared = {}

    def initialize_solver(self):
        """Initialize the reused native bridge once outside compared method calls."""
        started = time.perf_counter()
        settings = shared_allocation.solver_settings(self.solver_limits)
        native_delta = settings["delta_bps"] if settings["discretization"] == "bitrate" else 1
        shared_allocation._solver(settings["max_dp_cells"], 0., native_delta)
        return (time.perf_counter() - started)*1000

    def prepare(self, gop_id):
        gid = str(gop_id)
        if gid not in self._prepared:
            started = time.perf_counter()
            trajectory = self.trace.get_trajectory(gid)
            aggregate = self.visibility.aggregate(gid)
            rows = self.profiles.for_gop(gid)
            require({row.instance_id for row in rows} == set(aggregate["per_object"]),
                    "Profile/trace instance coverage mismatch")
            require(all(aggregate["per_object"][row.instance_id]["object_id"] == row.object_id for row in rows),
                    "Profile/trace object identity mismatch")
            if self.expected_object_count is not None:
                require(len(rows) == self.expected_object_count, "Frozen scene object count differs from configuration")
            self._prepared[gid] = trajectory, aggregate, (time.perf_counter() - started) * 1000
        return self._prepared[gid]

    def plan(self, method, gop_id, budget_bytes):
        require(method in SUPPORTED_METHODS, f"Unknown adaptation method: {method}")
        if method not in (SOLO_SSE, VISIBILITY_WEIGHTED_SSE):
            require(self.metadata is not None, "Selected baseline requires validated quality metadata")
            validate_quality_references(self.metadata.get("quality_references", {}), self.profiles, [method])
            provenance = self.metadata.get("provenance", {})
            require(provenance.get("information_role") == "OFFLINE_BASELINE_METADATA" and
                    provenance.get("joint_quality_used") is False, "Independent offline baseline provenance required")
        started = time.perf_counter()
        gid = str(gop_id)
        was_prepared = gid in self._prepared
        trajectory, aggregate, preparation_ms = self.prepare(gid)
        utility_started = time.perf_counter()
        constructed = construct_utilities(method, self.profiles, gid, trajectory, aggregate, self.metadata)
        utility_ms = (time.perf_counter() - utility_started) * 1000
        utilities = constructed["per_object"]
        allocation_started = time.perf_counter()
        options = {"cached_resource_ids": self.cached_resource_ids, "solver_limits": self.solver_limits,
                   "allowed_quality_ids": self.allowed_quality_ids}
        if method == VISIBILITY_WEIGHTED_SSE:
            visible = {iid: item["A_visible"] for iid, item in aggregate["per_object"].items()}
            solved = shared_allocation.allocate(self.profiles, gid, visible, budget_bytes,
                                                utilities=utilities, **options)
        else:
            solved = shared_allocation.allocate_utilities(self.profiles, gid, budget_bytes, utilities, **options)
        allocation_ms = (time.perf_counter() - allocation_started) * 1000
        # Runtime evidence is not part of the deterministic policy artifact.
        diagnostics = solved["diagnostics"]
        native_ms = diagnostics.pop("solver_elapsed_ms")
        adapter_ms = diagnostics.pop("allocation_adapter_elapsed_ms")
        solver_wall_ms = diagnostics.pop("solver_wall_ms")
        process_peak_rss_kib = diagnostics.pop("process_peak_rss_kib")
        selections = solved["allocations"]
        return {"method": method, "gop_id": gid, "budget_bytes": budget_bytes,
                "selected_quality": ({r["instance_id"]: r["selected_quality"] for r in selections}
                                     if solved["feasible"] else None),
                "estimated_utility_per_object": ({r["instance_id"]: r["weighted_utility"] for r in selections}
                                                 if solved["feasible"] else None),
                "estimated_total_utility": solved["selected_utility"],
                "total_required_bytes": solved["total_selected_bytes"],
                "allocation": solved, "utilities": utilities,
                "method_diagnostics": constructed["diagnostics"],
                "timings": {"shared_GOP_preparation_ms": preparation_ms,
                            "shared_GOP_preparation_reused": was_prepared,
                            "utility_construction_ms": utility_ms, "allocation_ms": allocation_ms,
                            "native_solver_ms": native_ms, "allocation_adapter_ms": adapter_ms,
                            "solver_wall_ms": solver_wall_ms,
                            "process_peak_rss_kib": process_peak_rss_kib,
                            "decision_call_ms": (time.perf_counter() - started) * 1000}}

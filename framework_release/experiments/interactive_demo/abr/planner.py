"""Isolated live Visibility × MSE planner over independent admission closures.

The validated experiment allocator is copied under ``validated`` and remains the
single residual-grid algorithm here. Native solver sources are reused read-only.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

from .objectives import construct_mse_utilities
from .validated.allocation import allocate_utilities, _solver
from .validated.profiles import OfflineProfiles, ObjectProfile, QualityProfile

CRFS = (40, 36, 28, 20)


class GOPPlanner:
    """One row per object, independent byte charges, signed measured MSE gains."""
    def __init__(self, catalog, quality, cache, *, mode='proposed-dp', fixed_crf=40,
                 safety_factor=.85, delta_bps=150000, startup_bandwidth_bps=20e6,
                 max_dp_cells=2000000, events=None):
        if mode not in ('proposed-dp', 'fixed'):
            raise ValueError('Unsupported ABR mode')
        if not math.isfinite(safety_factor) or not 0 < safety_factor <= 1:
            raise ValueError('safety_factor must be in (0, 1]')
        if not math.isfinite(startup_bandwidth_bps) or startup_bandwidth_bps <= 0:
            raise ValueError('startup bandwidth must be finite and positive')
        self.catalog, self.quality, self.cache = catalog, quality, cache
        self.mode, self.fixed_crf = mode, int(fixed_crf)
        self.safety_factor, self.startup_bandwidth_bps = safety_factor, startup_bandwidth_bps
        self.solver = _solver(max_dp_cells, 0., delta_bps)
        self.events = events
        self.instances = {i.id: i for i in catalog.instances}
        self.selected = {}
        for instance in catalog.instances:
            reps = catalog.contents[instance.content_id].representations
            self.selected[instance.id] = min(reps.values(), key=lambda r: (abs(r.crf-fixed_crf), r.id)).id
        self.last_decision = {}
        self.decisions = []

    def set(self, instance_id, representation_id):
        instance = self.instances[instance_id]
        if representation_id not in self.catalog.contents[instance.content_id].representations:
            raise KeyError(representation_id)
        self.selected[instance_id] = representation_id

    def _missing_resources(self, segment, scope):
        # Cold full closure is conservative if no exact admission identity exists.
        # Integrity SHA is unchanged: owner_key below is only an accounting key.
        refs = (self.cache.missing_resources(segment, scope=scope)
                if scope is not None and hasattr(self.cache, 'missing_resources') else segment.resources)
        return {ref.sha256: ref for ref in refs}

    def plan(self, targets, predicted_bandwidth_bps, visibility, *, context=None):
        began = time.perf_counter()
        context = dict(context or {})
        external_bytes = context.get('outstanding_external_bytes', 0)
        if type(external_bytes) is not int or external_bytes < 0:
            raise ValueError('outstanding_external_bytes must be a nonnegative integer')
        if not targets:
            return {}
        locked = context.get('locked_representations', {})
        scopes = context.get('delivery_scopes', {})
        bandwidth = self.startup_bandwidth_bps if predicted_bandwidth_bps is None else predicted_bandwidth_bps
        if not math.isfinite(bandwidth) or bandwidth < 0:
            raise ValueError('predicted bandwidth must be finite and nonnegative')
        ids = sorted(targets)
        measured = visibility is not None
        areas = tuple(float((visibility or {}).get(iid, 0.)) if measured else 1. for iid in ids)
        rows = []
        mse_rows = []
        for iid in ids:
            instance = self.instances[iid]
            content = self.catalog.contents[instance.content_id]
            by_crf = {rep.crf: rep for rep in content.representations.values()}
            if set(by_crf) != set(CRFS) or len(content.representations) != 4:
                raise ValueError('Validated MSE DP requires exactly CRF40, CRF36, CRF28, CRF20')
            alternatives = []
            mse_values = []
            for crf in CRFS:
                rep = by_crf[crf]
                segment = rep.segments[targets[iid]]
                distortion = self.quality.mse(content.id, segment.gop_index, crf)
                alternatives.append((rep, segment, distortion,
                                     self._missing_resources(segment, scopes.get(iid))))
                mse_values.append(distortion)
            rows.append(alternatives)
            mse_rows.append(mse_values)
        utilities, importance, weights = construct_mse_utilities(mse_rows, (areas,), (1.,))
        duration = min(a[1].duration for row in rows for a in row)
        budget_bytes = math.floor(bandwidth * self.safety_factor * duration / 8)
        budget_bps = math.floor(bandwidth * self.safety_factor)
        cohort_budget_bytes = max(0, budget_bytes - external_bytes)
        gid = 'interactive_cohort'
        resources, objects, utility_metadata, candidates = {}, {}, {}, []
        for row_index, (iid, row) in enumerate(zip(ids, rows)):
            candidate_indices = tuple(k for k, a in enumerate(row)
                                      if iid not in locked or a[0].id == locked[iid])
            if not candidate_indices:
                raise ValueError(f'Unknown locked representation for {iid}: {locked[iid]}')
            candidates.append(candidate_indices)
            qualities, values = [], []
            for k in candidate_indices:
                rep, segment, distortion, missing = row[k]
                resource_ids = []
                for sha, ref in missing.items():
                    # Include representation closure ownership. This is separate
                    # from the payload SHA and never reaches transport/cache.
                    key = f'{iid}:{segment.id}:{sha}'
                    resources[key] = SimpleNamespace(sha256=sha, bytes=ref.bytes, owner_key=key)
                    resource_ids.append(key)
                qid = f'CRF{rep.crf}'
                qualities.append(QualityProfile(qid, rep.id, distortion, tuple(resource_ids), (), tuple(resource_ids)))
                values.append(dict(quality_id=qid, representation_id=rep.id,
                                   weighted_utility=utilities[row_index][k]))
            objects[iid, gid] = ObjectProfile(self.instances[iid].content_id, iid, gid, 1., tuple(qualities), {}, targets[iid])
            utility_metadata[iid] = dict(object_id=self.instances[iid].content_id,
                                        instance_id=iid, gop_id=gid, weight=weights[row_index], qualities=values)
        profiles = OfflineProfiles(getattr(self.catalog, 'revision', 'interactive'), 'interactive', 'standalone',
                                   {}, resources, objects, {})
        result = None
        if self.mode == 'fixed':
            indices = []
            for iid, row, allowed in zip(ids, rows, candidates):
                indices.append(next((k for k in allowed if row[k][0].crf == self.fixed_crf),
                                    allowed[0] if iid in locked else None))
                if indices[-1] is None:
                    raise ValueError(f'Fixed CRF {self.fixed_crf} missing for {iid}')
            status = 'fixed'
        else:
            result = allocate_utilities(profiles, gid, cohort_budget_bytes, utility_metadata,
                solver_limits=dict(discretization='bitrate', delta_bps=self.solver.delta_bps,
                    gop_duration_seconds=duration, max_dp_cells=self.solver.max_dp_cells,
                    max_shared_resources=0, max_shared_patterns=1, timeout_seconds=0))
            status = ('infeasible_external_commitments' if external_bytes > budget_bytes
                      else result['status'])
            if result['feasible'] and external_bytes <= budget_bytes:
                selected_by_id = {a['instance_id']: a['representation_id'] for a in result['allocations']}
                indices = [next(k for k, a in enumerate(row) if a[0].id == selected_by_id[iid])
                           for iid, row in zip(ids, rows)]
            else:
                # Explicit uncertified fallback preserves every object and lock.
                indices = [min(allowed, key=lambda k: (sum(ref.bytes for ref in row[k][3].values()), k))
                           for row, allowed in zip(rows, candidates)]
        chosen = [row[k] for row, k in zip(rows, indices)]
        remaining_bytes = sum(sum(ref.bytes for ref in a[3].values()) for a in chosen)
        total_committed_bytes = remaining_bytes + external_bytes
        full_bytes = sum(sum(ref.bytes for ref in {r.sha256: r for r in a[1].resources}.values()) for a in chosen)
        actual_rate = math.ceil(remaining_bytes * 8 / duration)
        total_rate = math.ceil(total_committed_bytes * 8 / duration)
        feasible = status in ('optimal', 'fixed') and total_committed_bytes <= budget_bytes
        selected = {iid: a[0].id for iid, a in zip(ids, chosen)}
        self.selected.update(selected)
        allocation_diagnostics = result['diagnostics'] if result else {}
        selected_rows = {}
        for i, (iid, a, k) in enumerate(zip(ids, chosen, indices)):
            rep, segment, distortion, refs = a
            selected_rows[iid] = dict(representation_id=rep.id, crf=rep.crf,
                gop_index=targets[iid], mse=distortion, mse_improvement=mse_rows[i][0]-distortion,
                weighted_utility=utilities[i][k], missing_bytes=sum(r.bytes for r in refs.values()),
                full_admission_bytes=sum(r.bytes for r in {r.sha256:r for r in segment.resources}.values()),
                duration_s=segment.duration, cost_bps=math.ceil(sum(r.bytes for r in refs.values())*8/duration),
                cost_group=i, cost_scope='independent instance/representation/GOP admission', locked=iid in locked)
            if hasattr(self.quality, 'psnr'):
                selected_rows[iid]['psnr'] = self.quality.psnr(self.instances[iid].content_id, targets[iid], rep.crf)
        self.last_decision = dict(mode=self.mode, objective='VISIBILITY_MSE', status=status, feasible=feasible,
            fallback_policy=None if feasible else ('retain configured fixed CRF and allow explicit rebuffering'
                if self.mode == 'fixed' else 'request cheapest legal independent packages; uncertified fallback permits rebuffering'),
            predicted_bandwidth_bps=bandwidth,
            bandwidth_source='configured_startup' if predicted_bandwidth_bps is None else 'measured_harmonic_goodput',
            budget_bps=budget_bps, budget_bytes=budget_bytes, service_interval_s=duration,
            cohort_available_budget_bytes=cohort_budget_bytes,
            outstanding_external_bytes=external_bytes,
            external_commitment_policy='reserve noncohort pending admissions once before cohort allocation',
            service_interval_policy='shortest target interval for offset cohorts', safety_factor=self.safety_factor,
            visibility_policy='current_pose_hold' if measured else 'uniform_bootstrap_no_decoded_state',
            visibility_weights=dict(zip(ids, weights)), raw_areas_per_sample=[list(areas)],
            importance_per_sample=[list(sample) for sample in importance],
            quality_policy='fixed own-CRF20 foreground RGB pooled MSE; signed CRF40 distortion reduction',
            candidate_crf_order=list(CRFS), estimated_rate_bps=total_rate,
            selected_demand_rate_bps=actual_rate,
            exact_missing_independent_bytes=remaining_bytes, exact_missing_union_bytes=remaining_bytes,
            exact_total_committed_bytes=total_committed_bytes,
            full_admission_bytes=full_bytes, union_rate_upper_bound_bps=total_rate,
            mandatory_shared_rate_bps=0, selected_union_within_budget=total_committed_bytes <= budget_bytes,
            selected_cohort_within_budget=remaining_bytes <= cohort_budget_bytes,
            infeasibility_scope=None if feasible else 'independent physical byte budget or explicit native limit; fallback uncertified',
            cost_policy='sum scoped per-instance admission closures; no cross-object/GOP media credit or clone grouping',
            solver_costs_bps=[[math.ceil(sum(r.bytes for r in a[3].values())*8/duration) for a in row] for row in rows],
            solver_utilities=[list(row) for row in utilities],
            solver_budget_bps=math.floor(cohort_budget_bytes * 8 / duration),
            solver_selection=indices if feasible else None,
            solver_groups=[dict(instances=[iid], content_id=self.instances[iid].content_id,
                                gop_index=targets[iid], locked=iid in locked) for iid in ids],
            instance_selection_indices=dict(zip(ids, indices)), instance_utilities=[list(row) for row in utilities],
            selected=selected_rows, context=context,
            solver_elapsed_ms=allocation_diagnostics.get('solver_elapsed_ms', 0.),
            solver_diagnostics=allocation_diagnostics, delta_bps=self.solver.delta_bps,
            max_dp_cells=self.solver.max_dp_cells, duration_s=time.perf_counter()-began)
        self.decisions.append(self.last_decision)
        if len(self.decisions) > 1000:
            del self.decisions[:-1000]
        if self.events:
            self.events.emit('abr_decision', **self.last_decision)
        return selected

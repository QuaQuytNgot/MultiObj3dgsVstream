"""GOP utility/cost adaptation; decoder scheduling stays in PlaybackRuntime."""
from __future__ import annotations

import math
import time
from types import SimpleNamespace
from .solver import ProposedDPSolver


class GOPPlanner:
    """One replaceable planner over measured independent representations.

    Costs remove verified cache entries and charge assets mandatory for any
    choice once. Aligned unlocked clones with identical measured alternatives
    form one four-choice row: replacing their different choices with the best
    measured-PSNR choice already requested cannot raise union bytes or reduce
    nonnegative-weight utility. Other optional overlap stays conservative.
    The benchmark's MCKP recurrence and feasibility guarantee are unchanged.
    """
    def __init__(self, catalog, quality, cache, *, mode='proposed-dp', fixed_crf=40,
                 safety_factor=.85, delta_bps=250000, startup_bandwidth_bps=20e6,
                 events=None):
        if mode not in ('proposed-dp','fixed'):
            raise ValueError('Unsupported ABR mode')
        if not math.isfinite(safety_factor) or not 0 < safety_factor <= 1:
            raise ValueError('safety_factor must be in (0, 1]')
        if not math.isfinite(startup_bandwidth_bps) or startup_bandwidth_bps <= 0:
            raise ValueError('startup bandwidth must be finite and positive')
        self.catalog, self.quality, self.cache = catalog, quality, cache
        self.mode, self.fixed_crf = mode, int(fixed_crf)
        self.safety_factor, self.startup_bandwidth_bps = safety_factor, startup_bandwidth_bps
        self.solver = ProposedDPSolver(delta_bps=delta_bps)
        self.events = events
        self.instances = {i.id:i for i in catalog.instances}
        self.selected = {}
        for instance in catalog.instances:
            reps = catalog.contents[instance.content_id].representations
            self.selected[instance.id] = min(reps.values(),key=lambda r:(abs(r.crf-fixed_crf),r.id)).id
        self.last_decision = {}
        self.decisions = []

    def set(self, instance_id, representation_id):
        instance = self.instances[instance_id]
        if representation_id not in self.catalog.contents[instance.content_id].representations:
            raise KeyError(representation_id)
        self.selected[instance_id] = representation_id

    def _missing(self, ref):
        return self.cache.missing_bytes(SimpleNamespace(resource=ref,shared_assets=())) > 0

    def plan(self, targets, predicted_bandwidth_bps, visibility, *, context=None):
        began = time.perf_counter()
        context = dict(context or {})
        if not targets:
            return {}
        locked = context.get('locked_representations',{})
        bandwidth = self.startup_bandwidth_bps if predicted_bandwidth_bps is None else predicted_bandwidth_bps
        if not math.isfinite(bandwidth) or bandwidth < 0:
            raise ValueError('predicted bandwidth must be finite and nonnegative')
        budget = math.floor(bandwidth*self.safety_factor)
        ids = sorted(targets)
        weights = {}
        measured = bool(visibility)
        for identifier in ids:
            value = float(visibility.get(identifier, 0.) if measured else 1.)
            if not math.isfinite(value) or value < 0:
                raise ValueError('visibility weights must be finite and nonnegative')
            weights[identifier] = value
        total = sum(weights.values())
        if total:
            weights = {i:w/total for i,w in weights.items()}
        rows = []
        mandatory = {}
        for identifier in ids:
            instance = self.instances[identifier]
            content = self.catalog.contents[instance.content_id]
            reps = sorted(content.representations.values(),key=lambda r:r.crf)
            if len(reps) != 4:
                raise ValueError('Validated Proposed DP requires exactly four measured representations')
            if identifier in locked:
                chosen = content.representations[locked[identifier]]
                reps = [chosen]*4
            alternatives=[]
            for rep in reps:
                segment = rep.segments[targets[identifier]]
                psnr = self.quality.psnr(content.id, segment.gop_index, rep.crf)
                refs = {r.sha256:r for r in segment.resources if self._missing(r)}
                alternatives.append((rep,segment,psnr,refs))
            common = set.intersection(*(set(a[3]) for a in alternatives))
            for digest in common:
                ref = alternatives[0][3][digest]
                # Shortest service interval conservatively covers offset timelines.
                rate = max(math.ceil(ref.bytes*8/a[1].duration) for a in alternatives)
                mandatory[digest] = max(mandatory.get(digest,0),rate)
            rows.append(alternatives)
        fixed_rate = sum(mandatory.values())
        costs = [[sum(math.ceil(ref.bytes*8/segment.duration) for key,ref in refs.items()
                      if key not in mandatory) for _,segment,_,refs in row] for row in rows]
        utilities = [[weights[identifier]*a[2] for a in row] for identifier,row in zip(ids,rows)]
        groups = []
        group_keys = {}
        instance_group = {}
        for i, (identifier, row) in enumerate(zip(ids, rows)):
            # Identity includes actual remaining hashes/sizes and measured
            # alternatives. Different GOPs, durations, publications or manual
            # locks cannot acquire credit by merely sharing a content label.
            signature = ('locked', identifier) if identifier in locked else (
                'aligned_unlocked', self.instances[identifier].content_id, targets[identifier],
                tuple((rep.id, rep.crf, segment.duration, psnr,
                       tuple(sorted((digest, ref.bytes) for digest, ref in refs.items())))
                      for rep, segment, psnr, refs in row))
            if signature not in group_keys:
                group_keys[signature] = len(groups)
                groups.append([])
            index = group_keys[signature]
            groups[index].append(i)
            instance_group[i] = index
        grouped_costs = [list(costs[group[0]]) for group in groups]
        grouped_utilities = [[math.fsum(utilities[i][k] for i in group) for k in range(4)]
                             for group in groups]
        # Charge the fixed union once inside a real row. An artificial extra
        # row would add another ceil() and spuriously exclude N=1 allocations.
        solver_costs = [list(row) for row in grouped_costs]
        solver_costs[0] = [cost+fixed_rate for cost in solver_costs[0]]
        solver_utilities = grouped_utilities
        result = None
        if self.mode == 'fixed':
            indices=[]
            for identifier,row in zip(ids,rows):
                index=next((k for k,a in enumerate(row) if a[0].crf==self.fixed_crf),None)
                if index is None:
                    if identifier in locked:index=0
                    else:raise ValueError(f'Fixed CRF {self.fixed_crf} missing for {identifier}')
                indices.append(index)
            status='fixed'
        else:
            result=self.solver.solve(solver_costs,solver_utilities,budget)
            status=result.status
            group_indices = list(result.selection) if result.selection else [
                min(range(4), key=lambda k:(grouped_costs[g][k],-rows[group[0]][k][0].crf))
                for g,group in enumerate(groups)]
            indices = [group_indices[instance_group[i]] for i in range(len(ids))]
        selected={identifier:rows[i][k][0].id for i,(identifier,k) in enumerate(zip(ids,indices))}
        chosen=[rows[i][k] for i,k in enumerate(indices)]
        union={key:ref for a in chosen for key,ref in a[3].items()}
        additive_rate=fixed_rate+sum(grouped_costs[g][indices[group[0]]] for g,group in enumerate(groups))
        # Exact transfer bytes are unioned; different durations use the shortest
        # interval for an explicitly conservative physical feasibility diagnostic.
        union_rate=math.ceil(sum(r.bytes for r in union.values())*8/min(a[1].duration for a in chosen))
        feasible = (status in ('optimal','fixed') and additive_rate <= budget and union_rate <= budget)
        self.selected.update(selected)
        self.last_decision={
            'mode':self.mode,'status':status,'feasible':feasible,
            'fallback_policy':None if feasible else (
                'retain configured fixed CRF and allow explicit rebuffering' if self.mode=='fixed' else
                'request minimum additive-cost GOPs and allow explicit rebuffering; conservative model has no feasible certificate'),
            'predicted_bandwidth_bps':bandwidth,'bandwidth_source':'configured_startup' if predicted_bandwidth_bps is None else 'measured_harmonic_goodput',
            'budget_bps':budget,'safety_factor':self.safety_factor,
            'visibility_policy':'current_pose_hold' if measured else 'uniform_bootstrap_no_decoded_state',
            'visibility_weights':weights,'quality_policy':'measured PSNR arithmetic mean dB over all reference views; no camera correspondence',
            'estimated_rate_bps':additive_rate,'exact_missing_union_bytes':sum(r.bytes for r in union.values()),
            'union_rate_upper_bound_bps':union_rate,'mandatory_shared_rate_bps':fixed_rate,
            'selected_union_within_budget':union_rate <= budget,
            'infeasibility_scope':None if feasible else 'conservative additive MCKP model; actual selected union reported independently',
            'cost_policy':'verified resident cache excluded; mandatory shared hashes charged once; aligned unlocked identical GOP clones grouped exactly; other optional sharing conservatively additive',
            'solver_costs_bps':solver_costs,'solver_utilities':solver_utilities,
            'solver_budget_bps':budget,'solver_selection':list(result.selection) if result and result.selection else None,
            'solver_groups':[{'instances':[ids[i] for i in group],
                'content_id':self.instances[ids[group[0]]].content_id,
                'gop_index':targets[ids[group[0]]],
                'locked':ids[group[0]] in locked} for group in groups],
            'instance_selection_indices':dict(zip(ids,indices)),
            'instance_standalone_costs_bps':costs,'instance_utilities':utilities,
            'selected':{identifier:{'representation_id':a[0].id,'crf':a[0].crf,'gop_index':targets[identifier],
                'psnr':a[2],'missing_bytes':sum(r.bytes for r in a[3].values()),'duration_s':a[1].duration,
                'cost_bps':costs[i][indices[i]],'cost_group':instance_group[i],
                'cost_scope':'standalone optional rate before aligned-clone grouping',
                'locked':identifier in locked} for i,(identifier,a) in enumerate(zip(ids,chosen))},
            'context':context,'solver_elapsed_ms':result.elapsed_ms if result else 0.,
            'duration_s':time.perf_counter()-began}
        self.decisions.append(self.last_decision)
        if len(self.decisions)>1000:del self.decisions[:-1000]
        if self.events:self.events.emit('abr_decision',**self.last_decision)
        return selected

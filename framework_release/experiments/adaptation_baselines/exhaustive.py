"""Bounded enumeration of the existing Visibility x PSNR physical problem.

No DP, incumbent fallback, branch-and-bound or quality pruning is used. Only
complete enumeration certifies an allocation; limits discard every incumbent.
"""
from __future__ import annotations

from itertools import product
import math
from pathlib import Path
import time

from experiments.visibility_weighted_sse.common import finite, integer, require
from experiments.visibility_weighted_sse.allocation import _process_peak_rss

EXHAUSTIVE_SEARCH = 'EXHAUSTIVE_SEARCH'
DEFAULT_LIMITS = dict(max_seconds=5., max_extra_rss_bytes=64*1024**2,
                      max_process_rss_bytes=4*1024**3, check_interval=128)


def settings(value=None):
    value=dict(value or {})
    require(set(value)<=set(DEFAULT_LIMITS),'Unknown exhaustive-search limit')
    result={**DEFAULT_LIMITS,**value}
    require(type(result['max_seconds']) in (int,float) and
            math.isfinite(result['max_seconds']) and result['max_seconds']>0,'Positive exhaustive max_seconds required')
    for name in ('max_extra_rss_bytes','max_process_rss_bytes','check_interval'):
        integer(result[name],name,1)
    require(result['check_interval']<=1024,'Exhaustive deadline checks must be frequent')
    return result


def current_rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])*1024
    raise RuntimeError('Linux VmRSS required for bounded exhaustive search')


def enumerate_qualities(profiles,gop_id,budget_bytes,utilities,*,cached_resource_ids=(),
                        allowed_quality_ids=None,limits=None,clock=None,rss_reader=None):
    cfg=settings(limits)
    clock=clock or time.monotonic
    rss_reader=rss_reader or current_rss_bytes
    start=clock();deadline=start+cfg['max_seconds']
    initial_rss=rss_reader();peak_rss=initial_rss
    integer(budget_bytes,'GOP byte budget')
    rows=profiles.for_gop(gop_id)
    require(set(utilities)=={row.instance_id for row in rows},'Utility/profile instance coverage differs')
    cached_ids=tuple(cached_resource_ids)
    require(len(set(cached_ids))==len(cached_ids) and all(k in profiles.resources for k in cached_ids),
            'Invalid cached resource IDs')
    cached={profiles.resources[k].sha256 for k in cached_ids}
    sizes={r.sha256:r.bytes for r in profiles.resources.values()}
    closures=tuple(tuple(frozenset(profiles.resources[r].sha256 for r in q.resource_ids)-cached
                         for q in row.qualities) for row in rows)
    base_union=frozenset().union(*(row[0] for row in closures))
    base_bytes=sum(sizes[k] for k in base_union)
    values=[];candidates=[]
    if allowed_quality_ids is not None:
        require(set(allowed_quality_ids)==set(utilities),'Allowed qualities must cover all instances')
    for row in rows:
        supplied=utilities[row.instance_id]['qualities']
        require(len(supplied)==len(row.qualities),'Utility quality count differs')
        value=[]
        for q,u in zip(row.qualities,supplied):
            require(q.quality_id==u['quality_id'] and q.representation_id==u['representation_id'],
                    'Utility quality/profile identity differs')
            value.append(finite(u['weighted_utility'],'weighted utility'))
        values.append(value)
        allowed=(tuple(q.quality_id for q in row.qualities) if allowed_quality_ids is None
                 else tuple(allowed_quality_ids[row.instance_id]))
        known={q.quality_id for q in row.qualities}
        require(allowed and len(set(allowed))==len(allowed) and set(allowed)<=known and
                row.qualities[0].quality_id in allowed,'Invalid allowed quality subset')
        candidates.append(tuple(i for i,q in enumerate(row.qualities) if q.quality_id in allowed))
    # This is cost evaluation only: disjoint instance closures can be added.
    # Shared-resource inputs still evaluate the exact union at every leaf.
    families=[frozenset().union(*row) for row in closures]
    independent=all(not a.intersection(b) for i,a in enumerate(families) for b in families[i+1:])
    costs=tuple(tuple(sum(sizes[k] for k in q) for q in row) for row in closures)
    visited=0;best=None;status='optimal'
    for indices in product(*candidates):
        if visited % cfg['check_interval']==0:
            now=clock();rss=rss_reader();peak_rss=max(peak_rss,rss)
            if now>=deadline:
                status='TIMEOUT';break
            if rss>cfg['max_process_rss_bytes'] or rss-initial_rss>cfg['max_extra_rss_bytes']:
                status='MEMORY_LIMIT';break
        visited+=1
        total=(sum(costs[i][q] for i,q in enumerate(indices)) if independent else
               sum(sizes[k] for k in frozenset().union(*(closures[i][q] for i,q in enumerate(indices)))))
        if base_bytes>budget_bytes or total>budget_bytes:
            continue
        utility=sum(values[i][q] for i,q in enumerate(indices))
        tie=tuple(reversed(indices))
        if best is None or utility>best[0] or (utility==best[0] and tie<best[1]):
            best=(utility,tie,indices)
    elapsed=(clock()-start)*1000
    peak_rss=max(peak_rss,rss_reader())
    if status=='optimal' and elapsed>=cfg['max_seconds']*1000:
        status='TIMEOUT'
    if status=='optimal' and (peak_rss>cfg['max_process_rss_bytes'] or peak_rss-initial_rss>cfg['max_extra_rss_bytes']):
        status='MEMORY_LIMIT'
    if status=='optimal' and best is None:
        status='INFEASIBLE'
    complete=status=='optimal'
    process_peak,scope=_process_peak_rss()
    diagnostic=dict(algorithm='exhaustive_quality_cartesian_product',objective='VISIBILITY_PSNR',
        optimality_scope='original_exact_physical_byte_problem' if complete else 'unavailable',
        combinations_total=math.prod(map(len,candidates)),combinations_visited=visited,
        enumeration_complete=status in ('optimal','INFEASIBLE'),limits=cfg,
        native_calls=0,native_dp_cells_total=0,native_dp_cells_max=0,native_subproblems=[],
        native_requested_dp_cells_total=0,native_requested_dp_cells_max=0,
        native_budget_states_max=0,native_estimated_working_set_bytes_max=0,
        exact_byte_feasible=complete,solver_elapsed_ms=0.,solver_wall_ms=elapsed,
        allocation_adapter_elapsed_ms=elapsed,process_peak_rss_kib=process_peak,
        process_peak_rss_scope=scope,enumeration_peak_rss_bytes=peak_rss,
        enumeration_extra_peak_rss_bytes=max(0,peak_rss-initial_rss),
        tie_policy='strict utility, reversed-index lexicographic ties',
        incumbent_returned=False,dp_used=False)
    result=dict(gop_id=gop_id,status=status,solver_status=status,feasible=complete,budget_feasible=complete,
        available_budget_bytes=budget_bytes,base_bytes=base_bytes,total_selected_bytes=None,
        enhancement_bytes=None,selected_utility=None,allocations=[],cached_resource_ids=list(cached_ids),
        diagnostics=diagnostic)
    if not complete:
        return result
    indices=best[2];accounted=set();allocations=[]
    for i,(row,index) in enumerate(zip(rows,indices)):
        q=row.qualities[index];resources=closures[i][index];charged=resources-accounted
        accounted.update(resources)
        representations={candidate.quality_id:candidate.representation_id for candidate in row.qualities}
        allocations.append(dict(instance_id=row.instance_id,object_id=row.object_id,
            selected_quality=q.quality_id,representation_id=q.representation_id,
            dependencies=list(q.dependencies),required_representations=[representations[k] for k in (*q.dependencies,q.quality_id)],
            required_resource_ids=list(q.resource_ids),physical_byte_cost=sum(sizes[k] for k in charged),
            standalone_missing_bytes=sum(sizes[k] for k in resources),charged_resource_sha256=sorted(charged),
            weighted_utility=values[i][index]))
    total=sum(sizes[k] for k in accounted)
    require(total<=budget_bytes and sum(a['physical_byte_cost'] for a in allocations)==total,
            'Exhaustive selection failed exact physical-byte feasibility')
    result.update(allocations=allocations,total_selected_bytes=total,enhancement_bytes=total-base_bytes,
        selected_utility=sum(a['weighted_utility'] for a in allocations),selected_resource_sha256=sorted(accounted))
    return result

"""Bounded native-DP comparisons using frozen real Phase-A decision inputs.

This is a solver validation helper for the existing streaming experiment. It
does not generate visibility, decode, render, change a utility, or run playback.
Large exact references are read from checksum-verified historical decisions;
fresh exact calls are restricted to low-slack cases and 50 million DP cells.
Each new solve runs in an isolated process so its peak RSS is attributable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
from types import MappingProxyType

from experiments.adaptation_baselines.methods import METHODS, VISIBILITY_WEIGHTED_SSE
from experiments.visibility_weighted_sse import allocation
from experiments.visibility_weighted_sse.common import digest, file_digest, read_json, require
from experiments.visibility_weighted_sse.profiles import load_profiles
from .inputs import independent_instance_transfer_profiles

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT / 'experiments/streaming_evaluation/runs/runtime_six_second_h2_validation_v1'
DEFAULT_PROFILES = PROJECT / 'experiments/offline_profiles/runs/full_source_v1/prepared/profiles.json'
DEFAULT_OUTPUT = PROJECT / 'experiments/streaming_evaluation/runs/discretization_validation_v3'
DELTAS = (50_000, 250_000, 1_000_000)
FRESH_METHODS = ('NAVA', 'SOLO_SSE', VISIBILITY_WEIGHTED_SSE)
MAX_CELLS = 50_000_000
MODULE = 'experiments.streaming_evaluation.discretization_validation'


def write_new(path, value):
    """Never silently replace an input or completed result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read_json(path) == value, f'Existing validation artifact differs: {path}')
        return
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def peak_rss_kib():
    """Linux memory-map high-water since exec, excluding inherited ru_maxrss.

    A launcher that read large historical result JSON can pass its ru_maxrss
    high-water into fork+exec workers. /proc VmHWM belongs to the new address
    space and therefore measures this solver worker rather than its launcher.
    """
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1])
    raise RuntimeError('Linux /proc/self/status VmHWM is required for isolated RSS measurement')


def bind_case_profiles(source, case):
    """Use exactly the runtime's per-instance physical transfer namespace."""
    originals = {row.object_id: row for row in source.for_gop(case['source_gop_id'])}
    objects = {}
    for instance in case['instances']:
        row = replace(originals[instance['object_id']], instance_id=instance['instance_id'],
                      gop_id=case['gop_id'])
        objects[row.instance_id, row.gop_id] = row
    return independent_instance_transfer_profiles(replace(source, objects=MappingProxyType(objects)))


def selection_evidence(profiles, case, selected):
    """Independently check exact physical union and unchanged frozen objective."""
    rows = profiles.for_gop(case['gop_id'])
    require(set(selected) == {r.instance_id for r in rows}, 'Selection instance coverage differs')
    physical, values = {}, []
    for row in rows:
        index = next((i for i, q in enumerate(row.qualities)
                      if q.quality_id == selected[row.instance_id]), None)
        require(index is not None, 'Selection contains an unknown quality')
        quality = row.qualities[index]
        supplied = case['utilities'][row.instance_id]['qualities'][index]
        require(supplied['quality_id'] == quality.quality_id and
                supplied['representation_id'] == quality.representation_id,
                'Frozen utility/profile identity differs')
        if 'resource_ids' in supplied:
            require(supplied['resource_ids'] == list(quality.resource_ids),
                    'Frozen physical resource closure differs')
        if row.instance_id in case['base_only_instance_ids']:
            require(index == 0, 'Invisible proposed-policy instance escaped mandatory Q0')
        for rid in quality.resource_ids:
            ref = profiles.resources[rid]
            physical[ref.sha256] = ref.bytes
        values.append(supplied['weighted_utility'])
    total = sum(physical.values())
    return dict(exact_byte_feasible=total <= case['budget_bytes'],
                total_selected_bytes=total, utility=sum(values),
                selected_quality=dict(selected), physical_resource_count=len(physical))


def verify_source(folder):
    marker = read_json(folder / 'completion.json')
    for name in ('simulation.json', 'performance.json'):
        require(file_digest(folder / name) == marker['outputs_sha256'][name],
                f'Historical Phase-A checksum differs: {folder / name}')
    config_path = folder.parents[1] / 'configuration.json'
    require(file_digest(config_path) == marker['scene_file_sha256'],
            'Historical scene configuration checksum differs')
    return marker, config_path


def frozen_case(source_profiles, profile_path, folder, decision, *, kind):
    marker, config_path = verify_source(folder)
    configuration = read_json(config_path)
    scene = configuration['scene']
    require(configuration['config']['adaptation']['resource_sharing'] == 'per_instance',
            'This validation requires the existing independent-instance cache policy')
    require(not decision['allocation']['cached_resource_ids'], 'Unexpected resident video cache credit')
    case = dict(schema='streaming-dp-frozen-input.v1',
        case_id=f"N{scene['object_count']:02d}-{decision['method']}-{kind}",
        reference_kind=kind, object_count=scene['object_count'], method=decision['method'],
        scenario=scene['kind'], global_segment_index=decision['global_segment_index'],
        gop_id=decision['gop_id'], source_gop_id=decision['source_gop_id'],
        gop_duration_seconds=decision['duration_s'], budget_bytes=decision['budget_bytes'],
        original_budget_bytes=decision['budget_bytes'], profiles_path=str(profile_path),
        profiles_sha256=file_digest(profile_path),
        instances=[{key: row[key] for key in ('instance_id', 'object_id')}
                   for row in scene['instances']], utilities=decision['utilities'],
        base_only_instance_ids=([iid for iid, utility in decision['utilities'].items()
                                if utility['weight'] == 0]
                               if decision['method'] == VISIBILITY_WEIGHTED_SSE else []),
        source=dict(folder=str(folder), configuration_hash=marker['configuration_hash'],
            simulation_sha256=marker['outputs_sha256']['simulation.json'],
            performance_sha256=marker['outputs_sha256']['performance.json'],
            configuration_sha256=marker['scene_file_sha256'],
            utility_sha256=digest(decision['utilities'])))
    profiles = bind_case_profiles(source_profiles, case)
    if kind == 'reused_historical_exact':
        solved = decision['allocation']
        require(solved['status'] == 'optimal' and solved['feasible'], 'Exact reference is not certified')
        require(solved['diagnostics']['cost_units'] == 'exact_integer_bytes_with_GCD_reduction',
                'Historical reference did not use exact bytes')
        selected = {row['instance_id']: row['selected_quality'] for row in solved['allocations']}
        checked = selection_evidence(profiles, case, selected)
        require(checked['exact_byte_feasible'] and checked['total_selected_bytes'] == solved['total_selected_bytes'],
                'Historical exact-byte accounting cannot be reproduced')
        require(math.isclose(checked['utility'], solved['selected_utility'], abs_tol=1e-10, rel_tol=1e-12),
                'Historical objective cannot be reproduced')
        diagnostics = solved['diagnostics']
        require(diagnostics['native_calls'] == 1, 'Historical cell reconstruction requires one native subproblem')
        perf = read_json(folder / 'performance.json')['records'][decision['performance_reference']['record']]
        # The unchanged C++ code allocates N*(B+1) uint8 backtracking cells,
        # plus two double utility rows. Previous logs expose B+1, not RSS.
        states = diagnostics['native_budget_states_max']
        case['exact_reference'] = dict(**checked, feasible=True, status='optimal',
            measurement_kind='reused_historical_exact', native_solver_ms=perf['native_solver_ms'],
            solver_wall_ms=None, allocation_adapter_ms=perf['allocation_adapter_ms'],
            peak_rss_kib=None, rss_note='Not recorded by the historical Phase-A process',
            native_dp_cells=scene['object_count'] * states,
            native_dp_cells_provenance='reconstructed N*(B+1) from successful unchanged native allocation',
            native_estimated_working_set_bytes=states*(16 + scene['object_count']))
    else:
        base = {row.instance_id: row.qualities[0].quality_id for row in profiles.for_gop(case['gop_id'])}
        base_bytes = selection_evidence(profiles, case, base)['total_selected_bytes']
        if scene['object_count'] > 5:
            case['budget_bytes'] = base_bytes + 3_000_000
            case['budget_note'] = ('Counterfactual frozen-input solver check: exact mandatory base plus '
                                   '3,000,000 bytes; not a new playback experiment')
        else:
            case['budget_note'] = 'Original Phase-A bootstrap GOP budget, unchanged'
        case['mandatory_q0_bytes'] = base_bytes
    return case


def prepare(source_root, profile_path, output_root):
    historical_manifest = read_json(source_root / 'manifest.json')
    require(file_digest(profile_path) == historical_manifest['profiles_sha256'],
            'Prepared profiles differ from the historical exact-reference matrix')
    for name in ('experiments/adaptation_baselines/methods.py',
                 'experiments/streaming_evaluation/inputs.py'):
        require(file_digest(PROJECT / name) == historical_manifest['implementation_sha256'][name],
                f'Historical utility or physical-transfer implementation changed: {name}')
    source_profiles = load_profiles(profile_path)
    cases = []
    for count in (5, 7, 15):
        for method in METHODS:
            folder = source_root / f'matrix/camera_rotation/N{count:02d}/30s/{method}/phase_a'
            simulation = read_json(folder / 'simulation.json')
            decisions = simulation['decisions']
            # GOP3 repeats source GOP0 with its actual higher post-bootstrap
            # budget and the independently recorded contemporaneous utilities.
            reused = frozen_case(source_profiles, profile_path, folder, decisions[3],
                                 kind='reused_historical_exact')
            cases.append(reused)
            if method in FRESH_METHODS:
                cases.append(frozen_case(source_profiles, profile_path, folder, decisions[0],
                                         kind='new_bounded_exact'))
    paths = []
    for case in cases:
        path = output_root / 'inputs' / (case['case_id'] + '.json')
        write_new(path, case)
        paths.append(dict(path=str(path), sha256=file_digest(path), case_id=case['case_id']))
    manifest = dict(schema='streaming-dp-validation-manifest.v1', cases=paths,
        profiles_sha256=file_digest(profile_path), max_dp_cells=MAX_CELLS, delta_bps=list(DELTAS),
        scope='24 frozen solver inputs: 15 reused exact references and 9 fresh bounded exact references; no playback/rendering',
        rss_measurement='Each new solver mode runs in a fresh Python subprocess; Linux /proc/self/status VmHWM in KiB includes Python/profile loading; inherited ru_maxrss is recorded separately',
        prior_result_policy='Historical Phase-A artifacts are checksum verified and read only')
    write_new(output_root / 'manifest.json', manifest)
    return manifest


def solve_case(case_path, *, mode, delta_bps):
    case = read_json(case_path)
    require(file_digest(case['profiles_path']) == case['profiles_sha256'], 'Prepared profiles changed')
    profiles = bind_case_profiles(load_profiles(case['profiles_path']), case)
    limits = dict(discretization=mode, delta_bps=delta_bps,
                  gop_duration_seconds=case['gop_duration_seconds'],
                  max_dp_cells=MAX_CELLS, timeout_seconds=0)
    # Library initialization is a separate measurement, as in Phase A.
    initialized = time.perf_counter()
    allocation._solver(MAX_CELLS, 0., 1 if mode == 'exact_bytes' else delta_bps)
    initialization_ms = (time.perf_counter() - initialized)*1000
    baseline_rss = peak_rss_kib()
    start = time.perf_counter()
    solved = allocation.allocate_utilities(profiles, case['gop_id'], case['budget_bytes'], case['utilities'],
        base_only_instance_ids=case['base_only_instance_ids'], solver_limits=limits)
    wall_ms = (time.perf_counter() - start)*1000
    diag = solved['diagnostics']
    result = dict(case_id=case['case_id'], mode=mode, delta_bps=delta_bps,
        input_sha256=file_digest(case_path), allocation_implementation_sha256=file_digest(Path(allocation.__file__)),
        measurement_kind='new_isolated_process', feasible=solved['feasible'], status=solved['status'],
        native_solver_ms=diag['solver_elapsed_ms'], solver_wall_ms=diag['solver_wall_ms'],
        allocation_adapter_ms=diag['allocation_adapter_elapsed_ms'], call_wall_ms=wall_ms,
        initialization_ms=initialization_ms, baseline_peak_rss_kib=baseline_rss,
        peak_rss_kib=peak_rss_kib(), peak_rss_source='Linux /proc/self/status VmHWM since exec',
        ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        native_dp_cells=diag['native_dp_cells_total'], native_dp_cells_max=diag['native_dp_cells_max'],
        native_requested_dp_cells_max=diag['native_requested_dp_cells_max'],
        native_estimated_working_set_bytes=diag['native_estimated_working_set_bytes_max'],
        solver_diagnostics=diag)
    if solved['feasible']:
        selected = {r['instance_id']: r['selected_quality'] for r in solved['allocations']}
        checked = selection_evidence(profiles, case, selected)
        require(checked['exact_byte_feasible'] and checked['total_selected_bytes'] == solved['total_selected_bytes'],
                'New solve failed independent exact physical-byte check')
        require(math.isclose(checked['utility'], solved['selected_utility'], abs_tol=1e-10, rel_tol=1e-12),
                'New solve objective does not reproduce frozen utility')
        result.update(checked)
    else:
        result.update(exact_byte_feasible=False, utility=None, selected_quality=None,
                      total_selected_bytes=None)
    require(result['native_dp_cells_max'] <= MAX_CELLS, 'Fresh DP exceeded explicit cell cap')
    return result


def compare(case, reference, variants):
    rows = []
    require(reference['feasible'], 'Exact reference must be feasible')
    for variant in variants:
        gap = reference['utility'] - variant['utility'] if variant['feasible'] else None
        require(gap is None or gap >= -max(1e-9, abs(reference['utility'])*1e-10),
                'Discretized utility exceeds the exact optimum: mismatched inputs/reference')
        rows.append(dict(case_id=case['case_id'], object_count=case['object_count'], method=case['method'],
            reference_kind=case['reference_kind'], budget_bytes=case['budget_bytes'],
            delta_bps=variant['delta_bps'], utility_gap=gap,
            relative_utility_gap=(gap/abs(reference['utility']) if gap is not None and reference['utility'] else
                                  (0. if gap == 0 else None)),
            selected_crfs_changed=(sum(variant['selected_quality'][iid] != q
                                      for iid, q in reference['selected_quality'].items())
                                   if variant['feasible'] else None),
            exact_selected_quality=reference['selected_quality'], selected_quality=variant['selected_quality'],
            exact_utility=reference['utility'], utility=variant['utility'],
            exact_native_dp_cells=reference['native_dp_cells'], native_dp_cells=variant['native_dp_cells'],
            exact_native_solver_ms=reference['native_solver_ms'], native_solver_ms=variant['native_solver_ms'],
            exact_allocation_adapter_ms=reference['allocation_adapter_ms'],
            allocation_adapter_ms=variant['allocation_adapter_ms'],
            exact_peak_rss_kib=reference['peak_rss_kib'], peak_rss_kib=variant['peak_rss_kib'],
            feasible=variant['feasible'], exact_byte_feasible=variant['exact_byte_feasible'], status=variant['status']))
    return rows


def run(source_root, profile_path, output_root):
    manifest = prepare(source_root, profile_path, output_root)
    comparisons = []
    for row in manifest['cases']:
        path = Path(row['path'])
        require(file_digest(path) == row['sha256'], 'Frozen validation input changed')
        case = read_json(path)
        modes = [('bitrate', delta) for delta in DELTAS]
        if case['reference_kind'] == 'new_bounded_exact':
            modes.insert(0, ('exact_bytes', 1))
        results = []
        for mode, delta in modes:
            destination = output_root / 'solves' / f"{case['case_id']}-{mode}-{delta}.json"
            if not destination.exists():
                subprocess.run([sys.executable, '-m', MODULE, '--worker', '--case-file', str(path),
                    '--mode', mode, '--delta-bps', str(delta), '--result-file', str(destination)],
                    cwd=PROJECT, check=True, timeout=90)
            result = read_json(destination)
            require(result['case_id'] == case['case_id'] and result['mode'] == mode and result['delta_bps'] == delta,
                    'Stored solve identity differs')
            require(result['input_sha256'] == row['sha256'] and
                    result['allocation_implementation_sha256'] == file_digest(Path(allocation.__file__)),
                    'Stored solve input or adapter implementation changed; choose a new validation output root')
            results.append(result)
        reference = (case['exact_reference'] if case['reference_kind'] == 'reused_historical_exact'
                     else results.pop(0))
        comparisons.extend(compare(case, reference, results))
        print(json.dumps(dict(case=case['case_id'], completed=True,
            maximum_relative_gap=max(r['relative_utility_gap'] or 0 for r in comparisons[-3:]))), flush=True)
    summary = dict(schema='streaming-dp-validation-results.v1', comparison_count=len(comparisons),
        new_exact_solve_count=9, reused_exact_reference_count=15, new_discretized_solve_count=72,
        all_exact_byte_feasible=all(r['exact_byte_feasible'] for r in comparisons), comparisons=comparisons)
    write_new(output_root / 'comparison.json', summary)
    lines = ['# Bounded DP discretization comparison', '', manifest['scope'] + '.', '',
        'Every comparison preserves its exact frozen GOP budget, resource union, candidates, and recorded policy utilities. '
        'N=7/15 fresh checks use explicitly labeled base+3 MB counterfactual budgets; these are solver checks, not playback runs. '
        'The 15 historical high-budget exact references were checksum verified and never rerun.', '',
        '| Reference | Grid (Mbps) | Cases | Feasible | Max utility gap (%) | CRF vectors changed | Median native ms | Median adapter ms | Max cells | Peak RSS MiB (max) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind in ('new_bounded_exact', 'reused_historical_exact'):
        for delta in DELTAS:
            group = [r for r in comparisons if r['reference_kind'] == kind and r['delta_bps'] == delta]
            lines.append(f"| {kind} | {delta/1e6:g} | {len(group)} | {sum(r['exact_byte_feasible'] for r in group)} | "
                f"{100*max(r['relative_utility_gap'] or 0 for r in group):.4f} | "
                f"{sum(bool(r['selected_crfs_changed']) for r in group)} | "
                f"{statistics.median(r['native_solver_ms'] for r in group):.4f} | "
                f"{statistics.median(r['allocation_adapter_ms'] for r in group):.4f} | "
                f"{max(r['native_dp_cells'] for r in group):,} | "
                f"{max(r['peak_rss_kib'] for r in group)/1024:.2f} |")
    lines += ['', 'Fresh exact calls use at most 50,000,000 backtracking cells. '
        'Cells count the native N×(B+1) uint8 table; two double utility rows are included separately in the working-set estimate. '
        'RSS is the fresh worker memory-map high-water mark from Linux `/proc/self/status` VmHWM, including imports, source-profile loading, and native storage. '
        'Linux ru_maxrss may inherit launcher peaks across exec and is recorded separately, not used for isolated RSS comparisons. '
        'Historical exact RSS was not recorded and remains unverified. Native C++ elapsed time, adapter wall time, '
        'individual CRF vectors, objective values, feasibility checks, and cell counts are in `comparison.json` and `solves/`.', '',
        '| N | Fresh exact native ms (median) | Fresh exact adapter ms (median) | Fresh exact peak RSS MiB (max) | Fresh exact cells (max) | Historical exact native ms (median) | Historical exact cells (max) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for count in (5, 7, 15):
        fresh = [r for r in comparisons if r['object_count'] == count and r['reference_kind'] == 'new_bounded_exact'
                 and r['delta_bps'] == DELTAS[0]]
        reused = [r for r in comparisons if r['object_count'] == count and r['reference_kind'] == 'reused_historical_exact'
                  and r['delta_bps'] == DELTAS[0]]
        lines.append(f"| {count} | {statistics.median(r['exact_native_solver_ms'] for r in fresh):.3f} | "
            f"{statistics.median(r['exact_allocation_adapter_ms'] for r in fresh):.3f} | "
            f"{max(r['exact_peak_rss_kib'] for r in fresh)/1024:.2f} | "
            f"{max(r['exact_native_dp_cells'] for r in fresh):,} | "
            f"{statistics.median(r['exact_native_solver_ms'] for r in reused):.3f} | "
            f"{max(r['exact_native_dp_cells'] for r in reused):,} |")
    lines += ['', 'Tight-budget cases at the default 0.25 Mbps grid (CRF counts; ordered per-instance vectors are in `comparison.json`):', '',
        '| N | Method | Exact utility | Grid utility | Utility loss (%) | Exact CRF counts | Grid CRF counts |',
        '|---:|---|---:|---:|---:|---|---|']
    def crf_counts(selected):
        return ', '.join(f'{crf}×{count}' for crf, count in sorted(Counter(selected.values()).items()))
    for row in comparisons:
        if row['reference_kind'] == 'new_bounded_exact' and row['delta_bps'] == 250_000:
            lines.append(f"| {row['object_count']} | {row['method']} | {row['exact_utility']:.6g} | "
                f"{row['utility']:.6g} | {100*(row['relative_utility_gap'] or 0):.4f} | "
                f"{crf_counts(row['exact_selected_quality'])} | {crf_counts(row['selected_quality'])} |")
    lines += ['', 'Unverified: generalization beyond these frozen decisions, repeated-run latency confidence intervals, '
              'and historical exact RSS. These solver-only comparisons make no transport or playback claims.', '']
    report = output_root / 'COMPARISON.md'
    text = '\n'.join(lines)
    if report.exists():
        require(report.read_text() == text, 'Existing comparison report differs')
    else:
        report.write_text(text)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--case-file', type=Path)
    parser.add_argument('--mode', choices=('bitrate', 'exact_bytes'), default='bitrate')
    parser.add_argument('--delta-bps', type=int, default=250_000)
    parser.add_argument('--result-file', type=Path)
    args = parser.parse_args()
    if args.worker:
        require(args.case_file is not None and args.result_file is not None, 'Worker paths are required')
        write_new(args.result_file, solve_case(args.case_file, mode=args.mode, delta_bps=args.delta_bps))
    elif args.prepare_only:
        prepare(args.source_root.resolve(), args.profiles.resolve(), args.output_root.resolve())
    else:
        run(args.source_root.resolve(), args.profiles.resolve(), args.output_root.resolve())


if __name__ == '__main__':
    main()

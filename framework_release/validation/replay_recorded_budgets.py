"""Validate packaged original planners against recorded budgets; no experiments/rendering."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--release-root', type=Path, default=Path.cwd(), help='Packaged source root; default current directory')
    parser.add_argument('--reference-root', required=True, type=Path, help='External original streaming_http2_dp015_v1 run with visibility cache')
    parser.add_argument('--config', type=Path, help='Original runtime-matrix config with mounted external package/metadata paths')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    release = args.release_root.resolve()
    reference = args.reference_root.resolve()

    from experiments.streaming_evaluation.runtime_matrix import load_settings, validate_inputs
    from experiments.streaming_evaluation.runtime_scenes import make_scene, make_trace, bind_profiles
    from experiments.streaming_evaluation.runtime_adapter import ReadOnlyVisibilityEngine, RuntimeVisibilityAdapter, RuntimePlanner
    from experiments.streaming_evaluation.runtime_native import ExecutionBudget
    from experiments.streaming_evaluation.timeline import Timeline
    from experiments.visibility_weighted_sse.common import read_json, file_digest

    config = args.config or release/'experiments/streaming_evaluation/configs/streaming_http2_dp015.json'
    cfg = load_settings(config)
    profiles, metadata, _, records, _, _, _ = validate_inputs(cfg)
    scene = make_scene('camera_rotation', 3, records, seed=cfg['seed'])
    timeline = Timeline(6, 10, 2)
    trace = make_trace(scene, timeline, profiles.content_revision)
    bound, bound_metadata = bind_profiles(profiles, metadata, scene, timeline)
    methods = ('NAVA', 'VISIBILITY_PSNR', 'VISIBILITY_WEIGHTED_SSE', 'VISIBILITY_MSE', 'EXHAUSTIVE_SEARCH')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    budget_root = args.output.resolve().parent
    proof = dict(
        schema='framework-release-recorded-budget-replay.v1',
        case='camera_rotation/N03/10s',
        configuration_sha256=file_digest(config),
        validation_script_sha256=file_digest(Path(__file__)),
        reference_manifest_sha256=file_digest(reference/'manifest.json'),
        protocol='Existing RuntimePlanner.plan, original per-method recorded byte budgets, exact native visibility cache; no network or rendering',
        phase_b_rendered=False,
        excluded_allocation_fields=['diagnostics.enumeration_peak_rss_bytes', 'diagnostics.enumeration_extra_peak_rss_bytes'],
        excluded_field_reason='Measured process RSS depends on the executing process; numerical allocations and deterministic diagnostics are compared exactly',
        methods=[],
    )
    for method in methods:
        source = reference/f'matrix/camera_rotation/N03/10s/{method}/phase_a/simulation.json'
        marker = source.parent/'completion.json'
        checkpoint = read_json(marker)
        assert file_digest(source) == checkpoint['outputs_sha256']['simulation.json'], 'Reference simulation checksum mismatch'
        configuration = read_json(source.parent.parent.parent/'configuration.json')
        recorded_trace = read_json(source.parent.parent.parent/'trace.json')
        assert configuration['scene'] == scene, 'Generated scene differs from reference'
        assert recorded_trace == trace.trace, 'Generated trajectory differs from reference'
        engine = ReadOnlyVisibilityEngine(reference, ExecutionBudget(budget_root, cfg['limits']))
        visibility = RuntimeVisibilityAdapter(engine, trace, bound)
        planner = RuntimePlanner(
            bound, trace, visibility, baseline_metadata=bound_metadata,
            expected_object_count=3, cached_resource_ids=(),
            solver_limits=cfg['adaptation']['solver_limits'],
            exhaustive_limits=cfg['adaptation']['exhaustive_limits'],
        )
        rows = []
        for recorded in read_json(source)['decisions']:
            result = planner.plan(method, str(recorded['global_segment_index']), recorded['budget_bytes'])
            checks = {
                key: result[key] == recorded[key]
                for key in (
                    'selected_quality', 'estimated_total_utility', 'total_required_bytes',
                    'estimated_utility_per_object', 'utilities', 'method_diagnostics',
                )
            }
            actual_allocation, expected_allocation = deepcopy(result['allocation']), deepcopy(recorded['allocation'])
            measured_memory = {}
            for field in ('enumeration_peak_rss_bytes', 'enumeration_extra_peak_rss_bytes'):
                actual = actual_allocation['diagnostics'].pop(field, None)
                expected = expected_allocation['diagnostics'].pop(field, None)
                if actual is not None or expected is not None:
                    measured_memory[field] = dict(recorded=expected, replay=actual)
            checks['allocation_except_measured_rss'] = actual_allocation == expected_allocation
            rows.append(dict(
                global_segment_index=recorded['global_segment_index'],
                budget_bytes=recorded['budget_bytes'],
                selected_quality=result['selected_quality'],
                total_required_bytes=result['total_required_bytes'],
                estimated_total_utility=result['estimated_total_utility'],
                checks=checks, measured_memory=measured_memory,
            ))
        engine.close()
        proof['methods'].append(dict(
            method=method, reference_simulation_sha256=file_digest(source),
            reference_checkpoint_sha256=file_digest(marker),
            visibility_cache_hits=engine.stats['visibility_cache_hits'],
            gops=rows,
        ))
    local_modules = {}
    for name, module in sorted(sys.modules.items()):
        if name.split('.')[0] not in ('experiments', 'src_final', 'multiobj3dgs'):
            continue
        source = getattr(module, '__file__', None)
        if source:
            source = Path(source).resolve()
            assert source.is_relative_to(release), f'Local import escaped release: {name}: {source}'
            local_modules[name] = str(source.relative_to(release))
    proof['local_module_origins'] = local_modules
    proof['checked_local_modules'] = len(local_modules)
    proof['status'] = 'passed' if all(all(gop['checks'].values()) for method in proof['methods'] for gop in method['gops']) else 'failed'
    proof['decisions_compared'] = sum(len(method['gops']) for method in proof['methods'])
    args.output.write_text(json.dumps(proof, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=proof['status'], methods=len(proof['methods']), decisions_compared=proof['decisions_compared'], checked_local_modules=proof['checked_local_modules'], output=str(args.output))))
    return 0 if proof['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

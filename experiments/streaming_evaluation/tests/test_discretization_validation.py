"""The frozen-input benchmark certifies identity, byte unions and provenance."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.streaming_evaluation.discretization_validation import (
    bind_case_profiles, compare, selection_evidence, solve_case, verify_source, write_new,
)
from experiments.visibility_weighted_sse.common import file_digest, read_json
from experiments.visibility_weighted_sse.profiles import build_utilities, load_profiles

FIXTURE = Path(__file__).resolve().parents[2] / 'visibility_weighted_sse/fixtures/profiles.json'


def frozen_fixture():
    source = load_profiles(FIXTURE)
    case = dict(case_id='fixture', source_gop_id='g0', gop_id='global-3',
        instances=[dict(instance_id='copy-a', object_id='object-0'),
                   dict(instance_id='copy-b', object_id='object-0')],
        profiles_path=str(FIXTURE), profiles_sha256=file_digest(FIXTURE),
        budget_bytes=10_000, base_only_instance_ids=[], gop_duration_seconds=2.)
    profiles = bind_case_profiles(source, case)
    case['utilities'] = {row.instance_id: build_utilities(row, 100)
                         for row in profiles.for_gop(case['gop_id'])}
    return source, profiles, case


def test_rebinding_preserves_source_bytes_and_charges_both_independent_instances():
    source, profiles, case = frozen_fixture()
    rows = profiles.for_gop(case['gop_id'])
    selected = {r.instance_id: 'Q3' for r in rows}
    evidence = selection_evidence(profiles, case, selected)
    original = source.for_gop('g0')[0].qualities[3]
    source_union = {source.resources[rid].sha256: source.resources[rid].bytes
                    for rid in original.resource_ids}
    assert evidence['total_selected_bytes'] == 2*sum(source_union.values())
    assert evidence['physical_resource_count'] == 2*len(source_union)
    assert set(rows[0].qualities[3].resource_ids).isdisjoint(rows[1].qualities[3].resource_ids)
    assert len(rows[0].qualities[3].resource_ids) == len(original.resource_ids)


def test_evidence_rejects_different_utility_identity_and_base_only_selection():
    _, profiles, case = frozen_fixture()
    selected = {'copy-a': 'Q3', 'copy-b': 'Q0'}
    tampered = deepcopy(case)
    tampered['utilities']['copy-a']['qualities'][3]['representation_id'] = 'different'
    with pytest.raises(ValueError, match='identity'):
        selection_evidence(profiles, tampered, selected)
    case['base_only_instance_ids'] = ['copy-a']
    with pytest.raises(ValueError, match='mandatory Q0'):
        selection_evidence(profiles, case, selected)


def test_historical_references_require_both_result_and_scene_checksums(tmp_path):
    folder = tmp_path / '30s' / 'NAVA' / 'phase_a'
    folder.mkdir(parents=True)
    configuration = folder.parents[1] / 'configuration.json'
    write_new(configuration, {'scene': 'frozen'})
    for name in ('simulation.json', 'performance.json'):
        write_new(folder / name, {'frozen': name})
    marker = dict(outputs_sha256={name: file_digest(folder / name)
                  for name in ('simulation.json', 'performance.json')},
                  scene_file_sha256=file_digest(configuration))
    write_new(folder / 'completion.json', marker)
    assert verify_source(folder)[0] == marker
    configuration.write_text('{}')
    with pytest.raises(ValueError, match='configuration checksum'):
        verify_source(folder)


def test_bounded_native_worker_reports_actual_cells_rss_and_exact_union(tmp_path):
    _, profiles, case = frozen_fixture()
    selected = {row.instance_id: 'Q3' for row in profiles.for_gop(case['gop_id'])}
    case['budget_bytes'] = selection_evidence(profiles, case, selected)['total_selected_bytes']
    path = tmp_path / 'case.json'
    write_new(path, case)
    exact = solve_case(path, mode='exact_bytes', delta_bps=1)
    # A four-bit/s grid equals one byte for a two-second GOP, making this a
    # lossless fixture while still exercising the production grid interface.
    grid = solve_case(path, mode='bitrate', delta_bps=4)
    assert exact['feasible'] and grid['feasible']
    assert exact['utility'] == grid['utility']
    assert exact['selected_quality'] == grid['selected_quality']
    for result in (exact, grid):
        assert result['input_sha256'] == file_digest(path)
        assert result['exact_byte_feasible']
        assert 0 < result['native_dp_cells'] <= 50_000_000
        assert result['peak_rss_kib'] > 0
        assert result['solver_wall_ms'] >= 0
        assert result['allocation_adapter_ms'] >= result['solver_wall_ms']


def test_validation_outputs_cannot_overwrite_other_results(tmp_path):
    path = tmp_path / 'result.json'
    write_new(path, {'original': True})
    write_new(path, {'original': True})
    with pytest.raises(ValueError, match='differs'):
        write_new(path, {'original': False})
    assert read_json(path) == {'original': True}


def test_comparison_rejects_a_grid_objective_above_the_exact_reference():
    _, _, case = frozen_fixture()
    reference = dict(feasible=True, utility=1.)
    variant = dict(feasible=True, utility=2.)
    with pytest.raises(ValueError, match='exceeds the exact optimum'):
        compare(case, reference, [variant])

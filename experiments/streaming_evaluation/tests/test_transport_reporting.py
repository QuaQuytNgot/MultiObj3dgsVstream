"""Transport comparisons must verify and preserve their historical inputs."""
from copy import deepcopy
import shutil

import pytest

from experiments.adaptation_baselines.methods import METHODS
from experiments.streaming_evaluation.analyze_runtime_results import compare_transport_runs
from experiments.streaming_evaluation.network import NetworkTrace
from experiments.streaming_evaluation.runtime_native import commit
from experiments.streaming_evaluation.tests.test_two_phase import streaming_fixture
from experiments.visibility_weighted_sse.common import file_digest


def test_transport_comparison_preserves_baseline_and_rejects_corruption(streaming_fixture, tmp_path):
    runner, source, scene = streaming_fixture
    folder = runner.streaming(source.profiles, source.metadata, scene, 10, METHODS[0],
                              NetworkTrace([(0., 1000.)], duration_s=102.))
    baseline = tmp_path/'baseline'
    previous = baseline/folder.relative_to(runner.root)
    shutil.copytree(folder/'phase_a', previous)
    shutil.copyfile(folder.parent/'configuration.json', previous.parent/'configuration.json')
    manifest = dict(config=deepcopy(runner.cfg), network_sha256='fixture-trace',
        profiles_sha256='fixture-profiles', baseline_metadata_sha256='fixture-metadata', representations={})
    for root in (runner.root, baseline):
        commit(root/'manifest.json', manifest)
    before = {p: file_digest(p) for p in baseline.rglob('*') if p.is_file()}
    result = compare_transport_runs(runner.root, baseline)
    assert result['completed_streaming_runs'] == result['successful_streaming_runs'] == 1
    assert result['expected_runs'] == 5
    assert result['runs'][0]['previous'] == result['runs'][0]['multiplexed']
    assert (runner.root/'COMPARISON.md').exists()
    assert (runner.root/'transport_comparison.csv').exists()
    assert before == {p: file_digest(p) for p in baseline.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='separate output'):
        compare_transport_runs(baseline, baseline)
    (previous/'metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        compare_transport_runs(runner.root, baseline)


def test_new_phase_report_does_not_overwrite_paper_report(streaming_fixture, tmp_path, monkeypatch):
    from experiments.streaming_evaluation import two_phase_report, runtime_results
    runner, _, _ = streaming_fixture
    project = tmp_path/'project'
    original = project/'reports/STREAMING_EXPERIMENT_RESULTS.md'
    original.parent.mkdir(parents=True)
    original.write_text('Preserved historical paper results\n')
    checksum = file_digest(original)
    monkeypatch.setattr(two_phase_report, 'PROJECT', project)
    monkeypatch.setattr(runtime_results, '_plots', lambda *a: [])
    two_phase_report.update_report(runner.root, runner.state)
    assert (runner.root/'REPORT.md').exists()
    assert file_digest(original) == checksum

"""Analytical shared-capacity fixtures for virtual multiplexing and budgeting."""
from dataclasses import replace
import math

import pytest

from experiments.streaming_evaluation.network import NetworkLink, NetworkTrace, CausalBandwidthEstimator
from experiments.streaming_evaluation.tests.test_integration import synthetic_config, execute_fixture


def requests(sizes):
    return [dict(request_id=f"r{i}", requested_bytes=size) for i, size in enumerate(sizes)]


def test_equal_streams_overlap_latency_but_do_not_multiply_capacity():
    link = NetworkLink(NetworkTrace([(0., 8.)]), latency_s=.02, max_concurrent_streams=2)
    results = link.download_batch(requests([1_000_000, 1_000_000]))
    assert [r.start_s for r in results] == [0., 0.]
    assert [r.ready_s for r in results] == [.02, .02]
    assert [r.end_s for r in results] == pytest.approx([2.02, 2.02])
    assert link.now == pytest.approx(2.02)  # 2 MB / 1 MB/s, plus one shared delay
    assert max(r['active_bodies'] for r in link.ledger) == 2
    assert math.fsum(r['delivered_bytes'] for r in link.ledger) == pytest.approx(2_000_000)
    for row in link.ledger:
        assert row['delivered_bytes'] * 8 <= row['capacity_bits'] + 1e-6


def test_fifo_slot_refill_and_ready_body_fair_sharing():
    link = NetworkLink(NetworkTrace([(0., 8.)]), latency_s=.1, max_concurrent_streams=2)
    results = link.download_batch(requests([1_000_000, 500_000, 250_000]))
    assert [r.request_id for r in results] == ['r0', 'r1', 'r2']
    assert [r.queued_s for r in results] == [0., 0., 0.]
    assert [r.start_s for r in results] == pytest.approx([0., 0., 1.1])
    assert [r.end_s for r in results] == pytest.approx([1.85, 1.1, 1.7])
    # r0 takes the full capacity while r2 awaits its response, then shares it.
    assert any(r['start_s'] == pytest.approx(1.1) and r['end_s'] == pytest.approx(1.2)
               and r['active_bodies'] == 1 and r['delivered_bytes'] == pytest.approx(100_000)
               for r in link.ledger)
    assert link.expected_duration([1_000_000, 500_000, 250_000], 8e6) == pytest.approx(link.now)


def test_completed_stream_timestamp_survives_later_batch_deadline():
    link = NetworkLink(NetworkTrace([(0., 8.)]), latency_s=.1, max_concurrent_streams=2)
    results = link.download_batch(requests([1_000_000, 500_000, 250_000]), deadline_s=1.3)
    assert results[1].status == 'complete'
    assert results[1].end_s == pytest.approx(1.1)
    assert results[1].capacity_bytes == pytest.approx(1_100_000)
    assert results[0].status == results[2].status == 'time_limit'
    assert sum(r.delivered_bytes for r in results) == pytest.approx(1_200_000)


def test_trace_exhaustion_retains_partial_streams_and_unadmitted_requests():
    link = NetworkLink(NetworkTrace([(0., 8.)], duration_s=.5, end_policy='stop'),
                       latency_s=.1, max_concurrent_streams=2)
    results = link.download_batch(requests([1_000_000] * 3))
    assert [r.status for r in results] == ['trace_exhausted', 'trace_exhausted', 'cancelled_before_service']
    assert [r.delivered_bytes for r in results] == pytest.approx([200_000, 200_000, 0])
    assert results[2].start_s is results[2].ready_s is None
    assert results[2].to_dict()['queue_duration_s'] == .5
    estimator = CausalBandwidthEstimator()
    with pytest.raises(ValueError, match='partial'):
        estimator.observe_batch(results)
    assert estimator.samples == []
    with pytest.raises(ValueError, match='blocked'):
        link.download(1)


def test_body_estimator_uses_interval_union_and_goodput_keeps_latency():
    link = NetworkLink(NetworkTrace([(0., 8.)]), latency_s=.2, max_concurrent_streams=2)
    results = link.download_batch(requests([1_000_000, 1_000_000]))
    estimator = CausalBandwidthEstimator(startup_bps=1e6)
    sample = estimator.observe_batch(results)
    assert sample['duration_s'] == sample['body_duration_s'] == pytest.approx(2.)
    assert sample['application_duration_s'] == pytest.approx(2.2)
    assert sample['observed_body_bps'] == pytest.approx(8e6)
    assert sample['observed_goodput_bps'] == pytest.approx(16e6 / 2.2)
    assert estimator.predict(link.now) == pytest.approx(8e6)
    assert estimator.snapshot(link.now)['sample_count'] == 1
    assert sum(r.end_s - r.ready_s for r in results) == pytest.approx(4.)


def test_body_estimator_includes_outages_but_excludes_between_batch_idle():
    link = NetworkLink(NetworkTrace([(0., 8.), (.5, 0.), (1.5, 8.)]), max_concurrent_streams=2)
    results = link.download_batch(requests([500_000, 500_000]))
    estimator = CausalBandwidthEstimator(window=2)
    sample = estimator.observe_batch(results)
    assert sample['body_duration_s'] == pytest.approx(2.)
    assert sample['observed_body_bps'] == pytest.approx(4e6)
    more = link.download_batch([dict(request_id='later', requested_bytes=1_000_000)], start_time_s=10.)
    assert estimator.observe_batch(more)['body_duration_s'] == pytest.approx(1.)
    assert estimator.predict(link.now) == pytest.approx(2 / (1 / 4e6 + 1 / 8e6))


def test_estimator_rejects_bad_batch_atomically_before_either_history_changes():
    rows = NetworkLink(NetworkTrace([(0., 8.)])).download_batch(requests([1000, 1000]))
    estimator = CausalBandwidthEstimator()
    with pytest.raises(ValueError, match='intervals'):
        estimator.observe_batch([rows[0], replace(rows[1], delivered_bytes=1)])
    assert not estimator.samples and not estimator.goodput_predictor.samples
    estimator.observe_batch(rows)
    assert len(estimator.samples) == 1


@pytest.mark.parametrize('streams', [1, 2, 16])
def test_deterministic_variable_trace_schedule_and_conservation(streams):
    trace = NetworkTrace([(0., 8.), (.2, 2.), (.4, 0.), (.6, 12.)], duration_s=20, end_policy='stop')
    inputs = requests([0, 1, 100_000, 300, 1_000_000, 50_000] * 4)
    links = [NetworkLink(trace, latency_s=.02, max_concurrent_streams=streams) for _ in range(2)]
    outputs = [link.download_batch(inputs) for link in links]
    assert outputs[0] == outputs[1] and links[0].ledger == links[1].ledger
    link = links[0]
    assert all(r.status == 'complete' for r in outputs[0])
    assert sum(r.delivered_bytes for r in outputs[0]) == sum(r['requested_bytes'] for r in inputs)
    assert link.delivered_bytes == pytest.approx(sum(r.delivered_bytes for r in outputs[0]))
    for row in link.ledger:
        assert 0 <= row['active_bodies'] <= streams
        assert row['delivered_bytes'] * 8 <= row['capacity_bits'] + 1e-6
    # Waiting responses also occupy stream slots.
    for event in {r.start_s for r in outputs[0]}:
        assert sum(r.start_s <= event < r.end_s for r in outputs[0]) <= streams


@pytest.mark.parametrize('streams', [1, 2, 16])
def test_latency_budget_bound_covers_unequal_sizes_and_fifo_tails(streams):
    link = NetworkLink(NetworkTrace([(0., 8.)]), latency_s=.02, max_concurrent_streams=streams)
    for count in [1, 2, 17, 85, 119, 255]:
        sizes = [0 if i % 4 == 0 else (i * 7919) % 100_000 + 1 for i in range(count)]
        expected = link.expected_duration(sizes, 8e6)
        assert expected <= sum(sizes) * 8 / 8e6 + link.latency_reserve(count) + 1e-9


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_stream_limit_requires_positive_integer(limit):
    with pytest.raises(ValueError):
        NetworkLink(NetworkTrace([(0., 8.)]), max_concurrent_streams=limit)


def test_invalid_batch_does_not_submit_a_valid_prefix():
    link = NetworkLink(NetworkTrace([(0., 8.)]))
    with pytest.raises(ValueError, match='unique'):
        link.download_batch([dict(request_id='same', requested_bytes=1)] * 2)
    assert link.now == 0 and link.results == [] and link._link.idle


def test_end_to_end_budget_completion_and_causality(tmp_path):
    def configure(config):
        config['network'].update(max_concurrent_streams=2, request_latency_s=.02)
    simulation, _ = execute_fixture(synthetic_config(tmp_path, modify=configure))
    assert simulation['status'] == 'completed'
    assert simulation['transport_model']['connection_count'] == 1
    assert simulation['transport_model']['protocol_negotiated'] is False
    assert simulation['metrics']['bandwidth']['body_throughput_bps'] == pytest.approx(2000.)
    assert simulation['metrics']['bandwidth']['application_goodput_bps'] < 2000.
    for i, (decision, download) in enumerate(zip(simulation['decisions'], simulation['segment_downloads'])):
        assert decision['budget_bytes'] == math.floor(2000. *
            max(0., decision['safe_download_duration_s'] - decision['latency_reserve_s']) / 8)
        assert decision['expected_deadline_feasible']
        assert decision['expected_download_duration_s'] == pytest.approx(download['download_duration_s'])
        assert decision['bandwidth_estimate']['sample_count'] == min(i, 5)
        assert all(s['time_s'] <= decision['decision_wall_s'] for s in simulation['bandwidth_estimator_samples'][:i])

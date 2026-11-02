"""Small synthetic correctness fixtures, never measured-trace experiments."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments.streaming_evaluation.network import (
    CausalBandwidthEstimator, NetworkLink, NetworkTrace, load_trace,
)
from experiments.streaming_evaluation.playback import PlaybackBuffer


class TraceLoaderTests(unittest.TestCase):
    def test_existing_csv_schema_and_no_implicit_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            path.write_text("time_seconds,bandwidth_mbps\n0,8\n1,4\n")
            trace = load_trace({"path": "synthetic.csv", "duration_s": 2}, directory)
            self.assertEqual(trace.rate_bps(1.5), 4e6)
            self.assertEqual(trace.rate_bps(2), 0)
            self.assertEqual(trace.integral_bits(0, 2), 12e6)
            with self.assertRaisesRegex(ValueError, "duration_s"):
                load_trace({"path": "synthetic.csv"}, directory)

    def test_production_bps_json_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps([{"at_s": 0, "bps": 8_000_000, "latency_s": .01},
                                        {"at_s": 1, "bps": 2_000_000}]))
            trace = load_trace({"path": str(path), "duration_s": 2})
            self.assertEqual(trace.rate_bps(0), 8e6)
            self.assertEqual(trace.rate_bps(1), 2e6)

    def test_embedded_json_horizon_and_explicit_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"duration_s": 2, "points": [[0, 8], [1, 4]]}))
            trace = load_trace({"path": str(path)})
            self.assertEqual(trace.rate_bps(2), 0)
            self.assertEqual(load_trace({"points": [[0, 1]], "end_policy": "hold"}).rate_bps(99), 1e6)

    def test_loop_is_explicit_and_independent_of_video(self):
        trace = load_trace({"points": [[0, 8], [1, 4]], "duration_s": 2, "loop": True})
        self.assertEqual(trace.rate_bps(2), 8e6)
        self.assertEqual(trace.integral_bits(0, 4), 24e6)

    def test_invalid_curve_inputs(self):
        for config in ({"points": [[1, 8]], "duration_s": 2},
                       {"points": [[0, -1]], "duration_s": 2},
                       {"points": [[0, 1], [0, 2]], "duration_s": 2},
                       {"points": [[0, math.inf]], "duration_s": 2},
                       {"points": [[0, 1]], "loop": True},
                       {"mode": "unthrottled"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                load_trace(config)


class NetworkLinkTests(unittest.TestCase):
    def test_serial_rate_latency_cost_and_idle_time(self):
        link = NetworkLink(NetworkTrace([(0, 8)]), latency_s=.25)
        first = link.download(1_000_000, 0, request_id="a", kind="base")
        self.assertEqual(first.end_s, 1.25)
        self.assertEqual(first.ready_s, .25)
        self.assertEqual(first.capacity_bytes, 1_250_000)
        second = link.download(500_000, 2, request_id="b", kind="enhancement")
        self.assertEqual(second.end_s, 2.75)
        self.assertEqual(link.requested_bytes, 1_500_000)
        self.assertEqual(link.delivered_bytes, 1_500_000)
        self.assertEqual(second.to_dict()["kind"], "enhancement")

    def test_drop_zero_service_then_recovery(self):
        link = NetworkLink(NetworkTrace([(0, 8), (.5, 0), (1.5, 4)], duration_s=5, end_policy="stop"))
        result = link.download(1_000_000)
        self.assertEqual(result.end_s, 2.5)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.delivered_bytes, 1_000_000)
        for row in link.ledger:
            self.assertLessEqual(row["delivered_bytes"] * 8, row["capacity_bits"] + 1e-6)

    def test_partial_trace_exhaustion_is_explicit_and_no_retry_cache(self):
        link = NetworkLink(NetworkTrace([(0, 8)], duration_s=.5, end_policy="stop"))
        result = link.download(1_000_000)
        self.assertEqual(result.status, "trace_exhausted")
        self.assertEqual(result.end_s, .5)
        self.assertEqual(result.delivered_bytes, 500_000)
        self.assertIsNone(result.completed_at_s)
        with self.assertRaisesRegex(ValueError, "blocked"):
            link.download(1)

    def test_exact_trace_boundary_completes_and_error_trace_is_reported(self):
        for policy in ("stop", "error"):
            with self.subTest(policy=policy):
                link = NetworkLink(NetworkTrace([(0, 8)], duration_s=1, end_policy=policy))
                result = link.download(1_000_000)
                self.assertEqual(result.status, "complete")
                self.assertEqual(result.completed_at_s, 1)
                following = link.download(1)
                self.assertEqual(following.status, "trace_exhausted")
                self.assertEqual(following.delivered_bytes, 0)
                self.assertEqual(link.capacity_between(0, 2), 1_000_000)

    def test_no_capacity_and_absolute_deadline_terminate(self):
        result = NetworkLink(NetworkTrace([(0, 0)])).download(100)
        self.assertEqual(result.status, "no_future_service")
        result = NetworkLink(NetworkTrace([(0, 0)], duration_s=1, loop=True)).download(100)
        self.assertEqual(result.status, "no_future_service")
        result = NetworkLink(NetworkTrace([(0, 8)])).download(2_000_000, deadline_s=.5)
        self.assertEqual(result.status, "time_limit")
        self.assertEqual(result.delivered_bytes, 500_000)

    def test_repeated_source_requests_always_pay_bytes(self):
        link = NetworkLink(NetworkTrace([(0, 8)]))
        for cycle in range(3):
            result = link.download(100_000, request_id=f"object0:cycle{cycle}:source0")
            self.assertEqual(result.delivered_bytes, 100_000)
        self.assertAlmostEqual(link.now, .3)
        self.assertEqual(link.requested_bytes, 300_000)
        self.assertEqual(len(link.results), 3)

    def test_duplicate_ids_backward_time_invalid_sizes(self):
        link = NetworkLink(NetworkTrace([(0, 8)]))
        link.download(1, request_id="a")
        with self.assertRaisesRegex(ValueError, "unique"):
            link.download(1, request_id="a")
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            link.download(1, start_time_s=0)
        for size in (-1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                link.download(size)

    def test_zero_size_still_obeys_configured_request_latency(self):
        link = NetworkLink(NetworkTrace([(0, 8)]), latency_s=.25)
        result = link.download(0)
        self.assertEqual(result.completed_at_s, .25)
        self.assertEqual(result.delivered_bytes, 0)

    def test_deadline_or_trace_end_inside_latency_never_delivers_bytes(self):
        for trace, deadline, status in ((NetworkTrace([(0, 8)]), .5, "time_limit"),
                                        (NetworkTrace([(0, 8)], duration_s=.5, end_policy="stop"),
                                         None, "trace_exhausted")):
            with self.subTest(status=status):
                result = NetworkLink(trace, latency_s=1).download(1_000, deadline_s=deadline)
                self.assertEqual(result.status, status)
                self.assertEqual(result.end_s, .5)
                self.assertEqual(result.delivered_bytes, 0)


class CausalEstimatorTests(unittest.TestCase):
    def test_bootstrap_then_completed_transfer_only_and_staleness(self):
        estimator = CausalBandwidthEstimator(startup_bps=30e6, window=2, stale_s=2)
        link = NetworkLink(NetworkTrace([(0, 8)]), latency_s=1)
        self.assertEqual(estimator.snapshot(0)["bps"], 30e6)
        result = link.download(1_000_000)
        self.assertEqual(estimator.snapshot(1)["bps"], 30e6)
        estimator.observe(result)
        self.assertEqual(estimator.snapshot(2)["body_bps"], 8e6)
        self.assertEqual(estimator.snapshot(2)["application_goodput_bps"], 4e6)
        self.assertEqual(estimator.snapshot(5)["bps"], 30e6)
        self.assertTrue(estimator.snapshot(5)["stale"])

    def test_partial_is_not_a_completed_sample(self):
        estimator = CausalBandwidthEstimator(startup_bps=30e6)
        link = NetworkLink(NetworkTrace([(0, 8)], duration_s=.5, end_policy="stop"))
        result = link.download(1_000_000)
        self.assertIsNone(estimator.observe(result))
        self.assertEqual(estimator.snapshot(.5)["sample_count"], 0)

    def test_future_observation_duplicate_and_past_snapshot_rejected(self):
        result = NetworkLink(NetworkTrace([(0, 8)])).download(1_000_000)
        estimator = CausalBandwidthEstimator()
        with self.assertRaisesRegex(ValueError, "future"):
            estimator.observe(result, now=.5)
        estimator.observe(result)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            estimator.snapshot(.5)
        with self.assertRaisesRegex(ValueError, "twice"):
            estimator.observe(result)

    def test_mixed_resource_batch_uses_aggregate_bytes_and_busy_union(self):
        link = NetworkLink(NetworkTrace([(0, 8)]), latency_s=1)
        base = link.download(1_000_000, request_id="base", kind="base")
        enhancement = link.download(1_000, request_id="enhancement", kind="enhancement")
        estimator = CausalBandwidthEstimator()
        sample = estimator.observe_batch([base, enhancement])
        self.assertEqual(sample["transfer_count"], 2)
        self.assertEqual(sample["bytes"], 1_001_000)
        self.assertAlmostEqual(sample["body_duration_s"], 1.001)
        self.assertAlmostEqual(sample["application_duration_s"], 3.001)
        self.assertAlmostEqual(sample["observed_goodput_bps"], 8_008_000 / 3.001)
        self.assertAlmostEqual(estimator.predict(link.now), 8e6)
        self.assertEqual(estimator.snapshot(link.now)["sample_count"], 1)

    def test_empty_batch_has_no_sample_and_respects_clock(self):
        estimator = CausalBandwidthEstimator(startup_bps=17e6)
        self.assertIsNone(estimator.observe_batch([]))
        self.assertIsNone(estimator.observe_batch([], now=1))
        self.assertEqual(estimator.snapshot(1)["bps"], 17e6)
        self.assertEqual(estimator.samples, [])
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            estimator.observe_batch([], now=.5)

    def test_incomplete_future_and_duplicate_batches_reject_atomically(self):
        complete = NetworkLink(NetworkTrace([(0, 8)])).download(1_000_000, request_id="complete")
        incomplete = NetworkLink(NetworkTrace([(0, 8)], duration_s=.5, end_policy="stop")).download(
            1_000_000, request_id="partial")
        estimator = CausalBandwidthEstimator()
        with self.assertRaisesRegex(ValueError, "partial"):
            estimator.observe_batch([complete, incomplete])
        with self.assertRaisesRegex(ValueError, "future"):
            estimator.observe_batch([complete], now=.5)
        with self.assertRaisesRegex(ValueError, "twice"):
            estimator.observe_batch([complete, complete])
        self.assertEqual(estimator.samples, [])
        self.assertEqual(estimator.snapshot(0)["sample_count"], 0)
        estimator.observe_batch([complete])
        self.assertEqual(len(estimator.samples), 1)


class PlaybackBufferTests(unittest.TestCase):
    def test_startup_buffer_threshold_then_exact_drain(self):
        buffer = PlaybackBuffer(4, startup_buffer_s=2)
        buffer.append(0, 1, 1)
        self.assertFalse(buffer.started)
        buffer.append(1, 1, 2)
        self.assertEqual(buffer.startup_delay_s, 2)
        buffer.append(2, 2, 3)
        self.assertEqual(buffer.position_s, 1)
        self.assertEqual(buffer.finish(), 6)
        self.assertEqual(buffer.stall_count, 0)

    def test_stall_duration_and_completion(self):
        buffer = PlaybackBuffer(4)
        buffer.append(0, 2, 1)
        buffer.append(2, 2, 5)
        self.assertEqual(buffer.stall_count, 1)
        self.assertEqual(buffer.rebuffer_duration_s, 2)
        self.assertEqual(buffer.stalls, [dict(start_s=3., end_s=5., duration_s=2.)])
        self.assertEqual(buffer.finish(), 7)
        self.assertEqual(buffer.summary()["played_duration_s"], 4)

    def test_exact_arrival_at_empty_boundary_is_not_stall(self):
        buffer = PlaybackBuffer(4)
        buffer.append(0, 2, 1)
        buffer.append(2, 2, 3)
        self.assertEqual(buffer.finish(), 5)
        self.assertEqual(buffer.stall_count, 0)
        self.assertFalse(any(row["kind"] == "rebuffer_start" for row in buffer.events))

    def test_partial_final_gop_and_startup_threshold_larger_than_duration(self):
        buffer = PlaybackBuffer(2.5, startup_buffer_s=10)
        buffer.append(0, 2, .5)
        self.assertFalse(buffer.started)
        buffer.append(2, .5, 1)
        self.assertEqual(buffer.startup_delay_s, 1)
        self.assertEqual(buffer.finish(), 3.5)

    def test_incomplete_run_retains_active_stall(self):
        buffer = PlaybackBuffer(4)
        buffer.append(0, 2, 1)
        buffer.advance_to(5)
        self.assertEqual(buffer.stall_count, 1)
        self.assertEqual(buffer.rebuffer_duration_s, 2)
        self.assertIsNone(buffer.completion_wall_s)
        self.assertEqual(buffer.summary()["played_duration_s"], 2)

    def test_interval_buffer_integral_is_exact(self):
        buffer = PlaybackBuffer(4)
        buffer.append(0, 2, 1)
        buffer.append(2, 2, 2)
        buffer.finish()
        integral = sum((row["end_s"] - row["start_s"]) *
                       (row["buffer_start_s"] + row["buffer_end_s"]) / 2
                       for row in buffer.intervals)
        self.assertEqual(integral, 6)
        self.assertEqual(sum(row["end_s"] - row["start_s"] for row in buffer.intervals), 5)

    def test_contiguous_global_ranges_prevent_loop_cache_or_clock_reset(self):
        buffer = PlaybackBuffer(6)
        buffer.append(0, 2, 1)
        buffer.append(2, 2, 2)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            buffer.append(0, 2, 3)
        buffer.append(4, 2, 3)
        self.assertEqual(buffer.finish(), 7)
        self.assertEqual(len([row for row in buffer.events if row["kind"] == "gop_available"]), 3)

    def test_invalid_configs_and_admissions(self):
        for duration, startup in ((0, 0), (math.inf, 0), (True, 0), (1, -1), (1, math.nan)):
            with self.subTest(duration=duration, startup=startup), self.assertRaises(ValueError):
                PlaybackBuffer(duration, startup)
        buffer = PlaybackBuffer(2)
        for start, duration, wall in ((1, 1, 1), (0, 0, 1), (0, 3, 1), (0, 1, math.nan)):
            with self.subTest(start=start, duration=duration, wall=wall), self.assertRaises(ValueError):
                buffer.append(start, duration, wall)
        buffer.advance_to(1)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            buffer.advance_to(.5)


if __name__ == "__main__":
    unittest.main()

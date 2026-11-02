"""Actual local TLS sockets plus pure aggregate-shaper conservation checks."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from experiments.streaming_evaluation.http2_server import AggregateTraceShaper, fair_byte_grants
from experiments.streaming_evaluation.network import NetworkTrace, CausalBandwidthEstimator


class FairShaperTests(unittest.TestCase):
    def test_max_min_fair_grants_are_deterministic_and_conserve_capacity(self):
        self.assertEqual(fair_byte_grants([1, 100, 100], 101), [1, 50, 50])
        self.assertEqual(fair_byte_grants([100, 100, 100], 5), [2, 2, 1])
        for demands in ([0, 1, 37], [1000] * 16, [1, 1000, 17, 0]):
            for capacity in (0, 1, 19, 100, 10000):
                first = fair_byte_grants(demands, capacity)
                self.assertEqual(first, fair_byte_grants(demands, capacity))
                self.assertEqual(sum(first), min(capacity, sum(demands)))
                self.assertTrue(all(0 <= grant <= demand for grant, demand in zip(first, demands)))

    def test_piecewise_trace_integral_and_finite_exhaustion(self):
        trace = NetworkTrace([(0, 8), (.5, 0), (1, 4)], duration_s=2, end_policy="error")
        shaper = AggregateTraceShaper(trace)
        self.assertEqual(shaper.capacity(0, .5), 500000)
        self.assertEqual(shaper.capacity(.5, 1), 0)
        self.assertEqual(shaper.capacity(1, 3), 500000)
        self.assertEqual(shaper.capacity(3, 4), 0)

    def test_no_unthrottled_trace_and_bad_configuration(self):
        trace = NetworkTrace([(0, 8)])
        for tick in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                AggregateTraceShaper(trace, tick_s=tick)
        trace.unthrottled = True
        with self.assertRaises(ValueError):
            AggregateTraceShaper(trace)


DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("httpx", "h2", "hypercorn", "cryptography"))


@unittest.skipUnless(DEPENDENCIES, "Install experiments/streaming_evaluation/requirements-http2.txt")
class RealHTTP2Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.resource = Path(self.directory.name) / "payload.bin"
        self.resource.write_bytes(bytes(range(256)) * 250)

    def tearDown(self):
        self.directory.cleanup()

    def link(self, **kwargs):
        from experiments.streaming_evaluation.real_http2 import RealHTTP2Link
        return RealHTTP2Link(NetworkTrace([(0, 8)], duration_s=5, end_policy="stop"),
                             {"sha": self.resource}, **kwargs)

    def requests(self, count, prefix="r"):
        return [dict(request_id=f"{prefix}-{i}", resource_sha256="sha",
                     requested_bytes=self.resource.stat().st_size) for i in range(count)]

    def test_actual_h2_multiplexing_reuse_capacity_and_estimator(self):
        with self.link(max_concurrent_streams=4) as link:
            first = link.download_batch(self.requests(4), deadline_s=4)
            second = link.download_batch(self.requests(1, "second"), deadline_s=4)
            metadata = link.transport_metadata()
            self.assertTrue(metadata["actual_multiplexing_verified"])
            self.assertEqual(metadata["max_observed_server_streams"], 4)
            self.assertEqual(metadata["connection_count"], 1)
            self.assertEqual(len(metadata["observed_server_client_ports"]), 1)
            self.assertEqual([row.stream_id for row in first + second], [3, 5, 7, 9, 11])
            for row in first + second:
                self.assertEqual(row.http_version, "HTTP/2")
                self.assertEqual(row.alpn_protocol, "h2")
                self.assertEqual(row.delivered_bytes, self.resource.stat().st_size)
                self.assertGreaterEqual(row.first_body_s, row.ready_s)
                self.assertGreater(row.to_dict()["ttfb_s"], .02)
            self.assertLess(max(row.start_s for row in first), min(row.end_s for row in first))
            self.assertGreater(max(row.end_s for row in first) - min(row.start_s for row in first), .256)
            self.assertEqual(sum(tick["delivered_bytes"] for tick in link.ledger), link.delivered_bytes)
            for tick in link.ledger:
                self.assertLessEqual(tick["delivered_bytes"] * 8, tick["capacity_bits"] + 1e-6)
            sample = CausalBandwidthEstimator().observe_batch(first, max(row.end_s for row in first))
            body_interval = max(row.end_s for row in first) - min(row.ready_s for row in first)
            app_interval = max(row.end_s for row in first) - min(row.start_s for row in first)
            self.assertAlmostEqual(sample["observed_body_bps"], 4 * self.resource.stat().st_size * 8 / body_interval)
            self.assertAlmostEqual(sample["observed_goodput_bps"], 4 * self.resource.stat().st_size * 8 / app_interval)
            self.assertGreater(sample["observed_body_bps"], sample["observed_goodput_bps"])

    def test_two_stream_bound_and_fresh_requests_no_video_cache(self):
        with self.link(max_concurrent_streams=2) as link:
            rows = link.download_batch(self.requests(4), deadline_s=4)
            self.assertGreaterEqual(min(row.start_s for row in rows[2:]), min(row.end_s for row in rows[:2]))
            self.assertEqual(link.transport_metadata()["max_observed_server_streams"], 2)
            self.assertEqual(link.delivered_bytes, 4 * self.resource.stat().st_size)
            self.assertEqual(link.requested_bytes, link.delivered_bytes)

    def test_request_contract_validation_before_io(self):
        with self.link() as link:
            wrong = self.requests(1)
            wrong[0]["requested_bytes"] -= 1
            with self.assertRaisesRegex(ValueError, "exact byte size"):
                link.download_batch(wrong)
            duplicate = self.requests(1) * 2
            with self.assertRaisesRegex(ValueError, "unique"):
                link.download_batch(duplicate)
            self.assertEqual(link.results, [])

    def test_idle_capacity_is_discarded_and_clock_wait_is_real(self):
        import time
        with self.link() as link:
            start = time.monotonic()
            link.advance_to(link.now + .1)
            self.assertGreaterEqual(time.monotonic() - start, .095)
            row = link.download_batch(self.requests(1), deadline_s=4)[0]
            # Even after idle trace capacity > request size, all body service
            # must occur after this response becomes ready on the actual wire.
            self.assertGreaterEqual(row.end_s - row.ready_s, self.resource.stat().st_size / 1e6 - .002)

    def test_protocol_guard_rejects_http1_and_changed_connection(self):
        from types import SimpleNamespace
        with self.link() as link:
            with self.assertRaisesRegex(RuntimeError, "negotiated HTTP/1.1"):
                link._check_protocol(SimpleNamespace(http_version="HTTP/1.1", extensions={}))
            tls = SimpleNamespace(selected_alpn_protocol=lambda: "h2")
            foreign_stream = SimpleNamespace(get_extra_info=lambda _: tls)
            with self.assertRaisesRegex(RuntimeError, "connection was replaced"):
                link._check_protocol(SimpleNamespace(http_version="HTTP/2", extensions={
                    "network_stream": foreign_stream, "stream_id": 3}))

    def test_wall_deadline_is_partial_delivery_and_not_estimator_sample(self):
        with self.link(max_concurrent_streams=2) as link:
            rows = link.download_batch(self.requests(2), deadline_s=link.now + .04)
            self.assertTrue(link.failed)
            self.assertTrue(any(row.delivered_bytes < row.requested_bytes for row in rows))
            self.assertTrue(all(row.completed_at_s is None for row in rows))
            self.assertEqual(link.failure_status, "time_limit")
            self.assertEqual(link.origin_app.active, 0)
            self.assertFalse(link.shaper.pending)
            released_at_disconnect = link.shaper.released_bytes
            link.advance_to(link.now + .03)
            self.assertEqual(link.shaper.released_bytes, released_at_disconnect)
            self.assertIsNone(CausalBandwidthEstimator().observe(rows[0], now=link.now))
            with self.assertRaisesRegex(ValueError, "blocked"):
                link.download_batch(self.requests(1, "retry"))

    def test_finite_trace_horizon_returns_partial_and_cancels_queue(self):
        from experiments.streaming_evaluation.real_http2 import RealHTTP2Link
        for policy in ("stop", "error"):
            with self.subTest(end_policy=policy):
                trace = NetworkTrace([(0, 8)], duration_s=.065, end_policy=policy)
                with RealHTTP2Link(trace, {"sha": self.resource}, max_concurrent_streams=1) as link:
                    rows = link.download_batch(self.requests(3), deadline_s=3.)
                    self.assertEqual(link.failure_status, "trace_exhausted")
                    self.assertEqual(rows[0].status, "trace_exhausted")
                    self.assertTrue(0 < rows[0].delivered_bytes < rows[0].requested_bytes)
                    self.assertTrue(all(row.status == "cancelled_before_service" for row in rows[1:]))
                    self.assertTrue(all(row.start_s is None and row.delivered_bytes == 0 for row in rows[1:]))
                    self.assertTrue(all(row.completed_at_s is None for row in rows))
                    self.assertEqual(link.origin_app.active, 0)
                    self.assertFalse(link.shaper.pending)
                    released = link.shaper.released_bytes
                    link.advance_to(link.now + .03)
                    self.assertEqual(link.shaper.released_bytes, released)
                    self.assertTrue(all(tick["delivered_bytes"] == 0 for tick in link.ledger
                                        if tick["end_s"] >= trace.duration_s))
                    self.assertTrue(all(tick["end_s"] <= link.now for tick in link.ledger))
                    with self.assertRaisesRegex(ValueError, "partial delivery"):
                        CausalBandwidthEstimator().observe_batch(rows, link.now)

    def test_expired_trace_never_admits_real_request(self):
        from experiments.streaming_evaluation.real_http2 import RealHTTP2Link
        with RealHTTP2Link(NetworkTrace([(0, 8)], duration_s=.02, end_policy="stop"),
                           {"sha": self.resource}) as link:
            link.advance_to(.03)
            row = link.download_batch(self.requests(1))[0]
            self.assertEqual(row.status, "cancelled_before_service")
            self.assertEqual(link.failure_status, "trace_exhausted")
            self.assertEqual(link.origin_app.events, [])
            self.assertEqual(link.shaper.released_bytes, 0)

    def test_single_large_body_has_no_64kib_per_tick_ceiling(self):
        from experiments.streaming_evaluation.real_http2 import RealHTTP2Link
        self.resource.write_bytes(b"x" * 1048576)
        with RealHTTP2Link(NetworkTrace([(0, 1600)], duration_s=5, end_policy="stop"),
                           {"sha": self.resource}) as link:
            row = link.download_batch(self.requests(1), deadline_s=4)[0]
            self.assertEqual(row.delivered_bytes, 1048576)
            self.assertEqual(link.transport_metadata()["shaping_chunk_bytes"], 1048576)
            self.assertGreater(max(tick["delivered_bytes"] for tick in link.ledger), 65536)
            self.assertTrue(all(tick["delivered_bytes"] * 8 <= tick["capacity_bits"] + 1e-6
                                for tick in link.ledger))

    def test_more_than_1000_requests_keep_the_same_actual_h2_connection(self):
        from experiments.streaming_evaluation.real_http2 import RealHTTP2Link
        self.resource.write_bytes(b'x')
        with RealHTTP2Link(NetworkTrace([(0, 8)],duration_s=30,end_policy='stop'),
                           {'sha':self.resource},latency_s=0,max_concurrent_streams=16) as link:
            rows=link.download_batch(self.requests(1005),deadline_s=25)
            rows+=link.download_batch(self.requests(3,'after-threshold'),deadline_s=25)
            self.assertTrue(all(r.status=='complete' for r in rows))
            self.assertEqual(len({r.stream_id for r in rows}),1008)
            self.assertEqual(len({r.server_client_port for r in rows}),1)
            self.assertEqual(link.transport_metadata()['connection_count'],1)
            self.assertGreater(link.transport_metadata()['server_keep_alive_max_requests'],1009)

"""Actual traffic accounting and aggregate busy-interval goodput.

Concurrent requests are grouped until all admitted requests finish. The sample
uses total successful eligible bytes / union of their busy intervals, including
TTFB. It is effective application goodput, not a sum of individual rates.
"""
from collections import Counter
import time
from multiobj3dgs.shared.transfer import BandwidthSample, to_bandwidth_sample
from .bandwidth import BandwidthEstimator


class TransferObserver:
    KINDS = frozenset(('metadata','media','shared_asset','profile','other'))
    def __init__(self, estimator=None, concurrency=1, events=None):
        if isinstance(concurrency,bool) or not isinstance(concurrency,int) or concurrency<1:
            raise ValueError('concurrency must be positive')
        self.estimator = estimator or BandwidthEstimator()
        self.concurrency = concurrency
        self.events = events
        self.totals = Counter()
        self._active = set()
        self._batch = []

    def started(self, spec):
        self._active.add(spec.request_id)

    def __call__(self, result):
        kind = result.metadata.get('traffic_kind','other')
        kind = kind if kind in self.KINDS else 'other'
        self.totals[kind] += result.received_bytes
        self.totals['total_bytes'] += result.received_bytes
        self.totals['transfer_count'] += 1
        if not result.success:
            self.totals['failed_bytes'] += result.received_bytes
            self.totals['failed_transfer_count'] += 1
        eligible = kind in ('media','shared_asset') and result.metadata.get('bandwidth_eligible',True)
        if self.events:
            self.events.emit('transfer',request_id=result.request_id,url=result.url,
                             bytes=result.received_bytes,success=result.success,
                             duration_s=result.total_duration,ttfb_s=result.ttfb,
                             http_version=result.http_version,traffic_kind=kind,
                             error=str(result.error) if result.error else None)
        if self.concurrency==1:
            if eligible:
                self.estimator.accept_transfer_sample(to_bandwidth_sample(result))
            return
        # Failed media intervals remain in the denominator but cannot create a
        # successful capacity sample. Completion must correspond to started().
        if result.request_id not in self._active:
            raise ValueError('Concurrent observer requires started(spec) before fetch')
        self._active.remove(result.request_id)
        if eligible:
            self._batch.append(result)
        if not self._active and self._batch:
            batch,self._batch = self._batch,[]
            spans = sorted((r.start_time,r.end_time) for r in batch)
            start,end=spans[0];busy=0.
            for a,b in spans[1:]:
                if a>end:
                    busy+=end-start;start,end=a,b
                else:
                    end=max(end,b)
            busy+=end-start
            size=sum(r.received_bytes for r in batch)
            if busy>0 and size>0 and all(r.success for r in batch):
                rate=size*8/busy
                self.estimator.accept_transfer_sample(BandwidthSample(time.time(),size,None,busy,busy,rate,rate,None,True))

    def cancelled(self, spec):
        # The transport normally reports a cancellation completion. This also
        # handles failures before transport admission without manufacturing bytes.
        self._active.discard(spec.request_id)
        if not self._active:
            self._batch.clear()

    def snapshot(self):
        return dict(self.totals)

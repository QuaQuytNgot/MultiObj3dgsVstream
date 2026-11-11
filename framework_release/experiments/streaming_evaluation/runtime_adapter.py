"""GOP-time live visibility over the existing common adaptation planner."""
import time
from pathlib import Path

from experiments.adaptation_baselines.interface import AdaptationPlanner
from experiments.visibility_weighted_sse.visibility import CachedVisibilityAdapter
from experiments.visibility_weighted_sse.common import digest, read_json, require


class ReadOnlyVisibilityEngine:
    """Reuse exact native visibility inputs for transport-only Phase A checks.

    No network residency is granted. Each lookup verifies the complete physical
    render identity and payload hash in a historical, read-only cache namespace.
    Missing inputs fail rather than silently invoking rendering during a real
    network trace. The renderer implementation is not changed.
    """
    def __init__(self, source_root, budget):
        self.source_root, self.budget = Path(source_root).resolve(), budget
        self.fingerprint = digest(read_json(self.source_root/'manifest.json'))
        self.stats = dict(native_jobs_completed=0, native_jobs_failed=0, image_jobs=0,
                          visibility_jobs=0, visibility_cache_hits=0, native_wall_seconds=0.)

    def visibility(self, pose, frame):
        from .runtime_native import physical_inputs
        started = time.perf_counter()
        identity = dict(schema='streaming-native-job.v1', fingerprint=self.fingerprint,
            kind='proxy', **physical_inputs(pose, frame),
            allocation={r['instance_id']:40 for r in pose.instances})
        path = self.source_root/'cache/visibility'/f'{digest(identity)}.json'
        require(path.is_file(), f'Exact read-only visibility input missing: {path}')
        record = read_json(path)
        require(record['identity']==identity and digest(record['payload'])==record['payload_sha256'],
                'Read-only visibility identity/payload mismatch')
        self.stats['visibility_cache_hits'] += 1
        return record['payload'], dict(cache_hit=True, cache_lookup_ms=(time.perf_counter()-started)*1000,
            native_wall_ms=0., analysis_ms=0., original_analysis_ms=record['analysis_ms'],
            original_native_wall_ms=record['native_wall_ms'], source='verified_read_only_native_visibility')

    def close(self):
        pass


class RuntimeVisibilityAdapter(CachedVisibilityAdapter):
    """Compute only requested upcoming GOPs; exact repeated inputs may reuse work.

    The parent supplies unchanged validation and temporal aggregation. No
    content-preparation visibility file is required or produced.
    """
    def __init__(self, engine, trace_provider, profiles):
        self.engine, self.trace_provider = engine, trace_provider
        self.compatibility = dict(profiles.compatibility)
        self.data = dict(content_revision=profiles.content_revision)
        self.provenance = dict(information_role='VISIBILITY_PROXY', candidate_independent=True,
            joint_quality_used=False, source='runtime src_final VisibilityAnalyzer on native CRF40 states')
        self._results, self.timings, self.events = {}, {}, []

    def get_results(self, gop_id):
        gid = str(gop_id)
        if gid in self._results:
            return self._results[gid]
        samples, events = [], []
        for pose in self.trace_provider.get_trajectory(gid):
            self.engine.budget.check()
            frame = round(pose.timestamp*30) % 180
            payload, timing = self.engine.visibility(pose, frame)
            samples.append(dict(timestamp=pose.timestamp, pose_identity=pose.identity, per_object=payload))
            events.append(dict(global_gop_id=gid, global_time_s=pose.timestamp,
                               source_frame=frame, **timing))
        document = dict(schema='visibility-weighted-sse.visibility.v1',
            trace_id=self.trace_provider.trace_id, content_revision=self.data['content_revision'],
            compatibility=self.compatibility, provenance=self.provenance,
            gops=[dict(gop_id=gid, samples=samples)])
        validated = CachedVisibilityAdapter(document, self.trace_provider,
            expected_compatibility=self.compatibility, content_revision=self.data['content_revision'])
        self._results[gid] = validated.get_results(gid)
        self.events.extend(events)
        self.timings[gid] = dict(
            visibility_analysis_ms=sum(e['analysis_ms'] for e in events),
            visibility_native_pipeline_ms=sum(e['native_wall_ms'] for e in events),
            visibility_cache_lookup_ms=sum(e['cache_lookup_ms'] for e in events),
            visibility_original_analysis_ms=sum(e['original_analysis_ms'] for e in events),
            visibility_sample_count=len(events),
            visibility_computed_sample_count=sum(not e['cache_hit'] for e in events),
            visibility_cached_sample_count=sum(e['cache_hit'] for e in events))
        return self._results[gid]


class RuntimePlanner(AdaptationPlanner):
    """Add measured visibility timing without altering any utility or solver."""
    def plan(self, method, gop_id, budget_bytes):
        reused = str(gop_id) in self._prepared
        result = super().plan(method, gop_id, budget_bytes)
        timing = dict(self.visibility.timings[str(gop_id)])
        if reused:
            # Preparation happened on a previous call to this exact planner.
            for name in ('visibility_analysis_ms', 'visibility_native_pipeline_ms',
                         'visibility_cache_lookup_ms', 'visibility_computed_sample_count'):
                timing[name] = 0
        result['timings'].update(timing)
        result['timings']['decision_compute_excluding_preparation_ms'] = max(0.,
            result['timings']['decision_call_ms'] - (0. if reused else
                result['timings']['shared_GOP_preparation_ms']))
        return result

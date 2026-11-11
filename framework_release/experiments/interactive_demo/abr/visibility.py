"""Sample existing StreamSTGS alpha/transmittance analysis on displayed states.

No second decode or RGB reconstruction is performed. The production analyzer
uses identity-color contribution passes, including all scene occluders, on the
immutable Gaussian tensors already reconstructed for the native viewer.
"""
from __future__ import annotations

import math
import time


class RuntimeVisibility:
    """Current-pose hold with a bounded analysis viewport and sampling cadence."""

    def __init__(self, *, interval_s=1.0, analysis_width=256, analyzer=None):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError('visibility interval must be finite and positive')
        if type(analysis_width) is not int or not 16 <= analysis_width <= 2048:
            raise ValueError('analysis_width must be an integer in [16, 2048]')
        if analyzer is None:
            from multiobj3dgs.client.visibility.analyzer import VisibilityAnalyzer
            analyzer = VisibilityAnalyzer()
        self.analyzer = analyzer
        self.interval_s, self.analysis_width = float(interval_s), analysis_width
        self.last_result = None
        self.last_sample_wall = None
        self.sample_count = 0
        self.total_seconds = 0.0
        self.last_seconds = 0.0
        self.snapshot = None

    def sample(self, runtime, camera, *, force=False):
        """Call on the owner thread after render/refresh at the displayed pose.

        Timestamp and instance identities come from that actual reconstruction,
        rather than the already-advanced presentation clock. A missing snapshot
        remains missing; startup uniform weights are a separate planner policy.
        """
        runtime._assert_owner()
        if not runtime.last_states or runtime.displayed_time is None:
            return None
        now = time.monotonic()
        if not force and self.last_sample_wall is not None and now-self.last_sample_wall < self.interval_s:
            return None
        sources = tuple(runtime._last_frame_sources)
        if len(sources) != len(runtime.last_states):
            raise ValueError('displayed states and instance provenance differ')
        objects = {row['instance_id']: state for row, state in zip(sources, runtime.last_states)}
        width = min(self.analysis_width, camera.width)
        height = max(1, round(camera.height*width/camera.width))
        analysis_camera = camera.resized(width, height) if (width, height) != (camera.width, camera.height) else camera
        began = time.monotonic()
        result = self.analyzer.analyze(objects, analysis_camera,
            timestamp=runtime.displayed_time, revision=runtime.catalog.revision,
            diagnostics=False)
        self.last_seconds = time.monotonic()-began
        if set(result.per_object) != set(objects):
            raise ValueError('visibility analyzer did not return every displayed instance')
        pixel_count = width*height
        weights = {}
        for identifier, contribution in result.per_object.items():
            value = float(contribution.A_visible)
            if not math.isfinite(value) or value < 0:
                raise ValueError('visibility contribution must be finite and nonnegative')
            weights[identifier] = value/pixel_count
        runtime.visibility = weights
        for job in runtime.jobs.values():
            job.visibility = weights.get(job.instance_id, 0.0)
        self.last_result, self.last_sample_wall = result, now
        self.sample_count += 1
        self.total_seconds += self.last_seconds
        self.snapshot = {
            'timestamp': runtime.displayed_time,
            'camera_policy': 'current_pose_hold',
            'weight_policy': 'joint_alpha_transmittance_area_divided_by_viewport_pixels',
            'resolution': [width, height],
            'backend': result.metadata.get('backend', 'streamstgs_cuda_contribution'),
            'exact_fast_contribution': result.metadata.get('exact_fast_contribution', False),
            'weights': weights,
            'visible_area': {key: float(row.A_visible) for key, row in result.per_object.items()},
            'solo_area': {key: float(row.A_solo) for key, row in result.per_object.items()},
            'source_frames': list(sources),
            'duration_s': self.last_seconds,
            'sample_count': self.sample_count,
        }
        runtime.emit('abr_visibility', **self.snapshot)
        return result

    def invalidate(self):
        self.last_result = self.snapshot = self.last_sample_wall = None


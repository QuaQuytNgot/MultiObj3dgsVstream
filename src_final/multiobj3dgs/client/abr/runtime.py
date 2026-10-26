"""GOP quality authority over the production StreamSTGS playback runtime.

Only target selection is specialized. HTTP verification, shared immutable
payloads, independent decoder handles, progressive CPU preparation, incremental
GPU uploads, event readiness, byte budgets and clock controls remain owned by
``multiobj3dgs.client.runtime.PlaybackRuntime``.
"""
from __future__ import annotations

import time

from multiobj3dgs.client.adaptation import Candidate, PlannerInput, Selection, SelectionPlan
from multiobj3dgs.client.runtime import PlaybackRuntime
from multiobj3dgs.client.scene import media_time


class QualityAwarePlaybackRuntime(PlaybackRuntime):
    """Freeze decisions on admission, switching only at object GOP boundaries.

    ``selector.plan(targets, predicted_bandwidth_bps, visibility, context=...)``
    returns ``{instance_id: representation_id}``. Targets hold one GOP index per
    instance; already admitted peers are supplied as locked representations.
    A single upcoming GOP per instance bounds network/decode lookahead. This
    avoids assigning the entire bandwidth budget independently to every GOP in
    a longer prefetch window.
    """

    def __init__(self, *args, visibility_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.selector, 'plan'):
            raise TypeError('quality-aware runtime requires a GOP planner')
        self.visibility_sampler = visibility_sampler
        self._abr_choices = {}
        self._abr_scene_cycle = 0
        self._abr_last_scene_time = self.clock.position
        self._abr_generation = self.clock.generation
        self.abr_decision_count = 0
        self.abr_plan_seconds = 0.0
        self.abr_last_context = None
        self.effective_abr_prefetch_gops = min(1, self.prefetch)

    def _invalidate(self):
        super()._invalidate()
        if hasattr(self, '_abr_choices'):
            self._abr_choices.clear()
            self._abr_generation = self.clock.generation
            self._abr_scene_cycle = 0
            self._abr_last_scene_time = self.clock.position
            self.visibility = {}
            if self.visibility_sampler is not None:
                self.visibility_sampler.invalidate()

    def request_quality(self, instance_id, representation_id):
        """Keep existing native quality controls as a next-boundary override."""
        super().request_quality(instance_id, representation_id)
        # The displayed GOP is immutable; future requests are replaceable. The
        # inherited pump cancels their consumers and drains preparation owners.
        keep = self.required.get(instance_id)
        active_job = self.jobs.get(keep)
        for token in tuple(self._abr_choices):
            if token[2] == instance_id and (active_job is None or
                    (token[3], token[4]) != (active_job.key[-1][-1], active_job.segment.gop_index)):
                del self._abr_choices[token]

    def _token(self, instance, cycle, segment):
        return (self.clock.generation, self._abr_scene_cycle, instance.id, cycle, segment.gop_index)

    def _job_key(self, instance, cycle, segment, representation):
        return (instance.id, segment.id, representation,
                (self.clock.generation, self._abr_scene_cycle, cycle))

    def _plan(self, cohort, name):
        targets = {identifier: entry['segment'].gop_index for identifier, entry in cohort.items()}
        locked = {identifier: self._abr_choices[entry['token']]
                  for identifier, entry in cohort.items() if entry['token'] in self._abr_choices}
        for identifier, entry in cohort.items():
            if identifier not in locked and identifier in self.pending_representations:
                locked[identifier] = self.pending_representations[identifier]
        estimate = self.bandwidth_estimate
        bandwidth = estimate.bps if estimate is not None else None
        context = {'generation': self.clock.generation, 'scene_cycle': self._abr_scene_cycle,
                   'scene_time': self.clock.position, 'cohort': name,
                   'locked_representations': locked,
                   'cycles': {identifier: entry['cycle'] for identifier, entry in cohort.items()},
                   'visibility_policy': 'current_pose_hold' if self.visibility else 'startup_uniform_unmeasured',
                   'visibility_snapshot': self.visibility_sampler.snapshot if self.visibility_sampler else None}
        began = time.perf_counter()
        chosen = self.selector.plan(targets, bandwidth, dict(self.visibility), context=context)
        elapsed = time.perf_counter()-began
        self.abr_plan_seconds += elapsed
        self.abr_decision_count += 1
        self.abr_last_context = context
        if set(chosen) != set(targets):
            raise ValueError('planner must select exactly one representation per target instance')
        for identifier, representation in chosen.items():
            entry = cohort[identifier]
            content = self.catalog.contents[entry['instance'].content_id]
            if representation not in content.representations:
                raise ValueError(f'planner selected unknown representation {representation!r}')
            if identifier in locked and representation != locked[identifier]:
                raise ValueError('planner changed a committed GOP representation')
            self._abr_choices[entry['token']] = representation
        self.emit('abr_admission', cohort=name, scene_time=self.clock.position,
                  duration_s=elapsed, targets=targets, selected=chosen,
                  locked_representations=locked, decision_count=self.abr_decision_count)

    def _desired(self):
        # Explicit seek/restart changes generation. A natural scene loop keeps
        # the production clock generation and receives its own decision epoch.
        if self.clock.generation != self._abr_generation:
            self._abr_choices.clear()
            self._abr_generation = self.clock.generation
            self._abr_scene_cycle = 0
        elif self.clock.position < self._abr_last_scene_time-1e-9:
            self._abr_scene_cycle += 1
            self._abr_choices.clear()
        self._abr_last_scene_time = self.clock.position
        current, upcoming = {}, {}
        for instance in self.catalog.instances:
            content = self.catalog.contents[instance.content_id]
            timestamp = media_time(instance, content, self.clock.position)
            before_activation = timestamp is None and self.clock.position < instance.scene_start
            if timestamp is None:
                if not before_activation:
                    continue
                timestamp = media_time(instance, content, instance.scene_start)
            if timestamp is None:
                continue
            anchor = self.active_representations[instance.id]
            segment = content.segment_at(anchor, timestamp)
            raw_time = instance.media_offset+max(0.0, self.clock.position-instance.scene_start)
            cycle = max(0, round((raw_time-timestamp)/content.duration)) if instance.loop else 0
            distance = max(0.0, instance.scene_start-self.clock.position) if before_activation else 0.0
            entry = dict(instance=instance, segment=segment, cycle=cycle,
                         timestamp=timestamp, before_activation=before_activation,
                         deadline=time.monotonic()+distance)
            entry['token'] = self._token(instance, cycle, segment)
            current[instance.id] = entry
            if self.effective_abr_prefetch_gops:
                segments = content.representations[anchor].segments
                index = segment.gop_index+1
                next_cycle = cycle
                if index == len(segments):
                    if not instance.loop:
                        continue
                    index, next_cycle = 0, cycle+1
                candidate = segments[index]
                delta = segment.end-timestamp
                # Do not fetch beyond an independent instance's active window.
                target_scene_time = (instance.scene_start if before_activation else self.clock.position)+delta
                active_duration = instance.active_duration if instance.active_duration is not None else (
                    self.catalog.duration-instance.scene_start if instance.loop else content.duration-instance.media_offset)
                end_scene = instance.scene_start+active_duration
                if target_scene_time >= end_scene-1e-9:
                    continue
                future = dict(instance=instance, segment=candidate, cycle=next_cycle,
                              timestamp=candidate.start, before_activation=True,
                              deadline=time.monotonic()+distance+delta)
                future['token'] = self._token(instance, next_cycle, candidate)
                upcoming[instance.id] = future
        planned_current = any(row['token'] not in self._abr_choices for row in current.values())
        if planned_current:
            self._plan(current, 'current')
        desired, required = {}, {}

        def admit(identifier, entry, *, is_current):
            token = entry['token']
            if token not in self._abr_choices:
                return
            representation = self._abr_choices[token]
            content = self.catalog.contents[entry['instance'].content_id]
            segment = content.representations[representation].segments[entry['segment'].gop_index]
            key = self._job_key(entry['instance'], entry['cycle'], segment, representation)
            desired[key] = (entry['instance'], representation, segment, entry['deadline'])
            if is_current:
                self.active_representations[identifier] = representation
                # A manual request can equal the displayed GOP while replacing
                # a different prefetched representation. Consume it only when
                # entering the target job, not while readmitting the same GOP.
                if (self.pending_representations.get(identifier) == representation and
                        self.required.get(identifier) != key):
                    del self.pending_representations[identifier]
                if not entry['before_activation']:
                    required[identifier] = key

        for identifier, entry in current.items():
            admit(identifier, entry, is_current=True)
        self.required = required
        current_ready = bool(required) and all(key in self.jobs for key in required.values()) and self.ready
        if (not planned_current and current_ready and upcoming and
                any(row['token'] not in self._abr_choices for row in upcoming.values())):
            self._plan(upcoming, 'upcoming')
        for identifier, entry in upcoming.items():
            admit(identifier, entry, is_current=False)
        retained = {entry['token'] for entry in (*current.values(), *upcoming.values())}
        self._abr_choices = {key: value for key, value in self._abr_choices.items() if key in retained}
        candidates, selections = [], []
        buffers = {}
        for key, (instance, representation, segment, deadline) in desired.items():
            missing = self.cache.missing_bytes(segment) if hasattr(self.cache, 'missing_bytes') else segment.cold_bytes
            candidates.append(Candidate(instance.id, segment.id, representation, missing, segment.duration, deadline))
            selections.append(Selection(instance.id, segment.id, representation, deadline))
            if required.get(instance.id) == key:
                job = self.jobs.get(key)
                buffers[instance.id] = max(0., segment.end-current[instance.id]['timestamp']) if job and job.handle else 0.
        self.planner_input = PlannerInput(self.catalog.revision, tuple(candidates),
            self.bandwidth_estimate.bps if self.bandwidth_estimate is not None else None,
            dict(self.visibility), buffers,
            {'selection_policy': 'frozen_on_admission; independent_GOP_boundaries',
             'decision': getattr(self.selector, 'last_decision', None)})
        self.last_selection_plan = SelectionPlan(self.catalog.revision, tuple(selections), 'quality-aware-gop')
        return desired

    def update_visibility(self, camera, *, force=False):
        if self.visibility_sampler is None:
            return None
        return self.visibility_sampler.sample(self, camera, force=force)

    def diagnostics(self):
        result = super().diagnostics()
        result['network'] = dict(getattr(self, 'network_diagnostics', {}) or {})
        result['abr'] = {'decisions': self.abr_decision_count,
                         'planning_seconds': self.abr_plan_seconds,
                         'visibility_weights': dict(self.visibility),
                         'visibility_snapshot': self.visibility_sampler.snapshot if self.visibility_sampler else None,
                         'active_representations': dict(self.active_representations),
                         'pending_representations': dict(self.pending_representations),
                         'last_decision': getattr(self.selector, 'last_decision', None),
                         'latest_predicted_bandwidth_bps': self.bandwidth_estimate.bps if self.bandwidth_estimate else None,
                         'prefetch_requested_gops': self.prefetch,
                         'prefetch_effective_gops': self.effective_abr_prefetch_gops,
                         'prefetch_policy': 'at_most_one_budgeted_upcoming_GOP_per_instance'}
        return result

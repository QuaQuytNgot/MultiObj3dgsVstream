"""EDF admission with visibility tie-breaks and starvation protection.

Adapted from the priority ordering in the existing decode_scheduler.py; this
policy schedules preparation and deliberately never selects a representation.
"""
import time


class PreparationScheduler:
    def __init__(self, starvation_s=.75, clock=time.monotonic):
        if starvation_s <= 0:
            raise ValueError('starvation interval must be positive')
        self.starvation_s, self.clock = starvation_s, clock

    def order(self, jobs):
        now = self.clock()
        def rank(job):
            age = now - job.created
            if age >= self.starvation_s:
                return (0, -age, job.deadline, job.sequence)
            return (1, job.deadline, -job.visibility, job.sequence)
        return sorted(jobs, key=rank)

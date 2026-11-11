"""Explicit byte admission. Reservations include queued and running preparation."""
from dataclasses import dataclass


class ResourceLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryLimits:
    host_bytes: int = 2 * 1024**3
    gpu_bytes: int = 4 * 1024**3
    raw_bytes: int = 512 * 1024**2
    transient_gpu_bytes: int = 256 * 1024**2

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError('memory budgets must be positive integer bytes')
        if self.transient_gpu_bytes >= self.gpu_bytes:
            raise ValueError('GPU transient reserve must be smaller than GPU budget')


class ByteBudget:
    def __init__(self, capacity, name):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('capacity must be positive integer bytes')
        self.capacity, self.name = capacity, name
        self.reservations = {}
        self.peak = 0

    @property
    def used(self):
        return sum(self.reservations.values())

    def reserve(self, key, nbytes):
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError('reservation must be nonnegative integer bytes')
        if nbytes > self.capacity:
            raise ResourceLimitError(f'{self.name}: item needs {nbytes} bytes, budget is {self.capacity}')
        after = self.used - self.reservations.get(key, 0) + nbytes
        if after > self.capacity:
            return False
        self.reservations[key] = nbytes
        self.peak = max(self.peak, after)
        return True

    def release(self, key):
        self.reservations.pop(key, None)

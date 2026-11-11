"""Causal visibility inputs shared by every adaptation method.

Synthetic weights describe a reproducible camera workload in logical playback
time. They are not measured visibility or inferred viewport quality. Recorded
inputs use hold-current-sample semantics; before their first sample the provider
uses uniform bootstrap weights, and after their last sample it holds the last.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
from pathlib import Path


class VisibilityProvider:
    def __init__(self, seed=2027, mode="synthetic", trace_path=None):
        if mode not in ("uniform", "synthetic", "recorded", "trace"):
            raise ValueError("Visibility mode must be uniform, synthetic, or recorded")
        self.seed, self.mode = int(seed), mode
        self.time_basis = "media"
        self.samples = []
        self._times = []
        if mode in ("recorded", "trace"):
            if trace_path is None:
                raise ValueError("Recorded visibility needs trace_path")
            self._load(Path(trace_path))

    @staticmethod
    def _finite(value, label, nonnegative=True):
        number = float(value)
        if not math.isfinite(number) or (nonnegative and number < 0):
            raise ValueError(f"Invalid visibility {label}")
        return number

    def _load(self, path):
        # JSON: {"time_basis":"media"|"wall", "samples":[{"time_s":0,
        # "weights":{"object-000":0.8,...}},...]}. A bare sample list also works.
        # CSV: wide time_s,<instance ids...>, or long time_s,instance_id,weight.
        # CSV uses logical media time, making camera workload playback-aligned.
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text())
            if isinstance(value, dict):
                self.time_basis = value.get("time_basis", "media")
                records = value["samples"]
            else:
                records = value
            if self.time_basis not in ("media", "wall"):
                raise ValueError("Visibility time_basis must be media or wall")
            seen = set()
            for record in records:
                timestamp = self._finite(record["time_s"], "timestamp")
                if timestamp in seen:
                    raise ValueError("Duplicate visibility sample timestamp")
                seen.add(timestamp)
                self.samples.append((timestamp, {str(k): self._finite(v, "weight")
                                                for k, v in record["weights"].items()}))
        elif path.suffix.lower() == ".csv":
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames or []
                if "time_s" not in fields:
                    raise ValueError("Visibility CSV requires time_s")
                long_format = "instance_id" in fields and "weight" in fields
                samples = {}
                for record in reader:
                    timestamp = self._finite(record["time_s"], "timestamp")
                    if long_format:
                        weights = samples.setdefault(timestamp, {})
                        identifier = record["instance_id"]
                        if identifier in weights:
                            raise ValueError("Duplicate recorded instance at one timestamp")
                        weights[identifier] = self._finite(record["weight"], "weight")
                    else:
                        if timestamp in samples:
                            raise ValueError("Duplicate visibility sample timestamp")
                        samples[timestamp] = {k: self._finite(record[k], "weight")
                                              for k in fields if k != "time_s"}
                self.samples = list(samples.items())
        else:
            raise ValueError("Recorded visibility trace must be CSV or JSON")
        if not self.samples:
            raise ValueError("Visibility trace is empty")
        self.samples.sort(key=lambda row: row[0])
        self._times = [sample[0] for sample in self.samples]

    def weights(self, instance_ids: list[str], wall_time_s: float,
                media_time_s: float) -> dict[str, float]:
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("Visibility needs unique instance identifiers")
        wall = self._finite(wall_time_s, "wall time")
        media = self._finite(media_time_s, "media time")
        if not instance_ids:
            return {}
        if self.mode == "synthetic":
            values = {}
            for identifier in instance_ids:
                digest = hashlib.sha256(f"{self.seed}:{identifier}".encode()).digest()
                phase = int.from_bytes(digest[:8], "big") / 2**64 * 2 * math.pi
                period = 7.0 + int.from_bytes(digest[8:12], "big") / 2**32 * 10
                values[identifier] = 0.05 + ((1 + math.sin(2 * math.pi * media / period + phase)) / 2)**2
        elif self.mode in ("recorded", "trace"):
            timestamp = media if self.time_basis == "media" else wall
            index = bisect.bisect_right(self._times, timestamp + 1e-10) - 1
            if index < 0:
                return {identifier: 1.0 / len(instance_ids) for identifier in instance_ids}
            current = self.samples[index][1]
            values = {identifier: current.get(identifier, 0.0) for identifier in instance_ids}
        else:
            values = {identifier: 1.0 for identifier in instance_ids}
        total = math.fsum(values.values())
        if total <= 0:
            # Production preserves an explicitly all-invisible scene. Uniform
            # bootstrap applies only before any recorded sample has occurred.
            return {identifier: 0.0 for identifier in instance_ids}
        return {identifier: values[identifier] / total for identifier in instance_ids}

    def next_boundary(self, wall_time_s: float, media_time_s: float,
                      playing=True) -> float:
        """Next recorded transition as an absolute virtual wall timestamp.

        The simulator splits integration at this timestamp. This is visibility
        schedule information, never a network-capacity prediction. Frozen media
        time cannot reach a media-based boundary while the scene is stalled.
        """
        wall = self._finite(wall_time_s, "wall time")
        media = self._finite(media_time_s, "media time")
        if self.mode not in ("recorded", "trace") or (self.time_basis == "media" and not playing):
            return math.inf
        timestamp = media if self.time_basis == "media" else wall
        index = bisect.bisect_right(self._times, timestamp + 1e-10)
        if index == len(self._times):
            return math.inf
        boundary = self._times[index]
        return wall + boundary - media if self.time_basis == "media" else boundary

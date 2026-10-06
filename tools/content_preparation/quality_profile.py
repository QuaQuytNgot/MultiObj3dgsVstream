"""Raw view/scale/time-conditioned decoded quality measurements and lookup.

The highest delivered decoded quality is the reference. Aggregations are stored
as reporting conveniences; the original sampled rows remain queryable.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .metrics import aggregate


def _json_number(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        raise ValueError("NaN metrics are not allowed in quality profiles.")
    return "inf" if value == math.inf else "-inf" if value == -math.inf else value


def _safe_json(value):
    if isinstance(value, Mapping):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return _json_number(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(_safe_json(document), indent=2, sort_keys=True, allow_nan=False)+"\n")
    os.replace(temporary, path)


def _azimuth(value):
    return float(value) % 360


def _row_key(row):
    return (row.get("object", ""), row.get("mode", "independent"), str(row["quality"]),
            str(row["view_bin"]), str(row["scale_bin"]), str(row["time_bin"]))


def _metrics_average(rows, weights):
    names = sorted(set().union(*(row["metrics"].keys() for row in rows)))
    result = {}
    for name in names:
        present = [(row["metrics"].get(name), weight) for row, weight in zip(rows, weights) if weight > 0]
        if any(value is None for value, _ in present):
            result[name] = None
        else:
            result[name] = _json_number(sum(float(value)*float(weight) for value, weight in present)/sum(weight for _, weight in present))
    return result


class QualityProfile:
    def __init__(self, rows: Sequence[Mapping] | None = None, metadata: Mapping | None = None):
        self.metadata = dict(metadata or {})
        self.rows = []
        self._index = {}
        for row in rows or []:
            self.add(row)

    def add(self, row: Mapping):
        row = _safe_json(dict(row))
        for name in ("quality", "azimuth", "elevation", "scale", "timestamp", "metrics"):
            if name not in row:
                raise ValueError(f"Quality profile row is missing {name}.")
        if float(row["scale"]) <= 0:
            raise ValueError("Quality profile scale must be positive.")
        for name in ("azimuth", "elevation", "scale", "timestamp"):
            if not math.isfinite(float(row[name])):
                raise ValueError(f"Quality profile coordinate {name} must be finite.")
        row.setdefault("object", self.metadata.get("object_id", ""))
        row.setdefault("mode", "independent")
        row.setdefault("view_bin", f"az{row['azimuth']}_el{row['elevation']}")
        row.setdefault("scale_bin", f"scale{row['scale']}")
        row.setdefault("time_bin", str(row.get("frame", row["timestamp"])))
        if "mse" not in row["metrics"] or float(row["metrics"]["mse"]) < 0:
            raise ValueError("Quality profile metrics require nonnegative MSE.")
        key = _row_key(row)
        if key in self._index:
            if self._index[key] != row:
                raise ValueError(f"Duplicate profile index with different measurement: {key}")
            return
        self.rows.append(row)
        self._index[key] = row

    def indexed(self, object_id, quality, view_bin, scale_bin, time_bin, mode="independent"):
        return self._index[(object_id, mode, str(quality), str(view_bin), str(scale_bin), str(time_bin))]

    def to_dict(self, quality_order=None):
        document = {"schema": "dynamic-lapisgs.quality-profile.v1", "metadata": self.metadata,
                    "reference": "highest_quality_decoded_delivered_representation",
                    "coordinate_convention": {"azimuth": "degrees_periodic_360", "elevation": "degrees",
                                              "scale": "projected_size_multiplier", "timestamp": "seconds"},
                    "rows": self.rows, "aggregates": self.summarize()}
        if quality_order:
            document["transition_gains"] = self.transition_gains(quality_order)
        return _safe_json(document)

    def save(self, path, quality_order=None):
        _write_json(path, self.to_dict(quality_order))

    @classmethod
    def load(cls, path):
        document = json.loads(Path(path).read_text())
        if document.get("schema") != "dynamic-lapisgs.quality-profile.v1":
            raise ValueError("Unsupported quality profile schema.")
        return cls(document["rows"], document.get("metadata"))

    def summarize(self, weights: Mapping | None = None):
        result = []
        groups = {}
        for row in self.rows:
            groups.setdefault((row["object"], row["mode"], row["quality"]), []).append(row)
        for (object_id, mode, quality), rows in sorted(groups.items()):
            entry = {"object": object_id, "mode": mode, "quality": quality, "sample_count": len(rows), "metrics": {}}
            names = sorted(set().union(*(row["metrics"].keys() for row in rows)))
            for name in names:
                values = [row["metrics"].get(name) for row in rows]
                if any(value is None for value in values):
                    entry["metrics"][name] = None
                    continue
                numeric = [float(value) for value in values]
                summaries = {method: _json_number(aggregate(numeric, method))
                             for method in ("uniform_mean", "median", "p95", "worst")}
                if weights is not None:
                    sample_weights = [weights.get(_row_key(row), 1.) for row in rows]
                    summaries["weighted_mean"] = _json_number(aggregate(numeric, "weighted_mean", sample_weights))
                if name in {"psnr", "ssim"}:
                    # Higher-is-better reporting metrics need the minimum to
                    # express worst quality, unlike MSE and LPIPS distortion.
                    summaries["worst"] = _json_number(min(numeric))
                entry["metrics"][name] = summaries
            result.append(entry)
        return result

    def transition_gains(self, quality_order: Sequence[str]):
        """Distortion reductions retain all view/scale/time conditions and signs."""
        groups = {}
        for row in self.rows:
            key = (row["object"], row["mode"], str(row["view_bin"]), str(row["scale_bin"]), str(row["time_bin"]))
            groups.setdefault(key, {})[row["quality"]] = row
        gains = []
        for key, qualities in sorted(groups.items()):
            for q_from, q_to in zip(quality_order, quality_order[1:]):
                if q_from not in qualities or q_to not in qualities:
                    raise ValueError(f"Missing transition profile sample for {q_from} -> {q_to}: {key}")
                before, after = qualities[q_from], qualities[q_to]
                coordinate_names = ("azimuth", "elevation", "scale", "timestamp")
                if any(float(before[name]) != float(after[name]) for name in coordinate_names):
                    raise ValueError("Transition gains require identical sampled view/scale/time coordinates.")
                metrics = {}
                for name in ("mse", "lpips"):
                    a, b = before["metrics"].get(name), after["metrics"].get(name)
                    metrics[name] = None if a is None or b is None else _json_number(float(a)-float(b))
                gains.append({"object": key[0], "mode": key[1], "view_bin": key[2], "scale_bin": key[3], "time_bin": key[4],
                              "q_from": q_from, "q_to": q_to, "frame": before.get("frame"),
                              **{name: before[name] for name in coordinate_names}, "distortion_reduction": metrics})
        return gains

    def query(self, quality, azimuth, elevation, scale, timestamp, mode=None, method="nearest", object_id=None, neighbors=16):
        if not all(math.isfinite(float(v)) for v in (azimuth, elevation, scale, timestamp)) or scale <= 0:
            raise ValueError("Query coordinates must be finite and scale must be positive.")
        rows = [row for row in self.rows if row["quality"] == quality
                and (mode is None or row["mode"] == mode)
                and (object_id is None or row["object"] == object_id)]
        if not rows:
            raise KeyError(f"No quality profile samples for {quality} / {mode} / {object_id}.")
        if len({(row["object"], row["mode"]) for row in rows}) > 1:
            raise ValueError("Specify mode/object_id to resolve multiple profile grids.")
        query = [_azimuth(azimuth), float(elevation), float(scale), float(timestamp)]
        if method == "multilinear":
            selected, weights = self._multilinear(rows, query)
        elif method in {"nearest", "inverse_distance"}:
            coordinates = np.array([[_azimuth(row["azimuth"]), float(row["elevation"]), math.log(float(row["scale"])), float(row["timestamp"])] for row in rows])
            q = np.array([query[0], query[1], math.log(query[2]), query[3]])
            difference = coordinates-q
            difference[:, 0] = (difference[:, 0]+180) % 360-180
            span = np.maximum(np.ptp(coordinates, axis=0), [180., 45., 1., 1e-6])
            distance = np.linalg.norm(difference/span, axis=1)
            ordered = np.argsort(distance, kind="stable")
            if method == "nearest" or distance[ordered[0]] < 1e-12:
                selected, weights = [rows[int(ordered[0])]], [1.]
            else:
                if neighbors <= 0:
                    raise ValueError("Inverse-distance interpolation neighbors must be positive.")
                selected_indices = ordered[:int(neighbors)]
                selected = [rows[int(i)] for i in selected_indices]
                weights = 1/distance[selected_indices]**2
                weights /= weights.sum()
        else:
            raise ValueError("Profile query method must be nearest, inverse_distance, or multilinear.")
        return {"object": rows[0]["object"], "mode": rows[0]["mode"], "quality": quality,
                "azimuth": float(azimuth), "elevation": float(elevation), "scale": float(scale), "timestamp": float(timestamp),
                "metrics": _metrics_average(selected, weights), "interpolation": method,
                "source_samples": [{"view_bin": row["view_bin"], "scale_bin": row["scale_bin"], "time_bin": row["time_bin"], "weight": float(weight)}
                                   for row, weight in zip(selected, weights)]}

    @staticmethod
    def _multilinear(rows, query):
        coordinates = [tuple([_azimuth(row["azimuth"]), float(row["elevation"]), float(row["scale"]), float(row["timestamp"])]) for row in rows]
        grid = dict(zip(coordinates, rows))
        brackets = []
        for axis in range(4):
            values = sorted({coordinate[axis] for coordinate in coordinates})
            target = query[axis]
            exact = next((value for value in values if abs(value-target) < 1e-12), None)
            if exact is not None or len(values) == 1:
                brackets.append([(values[0] if exact is None else exact, 1.)])
                continue
            if axis == 0:
                # Periodic bracketing, including the seam between +180/-180.
                extended = [(values[-1]-360, values[-1])]+[(v, v) for v in values]+[(values[0]+360, values[0])]
                lower = max((v for v in extended if v[0] <= target), key=lambda v: v[0])
                upper = min((v for v in extended if v[0] >= target), key=lambda v: v[0])
                fraction = (target-lower[0])/(upper[0]-lower[0])
                brackets.append([(lower[1], 1-fraction), (upper[1], fraction)])
            elif target <= values[0] or target >= values[-1]:
                brackets.append([(values[0] if target <= values[0] else values[-1], 1.)])
            else:
                lower = max(value for value in values if value < target)
                upper = min(value for value in values if value > target)
                fraction = (target-lower)/(upper-lower)
                brackets.append([(lower, 1-fraction), (upper, fraction)])
        selected, weights = [], []
        for corner in itertools.product(*brackets):
            coordinate = tuple(value for value, _ in corner)
            weight = math.prod(weight for _, weight in corner)
            if weight <= 0:
                continue
            if coordinate not in grid:
                raise ValueError("Multilinear query needs a complete local view/scale/time grid; explicitly use inverse_distance for sparse samples.")
            selected.append(grid[coordinate])
            weights.append(weight)
        return selected, weights

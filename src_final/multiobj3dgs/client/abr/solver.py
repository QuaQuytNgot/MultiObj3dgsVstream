"""Small production ABI to the existing, validated Proposed-DP C++ solver.

Only the bridge and ``proposed_dp.cpp`` are built. No benchmark, workload
generation, metadata scan, warmup or processing-time experiment runs here.
Costs are integer bit/s; ceil(cost/delta), floor(budget/delta), strict first-index
ties and the final original-budget check belong to the original solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

from .config import ROOT

SOURCE_ROOT = Path(__file__).resolve().with_name("native")
INT64_MAX = (1 << 63) - 1
_BUILD_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class DPResult:
    status: str
    selection: tuple[int, ...] | None
    cost_bps: int | None
    utility: float | None
    detail: str
    elapsed_ms: float
    budget_states: int
    visited: int

    @property
    def feasible(self):
        return self.status == "optimal"


class _Output(ctypes.Structure):
    _fields_ = [("status", ctypes.c_char * 32), ("detail", ctypes.c_char * 512),
                ("cost_bps", ctypes.c_int64), ("utility", ctypes.c_double),
                ("visited", ctypes.c_uint64), ("budget_states", ctypes.c_uint64),
                ("elapsed_ms", ctypes.c_double)]


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= INT64_MAX:
        raise ValueError(f"{label} must be an integer in [{minimum}, {INT64_MAX}]")
    return value


def _library(cache_dir=None):
    sources = [SOURCE_ROOT / "solver_bridge.cpp",
               SOURCE_ROOT / "proposed_dp.cpp", SOURCE_ROOT / "common.hpp"]
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Proposed DP requires g++ to build its small production bridge")
    fingerprint = hashlib.sha256(b"abr-production-abi-v1;-std=c++17;-O3;-shared;-fPIC")
    hashes = {}
    for path in sources:
        data = path.read_bytes()
        hashes[path.name] = hashlib.sha256(data).hexdigest()
        fingerprint.update(data)
    directory = Path(cache_dir) if cache_dir else ROOT / ".cache" / "abr_solver"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ("proposed_dp_" + fingerprint.hexdigest()[:24] + ".so")
    with _BUILD_LOCK:
        if not target.is_file():
            handle, temporary = tempfile.mkstemp(prefix=".solver-", suffix=".so", dir=directory)
            os.close(handle)
            try:
                result = subprocess.run(
                    [compiler, "-std=c++17", "-O3", "-Wall", "-Wextra", "-Wpedantic",
                     "-shared", "-fPIC", "-I", str(SOURCE_ROOT),
                     str(sources[0]), str(sources[1]), "-o", temporary],
                    capture_output=True, text=True, timeout=60, check=False)
                if result.returncode:
                    raise RuntimeError(f"Proposed DP bridge compilation failed: {result.stderr.strip()}")
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        native = ctypes.CDLL(str(target))
    call = native.abr_proposed_dp
    call.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double),
                     ctypes.c_int64, ctypes.c_int64, ctypes.c_double, ctypes.c_uint64,
                     ctypes.POINTER(ctypes.c_int), ctypes.POINTER(_Output)]
    call.restype = ctypes.c_int
    return native, call, target, hashes


class ProposedDPSolver:
    """Deterministic four-CRF MCKP solver with explicit infeasibility statuses.

    ``optimal`` certifies the conservative discretized problem, rather than the
    original continuous-cost optimum. ``discretized_infeasible`` means physical
    feasibility exists but this grid excludes it. Neither failure returns an
    executable, over-budget selection.
    """
    def __init__(self, *, delta_bps=250_000, timeout_seconds=.5,
                 max_dp_cells=50_000_000, cache_dir=None):
        self.delta_bps = _integer(delta_bps, "delta_bps", 1)
        self.max_dp_cells = _integer(max_dp_cells, "max_dp_cells", 1)
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and nonnegative")
        self.timeout_seconds = float(timeout_seconds)
        self._native, self._call, self.library_path, self.source_hashes = _library(cache_dir)

    def solve(self, costs_bps, utilities, budget_bps):
        budget_bps = _integer(budget_bps, "budget_bps")
        costs = tuple(tuple(row) for row in costs_bps)
        values = tuple(tuple(row) for row in utilities)
        if not costs or len(costs) != len(values) or any(len(row) != 4 for row in (*costs, *values)):
            raise ValueError("exactly four costs/utilities per nonempty object row required")
        flat_costs = tuple(_integer(v, "cost_bps") for row in costs for v in row)
        flat_values = []
        for row in values:
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("utilities must be finite real measurements/weighted measurements")
                flat_values.append(float(value))
        n = len(costs)
        native_costs = (ctypes.c_int64 * (n * 4))(*flat_costs)
        native_values = (ctypes.c_double * (n * 4))(*flat_values)
        selected = (ctypes.c_int * n)()
        output = _Output()
        code = self._call(n, native_costs, native_values, budget_bps, self.delta_bps,
                          self.timeout_seconds, self.max_dp_cells, selected, ctypes.byref(output))
        detail = output.detail.decode("utf8", errors="replace")
        if code:
            raise RuntimeError(f"Native Proposed DP failed: {detail}")
        status = output.status.decode("ascii")
        if status != "optimal":
            return DPResult(status, None, None, None, detail, output.elapsed_ms,
                            output.budget_states, output.visited)
        selection = tuple(selected)
        actual = sum(costs[i][k] for i, k in enumerate(selection))
        utility = sum(values[i][k] for i, k in enumerate(selection))
        if any(k < 0 or k >= 4 for k in selection) or actual > budget_bps or actual != output.cost_bps:
            raise RuntimeError("Native Proposed DP returned an invalid original-budget selection")
        if not math.isclose(utility, output.utility, rel_tol=1e-12, abs_tol=1e-10):
            raise RuntimeError("Native Proposed DP utility/backtracking mismatch")
        return DPResult(status, selection, actual, output.utility, detail,
                        output.elapsed_ms, output.budget_states, output.visited)

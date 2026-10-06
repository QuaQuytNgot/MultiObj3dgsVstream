#!/usr/bin/env python3
"""Bounded GPU-0 readiness benchmark; no training or research-config edits.

Run with the freshly built project environment::

    CUDA_VISIBLE_DEVICES=0 python tools/benchmark_content_runtime.py

All configurations use the original renderer sequentially at 1024 x 1024 and
pretrained LPIPS VGG, batch one. Only CPU thread count and LPIPS device vary.
The tiny fixed fixture measures execution overhead, not full-content capacity.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import tempfile
import time

# Set before importing torch or any project module that might initialize CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def nvidia_query(fields, kind="gpu"):
    command = ["nvidia-smi", "--id=0", f"--query-{kind}={fields}",
               "--format=csv,noheader,nounits"]
    try:
        return subprocess.check_output(command, text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_processes():
    raw = nvidia_query("pid,process_name,used_memory", "compute-apps")
    records = []
    for line in (raw or "").splitlines():
        values = [item.strip() for item in line.split(",", 2)]
        if len(values) == 3 and values[0].isdigit():
            records.append({"pid": int(values[0]), "name": values[1],
                            "used_memory_mib": values[2]})
    return records


def process_tree():
    """Linux process-tree CPU ticks/RSS, including this process's native threads."""
    records = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            records[int(path.name)] = {
                "parent": int(fields[1]), "ticks": int(fields[11]) + int(fields[12]),
                "start": int(fields[19]), "rss": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
            }
        except (OSError, ValueError, IndexError):
            continue
    selected = {os.getpid()}
    while True:
        children = {pid for pid, item in records.items() if item["parent"] in selected}
        if children <= selected:
            break
        selected |= children
    return {(pid, records[pid]["start"]): records[pid] for pid in selected if pid in records}


class NVML:
    """Read physical GPU 0 directly from the installed driver, without a package."""

    class Memory(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]

    class Utilization(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint) for name in ("gpu", "memory")]

    def __init__(self):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.initialized = False
        functions = {
            "nvmlInit_v2": [], "nvmlShutdown": [],
            "nvmlDeviceGetHandleByIndex_v2": [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
            "nvmlDeviceGetMemoryInfo": [ctypes.c_void_p, ctypes.POINTER(self.Memory)],
            "nvmlDeviceGetUtilizationRates": [ctypes.c_void_p, ctypes.POINTER(self.Utilization)],
        }
        for name, arguments in functions.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        self.check(self.library.nvmlInit_v2())
        self.initialized = True
        self.handle = ctypes.c_void_p()
        try:
            self.check(self.library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)))
        except OSError:
            self.close()
            raise

    @staticmethod
    def check(status):
        if status:
            raise OSError(f"NVML returned status {status}")

    def read(self):
        memory, utilization = self.Memory(), self.Utilization()
        self.check(self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)))
        self.check(self.library.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(utilization)))
        return [float(utilization.gpu), memory.used / 2**20, memory.total / 2**20]

    def close(self):
        if self.initialized:
            self.library.nvmlShutdown()
            self.initialized = False


class Resources:
    """Sample shared-device GPU counters and process-tree CPU/RSS while working."""

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._last_cpu_sample = float("-inf")
        self.monitor_cpu_seconds = 0.
        self.nvml_error = None
        try:
            self.nvml = NVML()
        except (OSError, AttributeError) as exc:
            self.nvml = None
            self.nvml_error = str(exc)
        self.gpu_interval = .02 if self.nvml is not None else .5
        self.cpu_interval = .1

    def sample(self, force_cpu=False):
        cpu_started = time.thread_time()
        before = time.perf_counter()
        tree = None
        if force_cpu or before - self._last_cpu_sample >= self.cpu_interval:
            tree = process_tree()
            self._last_cpu_sample = before
        gpu = None
        if self.nvml is not None:
            try:
                gpu = self.nvml.read()
            except OSError as exc:
                self.nvml_error = str(exc)
                self.nvml.close()
                self.nvml = None
                self.gpu_interval = .5
        if self.nvml is None:
            gpu_raw = nvidia_query("utilization.gpu,memory.used,memory.total")
            try:
                gpu = [float(value.strip()) for value in gpu_raw.split(",")]
            except (AttributeError, ValueError):
                pass
        self.samples.append({"time": before, "tree": tree,
                             "rss": sum(item["rss"] for item in tree.values()) if tree is not None else None,
                             "gpu": gpu})
        self.monitor_cpu_seconds += time.thread_time() - cpu_started

    def __enter__(self):
        self.processes_before = gpu_processes()
        self.sample(force_cpu=True)
        self.started = time.perf_counter()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def _monitor(self):
        while not self._stop.wait(self.gpu_interval):
            self.sample()

    def __exit__(self, *unused):
        self.wall_seconds = time.perf_counter() - self.started
        self._stop.set()
        self._thread.join()
        self.sample(force_cpu=True)
        self.processes_after = gpu_processes()
        if self.nvml is not None:
            self.nvml.close()

    def report(self):
        percentages, cpu_seconds = [], 0.
        ticks_per_second = os.sysconf("SC_CLK_TCK")
        cpu_samples = [sample for sample in self.samples if sample["tree"] is not None]
        for before, after in zip(cpu_samples, cpu_samples[1:]):
            delta = sum(max(0, item["ticks"] - before["tree"].get(key, {"ticks": item["ticks"]})["ticks"])
                        for key, item in after["tree"].items()) / ticks_per_second
            elapsed = after["time"] - before["time"]
            cpu_seconds += delta
            if elapsed > 0:
                percentages.append(100 * delta / elapsed)
        measured_span = cpu_samples[-1]["time"] - cpu_samples[0]["time"]
        gpu = [sample["gpu"] for sample in self.samples if sample["gpu"] is not None]
        return {
            "wall_seconds": self.wall_seconds,
            "process_tree_cpu_percent_mean": 100 * cpu_seconds / measured_span if measured_span else None,
            "process_tree_cpu_percent_peak_sampled": max(percentages, default=None),
            "cpu_percent_convention": "100 percent is one fully occupied CPU core; may exceed 100",
            "process_tree_rss_baseline_bytes": cpu_samples[0]["rss"],
            "process_tree_rss_peak_sampled_bytes": max(sample["rss"] for sample in cpu_samples),
            "gpu0_utilization_percent_mean_sampled": sum(row[0] for row in gpu) / len(gpu) if gpu else None,
            "gpu0_utilization_percent_peak_sampled": max((row[0] for row in gpu), default=None),
            "gpu0_driver_memory_baseline_mib": gpu[0][1] if gpu else None,
            "gpu0_driver_memory_peak_sampled_mib": max((row[1] for row in gpu), default=None),
            "gpu0_total_memory_mib": gpu[0][2] if gpu else None,
            "gpu0_compute_processes_before": self.processes_before,
            "gpu0_compute_processes_after": self.processes_after,
            "samples": len(self.samples), "sample_interval_seconds": self.gpu_interval,
            "cpu_samples": len(cpu_samples), "cpu_sample_interval_seconds": self.cpu_interval,
            "gpu_sampling_backend": "NVML ctypes" if self.nvml is not None else "nvidia-smi fallback",
            "nvml_error": self.nvml_error, "monitor_cpu_seconds": self.monitor_cpu_seconds,
            "measurement_limits": "GPU counters cover the entire shared device; RSS/driver peaks are sampled lower bounds. NVML GPU utilization has its own driver averaging interval. Very short allocations/kernels may fall between samples. Monitor overhead is included.",
        }


def array_hash(value):
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def renderer_case(threads, states, views, settings, expected=None):
    import numpy as np
    import torch
    from tools.content_preparation.renderer_adapter import render

    torch.set_num_threads(threads)
    hashes, pairs, peaks = [], [], []
    # Exclude import, CUDA-context and first-kernel initialization from each case.
    warmup_started = time.perf_counter()
    render(states[0], views[0], settings)
    warmup_seconds = time.perf_counter() - warmup_started
    with Resources() as resources:
        for repeat in range(2):
            for view_index, view in enumerate(views):
                rendered = []
                for state_index, state in enumerate(states):
                    image = render(state, view, settings)
                    record = {key: array_hash(image[key]) for key in ("rgb", "alpha", "depth")}
                    hashes.append({"repeat": repeat, "view": view_index, "state": state_index, **record})
                    if any(not np.all(np.isfinite(image[key])) for key in ("rgb", "alpha", "depth")):
                        raise RuntimeError("Renderer returned nonfinite values")
                    peaks.append(image["metadata"]["peak_cuda_allocated_bytes"])
                    rendered.append(image["rgb"])
                if repeat == 0:
                    pairs.append(rendered)
    canonical = [{key: row[key] for key in ("rgb", "alpha", "depth")} for row in hashes[:6]]
    repeated = [{key: row[key] for key in ("rgb", "alpha", "depth")} for row in hashes[6:]]
    correct = canonical == repeated and (expected is None or canonical == expected)
    return {"stage": "render", "threads": threads, "device": "cuda:0", "batch_size": 1,
            "correct": correct, "render_arrays_bit_exact": correct, "render_hashes": hashes,
            "warmup_seconds_excluded": warmup_seconds,
            "peak_torch_cuda_allocated_bytes": max(peaks), **resources.report()}, pairs, canonical


def lpips_case(threads, device, pairs, baseline=None):
    import numpy as np
    import torch
    from tools.content_preparation.metrics import MetricEvaluator

    torch.set_num_threads(threads)
    # Pretrained initialization is outside timed steady-state comparisons.
    with MetricEvaluator({"lpips": {"net": "vgg", "device": device, "batch_size": 1,
                                    "allow_download": False}}) as evaluator:
        setup_started = time.perf_counter()
        evaluator._load_lpips()
        setup_seconds = time.perf_counter() - setup_started
        warmup_started = time.perf_counter()
        evaluator.compare(*pairs[0])
        warmup_seconds = time.perf_counter() - warmup_started
        torch.cuda.reset_peak_memory_stats()
        scores, peaks = [], []
        with Resources() as resources:
            image, reference = pairs[0]
            scores.append(evaluator.compare(image, reference).lpips)
            peaks.append(int(torch.cuda.max_memory_allocated()))
        expected = np.asarray(baseline if baseline is not None else scores)
        actual = np.asarray(scores).reshape(1, 1)
        differences = np.abs(actual - expected[None, :])
        tolerance = 1e-5
        correct = bool(np.isfinite(actual).all() and np.all(differences <= tolerance))
        result = {"stage": "lpips", "threads": threads, "device": device,
                  "net": "vgg", "batch_size": 1, "correct": correct,
                  "model_initialization_seconds_excluded": setup_seconds,
                  "warmup_seconds_excluded": warmup_seconds,
                  "lpips_scores": scores, "cpu1_baseline_scores": expected.tolist(),
                  "timed_comparisons": 1,
                  "absolute_tolerance": tolerance, "maximum_absolute_difference": float(differences.max()),
                  "peak_torch_cuda_allocated_bytes": max(peaks),
                  "metric_metadata": evaluator.metadata(), **resources.report()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/readiness/runtime_benchmark.json")
    args = parser.parse_args()
    import torch
    from tools.content_preparation.assets import synthetic_state, state_hash
    from tools.content_preparation.renderer_adapter import GPUExecutionGuard

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This benchmark requires CUDA with only physical GPU 0 visible")
    torch.cuda.set_device(0)
    torch.set_num_interop_threads(1)
    states = [synthetic_state(quality_count=128), synthetic_state(quality_count=256)]
    views = [{"azimuth": azimuth, "elevation": 10, "distance": 3, "scale": 1}
             for azimuth in (-30, 0, 30)]
    settings = {"backend": "upstream_cuda", "width": 1024, "height": 1024}
    report = {
        "scope": "bounded synthetic readiness fixture; no training; no full Longdress; no batching beyond one",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                        "nvidia_smi": nvidia_query("name,driver_version,memory.total,uuid"),
                        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "pid": os.getpid()},
        "fixture": {"state_hashes": [state_hash(state) for state in states], "views": views,
                    "renderer_settings": settings, "render_repeats": 2,
                    "lpips_timed_views": 1, "lpips_timed_repeats": 1},
        "configurations": [],
        "limits": "Results measure a tiny 128/256-Gaussian fixture at production render resolution. They do not establish full-scene VRAM, training speed, concurrency safety, or full-run capacity.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    expected, pairs, baseline = None, None, None
    try:
        for threads in (1, 2, 4):
            case, images, canonical = renderer_case(threads, states, views, settings, expected)
            report["configurations"].append(case)
            expected = canonical if expected is None else expected
            pairs = images if pairs is None else pairs
            save()
            print(f"render threads={threads}: {case['wall_seconds']:.3f}s correct={case['correct']}", flush=True)
        for threads, device in ((1, "cpu"), (2, "cpu"), (4, "cpu"), (1, "cuda")):
            case = lpips_case(threads, device, pairs, baseline)
            report["configurations"].append(case)
            baseline = case["lpips_scores"] if baseline is None else baseline
            save()
            print(f"LPIPS {device} threads={threads}: {case['wall_seconds']:.3f}s correct={case['correct']} delta={case['maximum_absolute_difference']:.3g}", flush=True)
        report["gpu_execution_guard"] = GPUExecutionGuard.statistics()
        report["correct"] = all(case["correct"] for case in report["configurations"])
        report["fastest_valid"] = {
            stage: min((case for case in report["configurations"] if case["stage"] == stage and case["correct"]),
                       key=lambda case: case["wall_seconds"], default=None)
            for stage in ("render", "lpips")
        }
        other_pids = {
            process["pid"] for case in report["configurations"]
            for name in ("gpu0_compute_processes_before", "gpu0_compute_processes_after")
            for process in case[name] if process["pid"] != os.getpid()
        }
        report["shared_gpu_service_pids"] = sorted(other_pids)
        report["shared_gpu_confounding"] = bool(other_pids)
        save()
        return 0 if report["correct"] else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["correct"] = False
        save()
        raise


if __name__ == "__main__":
    from tools.content_preparation.checkpoint import file_lock
    with file_lock(Path(tempfile.gettempdir()) / f"content-preparation-gpu-{os.getuid()}.lock"):
        raise SystemExit(main())

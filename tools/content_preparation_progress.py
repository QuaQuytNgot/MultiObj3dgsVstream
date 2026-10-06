"""Reporting only: never changes task inputs, outputs, scheduling or signatures.

Kept outside content_preparation so display changes do not invalidate its jobs.
The three explicitly marked observer hooks in pipeline.py are stripped when
hashing execution code; every other byte remains protected by the original hash.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import math
import os
from pathlib import Path
import sys
import threading
import time


def execution_source(source):
    """Recover the exact pre-observer source, including its original whitespace."""
    lines = source.splitlines(keepends=True)
    result = []
    inside = False
    for line in lines:
        marker = line.strip()
        if marker == b"# BEGIN PROGRESS DISPLAY ONLY":
            if inside:
                raise ValueError("Nested progress display markers")
            inside = True
        elif marker == b"# END PROGRESS DISPLAY ONLY":
            if not inside:
                raise ValueError("Unmatched progress display marker")
            inside = False
        elif not inside:
            result.append(line)
    if inside:
        raise ValueError("Unclosed progress display marker")
    return b"".join(result)


def implementation_hash(directory):
    from tools.content_preparation.config import config_hash
    hashes = {}
    for path in sorted(Path(directory).glob("*.py")):
        source = path.read_bytes()
        if path.name == "pipeline.py":
            source = execution_source(source)
        hashes[path.name] = hashlib.sha256(source).hexdigest()
    return config_hash(hashes)


def task_total(config, obj, stage, root=None):
    """Count journal tasks, including index writes, rather than CUDA iterations."""
    from tools.content_preparation.config import frame_records
    frames = frame_records(obj)
    n, q, modes = len(frames), len(obj["qualities"]), len(config["delivery_modes"])
    sampling = config["sampling"]
    sampled = sum(sampling["times"] == "all" or r["frame"] in sampling["times"] for r in frames)
    samples = sampled * len(sampling["azimuth"]) * len(sampling["elevation"]) * len(sampling["scales"])
    if stage == "preprocess":
        return 1 if obj["dataset"]["kind"] == "checkpoints" else n + 1
    if stage in ("train", "export"):
        return n * q + 1
    if stage in ("encode", "decode"):
        return n * q * modes + 1
    if stage == "profile":
        return samples * q * modes * 2 + 1
    if stage == "proxy":
        missing = 0
        if root is not None:
            mode, highest = config["delivery_modes"][0], obj["qualities"][-1]["id"]
            for record in frames:
                if sampling["times"] != "all" and record["frame"] not in sampling["times"]:
                    continue
                for a in range(len(sampling["azimuth"])):
                    for e in range(len(sampling["elevation"])):
                        for s in range(len(sampling["scales"])):
                            name = f"{record['frame']}_a{a}_e{e}_s{s}.npz"
                            missing += not (Path(root) / "profiles/images" / mode / highest / name).exists()
        return n + samples * 2 + missing + 2
    return 1


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class StageProgress:
    def __init__(self, label, total, mode="auto", interval=30, stream=None):
        if mode not in ("auto", "bar", "log", "off"):
            raise ValueError("Progress mode must be auto, bar, log or off")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Progress interval must be positive and finite")
        self.stream = sys.stderr if stream is None else stream
        self.mode = ("bar" if self.stream.isatty() else "log") if mode == "auto" else mode
        self.label, self.total, self.interval = label, total, interval
        self.started = time.monotonic()
        self.done = self.executed = self.skipped = 0
        self.work_seconds = 0.0
        self.current = "waiting"
        self.task_started = None
        self.last_print = -math.inf
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.thread = None

    def __enter__(self):
        if self.mode != "off":
            self.emit(force=True)
            self.thread = threading.Thread(target=self._heartbeat, daemon=True, name="preparation-progress")
            self.thread.start()
        return self

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.emit(force=True)

    def begin(self, key):
        with self.lock:
            first = self.task_started is None and self.done == 0
            self.current, self.task_started = key, time.monotonic()
            self.emit(force=first)

    def completed(self, skipped=False):
        with self.lock:
            now = time.monotonic()
            self.done += 1
            if skipped:
                self.skipped += 1
            else:
                self.executed += 1
                self.work_seconds += now - self.task_started
            self.task_started = None
            self.current = "between tasks"
            self.emit()

    def emit(self, force=False, status="running"):
        if self.mode == "off":
            return
        with self.lock:
            now = time.monotonic()
            if not force and now - self.last_print < self.interval:
                return
            self.last_print = now
            fraction = min(1.0, self.done / max(1, self.total))
            filled = int(20 * fraction)
            eta = "unknown"
            if self.executed and status == "running":
                active = now - self.task_started if self.task_started is not None else 0
                remaining = self.work_seconds / self.executed * (self.total - self.done) - active
                if remaining > 0:
                    finish = datetime.now().astimezone() + timedelta(seconds=remaining)
                    eta = f"~{duration(remaining)} (finish ~{finish:%Y-%m-%d %H:%M:%S %Z})"
            if status == "complete":
                eta = "00:00:00"
            active = f" task_elapsed={duration(now-self.task_started)}" if self.task_started is not None else ""
            line = (f"[{self.label}] [{'#'*filled}{'-'*(20-filled)}] "
                    f"{self.done}/{self.total} tasks {fraction:.1%} "
                    f"executed={self.executed} skipped={self.skipped} "
                    f"elapsed={duration(now-self.started)} ETA={eta} "
                    f"current={self.current}{active} status={status}")
            prefix = "\r\033[2K" if self.mode == "bar" else ""
            ending = "\n" if self.mode == "log" or status != "running" else ""
            self.stream.write(prefix + line + ending)
            self.stream.flush()

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        if exc_type is None:
            self.current = "finished"
        self.emit(force=True, status="complete" if exc_type is None else "failed")


def observe_stages(pipeline):
    """Wrap reporting around journal calls, preserving the actual journal logic."""
    mode = os.environ.get("CONTENT_PREPARATION_PROGRESS", "auto")
    interval = float(os.environ.get("CONTENT_PREPARATION_PROGRESS_INTERVAL", "30"))
    # Validate before replacing methods, even when the first stage has no tasks.
    StageProgress("settings", 1, mode, interval)
    journal = pipeline.journal
    original_stage = journal.stage

    @contextmanager
    def stage(name):
        stage_name = name.rsplit("/", 1)[-1]
        progress = StageProgress(name, task_total(pipeline.cfg, pipeline.obj, stage_name, pipeline.root), mode, interval)
        original_run = journal.run

        def run(key, *args, **kwargs):
            progress.begin(key)
            before = journal.skipped
            result = original_run(key, *args, **kwargs)
            progress.completed(skipped=journal.skipped > before)
            return result

        with progress:
            journal.run = run
            try:
                with original_stage(name):
                    yield
            finally:
                journal.run = original_run

    journal.stage = stage

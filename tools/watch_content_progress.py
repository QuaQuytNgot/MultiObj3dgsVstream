#!/usr/bin/env python3
"""Read-only progress monitor for a preparation process that is already running."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.content_preparation.config import STAGES, read_config, resolve_path, validate_config
from tools.content_preparation_progress import StageProgress, task_total


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def journal_progress(config, obj, stage, data, stream=None):
    """Use completed task durations for a rough ETA; never validate/alter outputs."""
    label = f"{obj['id']}/{stage}"
    prefix = label + "/"
    entries = [(key, row) for key, row in data.get("tasks", {}).items() if key.startswith(prefix)]
    completed = [row for _, row in entries if row["status"] == "complete"]
    active = [(key, row) for key, row in entries if row["status"] == "running"]
    failed = [(key, row) for key, row in entries if row["status"] == "failed"]
    stage_record = data.get("stages", {}).get(label, {})
    total = task_total(config, obj, stage, resolve_path(config["output"]) / obj["id"])
    progress = StageProgress(label, total, mode="log", stream=stream)
    clock, wall = time.monotonic(), time.time()
    if stage_record.get("started_at"):
        progress.started = clock - max(0, wall - timestamp(stage_record["started_at"]))
    progress.done = len(completed)
    timed = sorted((r for r in completed if r.get("completed_at") and r.get("started_at")),
                   key=lambda r: r["completed_at"])[-20:]
    durations = [max(0, timestamp(r["completed_at"]) - timestamp(r["started_at"])) for r in timed]
    progress.executed = len(completed)
    progress.work_seconds = sum(durations) / len(durations) * progress.executed if durations else 0
    status = stage_record.get("status", "waiting")
    if active:
        key, row = max(active, key=lambda item: item[1]["started_at"])
        progress.current = key
        progress.task_started = clock - max(0, wall - timestamp(row["started_at"]))
    elif status == "failed" and failed:
        progress.current = failed[-1][0]
    elif status == "complete":
        progress.current = "finished"
    else:
        progress.current = "waiting for next journal update"
    progress.emit(force=True, status=status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--stage", choices=STAGES, default="preprocess")
    parser.add_argument("--object", help="Object ID; defaults to all configured objects")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    args = parser.parse_args(argv)
    try:
        # Validate interval without initializing any CUDA/training backend.
        StageProgress("settings", 1, interval=args.interval)
        config = validate_config(read_config(args.config, args.runtime_config))
        objects = [o for o in config["objects"] if args.object is None or o["id"] == args.object]
        if not objects:
            raise ValueError(f"Unknown object: {args.object}")
        path = resolve_path(config["output"]) / ".preparation/journal.json"
        print(f"Read-only monitor: {path}; ETA uses the last 20 completed tasks and may change. "
              "A stale running entry does not prove the worker is still alive. Ctrl-C stops only this monitor.", flush=True)
        stamp, data = None, {}
        while True:
            if path.exists():
                stat = path.stat()
                if stamp != stat.st_mtime_ns:
                    data = json.loads(path.read_text())
                    stamp = stat.st_mtime_ns
                changed = datetime.fromtimestamp(stat.st_mtime).astimezone()
                print(f"Journal last updated: {changed:%Y-%m-%d %H:%M:%S %Z}", flush=True)
            else:
                print("Waiting for journal to be created.", flush=True)
            statuses = [journal_progress(config, obj, args.stage, data) for obj in objects]
            if "failed" in statuses:
                return 2
            if args.once or all(s == "complete" for s in statuses):
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    except (ValueError, OSError) as error:
        print(f"content progress: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

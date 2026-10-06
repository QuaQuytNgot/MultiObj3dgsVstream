"""Atomic, hash-checked task journal: checkpoint individual frames and views."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import hashlib
import json
import os
import tempfile
from .config import config_hash


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
        temp = f.name
    os.replace(temp, path)


def input_hashes(paths):
    result = {}
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Missing input {p}. Supply prepared/raw data or run the preceding stage.")
        if p.is_dir():
            result[str(p)] = config_hash({str(f.relative_to(p)): sha256(f) for f in sorted(p.rglob("*")) if f.is_file()})
        else:
            result[str(p)] = sha256(p)
    return result


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another preparation process holds {path}; GPU execution must be sequential")
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class Journal:
    def __init__(self, root, resume=False, overwrite=False):
        self.root = Path(root)
        self.path = self.root / ".preparation" / "journal.json"
        self.resume, self.overwrite = resume, overwrite
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"version": 1, "tasks": {}, "stages": {}}
        self.skipped = self.executed = 0

    def flush(self):
        write_json(self.path, self.data)

    def run(self, key, config, inputs, action, command=None, version="1.0.0"):
        hashes = input_hashes(inputs)
        fingerprint = config_hash({"config": config, "inputs": hashes, "version": version})
        previous = self.data["tasks"].get(key)
        if previous and self.resume and previous["status"] == "complete" and previous["fingerprint"] == fingerprint:
            if all((self.root / o["path"]).is_file() and sha256(self.root / o["path"]) == o["sha256"] for o in previous["outputs"]):
                self.skipped += 1
                return previous.get("result")
        if previous and previous["status"] == "complete" and not self.overwrite:
            raise FileExistsError(f"Task {key} already exists or changed. Use --resume for matching artifacts, --overwrite for recomputation, or a new output.")
        entry = {"status": "running", "input_hashes": hashes, "config_hash": config_hash(config), "fingerprint": fingerprint,
                 "started_at": now(), "command": command, "version": version, "outputs": []}
        self.data["tasks"][key] = entry
        self.flush()
        try:
            result, outputs = action()
            entry["outputs"] = [{"path": str(Path(p).resolve().relative_to(self.root.resolve())), "bytes": Path(p).stat().st_size, "sha256": sha256(p)} for p in outputs]
            if not outputs:
                raise RuntimeError(f"Task {key} produced no checkpoint artifacts")
            entry.update(status="complete", completed_at=now(), result=result)
            self.executed += 1
            self.flush()
            return result
        except BaseException as error:
            entry.update(status="failed", completed_at=now(), error=f"{type(error).__name__}: {error}")
            self.flush()
            raise

    @contextmanager
    def stage(self, name):
        self.data["stages"][name] = {"status": "running", "started_at": now()}
        self.flush()
        try:
            yield
        except BaseException as error:
            self.data["stages"][name].update(status="failed", error=str(error), completed_at=now())
            self.flush()
            raise
        self.data["stages"][name].update(status="complete", completed_at=now())
        self.flush()

#!/usr/bin/env python3
"""Freeze training environment before calling the unchanged ladder wrapper.

The sibling sidecar is deliberately outside work-root and is not part of the
original task fingerprints. Use this entry point for both fresh runs and resume.
Legacy work can be adopted only after completion, with explicit runtime proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.extend_quality_ladder import LadderExtension
from tools.content_preparation.checkpoint import file_lock, input_hashes, sha256, write_json
from tools.content_preparation.config import config_hash, load_config, resolve_path
from tools.content_preparation.upstream import runtime_provenance

SCHEMA = "quality-ladder-environment-guard-v1"


def guard_path(extension):
    return extension.work.parent / f".{extension.work.name}.quality-ladder-environment.json"


def environment(extension):
    runtime = runtime_provenance(extension.cfg)
    # GPU visibility differs between CPU prechecks and actual CUDA execution.
    runtime = {key: value for key, value in runtime.items() if key not in {"gpu", "cuda_runtime"}}
    return {"upstream_sources": extension.snapshot, "runtime": runtime,
            "preparation_implementation_hashes": input_hashes(sorted(
                (ROOT / "tools/content_preparation").glob("*.py"))),
            "wrapper_sha256": extension.wrapper_sha256}


def adoption_proof(extension, current, runtime_manifest):
    if runtime_manifest is None:
        raise FileExistsError("Existing work has no environment guard. Complete the original run, then supply --adopt-runtime-manifest with verified prior extension hashes; unfinished work cannot be adopted.")
    if not extension.manifest_path.is_file():
        raise FileExistsError("Cannot adopt unfinished work: completed extension manifest is missing")
    manifest = json.loads(extension.manifest_path.read_text())
    journal_path = extension.work / ".preparation" / "journal.json"
    if not journal_path.is_file():
        raise ValueError("Cannot adopt work without its completed training journal")
    journal = json.loads(journal_path.read_text())
    task = journal.get("tasks", {}).get("train/manifest", {})
    if (journal.get("stages", {}).get("extension/train", {}).get("status") != "complete"
            or task.get("status") != "complete" or manifest.get("lineage_verified") is not True):
        raise ValueError("Cannot adopt unfinished/unverified training work")
    recorded = next((row for row in task.get("outputs", [])
                     if (extension.work / row["path"]).resolve() == extension.manifest_path), None)
    if recorded is None or recorded.get("sha256") != sha256(extension.manifest_path):
        raise ValueError("Completed extension manifest differs from its journal checksum")
    if manifest.get("upstream_sources") != current["upstream_sources"]:
        raise ValueError("Cannot adopt: upstream sources differ from completed training provenance")
    if manifest.get("extension_wrapper_sha256") != current["wrapper_sha256"]:
        raise ValueError("Cannot adopt: original ladder wrapper differs from completed training provenance")
    proof_path = resolve_path(runtime_manifest)
    proof = json.loads(proof_path.read_text())
    prior_extensions = proof.get("provenance", {}).get("runtime", {}).get("extension_artifacts")
    current_extensions = current["runtime"].get("extension_artifacts")
    if not prior_extensions or not current_extensions or prior_extensions != current_extensions:
        raise ValueError("Cannot adopt: explicit runtime manifest does not prove identical CUDA extension artifacts")
    return {"extension_manifest": str(extension.manifest_path),
            "extension_manifest_sha256": sha256(extension.manifest_path),
            "runtime_manifest": str(proof_path), "runtime_manifest_sha256": sha256(proof_path),
            "scope": "current environment frozen after audited completed legacy run; prior runtime manifest verifies extension binaries"}


def run_guarded(extension, stage="all", dry_run=False, adopt_runtime_manifest=None):
    sidecar, current = guard_path(extension), environment(extension)

    def preflight():
        if sidecar.exists():
            recorded = json.loads(sidecar.read_text())
            if recorded.get("schema") != SCHEMA or recorded.get("work_root") != str(extension.work):
                raise ValueError("Environment guard belongs to another work directory/schema")
            if recorded.get("environment") != current:
                raise RuntimeError("Training environment changed since it was frozen (upstream source, runtime, CUDA extension, or wrapper). Use a new work-root; --overwrite does not bypass this guard.")
            return recorded, False
        adoption = None
        if extension.work.exists() and any(extension.work.iterdir()):
            adoption = adoption_proof(extension, current, adopt_runtime_manifest)
        return {"schema": SCHEMA, "work_root": str(extension.work),
                "environment": current, "adoption": adoption}, True

    if dry_run:
        # An unguarded in-progress legacy run can still be planned, but it cannot
        # be adopted or resumed through this entry point without completion.
        if sidecar.exists() or adopt_runtime_manifest is not None:
            preflight()
        result = extension.run(stage, dry_run=True)
        result["environment_guard"] = {"path": str(sidecar), "frozen": sidecar.exists(),
                                       "dry_run": True}
        return result
    with file_lock(sidecar.with_suffix(".lock")):
        recorded, create = preflight()
        if create:
            write_json(sidecar, recorded)
        result = extension.run(stage, dry_run=False)
        result["environment_guard"] = {"path": str(sidecar),
                                       "environment_sha256": config_hash(current),
                                       "adopted_existing_work": create and recorded["adoption"] is not None}
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--fullres-template", required=True)
    parser.add_argument("--poses-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--adopt-runtime-manifest", type=Path,
                        help="Explicit prior preparation manifest proving legacy-run CUDA extension hashes")
    args = parser.parse_args(argv)
    try:
        extension = LadderExtension(load_config(args.config), args.config, args.source_manifest,
                                    args.work_root, args.fullres_template, args.poses_root,
                                    args.resume, args.overwrite)
        result = run_guarded(extension, args.stage, args.dry_run, args.adopt_runtime_manifest)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError, KeyError, ImportError) as error:
        print(f"quality extension environment guard: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

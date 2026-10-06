#!/usr/bin/env python3
"""Validate a configured preparation run, including independent segment decoding."""
import argparse
import json
from pathlib import Path
import sys
import os
import tempfile
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.content_preparation.checkpoint import Journal, file_lock, sha256, write_json
from tools.content_preparation.config import VALIDATION_STAGES, configure_runtime, read_config, resolve_path, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime-config", type=Path, help="Execution-only machine settings; research fields are rejected")
    parser.add_argument("--stage", choices=("all",) + VALIDATION_STAGES, default="all")
    parser.add_argument("--report", type=Path, help="Default: prepared-root/checks/<stage>.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = read_config(args.config, args.runtime_config)
        configure_runtime(raw)
        config = validate_config(raw)
        from tools.content_preparation.prepared_validation import PreparedValidator
        from tools.content_preparation.upstream import source_snapshot, runtime_provenance
        extension = config["runtime"].get("extension_path")
        if extension:
            sys.path.insert(0, str(resolve_path(extension)))
        validator = PreparedValidator(config)
        owned = (validator.root / ".preparation/owner.json").is_file()
        if not owned and args.stage != "environment" and args.stage != "dataset":
            raise ValueError("Artifact validation requires a pipeline-owned output; run preparation first")
        # Preflight can run before preparation without creating a nonempty,
        # unowned pipeline output directory that would prevent --stage all.
        report = args.report.resolve() if args.report else (
            validator.root / "checks" / f"{args.stage}.json" if owned else
            ROOT / "output/checks/content_preparation" / f"{args.stage}.json")
        journal_root = validator.root if owned else report.parent
        if not report.is_relative_to(journal_root):
            raise ValueError(f"Report must reside within {journal_root}")
        if not owned and report.is_relative_to(validator.root):
            raise ValueError("Write preflight report outside unowned preparation output, or use the default report path")
        gpu_lock = file_lock(Path(tempfile.gettempdir()) / f"content-preparation-gpu-{os.getuid()}.lock") if args.stage in ("environment", "all") and config["renderer"]["backend"] == "upstream_cuda" else nullcontext()
        with file_lock(journal_root / ".preparation/run.lock"), gpu_lock:
            journal = Journal(journal_root, args.resume, args.overwrite)
            inputs = [args.config.resolve(), *validator.inputs(args.stage)]
            if args.runtime_config is not None:
                inputs.append(args.runtime_config.resolve())
            def action():
                result = validator.validate(args.stage)
                write_json(report, result)
                return result, [report]
            core = ROOT / "tools/content_preparation"
            code = [Path(__file__), *core.glob("*.py"), *core.glob("schemas/*"), *core.glob("native/*")]
            code = [p for p in code if p.is_file()]
            runtime = {k: v for k, v in runtime_provenance(config).items() if k not in {"gpu", "cuda_runtime"}}
            key = f"validation/{args.stage}/{report.relative_to(journal_root)}"
            if report.exists() and key not in journal.data["tasks"] and not args.overwrite:
                raise FileExistsError(f"Unowned validator report already exists: {report}; choose a new report path or --overwrite")
            result = journal.run(key,
                {"config": config, "code": {str(p.relative_to(ROOT)): sha256(p) for p in code}, "runtime": runtime, "upstream": source_snapshot()},
                inputs, action, command=[sys.executable, *sys.argv], version="1.0.0")
        print(json.dumps({"passed": result["passed"], "report": str(report),
                          "tasks_executed": journal.executed, "tasks_skipped": journal.skipped}, indent=2))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, ImportError, KeyError, StopIteration) as error:
        print(f"prepared validation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

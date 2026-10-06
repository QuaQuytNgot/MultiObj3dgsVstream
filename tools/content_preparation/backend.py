"""Run a pinned upstream entry point with explicit process-local compatibility."""
from pathlib import Path
import argparse
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.content_preparation.compatibility import configure_training_compatibility
from tools.content_preparation.paths import require_upstream


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True, help="train.py, render.py or metrics.py")
    args, forwarded = parser.parse_known_args(argv)
    upstream = require_upstream()
    entry = Path(args.entry)
    entry = entry.resolve() if entry.is_absolute() else (upstream / entry).resolve()
    allowed = {upstream / name for name in ("train.py", "render.py", "metrics.py")}
    if entry not in allowed:
        parser.error("--entry must select train.py, render.py or metrics.py in the pinned backend")
    configure_training_compatibility()
    previous_argv = sys.argv
    try:
        sys.argv = [str(entry), *forwarded]
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        sys.argv = previous_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

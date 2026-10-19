"""Run with ``python -m src.client.main --help`` from the project root."""
from __future__ import annotations

import asyncio
import json
import sys

from .playback_config import load_config


def main(argv=None):
    try:
        config = load_config(argv)
        from .playback import run_playback
        summary = asyncio.run(run_playback(config))
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Client failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

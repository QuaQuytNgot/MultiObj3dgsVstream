#!/usr/bin/env python3
"""Standalone isolated demo; existing publication/media are read-only inputs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src_final'))


def main(argv=None):
    from experiments.interactive_demo.abr.__main__ import main as run
    return run(argv)


if __name__ == '__main__':
    raise SystemExit(main())

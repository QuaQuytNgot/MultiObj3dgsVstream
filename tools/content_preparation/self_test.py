"""Run the bounded content preparation test suite without training datasets."""
from pathlib import Path
import argparse
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def successful(result, strict_optional=False):
    """Missing core/LPIPS checks fail readiness; optional browser skips can pass."""
    optional_reasons = ("Optional browser verification", "Sandbox forbids localhost sockets")
    skipped_required = [
        reason for _, reason in result.skipped
        if strict_optional or not reason.startswith(optional_reasons)
    ]
    return result.wasSuccessful() and not skipped_required


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-optional", action="store_true", help="Fail if optional browser checks skip")
    args = parser.parse_args(argv)
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"/"content_preparation"))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if successful(result, args.strict_optional) else 1


if __name__=="__main__":
    raise SystemExit(main())

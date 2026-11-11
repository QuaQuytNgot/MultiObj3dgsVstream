import argparse
import json
import sys

from .methods import METHODS, SUPPORTED_METHODS
from .runner import freeze_config, run, validate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen-input adaptation baseline comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("methods", help="List supported method IDs")
    freeze = commands.add_parser("freeze", help="Hash-bind existing input artifacts")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output", required=True)
    check = commands.add_parser("validate", aliases=["status"])
    check.add_argument("--config", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--config", required=True)
    execute.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "methods":
            result = {"methods": METHODS, "historical_methods": [m for m in SUPPORTED_METHODS if m not in METHODS]}
        elif args.command == "freeze":
            result = {"status": "frozen", "configuration_hash": freeze_config(args.config, args.output)["configuration_hash"]}
        elif args.command in ("validate", "status"):
            result = validate(args.config)
        else:
            decisions = run(args.config, args.output)
            result = {"status": "completed", "decisions": len(decisions["results"]),
                      "methods": decisions["methods"], "input_kind": decisions["input_kind"], "output": args.output}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "blocked_invalid_or_missing_input", "error": str(error),
                          "renderer_invoked": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

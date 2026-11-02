"""python -m experiments.visibility_weighted_sse --help"""
import argparse
import json
import sys

from .runner import freeze_config, run, validate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cached-input Visibility-Weighted SSE research planner")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Hash-bind existing local inputs")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output", required=True)
    check = commands.add_parser("validate", aliases=["status"], help="Validate frozen inputs without solving")
    check.add_argument("--config", required=True)
    execute = commands.add_parser("run", help="Plan one or multiple GOPs and budgets on CPU")
    execute.add_argument("--config", required=True)
    execute.add_argument("--output", required=True)
    benchmark = commands.add_parser("benchmark", help="Small synthetic CPU scaling check")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            config = freeze_config(args.config, args.output)
            result = {"status": "frozen", "configuration_hash": config["configuration_hash"]}
        elif args.command in ("validate", "status"):
            result = validate(args.config)
        elif args.command == "benchmark":
            from .benchmark import benchmark as scaling_check
            result = scaling_check(args.config, args.output)
        else:
            decisions = run(args.config, args.output)
            result = {"status": "completed", "decisions": len(decisions["results"]),
                      "output": args.output, "input_kind": decisions["input_kind"]}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked_invalid_or_missing_input", "error": str(exc),
                          "renderer_invoked": False}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

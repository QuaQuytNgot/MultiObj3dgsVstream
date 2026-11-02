import argparse
import json
import sys

from experiments.visibility_weighted_sse.common import write_json
from .config import freeze_config, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Streaming evaluation research framework")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "generate-scenario", "freeze", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name != "validate":
            command.add_argument("--output", required=True)
        if name == "run":
            command.add_argument("--method")
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = {"status": "frozen", "configuration_hash": freeze_config(args.config, args.output)["configuration_hash"]}
        elif args.command == "generate-scenario":
            _, _, scenario, _ = load_config(args.config, require_frozen=False)
            write_json(args.output, scenario)
            result = {"status": "geometry_generated", "scene_id": scenario["scene_id"], "visibility_generated": False}
        else:
            from .simulator import run, validate
            if args.command == "validate":
                result = validate(args.config)
            else:
                result = {"status": "finished", "methods": [r["method"] for r in run(args.config, args.output, method=args.method)]}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "blocked_invalid_or_missing_input", "error": str(error),
                          "renderer_invoked": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Run ``python -m multiobj3dgs.server --root dataset --host 127.0.0.1 --port 8080``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .origin import create_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve existing MPD and Gaussian media over HTTP/1.1")
    parser.add_argument("--root", type=Path, required=True, help="immutable publication root")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port; 0 chooses a free port (default: 8080)")
    parser.add_argument('--mbps', type=float, help='shared application-body rate, not per-request')
    parser.add_argument('--latency-ms', type=float, default=0.)
    parser.add_argument('--trace', type=Path, help='JSON list of {at_s,bps,latency_s}')
    args = parser.parse_args(argv)
    try:
        from .emulation import BodyPacer
        import json
        pacer = None
        if args.trace:
            pacer=BodyPacer(json.loads(args.trace.read_text()))
        elif args.mbps is not None:
            pacer=BodyPacer([dict(at_s=0,bps=args.mbps*1e6,latency_s=args.latency_ms/1000)])
        elif args.latency_ms:
            parser.error('--latency-ms requires --mbps or --trace')
        server = create_server(args.root, args.host, args.port, pacer=pacer)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    with server:
        print(f"Serving {server.root} at http://{args.host}:{server.server_port}/ (HTTP/1.1)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

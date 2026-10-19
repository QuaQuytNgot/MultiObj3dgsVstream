"""Application configuration; transport and renderer remain configuration-free."""
from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(argv=None):
    parser = argparse.ArgumentParser(description="Progressive multi-object Gaussian playback")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/client.yaml")
    parser.add_argument("--mpd", help="Local MPD path (offline) or HTTP(S) MPD URL (streaming)")
    parser.add_argument("--mode", choices=("offline", "streaming"))
    parser.add_argument("--display", choices=("headless", "interactive"))
    parser.add_argument("--backend", choices=("gsplat",))
    parser.add_argument("--frames", type=int, help="Output frames; 0 runs until interrupted")
    parser.add_argument("--quality", type=int, choices=range(4), help="Override every object's quality")
    parser.add_argument("--quality-cycle", help="Comma separated qualities, e.g. 0,1,2,3,0")
    parser.add_argument("--quality-interval", type=int, help="Frames per quality in the cycle")
    parser.add_argument("--objects", type=int, help="Duplicate the first source with independent scene states")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--http-version", choices=("h1", "h2", "h3", "h3-only"))
    parser.add_argument("--draco-executable", type=Path)
    parser.add_argument("--decoder-mode", choices=("baseline", "verified", "cached", "native", "native-cached"))
    parser.add_argument("--segment-cache-mib", type=int)
    parser.add_argument("--decode-ahead", action="store_true", help="Enable bounded pooled decoding of future presentation slots")
    parser.add_argument("--verification", choices=("strict", "runtime"))
    parser.add_argument("--workers", help="Fixed positive worker count or auto; enables pooled decode-ahead")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--decode-memory-mib", type=int, help="Worker admission budget; segment cache is additional")
    parser.add_argument("--worker-profile", type=Path, help="Offline measured worker scaling JSON")
    parser.add_argument("--ahead-frames", type=int, help="Maximum future presentation steps")
    parser.add_argument("--ahead-capacity", type=int, help="Bounded request/ready slots; checked against decode RAM allowance")
    parser.add_argument("--late-policy", choices=("wait", "hold"), help="Wait preserves every frame; hold reuses state at wall-clock deadlines")
    parser.add_argument("--startup-buffer-s", type=float, help="Decoded media window required before presentation starts")
    parser.add_argument("--target-buffer-s", type=float, help="Continuous decoded lookahead duration; 0 uses frame/startup horizon")
    parser.add_argument("--startup-timeout-s", type=float)
    parser.add_argument("--startup-fallback", choices=("shrink", "error"))
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable future admission; disabled still preserves startup-primed frames")
    parser.add_argument("--rebuffer-after-s", type=float, help="Hold policy: sustained underrun threshold; 0 disables rebuffering")
    parser.add_argument("--rebuffer-buffer-s", type=float, help="Decoded media window required on rebuffer recovery")
    parser.add_argument("--stream-package", type=Path, help="Experimental offline stream-reference index")
    parser.add_argument("--stream-base-url", help="HTTP base URL for experimental stream package")
    parser.add_argument("--decode-only", action="store_true", help="Explicit decoding experiment without a renderer")
    parser.add_argument("--no-loop", action="store_true")
    args = parser.parse_args(argv)
    import yaml
    config = yaml.safe_load(args.config.read_text()) or {}
    playback = config.setdefault("playback", {})
    rendering = config.setdefault("rendering", {})
    scene = config.setdefault("scene", {})
    network = config.setdefault("network", {})
    for name in ("mpd", "mode", "display", "frames", "quality_interval", "fps"):
        value = getattr(args, name)
        if value is not None:
            playback[name] = value
    playback.setdefault("mode", "offline")
    playback.setdefault("mpd", "dataset/manifest.mpd")
    playback.setdefault("display", "headless")
    playback.setdefault("frames", 0 if playback["display"] == "interactive" else 300)
    playback.setdefault("fps", 30.)
    playback.setdefault("quality_interval", 30)
    if args.decoder_mode is not None:
        from .optimized_codec import OPTIMIZATION_MODES
        config["decoder_optimization"] = OPTIMIZATION_MODES[args.decoder_mode]
    if args.segment_cache_mib is not None:
        if args.segment_cache_mib < 0: parser.error("--segment-cache-mib must be nonnegative")
        playback["buffer_bytes"] = args.segment_cache_mib * 1024**2
    if args.decode_ahead:
        playback["decode_ahead"] = True
    for key in ("max_workers", "decode_memory_mib", "ahead_frames", "ahead_capacity", "late_policy"):
        if getattr(args,key) is not None:
            playback[key] = getattr(args,key)
    for key in ("startup_buffer_s", "target_buffer_s", "startup_timeout_s", "startup_fallback", "prefetch",
                "rebuffer_after_s", "rebuffer_buffer_s"):
        if getattr(args,key) is not None:
            playback[key] = getattr(args,key)
    if args.worker_profile:
        playback["worker_profile"] = str(args.worker_profile.resolve())
    if args.workers is not None:
        try:
            playback["workers"] = "auto" if args.workers == "auto" else int(args.workers)
        except ValueError:
            parser.error("--workers must be auto or a positive integer")
        playback["decode_ahead"] = True
    if args.verification is not None:
        optimization = config.get("decoder_optimization")
        if optimization is None and args.verification == "runtime":
            parser.error("Runtime verification requires an optimized decoder mode")
        if optimization is not None:
            config["decoder_optimization"] = {**optimization, "verification": args.verification}
    playback.setdefault("workers", 1)
    playback.setdefault("max_workers", 6)
    playback.setdefault("decode_memory_mib", 2048)
    playback.setdefault("ahead_frames", 12)
    playback.setdefault("late_policy", "wait")
    playback.setdefault("startup_buffer_s", 0.)
    playback.setdefault("target_buffer_s", 0.)
    playback.setdefault("startup_timeout_s", 30.)
    playback.setdefault("startup_fallback", "shrink")
    playback.setdefault("prefetch", True)
    playback.setdefault("rebuffer_after_s", 0.)
    playback.setdefault("rebuffer_buffer_s", .5)
    if args.stream_package:
        playback["stream_package"] = str(args.stream_package.resolve())
    if args.stream_base_url:
        playback["stream_base_url"] = args.stream_base_url
    playback["loop"] = False if args.no_loop else playback.get("loop", True)
    playback["decode_only"] = args.decode_only or playback.get("decode_only", False)
    if args.quality_cycle is not None:
        try:
            playback["quality_cycle"] = [int(q) for q in args.quality_cycle.split(",")]
        except ValueError:
            parser.error("--quality-cycle requires comma separated integers 0..3")
    if args.quality is not None:
        playback["quality"] = args.quality
        if args.quality_cycle is None:
            playback["quality_cycle"] = []
    if args.objects is not None:
        playback["objects"] = args.objects
    for name in ("width", "height", "backend"):
        value = getattr(args, name)
        if value is not None:
            rendering[name] = value
    rendering.setdefault("backend", "gsplat")
    rendering.setdefault("width", 640)
    rendering.setdefault("height", 480)
    if args.http_version:
        network["http_version"] = args.http_version
    if args.draco_executable:
        config["decoder"] = {**config.get("decoder", {}), "name": "gaussian_attribute_draco_byteplanes",
                             "executable": str(args.draco_executable.resolve())}
    config["output"] = str(args.output or playback.get("output", "output/client_playback"))
    # Relative content/output paths follow the documented project-root convention.
    if playback["mode"] == "offline":
        playback["mpd"] = str((PROJECT_ROOT / playback["mpd"]).resolve())
    if playback.get("worker_profile"):
        playback["worker_profile"] = str((PROJECT_ROOT / playback["worker_profile"]).resolve())
    config["output"] = str((PROJECT_ROOT / config["output"]).resolve())
    if playback["mode"] not in ("offline", "streaming") or playback["display"] not in ("headless", "interactive"):
        parser.error("Unsupported playback mode/display")
    for key in ("frames", "quality_interval", "objects"):
        value = playback.get(key, 1)
        if type(value) is not int or value < (0 if key == "frames" else 1):
            parser.error(f"playback.{key} must be an integer >= {0 if key == 'frames' else 1}")
    import math
    if isinstance(playback["fps"], bool) or not math.isfinite(playback["fps"]) or playback["fps"] <= 0:
        parser.error("playback.fps must be positive and finite")
    cycle = playback.get("quality_cycle", [])
    if not isinstance(cycle, list) or any(type(q) is not int or q not in range(4) for q in cycle):
        parser.error("playback.quality_cycle must contain integers 0..3")
    quality = playback.get("quality", 0)
    if type(quality) is not int or quality not in range(4):
        parser.error("playback.quality must be an integer 0..3")
    if type(playback["decode_only"]) is not bool:
        parser.error("playback.decode_only must be boolean")
    if playback["decode_only"] and playback["display"] == "interactive":
        parser.error("--decode-only requires --display headless")
    for key in ("max_workers", "decode_memory_mib", "ahead_frames"):
        if type(playback[key]) is not int or playback[key] < 1:
            parser.error(f"playback.{key} must be a positive integer")
    if playback.get("ahead_capacity") is not None and (type(playback["ahead_capacity"]) is not int or playback["ahead_capacity"] < 1):
        parser.error("playback.ahead_capacity must be a positive integer")
    if playback["workers"] != "auto" and (type(playback["workers"]) is not int or playback["workers"] < 1):
        parser.error("playback.workers must be auto or a positive integer")
    if playback["late_policy"] not in ("wait", "hold"):
        parser.error("playback.late_policy must be wait or hold")
    for key in ("startup_buffer_s", "target_buffer_s", "rebuffer_after_s", "rebuffer_buffer_s", "startup_timeout_s"):
        value=playback[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            parser.error(f"playback.{key} must be finite and nonnegative")
    if playback["startup_timeout_s"] <= 0 or playback["rebuffer_buffer_s"] <= 0:
        parser.error("startup_timeout_s and rebuffer_buffer_s must be positive")
    if playback["startup_fallback"] not in ("shrink", "error") or type(playback["prefetch"]) is not bool:
        parser.error("Invalid startup fallback or prefetch selection")
    if playback["startup_buffer_s"] > 0 or playback["target_buffer_s"] > 0:
        playback["decode_ahead"] = True
    if playback["late_policy"] == "hold":
        playback["decode_ahead"] = True
    scene.setdefault("camera", {"azimuth": 0, "elevation": 0, "distance": 4, "target": [0, 0, 0]})
    return config

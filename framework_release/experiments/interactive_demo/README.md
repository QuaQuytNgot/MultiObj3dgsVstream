# Isolated interactive Visibility × MSE demo

This experimental demo copies the production HTTP/ABR/playback/cache Python modules and modifies the copies. The original `run_pipeline.py`, `src_final/`, production configurations, existing experiments, and evaluation runs are unchanged. Native StreamSTGS decoding, CUDA/FAST rendering, camera/window controls, publication parsing/materialization, and Request API dependencies are reused from the existing checkout.

The implementation and measured validation are described in [`reports/DEMO_PIPELINE_UPGRADE_RESULTS.md`](../../reports/DEMO_PIPELINE_UPGRADE_RESULTS.md). [`COPY_PROVENANCE.json`](COPY_PROVENANCE.json) records copied source files and checksums. No media archive, pretrained model, or published media tree is copied into this experiment. Runtime HTTP downloads use a temporary admission-owned spool and are removed when their consumers finish.

## Run

From the repository root, use the existing working GPU/display environment:

```bash
PYTHONPATH=src_final DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia /home/fil/miniconda3/envs/Hoang/bin/python experiments/interactive_demo/run_demo.py --config experiments/interactive_demo/configs/n4.json --server-mbps 24
```

The standalone entry point inserts the repository and `src_final` into its own import path. `python -m experiments.interactive_demo` is also available from the repository root. The production command continues to use its original PSNR/shared-cache behavior.

Examples with different fixed instance lists:

```bash
# N=3, default seeded random placement; first three configured contents.
PYTHONPATH=src_final DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia /home/fil/miniconda3/envs/Hoang/bin/python experiments/interactive_demo/run_demo.py --config experiments/interactive_demo/configs/n4.json --objects 3 --server-mbps 24

# N=5, alternating content identities, circular placement.
PYTHONPATH=src_final DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia /home/fil/miniconda3/envs/Hoang/bin/python experiments/interactive_demo/run_demo.py --config experiments/interactive_demo/configs/n5.json --contents longdress,soldier --placement camera_rotation --server-mbps 24

# N=10, all four contents repeated independently, explicit random seed.
PYTHONPATH=src_final DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia /home/fil/miniconda3/envs/Hoang/bin/python experiments/interactive_demo/run_demo.py --config experiments/interactive_demo/configs/n10.json --placement random_spatial --seed 238019 --server-mbps 24
```

N=5/10 commands are supported configurations, not claims of validated native capacity or realtime delivery. Native smoke validation was limited to N=3/4.

Use `--placement long_line` for the old centered 1.8-spacing layout. Omit `--server-mbps` for ordinary unthrottled localhost HTTP; `--unthrottled` explicitly removes shaping inherited from a config. The default configurations retain the original 80 Mbps startup prior. `--startup-mbps 24` makes the prior conservative for a 24 Mbps shaped link, but cannot make an oversized base representation set fit.

The demo automatically references an existing compatible publication under `output/abr_publications/`. Supply `--mpd /path/to/existing/manifest.mpd` or an HTTP(S) MPD for another existing publication. It never publishes/extracts archives or changes that publication. Missing media or incompatible MSE metadata produces an explicit error. `--publication` accepts an existing read-only publication directory; archive-only publication is deliberately unsupported here.

## Configuration and controls

| Option | Behavior |
|---|---|
| `--objects N` | Positive count, fixed before playback. The ordered content list is repeated cyclically. Without a count in CLI/config, use the list length. |
| `--contents longdress,soldier,loot,redandblack` | Select supported contents; repeats create distinct instances with different IDs and owners. |
| `--scene FILE` | Explicit named instances, transforms, offsets, duration and loops. Scene content identities determine required assets. Explicit CLI count/content/layout/seed overrides conflict and are rejected; inherited defaults do not override scene instances. |
| `--placement random_spatial` | Default: local PCG64(238019), bounded seeded rejection and deterministic X/Z domain expansion. |
| `--placement camera_rotation` | Fixed circular X/Z arrangement; radius grows for clearance. No automatic camera motion. |
| `--placement long_line` | Preserve the original centered X line and 1.8 spacing; report overlaps without moving explicit/legacy positions. |
| `--seed INTEGER` | Nonnegative placement seed; no global RNG mutation. |
| `--mse-profiles DIRECTORY` | Prepared bank; default `experiments/offline_profiles/runs/full_source_v1/prepared`. It must retain its saved preparation validation inventory. |
| `--delta-mbps 0.15` | Fixed validated 150,000 bps grid; other values are rejected in this isolated demo. |
| `--frames N`, `--wall-time-s SECONDS` | Bound fresh frames and cooperative native-loop duration. Use external `timeout` as well for automated tests that must terminate blocking CUDA/codec calls. |
| `--report PATH`, `--cache PATH`, `--capture-dir PATH` | Outputs must remain under `experiments/interactive_demo/output/`. |

Existing controls remain available: LMB orbit, RMB pan, wheel zoom, F fly mode, WASD/E/C movement, Space pause/resume, arrows/Home seek, L loop, Tab select object, Q quality request, and the existing playback panel. Quality changes apply at an object GOP boundary. There is no runtime object add/remove interface.

## Adaptation and isolation

`abr/objectives.py` preserves the validated temporal arithmetic: per-sample cross-object alpha normalization and `math.fsum(tau * importance * (MSE_Q0 - MSE_q))`. Gains remain signed. MSE uses the fixed own-CRF20 alpha > 0.01 foreground mask, clipped RGB, three channels, data range one, pooled SSE/scalar count. Candidate order is CRF40/36/28/20, including strict first-index ties. Missing startup visibility uses uniform importance; a measured empty view stays zero.

The copied experiment allocator in `abr/validated/` reserves exact mandatory bytes and row-minimum offsets before residual byte-to-bitrate conversion and the 150 kbps grid. It calls the original C++ DP sources through the copied bridge, with a 2M-cell limit and an isolated build directory. Final physical-byte feasibility is checked. Existing external in-flight admissions are reserved before a cohort is allocated. Infeasible/limit cases retain every object, request the cheapest legal independent packages, and explicitly report uncertified fallback/rebuffering.

Runtime costs use the published CPSEG plus shared-asset closures, rather than original package-size or nominal bitrate estimates. Complete MSE coverage (80 content/GOP/CRF entries), reference protocol, prepared checksums, source manifest/member hashes, and CRF20 bounds/reconstruction bindings are validated. PSNR remains a separately labeled diagnostic; it is not the adaptation objective.

Delivery ownership is `(instance, generation, scene cycle, source cycle, GOP, segment/representation)` plus a run-local spool. Original SHA256 values still verify payloads. Different objects, GOPs, source loops, seeks, representations and runs fetch their own closures, including shared weights/assets. Duplicate materialization of the same live admission may retain required buffers. The cache never discovers previous sessions or gives another admission byte credit. CPU producers, prepared arrays, host/raw reservations and GPU handles are independent per job; two shared pinned upload staging buffers remain essential transfer machinery.

Current and at most one upcoming GOP remain bounded playback buffers. Progressive feature-window readiness, producer-drain leases, canceled-transfer accounting, owner-thread CUDA, and last displayed framebuffer/Gaussian state are retained. Per-instance/per-admission request and byte ledgers reconcile actual transfers; pending shielded transfers remain commitments after wrapper cancellation. All successful test runs finish with an empty scoped spool.

## Placement and known limits

Position generation is adapted from `streaming_evaluation/runtime_scenes.py`, preserving seed 238019, the original random domain and minimum 0.8 origin separation. Generated random/circular scenes also avoid padded intersections of complete measured all-GOP CRF20 Gaussian-mean AABBs. The fixed padding is 0.15 world units; it is not a guarantee for infinite Gaussian tails. Explicit scenes and the old line are preserved and report any intersections.

The experiment's historical ring is elliptical; this demo uses equal circular radii. Experiment oracle camera paths and recorded-center rescaling are not copied into interactive playback. Current-pose visibility reuses actual displayed states and is held as a single sample; it cannot reproduce a full oracle trajectory or candidate-independent CRF40 visibility.

Redandblack GOP2 has a verified bound extending to Y≈23.01 and X≈−6.20. These outliers are retained. Full-envelope clearance/framing can spread instances and make bodies small, particularly in N=4. Use the normal camera controls to explore. No compact-envelope rule or silent geometry trimming was introduced.

At 24 Mbps and safety 0.85, a two-second cohort has a 5.1 MB estimated budget. The actual first-GOP Q0 closures total 7,273,830 bytes for the N=3 contents and 9,438,615 bytes for N=4. Both exceed that budget; network overhead makes achieved goodput lower still. More objects add real download, CPU decode and VRAM load. Short smoke measurements do not establish sustained FPS or larger-N feasibility.

## Validation and artifacts

Final regression: **157 passed**, comprising 111 isolated tests and 46 focused original-runtime/solver/viewer/camera regressions. Final native N=3/4 each passed eight fresh frames/swaps and programmatic injection of existing camera, pause, quality and seek controls. An initial N=3 attempt timed out during startup; later traced/final runs passed, and the initial cause remains unproven. See the results report for timings and limitations.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src_final:. /home/fil/miniconda3/envs/Hoang/bin/python -m pytest experiments/interactive_demo/tests tests_final/unit/test_abr_migration_solver.py tests_final/unit/test_runtime.py tests_final/unit/test_runtime_pipeline.py tests_final/unit/test_geometry.py tests_final/unit/test_timeline_precision.py tests_final/test_abr_migration_viewer.py tests/test_abr_pipeline_camera.py --junitxml=experiments/interactive_demo/output/regression.xml -q -p no:cacheprovider

# Use a fresh output directory for each invocation; repeat with --objects 4.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src_final:. DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia timeout 65s /home/fil/miniconda3/envs/Hoang/bin/python -u experiments/interactive_demo/tools/native_smoke.py --objects 3 --wall-time-s 45 --output experiments/interactive_demo/output/new_native_n3

# Re-read the recorded validation artifacts without starting playback.
PYTHONDONTWRITEBYTECODE=1 /home/fil/miniconda3/envs/Hoang/bin/python experiments/interactive_demo/tools/summarize_validation.py
```

The execution environment's network sandbox blocks asyncio socketpair wakeups and localhost tests; those checks were run with host loopback/display/GPU access. No large benchmark, new quality profiling, or ongoing evaluation modification was performed.

Artifacts live under `output/`: `regression.xml`, `validation_summary.json`, `source_baseline.json`, `network_n3/`, `final_native_n3/`, and `final_native_n4/`. Native directories contain report JSON, request/stage JSONL, configuration, summary assertions, and framebuffer captures. Native libraries and temporary media spools are ignored by this experiment's `.gitignore`.

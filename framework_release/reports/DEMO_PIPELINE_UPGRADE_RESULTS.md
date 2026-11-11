# Isolated Interactive Demo Upgrade: Implementation and Validation

Date: 2026-11-06. Implementation is confined to `experiments/interactive_demo/` plus this requested report. The original demo remains unchanged. No large-scale benchmark, quality reprofiling, media republishing, or modification of ongoing evaluation runs was performed.

## Outcome

Implemented Visibility × MSE with the validated 150,000 bps residual DP grid, fixed startup object configuration, independent HTTP/media/CPU ownership per instance and GOP, and three deterministic placement modes. The existing native CUDA/FAST renderer, StreamSTGS codec, free-camera controls and progressive playback remain in use.

Final validation passed **157 tests** in **4.47 s**, including **111 isolated tests** and **46 original-runtime/solver/viewer/camera regressions**. Final native smoke tests passed for **N=3 and N=4**, each presenting eight fresh frames. A checksum audit found **zero changes among 1,692 protected existing source/configuration files**.

The standalone working N=4 command is:

```bash
PYTHONPATH=src_final DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia /home/fil/miniconda3/envs/Hoang/bin/python experiments/interactive_demo/run_demo.py --config experiments/interactive_demo/configs/n4.json --server-mbps 24
```

The original `run_pipeline.py --config configs/abr_pipeline/n4.json --server-mbps 24` still runs the original production implementation. Its 0.25 Mbps default/config, PSNR objective and sharing policies were preserved.

## Implementation and reuse

The isolated package uses its own namespace, `experiments.interactive_demo`. It does not replace production imports or monkey-patch production modules during playback. The bounded test harness subclasses/replaces only the isolated diagnostic window for input injection. Every normal control handler and renderer/codec method remains the existing implementation.

| Component | Reused source | Isolated modifications |
|---|---|---|
| CLI/composition | `src_final/multiobj3dgs/client/abr/{__main__,config,runner}.py` | Own entry point/configs/output paths; startup count/content/scene normalization; placement and MSE-bank options; existing publication discovery without writing media. |
| Quality binding | Production `catalog.py`, `quality_metadata.py`, `binding.py`; experiment baseline metadata validation | Keep original PSNR/media integrity binding; attach prepared MSE, protocol/checksum validation and CRF20 bounds/reconstruction identity. Forward admission scope through materialization. |
| MSE objective | `adaptation_baselines/methods.py::construct_utilities` MSE branch and existing importance arithmetic | Copy normalization, signed gain and temporal `math.fsum` operation order into the isolated objective kernel. Runtime supplies held current-pose visibility. |
| Physical allocation | Copied `visibility_weighted_sse/{allocation,profiles,common}.py` and baseline metadata reader | Preserve exact mandatory/minimum offsets, rate conversion, grid, ties and feasibility checks. Accounting uses an ownership identity separately from the true payload SHA. |
| Native DP | Original `client/abr/native/{proposed_dp.cpp,solver_bridge.cpp,common.hpp}` | Sources read-only; copied Python bridge builds its small library under isolated `output/solver`. Entry point enforces 150000 bps and 2M cells. |
| Media spool/transport | Production `client/cache/resources.py`, ABR network/binding adapters | Run/admission/representation-scoped inputs, singleflight, leases and integrity state; no cross-owner or prior-session reuse. Per-instance/per-scope request/byte ledgers; pending shielded transfers remain reservations. |
| Playback | Copied `client/runtime/*.py` and `client/abr/runtime.py` | Disable CPU preparation sharing; thread admission scopes; reserve outside-cohort pending demand; keep independent handles, progressive readiness, host/raw/GPU budgets and owner-thread CUDA. |
| Placement | `streaming_evaluation/runtime_scenes.py` position-generation structure | Local PCG64(238019), random domain/rejection, deterministic bounded growth, true circular ring, legacy 1.8 line, full measured envelope clearance and calibrated framing. |
| Diagnostics/capture | Copied ABR visibility/metrics/diagnostic/capture/origin modules | Display MSE; separately label optional PSNR; keep actual interactive viewport ground-truth quality unavailable. Disable ImGui persistence into the original checkout. |
| Native dependencies | Original StreamSTGS adapter/vendor, FAST renderer, window/camera, shared geometry/metadata, Request API, publication/materializer | Imported unchanged. Existing published media, prepared profile bank and native pretrained assets are read-only inputs. |

Detailed file origins and original hashes are in [`COPY_PROVENANCE.json`](../experiments/interactive_demo/COPY_PROVENANCE.json). The copied allocator's only delivery-identity extension is `owner_key`; integrity still uses original SHA256 values. Existing experiment modules were not edited to delegate to new helpers: isolation supersedes that part of the earlier production-oriented plan.

## Numerical and metadata contract

The objective is the signed distortion improvement from CRF40/Q0:

\[
U_{iq}=\sum_t\tau_t\frac{A^{visible}_{it}}{\sum_jA^{visible}_{jt}}
        (MSE_{i,CRF40}-MSE_{iq}),
\]

with zero importance when a measured sample has no visible contribution. An absent startup snapshot uses uniform importance explicitly. The implementation preserves per-sample normalization and `math.fsum(tau * importance * gain)`, rather than factoring the gain after temporal aggregation. Candidate order remains CRF40/36/28/20; negative gains and nonmonotonic measured profiles remain valid.

MSE comes from the existing complete prepared bank: four contents × five GOPs × four CRFs = **80 entries**. It uses fixed own-CRF20 alpha > 0.01 foreground, clipped RGB, three channels, range one, pooled SSE/foreground-scalar count, eight camera IDs and four timestamps/GOP. The prepared profile SHA256 remains `27b60c29ca12e3c941849e1784a21a5d13b3a4434b0c51faf84df90ad8b951a3`.

Startup validates profile/baseline bindings, saved output checksums, protocol/distribution, source coverage, original representation manifests and member hashes/sizes, and CRF20 bounds/reconstruction identity. It does not compare CPSEG publication sizes/revision to original-package sizes/revision. Costs use the actual published resources; this publication has **15 requests per selected object/GOP**, although the original offline package lists 17 file records.

The allocator reserves exact fixed/common bytes and per-row minimum offsets before converting residual byte costs using `ceil(8*c/d)` and quantizing at 150000 bps. Residual budgets use floor conversion and the existing maximum-useful-capacity bound. For two seconds, one residual step is 37,500 bytes. Backtracking rechecks actual physical bytes. Each object is an independent row. Existing pending work outside a replanned cohort reserves bytes once before allocation, including shielded HTTP that outlives cancellation of its consumer.

If the mandatory set cannot fit, status/fallback remain explicit: the demo requests the cheapest legal independent packages and allows buffering. A completed playback test therefore does not imply every adaptation decision was budget-feasible.

Interactive visibility analyzes actual displayed selected Gaussian states at the live camera and holds that sample. The experiment used candidate-independent CRF40 states over a pre-known camera trace. Numerical parity is demonstrated for identical inputs; full oracle-trajectory or viewport-quality parity is not claimed.

## Independent ownership and its proof

Media delivery scope includes instance, seek generation, scene/source cycle, GOP and exact segment/representation, under a unique run-local spool. The true SHA verifies payloads; it does not identify a reusable cross-object download. Different instances/GOPs/loops/seeks/representations/runs fetch their complete closures afresh, including weights and shared assets. Same live admission buffers are retained only for actual playback/preparation consumers.

CPU preparation uses separate producers, prepared arrays and host/raw reservations per job, even for identical `preparation_key` values. Decoder/GPU handles remain independent. Progressive feature buffers, two pinned upload staging buffers, current/one-upcoming GOPs and the last displayed framebuffer/state remain necessary playback machinery. Files are released after the last producer/consumer drains; final smoke reports show **zero spool bytes, reservations, resources, packages or pending transfers after cleanup**.

Real loopback tests demonstrated:

- Two identical objects requesting the same bundle and weight payload issue four distinct requests with distinct owner/request IDs and unchanged payload SHA values.
- Adjacent GOP, source-loop, scene-loop, seek and representation changes issue new requests despite identical resource hashes. The fixture made 14 transfers over seven independent closures; restart required two additional transfers.
- Repeated materialization of one live admission issues no extra HTTP requests; releasing one owner's lease leaves its peer intact.
- Expected closure sums, per-instance/per-admission observer ledgers and origin body bytes reconcile exactly. Integrity failure, disk budget rejection and cancellation/draining do not mint cache credit or leak leases.
- Pending external admission demand is reserved during offset/manual replanning; canceled wrappers cannot hide a still-running shielded transfer.

## Test outcomes

| Validation | Result | Evidence |
|---|---|---|
| Isolated metadata, utility/DP, placement/config, cache/runtime and entry point tests | 111 passed | `experiments/interactive_demo/tests/` |
| Focused original solver/runtime/progressive/geometry/timeline/viewer/camera tests | 46 passed | Original test files executed unchanged |
| Final combined regression | 157 passed, zero errors/failures/skips, 4.47 s | [`regression.xml`](../experiments/interactive_demo/output/regression.xml) |
| Protected source/config checksum audit | 1,692 checked; zero changed | [`source_baseline.json`](../experiments/interactive_demo/output/source_baseline.json), [`validation_summary.json`](../experiments/interactive_demo/output/validation_summary.json) |
| Shaped N=3 HTTP integration | Completed; exact independent-byte reconciliation; base infeasibility reported | [`network_n3/report.json`](../experiments/interactive_demo/output/network_n3/report.json) |
| Final native N=3 | Passed eight fresh frames and eight swaps | [`final_native_n3/summary.json`](../experiments/interactive_demo/output/final_native_n3/summary.json) |
| Final native N=4 | Passed eight fresh frames and eight swaps | [`final_native_n4/summary.json`](../experiments/interactive_demo/output/final_native_n4/summary.json) |

The numerical suite includes frozen helper parity over seeded 150 kbps cases, an exhaustive three-row oracle for the discretized feasible set, byte feasibility, signed/nonmonotonic gains, all-invisible samples, explicit native cell-limit fallback, exact 37,500-byte residual boundaries, and tampered/missing metadata rejection. Generic original 250 kbps solver/source-hash cases remain unchanged and pass.

Placement tests cover N=1/3/4/5/10, deterministic local RNG, minimum 0.8 origin separation, padded envelope intersections, circular equal radii, bounded dense layouts, legacy line coordinates, explicit-scene precedence, source-bound binding and calibrated initial framing. No native benchmark was run for N=5/10.

## Bounded measurements

Native tests used the existing GTX 1660 (6 GiB), PyTorch 2.3.1+cu118, DISPLAY=:1, the existing Xauthority and NVIDIA PRIME/GLX environment. GL reported `NVIDIA Corporation` and `NVIDIA GeForce GTX 1660/PCIe/SSE2`. Another process already held roughly 4.2–4.6 GiB during the work; it was not interrupted.

Final smoke settings were 256×192 native viewport, 64-wide visibility analysis, two workers/HTTP slots, prefetch=0, startup prior=24 Mbps, unthrottled local HTTP, eight fresh frames, an injected short pause/camera change/manual-quality request/seek, 45-second cooperative limit and 65-second external process cap. The smoke deliberately did not use a shaped 24 Mbps link. These short checks include cold setup, a seek/rebuffer and first-render overhead; they are not sustained performance benchmarks.

| Measurement | N=3 | N=4 |
|---|---:|---:|
| Total wall time, including setup/cleanup | 6.705 s | 7.701 s |
| Native-loop start to first frame | 2.285 s | 3.017 s |
| Application start to first frame | 4.442 s | 5.214 s |
| Fresh frames / native swaps | 8 / 8 | 8 / 8 |
| Media GETs over initial GOP + seek admission | 90 | 120 |
| Media response body bytes | 31,228,361 | 42,868,381 |
| Prepared-host reservation peak | 900.08 MiB | 939.06 MiB |
| Raw/staging reservation peak | 121.95 MiB | 126.14 MiB |
| GPU model reservation peak | 931.79 MiB | 1,214.85 MiB |
| PyTorch process peak allocated | 820.92 MiB | 1,214.14 MiB |
| PyTorch process peak reserved | 1,080 MiB | 1,470 MiB |
| Fresh-scene compute median | 33.26 ms | 43.17 ms |
| Fresh-scene compute mean / maximum | 64.77 / 285.08 ms | 65.07 / 215.20 ms |
| Observed short-window fresh wall FPS | 3.15 | 2.88 |
| Runtime readiness/rebuffer events / seconds | 3 / 2.932 s | 2 / 3.758 s |

Reservation peaks are logical admission accounting; PyTorch peaks measure this process's allocator, not all CUDA/GL/system allocations or physical display performance. Fresh wall FPS includes the injected pause/seek and cold transitions. No steady 30 FPS claim follows from these results.

The native harness verified distinct handles/positions, existing mouse orbit/pan, wheel zoom, fly movement, resize, pause/resume, next-boundary quality request, actual requested representation rendering after seek, seek generation and loop-toggle callbacks. Unit regressions separately preserve automatic GOP-boundary switching, progressive readiness, source/scene loops and offset timing. Input was programmatically injected into existing callbacks/poll paths; no physical input or physical-display-latency claim is made.

The separate shaped N=3 network-only check used a 24 Mbps server and 24 Mbps prior. It transferred exactly **7,273,830 media bytes in 45 successful media requests**; two MPD/sidecar requests contributed another 789,794 bytes. Selected-GOP transfer/materialization took **19.218 s** on this invocation; aggregate network/host/request overhead meant the nominal capacity was not achieved. Status was `infeasible_base`, correctly exceeding the **5,100,000-byte** two-second safety budget. N=4 Q0 first-GOP demand is **9,438,615 bytes**. The original server shaping remains aggregate across all independent responses.

## Placement behavior and limitations

Random placement starts from the validated X/Y/Z domain, uses PCG64 seed **238019**, then expands X/Z deterministically when full envelopes do not fit. Final N=3 required zero expansions; N=4 required one. Both had zero padded AABB intersections. Circular placement uses equal X/Z radii and grows as needed; `long_line` retains the original 1.8 spacing and reports intersections without moving instances.

The complete Redandblack GOP2 bounds include X≈[−6.20,0.56], Y≈[−1.01,23.01], Z≈[−3.03,0.51]. No outlier trimming was introduced. The final N=4 padded world envelope extends approximately Y=[−1.18,23.18], making bodies small after full-envelope camera fitting. The captures show this directly: [N=3 first frame](../experiments/interactive_demo/output/final_native_n3/captures/frame-000000.png), [N=4 first frame](../experiments/interactive_demo/output/final_native_n4/captures/frame-000000.png). Normal camera controls remain usable for exploring/zooming. A compact visible-extent policy would require a separately defined, validated rule; this implementation retains the conservative source geometry.

An initial N=3 native attempt stalled after startup upload admission and hit its 120-second external process cap before any fresh frame. No thread trace was captured for that first attempt, so its root cause remains unestablished. A traced retry passed, followed by successful N=4 and final N=3/4 reruns. The timed-out attempt's run-local media spool was removed after termination; its log and [`timeout.json`](../experiments/interactive_demo/output/native_n3/timeout.json) were retained. This intermittent startup outcome and GPU contention remain operational limitations, rather than evidence of long-run stability.

Larger N remains constrained by independent CPU decoding, host/raw limits, progressive upload pacing, actual free VRAM, per-frame reconstruction/concatenation and network startup/request overhead. Smaller compressed representations do not necessarily shrink decoded grids. Unavailable metadata/resources fail explicitly; an impossible resource set must not be hidden through deduplication or a reduced object count. No sustained/larger-N validation was attempted.

## Commands and artifacts

The isolated [`README.md`](../experiments/interactive_demo/README.md) contains N=3/5/10 commands and all flags. Append `--placement camera_rotation`, `--placement long_line`, or `--seed VALUE` for generated scenes. All instance definitions are built before playback; there is no object-management endpoint.

Reproduce the bounded final smoke using a new output directory each time:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src_final:. DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia timeout 65s /home/fil/miniconda3/envs/Hoang/bin/python -u experiments/interactive_demo/tools/native_smoke.py --objects 3 --wall-time-s 45 --output experiments/interactive_demo/output/new_native_n3

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src_final:. DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia timeout 65s /home/fil/miniconda3/envs/Hoang/bin/python -u experiments/interactive_demo/tools/native_smoke.py --objects 4 --wall-time-s 45 --output experiments/interactive_demo/output/new_native_n4
```

`output/validation_summary.json` is generated by `tools/summarize_validation.py` from existing raw reports, request/stage JSONL, JUnit output and the protected-source checksum baseline. It does not launch playback. Native reports include configuration, source/representation mapping, MSE/grid choices, scopes, memory telemetry, actual request ledgers, origin observations and empty-spool cleanup evidence. Automated tests required host loopback/thread-wakeup access because those facilities stall or fail in the network sandbox.

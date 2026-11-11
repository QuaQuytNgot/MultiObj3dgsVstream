# Additional network-trace streaming evaluation

Updated 2026-11-07T07:29:39Z. NAVA extension **complete**, phase **B**.
NAVA Phase A terminal **189/189**; statuses {'SUCCESS': 189}; Phase B {'COMPLETE': 189}.
The original 567 Visibility × MSE/PSNR/Exhaustive cases remain complete and are not rescheduled.

NAVA output: `/home/fil/Hoang/MultiObj3dgsVstream/experiments/streaming_evaluation/runs/streaming_additional_network_nava_v1`. Original results and combined tables: `/home/fil/Hoang/MultiObj3dgsVstream/experiments/streaming_evaluation/runs/streaming_additional_network_v1`.
Full combined matrix: 3 traces × 7 object counts × 3 scenes × 3 durations × 4 methods = **756 cases**; only the **189 NAVA** cases are new. Smoke uses six separate two-second cases, not matrix results.

HS/LS/LV retain the original measured CSVs, timestamps, offset 0 s, period 120 s, loop=true and end_policy=error. Real HTTPX/Hypercorn TLS/ALPN h2 uses one persistent connection/case, 16 streams, shared aggregate trace shaping and 20 ms overlapping request latency. Bootstrap 80 Mbps, harmonic window 5/staleness 30 s, safety 0.9 and the original 102 s streaming bound are unchanged. Host CPU limitations and payload-only shaping still apply.

Longdress/Soldier; N=3/5/7/10/12/15/20; camera_rotation/long_line/random_spatial; playback 10/20/30 s; seed 238019; 640×480; fx=fy=772.5483399593904; cx=319.5; cy=239.5; horizontal FOV≈45°. Original poses, object transforms, 30 fps, six-second source loops, two-second GOPs, CRF20/28/36/40, 17 resources/object/GOP, independent transfers, startup 2 s and max buffer 6 s are preserved.

NAVA retains the historical projected-AABB `paper_discard` policy, temporal/cross-object normalization and weighted profile-PSNR objective. No clipping or FOV change is introduced. DP Δ=150000 bps, max 2000000 cells and exact physical-byte checks remain. The compatibility smoke replays historical HV decisions at their original budgets, checking exact utility, diagnostics, allocation and feasibility equality across all 21 scenes. The historical AABB zero-importance limitation remains part of this baseline.

All new Phase A trace matrices finish before Phase B. Recorded displayed-frame poses, intrinsics, transforms, timestamps and CRFs are replayed by the existing native joint renderer. Full-viewport clipped RGB PSNR uses the joint all-CRF20 reference and pooled SSE/sample counts. Cache identities include exact camera K/poses, source frames and allocations; payload SHA checks are retained. Used cache JSONs are copied into the new output and immutable NPZs linked after checksum verification. Historical case/cache metadata is never rewritten. No Phase B information feeds adaptation.

Resume uses immutable Phase A and completed Phase B markers, a persistent stat-bound checksum index and the existing compact pending-job planner. One suite/GPU owner; max two resident models, one pending image write, 3 GiB RSS guard and original wall/native/disk/free-space safeguards. TIMEOUT/INFEASIBLE/MEMORY_LIMIT rows remain NA; DP Q0 delivery overruns remain explicitly reported as infeasible fallbacks, not feasible solutions.

Active: `None`.

| Trace | Method | Success/cases | PSNR cases | Body Mbps | Goodput Mbps | Startup s | Rebuffer s | Solver ms | Q0 GOPs | PSNR dB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HS | NAVA | 63/63 | 63 | 369.323 | 350.011 | 0.985 | 0.000 | 0.146 | 68 | 23.516 |
| HS | Visibility × MSE | 63/63 | 63 | 441.249 | 426.824 | 0.766 | 0.000 | 0.188 | 53 | 34.096 |
| HS | Visibility × PSNR | 63/63 | 63 | 443.297 | 428.735 | 0.763 | 0.000 | 0.197 | 55 | 33.835 |
| HS | Exhaustive Search | 18/63 | 18 | 492.677 | 468.094 | 0.358 | 0.000 | 1.822 | 0 | 43.656 |
| LS | NAVA | 63/63 | 63 | 89.462 | 87.988 | 2.534 | 11.026 | 0.044 | 378 | 23.078 |
| LS | Visibility × MSE | 63/63 | 63 | 93.063 | 91.786 | 2.414 | 8.764 | 0.058 | 378 | 25.655 |
| LS | Visibility × PSNR | 63/63 | 63 | 93.116 | 91.891 | 2.425 | 8.747 | 0.053 | 378 | 25.443 |
| LS | Exhaustive Search | 12/63 | 12 | 110.826 | 108.677 | 1.429 | 0.000 | 1.180 | 0 | 35.030 |
| LV | NAVA | 63/63 | 63 | 332.081 | 316.453 | 0.995 | 0.034 | 0.141 | 74 | 23.511 |
| LV | Visibility × MSE | 63/63 | 63 | 388.802 | 377.033 | 0.774 | 0.000 | 0.176 | 75 | 33.792 |
| LV | Visibility × PSNR | 63/63 | 63 | 389.987 | 378.281 | 0.777 | 0.000 | 0.172 | 74 | 33.509 |
| LV | Exhaustive Search | 18/63 | 18 | 426.408 | 408.293 | 0.376 | 0.000 | 2.145 | 0 | 43.656 |

Paired deltas below are **NAVA minus the visibility method**, matched by trace/scene/N/duration. Streaming deltas require two SUCCESS cases; PSNR additionally requires two completed Phase B results. All original Visibility × MSE versus PSNR/Exhaustive paired rows remain in paired_comparisons.csv.

| Trace | Versus | Streaming pairs | PSNR pairs | Δ PSNR dB | Δ body Mbps | Δ goodput Mbps | Δ startup s | Δ rebuffer s | Δ solver ms | Δ Q0 GOPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HS | Visibility × MSE | 63 | 63 | -10.580 | -71.926 | -76.813 | 0.219 | 0.000 | -0.042 | 0.238 |
| HS | Visibility × PSNR | 63 | 63 | -10.319 | -73.973 | -78.723 | 0.222 | 0.000 | -0.051 | 0.206 |
| LS | Visibility × MSE | 63 | 63 | -2.577 | -3.601 | -3.799 | 0.120 | 2.262 | -0.014 | 0.000 |
| LS | Visibility × PSNR | 63 | 63 | -2.365 | -3.654 | -3.904 | 0.109 | 2.279 | -0.008 | 0.000 |
| LV | Visibility × MSE | 63 | 63 | -10.281 | -56.721 | -60.580 | 0.221 | 0.034 | -0.035 | -0.016 |
| LV | Visibility × PSNR | 63 | 63 | -9.999 | -57.905 | -61.828 | 0.218 | 0.034 | -0.032 | 0.000 |

Success-only aggregate coverage differs for Exhaustive Search. Process RSS includes prior native allocations. Historical HV (offset 18 s) remains separate. Original aggregate/report versions are archived under `preserved/aggregate/` in the NAVA output; original per-case results, configurations and completion markers stay in place.

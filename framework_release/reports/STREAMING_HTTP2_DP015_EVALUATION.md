# Real HTTP/2 streaming evaluation — DP Δ = 0.15 Mbps

Updated 2026-11-03T19:54:36Z. Run **full_315**; status **complete**, phase **B**.

Cases terminal: **315/315**; Phase A statuses `{'SUCCESS': 270, 'INFEASIBLE': 18, 'TIMEOUT': 27}`; rendered-quality statuses `{'COMPLETE': 270, 'NA': 45}`. Active: `None`.

Output: `/home/fil/Hoang/MultiObj3dgsVstream/experiments/streaming_evaluation/runs/streaming_http2_dp015_v1`. Checkpoints, raw requests/decisions/buffer/solver diagnostics, display records, per-frame PSNR CSV/JSON, lossless image/metric caches and logs remain versioned here.

The full matrix is N=3/5/7/10/12/15/20 × camera_rotation/long_line/random_spatial × 10/20/30 s × exactly NAVA, Visibility × PSNR, Visibility × SSE, Visibility × MSE, Exhaustive Search. Smoke mode is explicitly a two-second N=3/N=20 subset and is not a completed matrix.

Protocol retains Longdress/Soldier, six-second source, 30 FPS, 640×480, original seeded placements and recorded camera trajectories, Lumos5G HV offset 18 s (102 s finite service), 80 Mbps causal bootstrap, safety 0.9, 20 ms per-resource latency, 17 resources/object/GOP and independent instance/loop transfers. HTTPX negotiates TLS/ALPN h2 against Hypercorn with one persistent connection per case and 16 multiplexed streams. One aggregate application-body shaper serves the trace; it is not packet/radio emulation. The measured Python host CPU ceiling still applies.

The four utility functions are unchanged. DP uses conservative Δ=150000 bps and exact-byte feasibility. Exhaustive Search uses the identical Visibility × PSNR utility rows and exact physical unions, enumerates combinations without DP, and returns a selection only after complete enumeration. Per-GOP limits are 5 s, 64 MiB extra process RSS, 4 GiB absolute RSS; no incumbent, DP or Q0 substitution is returned on a limit. TIMEOUT/MEMORY_LIMIT/INFEASIBLE cases have NA QoE/PSNR, while measured attempt latency/memory remains available. The common 80 Mbps bootstrap may itself make Exhaustive Search infeasible at larger N; this is retained explicitly rather than relaxing its bandwidth budget. Existing DP Q0 overrun behavior remains explicitly identified as a non-feasible delivery fallback.

Phase A prepares candidate-independent oracle-trace visibility before the media clock, recording preparation cost separately. Online utility/optimization, requests and playback waits consume real wall time. TLS setup/warmup is separate. It saves each scheduled displayed frame’s wall/media timestamp, source frame/GOP, selected CRFs, exact camera intrinsics/extrinsics and all transforms in phase_a/played_frames.json. Stall holds remain in the continuous buffer intervals. These are the existing modeled playback display events, not a live interactive renderer.

Phase B reads those persisted poses without regenerating trajectories. It renders only selected joint scenes and corresponding all-CRF20 joint references, uses validated full-viewport clipped RGB SSE/sample-count PSNR and checksum-verified exact-identity caches. All-CRF20 matches retain +∞ flags. Pooled PSNR sums SSE/counts. No Phase B image or score feeds adaptation.

One writer lock, atomic terminal/phase/image checkpoints, bounded native/GPU/disk/wall resources, two attempts per failed case/job and report heartbeats support continuation. Repeating the same command resumes verified completed cases and image caches. Historical results and production codec/renderer remain untouched. No full-matrix result is claimed before completion.

| Scene | N | Duration s | Method | Status | Startup s | Rebuffer s | Body Mbps | Solver ms | PSNR dB | Frames |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| camera_rotation | 3 | 10 | NAVA | SUCCESS/COMPLETE | 0.385 | 0.000 | 377.711 | 0.101 | 25.856 | 300 |
| camera_rotation | 3 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.322 | 0.000 | 499.153 | 0.112 | 42.798 | 300 |
| camera_rotation | 3 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.321 | 0.000 | 505.579 | 0.095 | 42.798 | 300 |
| camera_rotation | 3 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.317 | 0.000 | 504.803 | 0.101 | 42.798 | 300 |
| camera_rotation | 3 | 10 | Exhaustive Search | SUCCESS/COMPLETE | 0.329 | 0.000 | 492.587 | 0.368 | 42.798 | 300 |
| camera_rotation | 3 | 20 | NAVA | SUCCESS/COMPLETE | 0.384 | 0.000 | 372.497 | 0.102 | 26.077 | 600 |
| camera_rotation | 3 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.315 | 0.000 | 503.122 | 0.167 | 45.808 | 600 |
| camera_rotation | 3 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.320 | 0.000 | 509.335 | 0.172 | 45.808 | 600 |
| camera_rotation | 3 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.318 | 0.000 | 510.585 | 0.370 | 45.808 | 600 |
| camera_rotation | 3 | 20 | Exhaustive Search | SUCCESS/COMPLETE | 0.315 | 0.000 | 513.695 | 1.517 | 45.808 | 600 |
| camera_rotation | 3 | 30 | NAVA | SUCCESS/COMPLETE | 0.376 | 0.000 | 364.414 | 0.101 | 25.689 | 900 |
| camera_rotation | 3 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.312 | 0.000 | 503.636 | 0.450 | 47.569 | 900 |
| camera_rotation | 3 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.311 | 0.000 | 508.870 | 0.307 | 47.569 | 900 |
| camera_rotation | 3 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.317 | 0.000 | 502.030 | 0.437 | 47.569 | 900 |
| camera_rotation | 3 | 30 | Exhaustive Search | SUCCESS/COMPLETE | 0.318 | 0.000 | 502.410 | 1.536 | 47.569 | 900 |
| long_line | 3 | 10 | NAVA | SUCCESS/COMPLETE | 0.331 | 0.000 | 451.072 | 0.100 | 30.454 | 300 |
| long_line | 3 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.324 | 0.000 | 496.843 | 0.104 | 46.100 | 300 |
| long_line | 3 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.321 | 0.000 | 508.908 | 0.135 | 46.100 | 300 |
| long_line | 3 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.318 | 0.000 | 511.656 | 0.101 | 46.100 | 300 |
| long_line | 3 | 10 | Exhaustive Search | SUCCESS/COMPLETE | 0.318 | 0.000 | 511.723 | 0.325 | 46.100 | 300 |
| long_line | 3 | 20 | NAVA | SUCCESS/COMPLETE | 0.329 | 0.000 | 442.058 | 0.103 | 30.921 | 600 |
| long_line | 3 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.376 | 0.000 | 510.383 | 0.429 | 49.110 | 600 |
| long_line | 3 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.351 | 0.000 | 510.701 | 0.326 | 49.110 | 600 |
| long_line | 3 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.317 | 0.000 | 517.201 | 0.440 | 49.110 | 600 |
| long_line | 3 | 20 | Exhaustive Search | SUCCESS/COMPLETE | 0.319 | 0.000 | 518.364 | 1.526 | 49.110 | 600 |
| long_line | 3 | 30 | NAVA | SUCCESS/COMPLETE | 0.340 | 0.000 | 447.633 | 0.104 | 30.989 | 900 |
| long_line | 3 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.317 | 0.000 | 518.135 | 0.440 | 50.871 | 900 |
| long_line | 3 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.319 | 0.000 | 515.285 | 0.440 | 50.871 | 900 |
| long_line | 3 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.316 | 0.000 | 515.007 | 0.436 | 50.871 | 900 |
| long_line | 3 | 30 | Exhaustive Search | SUCCESS/COMPLETE | 0.312 | 0.000 | 525.270 | 1.555 | 50.871 | 900 |
| random_spatial | 3 | 10 | NAVA | SUCCESS/COMPLETE | 0.274 | 0.000 | 367.841 | 0.101 | 26.081 | 300 |
| random_spatial | 3 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.353 | 0.000 | 511.928 | 0.132 | 41.021 | 300 |
| random_spatial | 3 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.272 | 0.000 | 484.689 | 0.144 | 44.330 | 300 |
| random_spatial | 3 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.274 | 0.000 | 497.946 | 0.099 | 44.330 | 300 |
| random_spatial | 3 | 10 | Exhaustive Search | SUCCESS/COMPLETE | 0.312 | 0.000 | 503.728 | 0.313 | 41.021 | 300 |
| random_spatial | 3 | 20 | NAVA | SUCCESS/COMPLETE | 0.269 | 0.000 | 363.117 | 0.102 | 26.263 | 600 |
| random_spatial | 3 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.316 | 0.000 | 508.218 | 0.307 | 44.031 | 600 |
| random_spatial | 3 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.309 | 0.000 | 503.443 | 0.223 | 47.340 | 600 |
| random_spatial | 3 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.271 | 0.000 | 497.364 | 0.423 | 47.340 | 600 |
| random_spatial | 3 | 20 | Exhaustive Search | SUCCESS/COMPLETE | 0.314 | 0.000 | 496.757 | 0.434 | 44.031 | 600 |
| random_spatial | 3 | 30 | NAVA | SUCCESS/COMPLETE | 0.334 | 0.000 | 379.627 | 0.103 | 26.147 | 900 |
| random_spatial | 3 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.323 | 0.000 | 490.982 | 0.411 | 45.792 | 900 |
| random_spatial | 3 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.271 | 0.000 | 504.110 | 0.422 | 49.101 | 900 |
| random_spatial | 3 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.272 | 0.000 | 503.707 | 0.145 | 49.101 | 900 |
| random_spatial | 3 | 30 | Exhaustive Search | SUCCESS/COMPLETE | 0.313 | 0.000 | 515.769 | 1.526 | 45.792 | 900 |
| camera_rotation | 5 | 10 | NAVA | SUCCESS/COMPLETE | 0.504 | 0.000 | 342.419 | 0.113 | 24.243 | 300 |
| camera_rotation | 5 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.400 | 0.000 | 503.756 | 0.126 | 37.162 | 300 |
| camera_rotation | 5 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.417 | 0.000 | 483.642 | 0.119 | 37.162 | 300 |
| camera_rotation | 5 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.455 | 0.000 | 486.160 | 0.124 | 37.162 | 300 |
| camera_rotation | 5 | 10 | Exhaustive Search | SUCCESS/COMPLETE | 0.396 | 0.000 | 508.162 | 1.898 | 37.162 | 300 |
| camera_rotation | 5 | 20 | NAVA | SUCCESS/COMPLETE | 0.495 | 0.000 | 343.527 | 0.121 | 24.445 | 600 |
| camera_rotation | 5 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.400 | 0.000 | 495.802 | 0.131 | 40.173 | 600 |
| camera_rotation | 5 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.396 | 0.000 | 510.996 | 0.214 | 40.173 | 600 |
| camera_rotation | 5 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.396 | 0.000 | 507.801 | 0.280 | 40.173 | 600 |
| camera_rotation | 5 | 20 | Exhaustive Search | SUCCESS/COMPLETE | 0.397 | 0.000 | 502.753 | 3.635 | 40.173 | 600 |
| camera_rotation | 5 | 30 | NAVA | SUCCESS/COMPLETE | 0.496 | 0.000 | 348.763 | 0.122 | 24.136 | 900 |
| camera_rotation | 5 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.404 | 0.000 | 494.042 | 0.134 | 41.934 | 900 |
| camera_rotation | 5 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.451 | 0.000 | 501.929 | 0.517 | 41.934 | 900 |
| camera_rotation | 5 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.478 | 0.000 | 500.499 | 0.141 | 41.934 | 900 |
| camera_rotation | 5 | 30 | Exhaustive Search | SUCCESS/COMPLETE | 0.417 | 0.000 | 497.744 | 3.908 | 41.934 | 900 |
| long_line | 5 | 10 | NAVA | SUCCESS/COMPLETE | 0.468 | 0.000 | 430.336 | 0.123 | 26.864 | 300 |
| long_line | 5 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.389 | 0.000 | 488.161 | 0.123 | 39.735 | 300 |
| long_line | 5 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.389 | 0.000 | 508.139 | 0.128 | 39.735 | 300 |
| long_line | 5 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.388 | 0.000 | 504.207 | 0.125 | 39.735 | 300 |
| long_line | 5 | 10 | Exhaustive Search | SUCCESS/COMPLETE | 0.397 | 0.000 | 498.058 | 1.898 | 39.735 | 300 |
| long_line | 5 | 20 | NAVA | SUCCESS/COMPLETE | 0.464 | 0.000 | 407.309 | 0.123 | 27.025 | 600 |
| long_line | 5 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.388 | 0.000 | 504.751 | 0.148 | 42.745 | 600 |
| long_line | 5 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.463 | 0.000 | 511.911 | 0.490 | 42.745 | 600 |
| long_line | 5 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.389 | 0.000 | 491.979 | 0.561 | 42.745 | 600 |
| long_line | 5 | 20 | Exhaustive Search | SUCCESS/COMPLETE | 0.388 | 0.000 | 510.778 | 3.906 | 42.745 | 600 |
| long_line | 5 | 30 | NAVA | SUCCESS/COMPLETE | 0.460 | 0.000 | 433.259 | 0.123 | 26.994 | 900 |
| long_line | 5 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.383 | 0.000 | 496.248 | 0.132 | 44.506 | 900 |
| long_line | 5 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.397 | 0.000 | 499.471 | 0.147 | 44.506 | 900 |
| long_line | 5 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.379 | 0.000 | 511.534 | 0.134 | 44.506 | 900 |
| long_line | 5 | 30 | Exhaustive Search | SUCCESS/COMPLETE | 0.389 | 0.000 | 513.986 | 3.403 | 44.506 | 900 |
| random_spatial | 5 | 10 | NAVA | SUCCESS/COMPLETE | 0.507 | 0.000 | 365.907 | 0.116 | 24.954 | 300 |
| random_spatial | 5 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.382 | 0.000 | 499.735 | 0.122 | 39.556 | 300 |
| random_spatial | 5 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.385 | 0.000 | 497.245 | 0.125 | 39.556 | 300 |
| random_spatial | 5 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.395 | 0.000 | 481.409 | 0.130 | 39.556 | 300 |
| random_spatial | 5 | 10 | Exhaustive Search | SUCCESS/COMPLETE | 0.398 | 0.000 | 476.695 | 1.968 | 39.556 | 300 |
| random_spatial | 5 | 20 | NAVA | SUCCESS/COMPLETE | 0.455 | 0.000 | 367.435 | 0.121 | 25.295 | 600 |
| random_spatial | 5 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.384 | 0.000 | 504.275 | 0.270 | 42.566 | 600 |
| random_spatial | 5 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.465 | 0.000 | 481.384 | 0.510 | 42.566 | 600 |
| random_spatial | 5 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.403 | 0.000 | 488.701 | 0.184 | 42.566 | 600 |
| random_spatial | 5 | 20 | Exhaustive Search | SUCCESS/COMPLETE | 0.380 | 0.000 | 504.262 | 3.714 | 42.566 | 600 |
| random_spatial | 5 | 30 | NAVA | SUCCESS/COMPLETE | 0.441 | 0.000 | 372.456 | 0.122 | 25.119 | 900 |
| random_spatial | 5 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.389 | 0.000 | 500.542 | 0.246 | 44.327 | 900 |
| random_spatial | 5 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.386 | 0.000 | 497.689 | 0.514 | 44.327 | 900 |
| random_spatial | 5 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.384 | 0.000 | 497.529 | 0.146 | 44.327 | 900 |
| random_spatial | 5 | 30 | Exhaustive Search | SUCCESS/COMPLETE | 0.389 | 0.000 | 487.849 | 3.900 | 44.327 | 900 |
| camera_rotation | 7 | 10 | NAVA | SUCCESS/COMPLETE | 0.649 | 0.000 | 398.588 | 0.145 | 22.836 | 300 |
| camera_rotation | 7 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.551 | 0.000 | 494.327 | 0.158 | 29.824 | 300 |
| camera_rotation | 7 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.486 | 0.000 | 487.876 | 0.151 | 29.824 | 300 |
| camera_rotation | 7 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.496 | 0.000 | 484.563 | 0.146 | 29.824 | 300 |
| camera_rotation | 7 | 10 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 14.442 | NA | NA |
| camera_rotation | 7 | 20 | NAVA | SUCCESS/COMPLETE | 0.649 | 0.000 | 403.309 | 0.154 | 23.121 | 600 |
| camera_rotation | 7 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.489 | 0.000 | 506.028 | 0.163 | 32.834 | 600 |
| camera_rotation | 7 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.491 | 0.000 | 507.299 | 0.616 | 32.834 | 600 |
| camera_rotation | 7 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.483 | 0.000 | 506.432 | 0.160 | 32.834 | 600 |
| camera_rotation | 7 | 20 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 14.617 | NA | NA |
| camera_rotation | 7 | 30 | NAVA | SUCCESS/COMPLETE | 0.657 | 0.000 | 392.175 | 0.153 | 22.998 | 900 |
| camera_rotation | 7 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.486 | 0.000 | 500.478 | 0.182 | 34.595 | 900 |
| camera_rotation | 7 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.492 | 0.000 | 502.255 | 0.437 | 34.595 | 900 |
| camera_rotation | 7 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.504 | 0.000 | 497.810 | 0.241 | 34.595 | 900 |
| camera_rotation | 7 | 30 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 14.357 | NA | NA |
| long_line | 7 | 10 | NAVA | SUCCESS/COMPLETE | 0.633 | 0.000 | 424.926 | 0.147 | 24.356 | 300 |
| long_line | 7 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.494 | 0.000 | 484.242 | 0.145 | 31.318 | 300 |
| long_line | 7 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.555 | 0.000 | 482.428 | 0.140 | 31.318 | 300 |
| long_line | 7 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.489 | 0.000 | 472.699 | 0.149 | 31.318 | 300 |
| long_line | 7 | 10 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 14.485 | NA | NA |
| long_line | 7 | 20 | NAVA | SUCCESS/COMPLETE | 0.638 | 0.000 | 420.850 | 0.160 | 24.520 | 600 |
| long_line | 7 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.487 | 0.000 | 495.488 | 0.162 | 34.328 | 600 |
| long_line | 7 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.492 | 0.000 | 496.437 | 0.203 | 34.328 | 600 |
| long_line | 7 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.496 | 0.000 | 486.351 | 0.165 | 34.328 | 600 |
| long_line | 7 | 20 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 14.472 | NA | NA |
| long_line | 7 | 30 | NAVA | SUCCESS/COMPLETE | 0.748 | 0.000 | 412.314 | 0.154 | 24.519 | 900 |
| long_line | 7 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.503 | 0.000 | 488.331 | 0.165 | 36.089 | 900 |
| long_line | 7 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.497 | 0.000 | 488.134 | 0.184 | 36.089 | 900 |
| long_line | 7 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.499 | 0.000 | 487.340 | 0.168 | 36.089 | 900 |
| long_line | 7 | 30 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 14.277 | NA | NA |
| random_spatial | 7 | 10 | NAVA | SUCCESS/COMPLETE | 0.693 | 0.000 | 377.551 | 0.147 | 23.884 | 300 |
| random_spatial | 7 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.506 | 0.000 | 481.336 | 0.146 | 34.156 | 300 |
| random_spatial | 7 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.511 | 0.000 | 475.350 | 0.145 | 34.188 | 300 |
| random_spatial | 7 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.565 | 0.000 | 481.205 | 0.156 | 34.188 | 300 |
| random_spatial | 7 | 10 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 14.610 | NA | NA |
| random_spatial | 7 | 20 | NAVA | SUCCESS/COMPLETE | 0.639 | 0.000 | 376.735 | 0.152 | 23.960 | 600 |
| random_spatial | 7 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.492 | 0.000 | 487.412 | 0.167 | 37.198 | 600 |
| random_spatial | 7 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.490 | 0.000 | 496.118 | 0.156 | 37.198 | 600 |
| random_spatial | 7 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.491 | 0.000 | 499.531 | 0.171 | 37.198 | 600 |
| random_spatial | 7 | 20 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 14.523 | NA | NA |
| random_spatial | 7 | 30 | NAVA | SUCCESS/COMPLETE | 0.621 | 0.000 | 386.711 | 0.150 | 23.711 | 900 |
| random_spatial | 7 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.492 | 0.000 | 507.586 | 0.170 | 38.927 | 900 |
| random_spatial | 7 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.493 | 0.000 | 500.645 | 0.372 | 38.959 | 900 |
| random_spatial | 7 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.505 | 0.000 | 497.034 | 0.158 | 38.959 | 900 |
| random_spatial | 7 | 30 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 14.531 | NA | NA |
| camera_rotation | 10 | 10 | NAVA | SUCCESS/COMPLETE | 0.958 | 0.000 | 394.869 | 0.166 | 22.321 | 300 |
| camera_rotation | 10 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.716 | 0.000 | 453.623 | 0.207 | 30.093 | 300 |
| camera_rotation | 10 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.707 | 0.000 | 452.974 | 0.168 | 30.192 | 300 |
| camera_rotation | 10 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.723 | 0.000 | 457.593 | 0.167 | 30.173 | 300 |
| camera_rotation | 10 | 10 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 1002.962 | NA | NA |
| camera_rotation | 10 | 20 | NAVA | SUCCESS/COMPLETE | 0.940 | 0.000 | 403.616 | 0.170 | 22.533 | 600 |
| camera_rotation | 10 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.719 | 0.000 | 475.501 | 0.201 | 32.935 | 600 |
| camera_rotation | 10 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.715 | 0.000 | 474.822 | 0.342 | 33.119 | 600 |
| camera_rotation | 10 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.711 | 0.000 | 477.774 | 0.195 | 33.088 | 600 |
| camera_rotation | 10 | 20 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 991.469 | NA | NA |
| camera_rotation | 10 | 30 | NAVA | SUCCESS/COMPLETE | 0.948 | 0.000 | 391.166 | 0.168 | 22.347 | 900 |
| camera_rotation | 10 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.709 | 0.000 | 493.213 | 0.202 | 34.684 | 900 |
| camera_rotation | 10 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.708 | 0.000 | 495.957 | 0.352 | 34.935 | 900 |
| camera_rotation | 10 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.729 | 0.000 | 475.518 | 0.202 | 34.749 | 900 |
| camera_rotation | 10 | 30 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 1108.945 | NA | NA |
| long_line | 10 | 10 | NAVA | SUCCESS/COMPLETE | 0.925 | 0.000 | 401.541 | 0.163 | 23.334 | 300 |
| long_line | 10 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.812 | 0.000 | 427.575 | 0.168 | 30.258 | 300 |
| long_line | 10 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.825 | 0.000 | 434.059 | 0.160 | 30.245 | 300 |
| long_line | 10 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.720 | 0.000 | 456.728 | 0.173 | 30.574 | 300 |
| long_line | 10 | 10 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 989.001 | NA | NA |
| long_line | 10 | 20 | NAVA | SUCCESS/COMPLETE | 0.921 | 0.000 | 405.085 | 0.168 | 23.549 | 600 |
| long_line | 10 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.720 | 0.000 | 470.260 | 0.257 | 33.490 | 600 |
| long_line | 10 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.721 | 0.000 | 473.129 | 0.273 | 33.585 | 600 |
| long_line | 10 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.716 | 0.000 | 455.510 | 0.246 | 33.545 | 600 |
| long_line | 10 | 20 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 1013.293 | NA | NA |
| long_line | 10 | 30 | NAVA | SUCCESS/COMPLETE | 0.916 | 0.000 | 398.937 | 0.172 | 23.415 | 900 |
| long_line | 10 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.725 | 0.000 | 472.361 | 0.196 | 35.235 | 900 |
| long_line | 10 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.711 | 0.000 | 489.239 | 0.234 | 35.346 | 900 |
| long_line | 10 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.715 | 0.000 | 479.755 | 0.196 | 35.346 | 900 |
| long_line | 10 | 30 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 986.022 | NA | NA |
| random_spatial | 10 | 10 | NAVA | SUCCESS/COMPLETE | 0.928 | 0.000 | 371.614 | 0.166 | 23.004 | 300 |
| random_spatial | 10 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.719 | 0.000 | 443.453 | 0.175 | 32.025 | 300 |
| random_spatial | 10 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.842 | 0.000 | 424.282 | 0.153 | 31.697 | 300 |
| random_spatial | 10 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.729 | 0.000 | 453.330 | 0.165 | 32.338 | 300 |
| random_spatial | 10 | 10 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 982.723 | NA | NA |
| random_spatial | 10 | 20 | NAVA | SUCCESS/COMPLETE | 0.928 | 0.000 | 369.898 | 0.164 | 23.138 | 600 |
| random_spatial | 10 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.728 | 0.000 | 466.395 | 0.192 | 34.482 | 600 |
| random_spatial | 10 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.707 | 0.000 | 478.393 | 0.234 | 34.974 | 600 |
| random_spatial | 10 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.702 | 0.000 | 465.764 | 0.196 | 34.917 | 600 |
| random_spatial | 10 | 20 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 994.637 | NA | NA |
| random_spatial | 10 | 30 | NAVA | SUCCESS/COMPLETE | 0.923 | 0.000 | 365.128 | 0.167 | 22.866 | 900 |
| random_spatial | 10 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.723 | 0.000 | 475.123 | 0.203 | 35.612 | 900 |
| random_spatial | 10 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.728 | 0.000 | 483.374 | 0.214 | 36.442 | 900 |
| random_spatial | 10 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.712 | 0.000 | 486.992 | 0.205 | 36.478 | 900 |
| random_spatial | 10 | 30 | Exhaustive Search | INFEASIBLE/NA | NA | NA | NA | 991.919 | NA | NA |
| camera_rotation | 12 | 10 | NAVA | SUCCESS/COMPLETE | 1.144 | 0.000 | 392.443 | 0.172 | 21.895 | 300 |
| camera_rotation | 12 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.854 | 0.000 | 423.606 | 0.179 | 29.107 | 300 |
| camera_rotation | 12 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.865 | 0.000 | 420.982 | 0.173 | 29.043 | 300 |
| camera_rotation | 12 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.865 | 0.000 | 415.945 | 0.173 | 29.069 | 300 |
| camera_rotation | 12 | 10 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.048 | NA | NA |
| camera_rotation | 12 | 20 | NAVA | SUCCESS/COMPLETE | 1.153 | 0.000 | 375.408 | 0.167 | 22.120 | 600 |
| camera_rotation | 12 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.863 | 0.000 | 427.922 | 0.537 | 31.280 | 600 |
| camera_rotation | 12 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.879 | 0.000 | 443.943 | 0.193 | 31.594 | 600 |
| camera_rotation | 12 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.853 | 0.000 | 443.944 | 0.227 | 31.770 | 600 |
| camera_rotation | 12 | 20 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.151 | NA | NA |
| camera_rotation | 12 | 30 | NAVA | SUCCESS/COMPLETE | 1.142 | 0.000 | 383.537 | 0.181 | 21.971 | 900 |
| camera_rotation | 12 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.847 | 0.000 | 464.880 | 0.206 | 33.072 | 900 |
| camera_rotation | 12 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.857 | 0.000 | 445.849 | 0.213 | 33.075 | 900 |
| camera_rotation | 12 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.861 | 0.000 | 447.305 | 0.201 | 33.082 | 900 |
| camera_rotation | 12 | 30 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.064 | NA | NA |
| long_line | 12 | 10 | NAVA | SUCCESS/COMPLETE | 1.109 | 0.000 | 366.611 | 0.162 | 22.199 | 300 |
| long_line | 12 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.878 | 0.000 | 414.154 | 0.173 | 28.961 | 300 |
| long_line | 12 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.869 | 0.000 | 422.851 | 0.197 | 29.234 | 300 |
| long_line | 12 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.851 | 0.000 | 426.765 | 0.175 | 29.220 | 300 |
| long_line | 12 | 10 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.152 | NA | NA |
| long_line | 12 | 20 | NAVA | SUCCESS/COMPLETE | 1.108 | 0.000 | 367.322 | 0.170 | 22.352 | 600 |
| long_line | 12 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.990 | 0.000 | 421.957 | 0.188 | 31.026 | 600 |
| long_line | 12 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.861 | 0.000 | 432.816 | 0.201 | 31.852 | 600 |
| long_line | 12 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.853 | 0.000 | 432.543 | 0.200 | 31.891 | 600 |
| long_line | 12 | 20 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.149 | NA | NA |
| long_line | 12 | 30 | NAVA | SUCCESS/COMPLETE | 1.100 | 0.000 | 368.908 | 0.170 | 22.264 | 900 |
| long_line | 12 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.850 | 0.000 | 451.021 | 0.235 | 33.119 | 900 |
| long_line | 12 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 1.022 | 0.000 | 446.392 | 0.209 | 33.367 | 900 |
| long_line | 12 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.996 | 0.000 | 443.155 | 0.197 | 32.814 | 900 |
| long_line | 12 | 30 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.110 | NA | NA |
| random_spatial | 12 | 10 | NAVA | SUCCESS/COMPLETE | 1.105 | 0.000 | 370.633 | 0.169 | 21.955 | 300 |
| random_spatial | 12 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 0.863 | 0.000 | 408.380 | 0.170 | 29.031 | 300 |
| random_spatial | 12 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 0.859 | 0.000 | 424.170 | 0.171 | 29.098 | 300 |
| random_spatial | 12 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 0.853 | 0.000 | 426.360 | 0.176 | 29.166 | 300 |
| random_spatial | 12 | 10 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.135 | NA | NA |
| random_spatial | 12 | 20 | NAVA | SUCCESS/COMPLETE | 1.098 | 0.000 | 363.281 | 0.171 | 22.097 | 600 |
| random_spatial | 12 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 0.881 | 0.000 | 429.602 | 0.216 | 30.845 | 600 |
| random_spatial | 12 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 0.867 | 0.000 | 437.778 | 0.195 | 31.414 | 600 |
| random_spatial | 12 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 0.861 | 0.000 | 443.810 | 0.201 | 31.333 | 600 |
| random_spatial | 12 | 20 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.122 | NA | NA |
| random_spatial | 12 | 30 | NAVA | SUCCESS/COMPLETE | 1.273 | 0.000 | 378.949 | 0.172 | 21.952 | 900 |
| random_spatial | 12 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 0.860 | 0.000 | 448.441 | 0.200 | 32.175 | 900 |
| random_spatial | 12 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 0.861 | 0.000 | 448.407 | 0.200 | 32.611 | 900 |
| random_spatial | 12 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 0.860 | 0.000 | 441.676 | 0.199 | 32.449 | 900 |
| random_spatial | 12 | 30 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.106 | NA | NA |
| camera_rotation | 15 | 10 | NAVA | SUCCESS/COMPLETE | 1.429 | 0.000 | 334.528 | 0.173 | 21.294 | 300 |
| camera_rotation | 15 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 1.206 | 0.000 | 370.055 | 0.161 | 25.920 | 300 |
| camera_rotation | 15 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 1.062 | 0.000 | 385.016 | 0.173 | 27.007 | 300 |
| camera_rotation | 15 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 1.201 | 0.000 | 372.739 | 0.159 | 26.470 | 300 |
| camera_rotation | 15 | 10 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.096 | NA | NA |
| camera_rotation | 15 | 20 | NAVA | SUCCESS/COMPLETE | 1.423 | 0.000 | 341.194 | 0.166 | 21.550 | 600 |
| camera_rotation | 15 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 1.223 | 0.000 | 385.521 | 0.178 | 27.602 | 600 |
| camera_rotation | 15 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 1.205 | 0.000 | 377.162 | 0.189 | 27.935 | 600 |
| camera_rotation | 15 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 1.083 | 0.000 | 383.123 | 0.179 | 28.237 | 600 |
| camera_rotation | 15 | 20 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.119 | NA | NA |
| camera_rotation | 15 | 30 | NAVA | SUCCESS/COMPLETE | 1.534 | 0.000 | 351.306 | 0.168 | 21.395 | 900 |
| camera_rotation | 15 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 1.052 | 0.000 | 417.236 | 0.471 | 29.236 | 900 |
| camera_rotation | 15 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 1.060 | 0.000 | 401.544 | 0.191 | 29.231 | 900 |
| camera_rotation | 15 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 1.050 | 0.000 | 406.595 | 0.197 | 29.382 | 900 |
| camera_rotation | 15 | 30 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.145 | NA | NA |
| long_line | 15 | 10 | NAVA | SUCCESS/COMPLETE | 1.343 | 0.000 | 355.884 | 0.171 | 21.845 | 300 |
| long_line | 15 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 1.063 | 0.000 | 382.017 | 0.168 | 26.716 | 300 |
| long_line | 15 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 1.055 | 0.000 | 393.098 | 0.183 | 26.713 | 300 |
| long_line | 15 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 1.064 | 0.000 | 381.675 | 0.162 | 26.591 | 300 |
| long_line | 15 | 10 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.153 | NA | NA |
| long_line | 15 | 20 | NAVA | SUCCESS/COMPLETE | 1.362 | 0.000 | 366.068 | 0.172 | 21.901 | 600 |
| long_line | 15 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 1.060 | 0.000 | 398.903 | 0.183 | 28.723 | 600 |
| long_line | 15 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 1.077 | 0.000 | 393.817 | 0.209 | 28.610 | 600 |
| long_line | 15 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 1.049 | 0.000 | 399.977 | 0.191 | 28.825 | 600 |
| long_line | 15 | 20 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.074 | NA | NA |
| long_line | 15 | 30 | NAVA | SUCCESS/COMPLETE | 1.552 | 0.000 | 355.261 | 0.177 | 21.904 | 900 |
| long_line | 15 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 1.056 | 0.000 | 408.696 | 0.216 | 29.675 | 900 |
| long_line | 15 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 1.054 | 0.000 | 409.003 | 0.190 | 29.658 | 900 |
| long_line | 15 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 1.071 | 0.000 | 397.209 | 0.199 | 29.490 | 900 |
| long_line | 15 | 30 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.071 | NA | NA |
| random_spatial | 15 | 10 | NAVA | SUCCESS/COMPLETE | 1.368 | 0.000 | 365.556 | 0.167 | 21.400 | 300 |
| random_spatial | 15 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 1.068 | 0.000 | 388.594 | 0.166 | 26.056 | 300 |
| random_spatial | 15 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 1.060 | 0.000 | 374.226 | 0.178 | 26.234 | 300 |
| random_spatial | 15 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 1.057 | 0.000 | 397.630 | 0.169 | 26.245 | 300 |
| random_spatial | 15 | 10 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.074 | NA | NA |
| random_spatial | 15 | 20 | NAVA | SUCCESS/COMPLETE | 1.527 | 0.000 | 368.203 | 0.165 | 21.626 | 600 |
| random_spatial | 15 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 1.073 | 0.000 | 388.457 | 0.183 | 27.704 | 600 |
| random_spatial | 15 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 1.068 | 0.000 | 398.607 | 0.191 | 27.978 | 600 |
| random_spatial | 15 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 1.074 | 0.000 | 390.714 | 0.303 | 27.780 | 600 |
| random_spatial | 15 | 20 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.093 | NA | NA |
| random_spatial | 15 | 30 | NAVA | SUCCESS/COMPLETE | 1.604 | 0.000 | 361.361 | 0.167 | 21.529 | 900 |
| random_spatial | 15 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 1.098 | 0.000 | 390.249 | 0.187 | 28.111 | 900 |
| random_spatial | 15 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 1.075 | 0.000 | 402.120 | 0.205 | 28.701 | 900 |
| random_spatial | 15 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 1.071 | 0.000 | 409.529 | 0.192 | 28.978 | 900 |
| random_spatial | 15 | 30 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.066 | NA | NA |
| camera_rotation | 20 | 10 | NAVA | SUCCESS/COMPLETE | 1.896 | 0.000 | 311.974 | 0.141 | 21.089 | 300 |
| camera_rotation | 20 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 1.434 | 0.000 | 324.917 | 0.138 | 22.857 | 300 |
| camera_rotation | 20 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 1.421 | 0.000 | 321.283 | 0.142 | 23.040 | 300 |
| camera_rotation | 20 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 1.427 | 0.000 | 332.568 | 0.151 | 23.317 | 300 |
| camera_rotation | 20 | 10 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.073 | NA | NA |
| camera_rotation | 20 | 20 | NAVA | SUCCESS/COMPLETE | 1.899 | 0.000 | 316.900 | 0.139 | 21.287 | 600 |
| camera_rotation | 20 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 1.624 | 0.000 | 317.344 | 0.131 | 22.564 | 600 |
| camera_rotation | 20 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 1.411 | 0.000 | 329.002 | 0.146 | 23.406 | 600 |
| camera_rotation | 20 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 1.626 | 0.000 | 328.998 | 0.142 | 23.008 | 600 |
| camera_rotation | 20 | 20 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.108 | NA | NA |
| camera_rotation | 20 | 30 | NAVA | SUCCESS/COMPLETE | 1.910 | 0.000 | 314.213 | 0.141 | 21.173 | 900 |
| camera_rotation | 20 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 1.435 | 0.000 | 326.008 | 0.145 | 23.047 | 900 |
| camera_rotation | 20 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 1.445 | 0.000 | 321.049 | 0.150 | 22.947 | 900 |
| camera_rotation | 20 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 1.416 | 0.000 | 323.486 | 0.158 | 23.029 | 900 |
| camera_rotation | 20 | 30 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.036 | NA | NA |
| long_line | 20 | 10 | NAVA | SUCCESS/COMPLETE | 1.826 | 0.000 | 314.444 | 0.142 | 21.494 | 300 |
| long_line | 20 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 1.433 | 0.000 | 317.567 | 0.154 | 22.844 | 300 |
| long_line | 20 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 1.512 | 0.000 | 331.943 | 0.144 | 23.592 | 300 |
| long_line | 20 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 1.427 | 0.000 | 333.935 | 0.140 | 23.463 | 300 |
| long_line | 20 | 10 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.143 | NA | NA |
| long_line | 20 | 20 | NAVA | SUCCESS/COMPLETE | 1.997 | 0.000 | 309.207 | 0.132 | 21.465 | 600 |
| long_line | 20 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 1.443 | 0.000 | 321.033 | 0.139 | 22.925 | 600 |
| long_line | 20 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 1.409 | 0.000 | 333.992 | 0.162 | 23.547 | 600 |
| long_line | 20 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 1.418 | 0.000 | 332.776 | 0.149 | 23.615 | 600 |
| long_line | 20 | 20 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.187 | NA | NA |
| long_line | 20 | 30 | NAVA | SUCCESS/COMPLETE | 1.793 | 0.000 | 315.682 | 0.141 | 21.512 | 900 |
| long_line | 20 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 1.403 | 0.000 | 329.886 | 0.149 | 23.445 | 900 |
| long_line | 20 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 1.423 | 0.000 | 320.622 | 0.147 | 23.178 | 900 |
| long_line | 20 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 1.442 | 0.000 | 316.869 | 0.140 | 22.773 | 900 |
| long_line | 20 | 30 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.182 | NA | NA |
| random_spatial | 20 | 10 | NAVA | SUCCESS/COMPLETE | 1.935 | 0.000 | 306.523 | 0.145 | 20.585 | 300 |
| random_spatial | 20 | 10 | Visibility × PSNR | SUCCESS/COMPLETE | 1.481 | 0.000 | 319.401 | 0.139 | 21.810 | 300 |
| random_spatial | 20 | 10 | Visibility × SSE | SUCCESS/COMPLETE | 1.480 | 0.000 | 306.855 | 0.139 | 21.642 | 300 |
| random_spatial | 20 | 10 | Visibility × MSE | SUCCESS/COMPLETE | 1.499 | 0.000 | 313.517 | 0.142 | 21.538 | 300 |
| random_spatial | 20 | 10 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.181 | NA | NA |
| random_spatial | 20 | 20 | NAVA | SUCCESS/COMPLETE | 1.842 | 0.000 | 325.330 | 0.149 | 20.829 | 600 |
| random_spatial | 20 | 20 | Visibility × PSNR | SUCCESS/COMPLETE | 1.442 | 0.000 | 327.581 | 0.144 | 22.135 | 600 |
| random_spatial | 20 | 20 | Visibility × SSE | SUCCESS/COMPLETE | 1.421 | 0.000 | 328.753 | 0.153 | 22.201 | 600 |
| random_spatial | 20 | 20 | Visibility × MSE | SUCCESS/COMPLETE | 1.422 | 0.000 | 330.335 | 0.158 | 22.123 | 600 |
| random_spatial | 20 | 20 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.108 | NA | NA |
| random_spatial | 20 | 30 | NAVA | SUCCESS/COMPLETE | 1.857 | 0.000 | 317.473 | 0.142 | 20.725 | 900 |
| random_spatial | 20 | 30 | Visibility × PSNR | SUCCESS/COMPLETE | 1.452 | 0.000 | 305.619 | 0.158 | 21.779 | 900 |
| random_spatial | 20 | 30 | Visibility × SSE | SUCCESS/COMPLETE | 1.416 | 0.000 | 319.613 | 0.145 | 22.045 | 900 |
| random_spatial | 20 | 30 | Visibility × MSE | SUCCESS/COMPLETE | 1.394 | 0.000 | 329.155 | 0.151 | 22.131 | 900 |
| random_spatial | 20 | 30 | Exhaustive Search | TIMEOUT/NA | NA | NA | NA | 5000.113 | NA | NA |

Measured process RSS belongs to the experiment process (including preceding native allocations); it is not isolated solver-only RAM. Solver tables, utility/CRF choices, startup/rebuffering and throughput are retained in summary.csv/JSON and immutable per-case Phase A artifacts. Failures have adjacent error JSON; unavailable values remain blank in CSV and null in JSON.

# CPU ABR solver processing-time experiment

This is an isolated object-level MCKP microbenchmark using measured StreamSTGS PSNR and GOP transfer costs. It makes no playback, rendering, decoding, network, or full-system QoE claims.

## Inputs and common problem

The loader verified 80 profiles (four contents × five 60-frame GOPs × four independent CRFs), all 115,200 frame/view metric rows, and metadata hash links. PSNR is the arithmetic mean in dB across all 24 views, including training and held-out views, at scale 1.0. Costs include manifest-listed payload components plus the representation manifest itself. Each GOP lasts exactly two seconds, so R=4×transfer_bytes bit/s, without rounding. The observed PSNR range is 23.7903–35.7771 dB, and rate range is 5.2468–60.1562 Mb/s. See [DATA_AUDIT.md](DATA_AUDIT.md).

Every solver maximizes ΣwᵢPSNRᵢₖ under one shared total-rate budget, choosing exactly one CRF per object. No temporal, buffer, switching, compute, or representation-dependency constraints are added. Utility uses absolute PSNR, so a percent gap is a fraction of the entire quality objective, not just achievable quality improvement.

There are 72 cases: N=[2, 4, 8, 16, 32, 64], seeds=[2027, 2028, 2029], four budgets each. Contents are balanced by shuffled blocks of four; GOPs are sampled uniformly. All clones reuse complete measured four-CRF rows with no quality/rate perturbation. Smaller N uses a common prefix within each seed. Repeated instances are independent decision variables, not independently captured scenes. Visibility is synthetic: offscreen zero (10%); peripheral U[.01,.10] (20%); visible U[.20,.70] (45%); foreground U[.75,1] (25%), normalized to sum one. All solvers receive the same immutable arrays; hashes are recorded per case.

Effective budgets are Rmin+floor(α(Rmax−Rmin)), with α=.08,.35,.75,1.10 for tight, balanced, generous and overprovisioned. All original-problem cases are feasible. η=.9 and Ĉ=budget/η are informational; solvers directly receive the exact integer effective budget. Offscreen objects still require one representation.

## Algorithms and complexity

| Method | Implementation and guarantee | Complexity |
|---|---|---|
| Brute-force | Complete-vector odometer; every K^N vector checked; exact when completed. | O(N K^N) time, O(N) traversal memory. |
| NAVA | Section III-G/Algorithm 1 layered graph and recursive pulse with same-node label dominance, suffix minimum-rate feasibility, and maximum-utility upper bounds; exact when completed. | Exponential worst case; nondominated labels can grow exponentially; map operations add log-label overhead. |
| LTS-inspired | Independent recursive full-space enumeration and feasibility/objective evaluation; exact when completed. | Exponential; O(N K^N) upper bound and O(N) traversal memory. |
| Proposed DP | ceil(R/Δ), floor(budget/Δ), rolling value rows and byte-valued backtracking decisions. Exact for the conservative discretized problem. | O(N K B) time, O(B) value memory plus O(NB) parent memory, B=floor(budget/Δ). Pseudo-polynomial. |

NAVA inherits its graph/pulse pruning structure, replacing measured geometry PSNR with weighted measured RGB-reference PSNR and measured StreamSTGS costs. No NAVA Lagrange-multiplier solver or new heuristic bound is substituted. LTS originally selects layer–segment variables, with tile visibility, cached data, layer prerequisites and temporal priority. After adapting to this common independent-CRF problem, its transferable full-space search coincides conceptually with brute-force; timing differences reflect enumeration details. Author source code is unavailable, so this baseline is always named LTS-inspired. Detailed mapping and primary-paper links are in [NAVA_METHOD.md](../references/NAVA_METHOD.md) and [LTS_METHOD.md](../references/LTS_METHOD.md).

DP uses exact integer ceil/floor arithmetic, unreachable −∞ states and deterministic first-index ties. Backtracking rechecks the original budget. Conservative quantization cannot admit an over-budget solution, but can exclude original optima or every feasible original selection; the latter is explicitly discretized_infeasible. There is no general relative approximation bound from this discretization.

## Measurement and censoring

All solvers are independent C++17 implementations compiled together with -std=c++17 -O3 -Wall -Wextra -Wpedantic, using g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0. This run pins one thread to CPU 0 on Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz. No GPU work is invoked. Each case/job has 3 warmups and 30 measured calls, with seeded shuffled job order in each round. Individual calls use std::chrono::steady_clock. The timed interval includes validation, solver-specific graph/quantization setup, allocations, solving, and backtracking. Files, workload generation, scheduling, result validation/retention and output are outside. No per-case solver state is cached between repetitions.

Both exhaustive solvers skip before enumeration when K^N exceeds 1,000,000; see coverage for the skipped object counts. All solvers have 0.5s search deadlines (DP checks recurrence time, while its setup remains included in measured latency); NAVA also has 100,000 live-label cap and DP has 50,000,000 parent-cell cap. Deadline checks are periodic, so slight overruns are possible. A capped NAVA run is recorded as timeout with its detailed reason, never certified optimal. The first timeout/skip stops further calls for that case/job, including during warmup. Any partial measured series is excluded from completed-case latency summaries. Raw samples retain phase, duration, incumbent, status and reason.

Figure medians and P95 pool measured invocations over completed cases, with equal repetitions per case. P95 uses the nearest-rank definition; bars describe workload plus invocation variation, not confidence intervals or independent-sample statistical significance. Censored coverage must be read alongside latency, since easy completed cases alone can look fast. Maximum, per-case and per-budget summaries are also saved. Timer overhead is not subtracted; CPU frequency, other processes and thermal state are not controlled. Peak solver memory was not reliably isolated and is null, rather than reporting process RSS as solver memory.

## Results

![Processing time](../figures/processing_time_vs_n.png)

| Method | Δ (Mb/s) | N | Completed cases | Median ms | P95 ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|
| Brute-force | 0 | 2 | 12/12 | 0.0003 | 0.0003 | 0.0004 |
| Brute-force | 0 | 4 | 12/12 | 0.0039 | 0.0041 | 0.0175 |
| Brute-force | 0 | 8 | 12/12 | 1.6433 | 1.7578 | 1.9688 |
| Brute-force | 0 | 16 | 0/12 | — | — | — |
| Brute-force | 0 | 32 | 0/12 | — | — | — |
| Brute-force | 0 | 64 | 0/12 | — | — | — |
| NAVA | 0 | 2 | 12/12 | 0.0006 | 0.0010 | 0.0012 |
| NAVA | 0 | 4 | 12/12 | 0.0029 | 0.0047 | 0.0276 |
| NAVA | 0 | 8 | 12/12 | 0.0198 | 0.1511 | 0.1569 |
| NAVA | 0 | 16 | 12/12 | 1.1451 | 11.6267 | 13.9090 |
| NAVA | 0 | 32 | 7/12 | 9.6804 | 302.3979 | 331.1837 |
| NAVA | 0 | 64 | 5/12 | 0.3113 | 418.1292 | 430.9282 |
| LTS-inspired | 0 | 2 | 12/12 | 0.0003 | 0.0004 | 0.0005 |
| LTS-inspired | 0 | 4 | 12/12 | 0.0028 | 0.0030 | 0.0032 |
| LTS-inspired | 0 | 8 | 12/12 | 0.6653 | 0.7353 | 0.9135 |
| LTS-inspired | 0 | 16 | 0/12 | — | — | — |
| LTS-inspired | 0 | 32 | 0/12 | — | — | — |
| LTS-inspired | 0 | 64 | 0/12 | — | — | — |
| Proposed-DP | 0.25 | 2 | 12/12 | 0.0020 | 0.0043 | 0.0163 |
| Proposed-DP | 0.25 | 4 | 12/12 | 0.0093 | 0.0142 | 0.0270 |
| Proposed-DP | 0.25 | 8 | 12/12 | 0.0330 | 0.0556 | 0.1016 |
| Proposed-DP | 0.25 | 16 | 12/12 | 0.1519 | 0.2173 | 0.2631 |
| Proposed-DP | 0.25 | 32 | 12/12 | 0.6420 | 0.8847 | 1.1492 |
| Proposed-DP | 0.25 | 64 | 12/12 | 2.1283 | 3.5442 | 4.2493 |

At N=64, Proposed DP with Δ=0.25 Mb/s completes 12/12 cases: median 2.1283 ms, P95 3.5442 ms, maximum 4.2493 ms. Across 61 cases with certified optima, its mean gap is 0.0556% and maximum gap is 0.6022%. These known-gap cases are a subset, with large-N coverage favoring cases completed by NAVA. NAVA coverage is N=2: 12/12; N=4: 12/12; N=8: 12/12; N=16: 12/12; N=32: 7/12; N=64: 5/12. Survivor medians are not evidence that latency improves with more objects. NAVA censor reasons: 12 wall-clock deadlines and 0 live-label caps.

Brute-force supplies the independent small-N oracle; completed NAVA additionally certifies some larger cases. No gap is imputed where an exact oracle is unavailable. There are 60 complete NAVA case series and 61 certified cases. An exact result returned before a later timeout remains a valid oracle certificate; that censored case series is still excluded from latency aggregation. Exact-method comparisons at small N match utility and constraints; different tied selections are accepted.

| Method | Δ (Mb/s) | Cases with exact gap | Mean gap % | Max gap % | Cases checked against brute-force |
|---|---:|---:|---:|---:|---:|
| Brute-force | 0 | 36 | 0.0000 | 0.0000 | 36 |
| NAVA | 0 | 60 | 0.0000 | 0.0000 | 36 |
| LTS-inspired | 0 | 36 | 0.0000 | 0.0000 | 36 |
| Proposed-DP | 0.05 | 61 | 0.0100 | 0.1469 | 36 |
| Proposed-DP | 0.25 | 61 | 0.0556 | 0.6022 | 36 |
| Proposed-DP | 1 | 61 | 0.2489 | 4.8319 | 36 |

![DP trade-off](../figures/dp_discretization_tradeoff.png)

All completed exact methods attain the same optimum where an independent oracle exists. DP trades quantization conservatism for runtime; finer Δ increases the number of budget states and can improve the objective. Individual quantization grids are not nested for arbitrary Δ, so finer steps need not improve every individual selection monotonically. Latency comparisons characterize these implementations on this workload, rather than universal ordering of algorithms.

## Validation and reproducibility

Native correctness tests passed 350 randomized small instances and boundary cases with 13,873 assertions, using independent full-space and quantized oracles. Python metadata and harness tests validate source isolation, missing/invalid metadata, matrix identity and wire round-trips. An AddressSanitizer/UndefinedBehaviorSanitizer build also passed; LeakSanitizer was disabled because of the environment's ptrace restriction. Runtime validation checked 11,508 returned selections with **0 feasibility violations**. 61 of 72 cases have certified exact objectives. Logs and [validation.json](../results/validation.json) preserve evidence.

Reproduce with `python3 benchmark.py all` from this directory. See [README.md](../README.md) for configuration, independent stage commands, external metadata paths and artifact descriptions. Raw JSONL includes warmups and failure/censor samples; validated JSON and CSV attach input hashes, utilities, budget utilization and exact gaps. Machine details, compiler flags, source hashes, config and input SHA256 are in results/run_manifest.json.

## Limits on interpretation

Only four real source contents and five GOPs per content are available; synthetic scaling reuses rows. Visibility is synthetic and independent of actual pose, occlusion and PSNR view dependence. The experiment intentionally omits cache sharing and scene interactions. All views are averaged, and visibility weights do not make quality view-conditioned. Normalization means total utility remains comparable as N grows. The configured seeds provide workload examples, not a population study. All budgets scale with N and actual rates: DP state count B therefore also grows with N at fixed Δ. Fixed-Δ scalability can approach quadratic growth rather than linear scaling in N.

NAVA has exponential worst-case label growth and time/label censoring. Exhaustive and LTS-inspired large-N results are unavailable when the candidate cap is exceeded. A missing exact oracle at larger N means the DP gap is unknown, even if it returns a feasible solution. Reports do not claim DP dominates NAVA on every budget; overprovisioned instances can be especially easy for bounding. The benchmark establishes CPU solver latency and the observed feasibility/optimality trade-off; it does not establish end-to-end ABR performance or mobile-device latency.

# Isolated ABR solver processing-time benchmark

Runnable CPU experiment for one-epoch, object-level Multiple-Choice Knapsack optimization. All code, prepared inputs, configuration, tests, logs, raw results, figures and reports live here. No production ABR, `src/`, `src_final/`, playback, renderer, decoder, network runtime or GPU is imported or executed.

The completed run uses all 80 measured StreamSTGS quality profiles from Longdress, Soldier, Loot and RedAndBlack: five 60-frame GOPs, four CRFs each. It contains 72 decision instances, three visibility seeds, four feasible budget regimes, and N=2,4,8,16,32,64. Each solver receives identical prepared cost/utility arrays. All four methods are implemented independently in C++17.

Read [reports/REPORT.md](reports/REPORT.md) for methods and results, [reports/PAPER_TABLE.md](reports/PAPER_TABLE.md) for the compact latency table, and [reports/DATA_AUDIT.md](reports/DATA_AUDIT.md) for metadata provenance. Figures are exported as PNG, SVG and PDF.

## Reproduce

Requirements: Python 3.10+, a C++17 `g++`, Linux `taskset`/`lscpu`, and Matplotlib for figures. Python preparation/tests and the native solver have no third-party runtime dependencies. The measured bundle remains an external read-only input.

```bash
cd /home/fil/Hoang/MultiObj3dgsVstream/experiments/abr_solver_processing_time
python3 benchmark.py all
```

This rebuilds the binaries, verifies the actual metadata, regenerates identical cases, runs correctness tests, executes CPU measurements, validates every selection, and generates reports/figures. It overwrites generated artifacts in this experiment directory. The seed and arrays reproduce exactly; execution times naturally vary.

Run stages separately:

```bash
python3 benchmark.py prepare
python3 benchmark.py build
python3 benchmark.py test
python3 benchmark.py run
python3 benchmark.py report
```

`configs/default.json` controls object counts, seeds, repetitions, time/label/candidate caps, DP steps and CPU affinity. The current run uses three warmups and 30 measurements per completed case/solver, Δ=50,000/250,000/1,000,000 bit/s, and a 0.5s solve deadline. The first allowed CPU is pinned and recorded in `results/run_manifest.json`. The `run` stage expects prepared inputs and compiled binaries. Prepare again after changing workload settings; build again after editing C++.

Use another metadata location without changing code:

```bash
python3 benchmark.py all --data-root /path/to/streamstgs_8i_quality_metadata
python3 benchmark.py prepare --data-root /path/to/quality_bundle --cost-root /path/to/cost_metadata
```

Paths in config resolve relative to this experiment directory. `--data-root`/`--cost-root` are absolute after CLI resolution. Missing or inconsistent PSNR/cost metadata is an error; there is no invented fallback. Cost-root layouts are documented in `metadata.py`; relocated representations must match the quality-linked hashes and publication inventory. No media extraction is needed for the supplied bundle.

To configure budget fractions or a different η, call the standalone generator (the standard CLI uses the four documented regimes):

```python
from metadata import load_dataset
from workloads import generate_workloads, write_runner_input
profiles, audit = load_dataset('/path/to/metadata')
cases = generate_workloads(profiles, sizes=(2, 4, 8), seeds=(2027,),
                           budget_specs=(('tight', .08), ('balanced', .35),
                                         ('generous', .75), ('overprovisioned', 1.10)), eta=.9)
```

Custom calls must also write the prepared JSON wrapper with matching case IDs before using the report/validation CLI; `workloads.prepare(...)` is the standard complete writer. Effective solver budgets are exact integer bit/s; η×Ĉ is informationally reconstructed from them, avoiding floating rounding of the actual constraint.

## Solvers and guarantees

| Algorithm | File | Guarantee |
|---|---|---|
| Brute-force | `cpp/brute_force.cpp` | Evaluates every complete K^N assignment; original optimum when completed. |
| NAVA | `cpp/nava.cpp` | Layered-graph `pulse()` with dominance, suffix feasibility and upper-bound pruning; original optimum when completed. |
| LTS-inspired | `cpp/lts_inspired.cpp` | Independent recursive full-space enumeration; original optimum when completed. |
| Proposed DP | `cpp/proposed_dp.cpp` | O(NKB) conservative bandwidth DP; optimum of the discretized problem, feasible for the original problem. |

The interface in `cpp/common.hpp` is `Result solver(const Instance&, const Options&)`. Each object has four actual rates in CRF20/28/36/40 order and four utilities `weight * measured_psnr`. Result records status, selection, actual rate, utility and search counters. DP keeps two utility rows and a compact backtracking table, then rechecks actual bandwidth. Ties are deterministic within each solver; tests compare optimal utilities rather than requiring equal tied selections.

`optimal` for DP always refers to its discretized problem. `discretized_infeasible` means a feasible original instance was excluded by conservative quantization. `infeasible` means no original feasible selection. `timeout` includes either the deadline or NAVA label cap, with the reason retained; any incumbent is unproven. `skipped` records candidate/DP resource caps. BF and LTS-inspired skip N≥16 under the default one-million-candidate cap. Other jobs continue independently.

NAVA and LTS are adaptations of the cited primary papers, not complete system reproductions. NAVA keeps its pulse/pruning structure. LTS originally makes layer–segment decisions, with tiles affecting visibility and cost; independent object/CRF adaptation collapses its full-space search to the same conceptual search as exhaustive MCKP. No authors' unpublished implementation is claimed. See [references/NAVA_METHOD.md](references/NAVA_METHOD.md) and [references/LTS_METHOD.md](references/LTS_METHOD.md).

## Timing and reporting

`cpp/benchmark.cpp` loads all arrays before timing. Each high-resolution monotonic `steady_clock` interval contains exactly a solver call through returned selection, including its per-call validation, setup, allocations and backtracking. Workload preparation, file I/O, job ordering, logging and independent result checks are outside. Solver order is shuffled deterministically each round; no solver caches results between invocations.

A warmup timeout/skip stops that job and is still recorded. Any timeout after partial measurements excludes the entire case/job from completed-case latency summaries. The plots show completed-case median and nearest-rank P95; they never graph skip/preflight durations as solve latency. Coverage counts identify censored groups, and incomplete-coverage markers are unconnected. P95 includes workload and invocation variation, and is not a confidence interval. Raw warmups and failed attempts remain available. Individual solver peak memory is `null`: process RSS cannot reliably attribute memory to one call.

The exact oracle is brute-force on N≤8; completed NAVA searches certify additional larger cases. Exact values from warmup are valid certificates even if a later call times out, while that case/job remains excluded from latency aggregates. No gap is fabricated for cases without a certificate. Full `case_summary.csv` includes objective and utilization; raw JSON preserves every selection and actual rate.

## Artifacts

| Location | Contents |
|---|---|
| `cpp/` | Four solvers, common interface and native timing runner. |
| `metadata.py`, `workloads.py` | Standalone strict loader and seeded synthetic visibility/instance scaling. |
| `inputs/` | 80 normalized measured profiles, consumed-file hashes/audit, 72 prepared cases and exact text wire input. |
| `tests/` | Native independent exact/quantized oracles; Python metadata, matrix identity, source isolation and timing-boundary checks. Invented values occur only in test fixtures. |
| `configs/` | Checked-in reproducible settings. |
| `results/raw.jsonl` | Native per-call results, including warmups and timeout/skip records. |
| `results/raw.csv`, `validated_results.json` | Enriched measurements with matrix hash, exact gap, budget utilization and feasibility checks. |
| `results/case_summary.csv`, `summary.csv`, `by_budget.csv` | Per-case and pooled latency, completion counts and censoring. |
| `results/optimality_gap.csv` | Gaps against available certified optima; independent brute-force subset also shown. |
| `results/run_manifest.json`, `validation.json` | CPU, compiler, config, command/source/input hashes and runtime correctness evidence. |
| `figures/` | Processing time vs N with P95, per-budget timings, optimality gap, DP latency/discretization trade-off; PNG/PDF/SVG. |
| `reports/` | Method/results/limits, dataset audit and compact paper table. |
| `logs/` | Builds, native/Python tests, sanitizer validation and benchmark logs. |
| `references/` | Paper DOI/URLs, provenance and adaptation notes. Local downloaded PDF/text are ignored. |
| `build/` | Ignored CPU executables; rebuilt by the CLI. |

To rerun the sanitizer check:

```bash
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Icpp cpp/brute_force.cpp cpp/nava.cpp cpp/lts_inspired.cpp cpp/proposed_dp.cpp \
  tests/solver_tests.cpp -o build/solver_tests_sanitized
ASAN_OPTIONS=detect_leaks=0 build/solver_tests_sanitized
```

LeakSanitizer is disabled for this environment's ptrace restriction. This experiment does not modify production modules or infer solver processing time from FPS, playback or quality-pipeline runtime.

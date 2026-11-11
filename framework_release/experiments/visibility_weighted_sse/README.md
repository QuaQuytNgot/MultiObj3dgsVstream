# Visibility-Weighted SSE planning experiment

An isolated research planner using frozen camera geometry, cached alpha contributions,
offline Solo SSE, and the existing `src_final` native DP. It exports decisions without
a server, viewer, renderer, decoder, CUDA, or learned predictor.

From the repository root, with Python containing NumPy and an available `g++`:

```bash
python -m experiments.visibility_weighted_sse validate --config experiments/visibility_weighted_sse/configs/fixture.frozen.json
python -m experiments.visibility_weighted_sse run --config experiments/visibility_weighted_sse/configs/fixture.frozen.json --output /tmp/weighted_sse_fixture
python -m pytest experiments/visibility_weighted_sse/tests -q
```

On this workstation use `/home/fil/miniconda3/envs/Hoang/bin/python`; the default
Python does not contain the test dependencies. The fixture has two GOPs, three
instances, progressive Q0/E1/E2/E3 dependency chains, and four budgets, including
an intentionally infeasible base budget. Its contributions and SSE are synthetic.

For the previously measured independent cache available in this checkout:

```bash
python -m experiments.visibility_weighted_sse.real_cached --source experiments/visibility_guided_sse/runs/pilot_v1 --output /tmp/weighted_sse_inputs
python -m experiments.visibility_weighted_sse run --config /tmp/weighted_sse_inputs/config.json --output /tmp/weighted_sse_cached
```

The bounded converter handles the frozen `recorded_source_control` GOP only. It
verifies metadata, independent contribution-only NPZ payloads, and actual encoded
component hashes/sizes, retaining the original offline SSE convention. It reads
no RGB images or joint-quality evaluation scores. The included converted inputs
at `configs/cached_recorded_source_control/config.json` can also be run directly.
Their provenance identifies the original inputs; running the converted snapshot
does not require opening those source payloads again.

To use another trace or scene, supply the three input documents in the versioned
schemas, declare the same compatibility metadata, and freeze a configuration:

```bash
python -m experiments.visibility_weighted_sse freeze --config my_config.json --output my_config.frozen.json
python -m experiments.visibility_weighted_sse run --config my_config.frozen.json --output /tmp/my_decisions
python -m experiments.visibility_weighted_sse benchmark --config experiments/visibility_weighted_sse/configs/fixture.frozen.json --output /tmp/weighted_sse_scaling.json
```

`--help` lists the commands. `status` is an alias of `validate`; neither command
creates visibility. Missing or invalid inputs produce a structured error and
exit code 2. A finished planning command exports each solver status, including
explicit infeasibility or resource-limit failures.

The trace uses absolute timestamps in half-open GOPs, requires a sample exactly
at each GOP start, and supplies positive-Z calibrated world cameras plus explicit
object-to-world similarity transforms and content-state identities. The provider
returns every frozen sample under `ORACLE_CAMERA_TRACE`. Temporal weights hold
each sample until the next sample or GOP end. Integer GOP identities are
canonicalized to strings when joined to profiles and configuration budgets.

Both profile and visibility metadata must declare resolution, coordinate and
transform conventions, continuous alpha-weighted pixel semantics, proxy
representation identity, and temporal aggregation. Profiles additionally declare
the offline distribution ID and camera/time sample IDs with normalized weights;
one shared distribution describes their SSE and mean solo contribution. Optional
additional compatibility fields are checked too. Q0 is the first quality in
each profile. One run uses a single representation family. Physical resources
use real SHA256 identities and byte counts; fixtures explicitly use synthetic
identities. Progressive qualities declare dependencies; standalone qualities
contain their full physical resource closure. See the fixture JSON for examples.

The only utility is `U(q) = (GOP_visible / offline_mean_solo) * (SSE(Q0)-SSE(q))`.
Weights and signed measured gains remain unchanged. Zero visibility retains Q0.
Hash-shared and declared resident resources are charged once or excluded,
respectively. The cache is a fixed resident snapshot for each decision; delivery
and cache evolution are outside this experiment. The existing native solver
accepts at most four qualities per object. State and shared-activation bounds
produce an explicit incomplete status without a selected vector. Timeouts are
disabled to preserve deterministic statuses.

`decisions.json` contains complete identity, visibility time series, utilities,
and resource-validated allocations. `decisions.csv` has a lossless GOP/budget
record; `allocations.csv` has object selections and byte diagnostics;
`visibility.csv` has object time series; `utilities.csv` has object/quality rows.
These artifacts are byte-identical on repeated frozen runs. Measured timings
are separated in `performance.json` and referenced by each decision; timing
values naturally vary. The scaling artifact labels synthetic CPU timings and
excludes native compilation/warmup. All quality evaluation metrics are marked
unavailable. Cached-input timing does not measure new visibility acquisition.

Architecture, formulas, verification, measured timings, data limitations, and
reproduction details are in
[`VISIBILITY_WEIGHTED_SSE_IMPLEMENTATION.md`](../../reports/VISIBILITY_WEIGHTED_SSE_IMPLEMENTATION.md).

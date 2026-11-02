# Adaptation baselines for paper experiments

One frozen-input interface for `NAVA`, `VISIBILITY_PSNR`, `VISIBILITY_MSE`,
`SOLO_SSE`, and `VISIBILITY_WEIGHTED_SSE`. This module exports adaptation
decisions and runtime diagnostics on CPU, using the existing trace, visibility,
profile, physical-resource and native solver components.

Run from the repository root with Python containing NumPy and an available
`g++`; tests additionally need PyTest. On this workstation use
`/home/fil/miniconda3/envs/Hoang/bin/python`.

```bash
python -m experiments.adaptation_baselines methods
python -m experiments.adaptation_baselines validate --config experiments/adaptation_baselines/configs/fixture.frozen.json
python -m experiments.adaptation_baselines run --config experiments/adaptation_baselines/configs/fixture.frozen.json --output /tmp/baselines_fixture
python -m pytest experiments/adaptation_baselines/tests -q
```

The synthetic fixture reuses the proposed method's exact profiles, trace and
visibility documents. It adds synthetic PSNR/MSE/AABB metadata, two GOPs and
four budgets, including a deliberately infeasible mandatory-base budget. These
fixtures are mathematical tests, not empirical paper results.

For the actual audited recorded-source cache:

```bash
python -m experiments.adaptation_baselines.real_cached --source experiments/visibility_guided_sse/runs/pilot_v1 --output /tmp/baselines_inputs
python -m experiments.adaptation_baselines run --config /tmp/baselines_inputs/config.json --output /tmp/baselines_cached
```

The converter calls the proposed method's converter and adds audited scalar
PSNR/MSE and full-GOP AABB metadata. It verifies 279 source files without reading
RGB payloads or joint-quality scores, invoking rendering or decoding. The
included snapshot at `configs/cached_recorded_source_control/config.json` can
also be run directly.

For another frozen scene, provide the same three proposed-method input schemas
plus a `baseline_metadata` sidecar when selecting NAVA, PSNR or MSE. The sidecar
binds its profiles to the physical/SSE profile file SHA256, representation IDs,
content revision and explicit quality references. `configs/fixture.json` and
`fixtures/baseline_metadata.json` show the configuration and metadata schemas.
`freeze` binds the supplied input files; it does not acquire missing data.

```bash
python -m experiments.adaptation_baselines freeze --config my_config.json --output my_config.frozen.json
python -m experiments.adaptation_baselines run --config my_config.frozen.json --output /tmp/my_baselines
```

`methods` selects any nonempty subset of the five IDs. `gop_ids`, byte `budgets`
(global or `per_gop`), declared resident `cached_resource_ids`, and native
`solver_limits` are configurable. Object count comes from the compatible frozen
scene's instances; `expected_object_count` checks that count. To change a scene's
object count, provide matching profiles, trace, contributions and sidecar;
removing an occluder from an existing cached scene requires a compatible cache.
No count is hardcoded. Synthetic N=1 and N=5 scenes are tested.

`quality_levels` optionally maps every instance to eligible quality IDs, e.g.
`{"instance-0":["Q0","Q2"],"instance-1":["Q0"],"instance-2":["Q0","Q3"]}`.
Every set includes Q0. Excluding Q1 as a candidate still charges its physical
dependency when Q2 needs it. The unchanged native solver supports one to four
quality candidates per object. Dependencies, shared physical SHA256 resources
and declared resident data are accounted consistently for every method.

The shared experiment allocator now defaults to the production solver's
0.25 Mbps grid. Configure the existing `solver_limits` object with
`{"delta_bps":250000,"discretization":"bitrate","gop_duration_seconds":2.0}`;
the streaming runtime uses its actual GOP duration. Mandatory resources,
optional shared-resource activation costs and row minimum costs are removed
exactly, then the unchanged native solver applies `ceil(rate/delta)` to each
remaining choice and `floor(budget/delta)` to the remaining budget. The final
physical SHA256 union is checked against the original byte budget. This
certifies the conservative grid optimum, which may differ from the byte optimum.
For a bounded exact reference use `{"discretization":"exact_bytes"}` to retain
the earlier GCD reduction. `max_dp_cells` remains an explicit memory bound.
Diagnostics report allocated and requested native DP cells, visited transitions,
table memory from the native layout and exact-byte feasibility. Timing and
measured executable peak RSS (including native allocations) live in the
separate performance artifact; the peak is not an incremental solver-only RSS.
Linux uses `/proc/self/status` `VmHWM` so a fresh benchmark worker does not
inherit a launcher's pre-exec high-water mark. A named `getrusage` fallback is
used if that measurement is unavailable, with its process lifetime scope.

The direct Python interface is
`AdaptationPlanner(profiles, trace_provider, visibility_adapter, baseline_metadata=...).plan(method, gop_id, budget_bytes)`.
Use `runner.load_experiment(config_path)["planner"]` to obtain a fully validated
instance, or `metadata.load_metadata(...)` for a programmatic sidecar. Call
`initialize_solver()` before timing method comparisons; `run` does this once
and records native bridge initialization separately. All methods return selected
qualities, selected per-object utilities, required physical bytes, solver status,
candidate utilities and timings. No evaluator callback is accepted.

NAVA retains audited `paper_discard` AABB visibility and the original-PLY
image-PSNR adaptation. Visibility × PSNR and Visibility × MSE both use normalized
alpha contribution per timestamp. Their visibility source and unchanged quality
reference conventions are exported explicitly. Solo and Proposed delegate signed SSE gains
to the existing proposed builder; Proposed also calls its existing guarded
allocation entry point. All use the same physical-resource adapter and native solver, which is an
explicit solver adaptation from historical NAVA graph DP / MSE sparse DP.

`AABB_MSE_HISTORICAL` separately preserves the old `AVERAGE_MSE` AABB weighting,
with the same MSE quality metric. It is accepted explicitly by the common API
and configuration, outside the five primary `METHODS`. Existing saved
`VISIBILITY_MSE` outputs produced before the correction use this historical
weighting; select the explicit historical ID to reproduce it.

`decisions.json`, `decisions.csv`, `selections.csv` and `utilities.csv` are
deterministic for frozen inputs. `performance.json` separately records input,
shared geometry/visibility preparation, native initialization, utility and
solver timings. Utility units differ between methods and are not measured
viewport quality. Rendering quality is explicitly unavailable.

See [the implementation report](../../reports/ADAPTATION_BASELINES_IMPLEMENTATION.md)
for formulas, fidelity differences, test results, actual-cache decisions and
limitations.

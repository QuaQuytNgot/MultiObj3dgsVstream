# Adaptation baselines implementation

**Status: implemented and verified for paper-experiment planning, 2026-10-31.**
All five methods are available through `experiments/adaptation_baselines/`.
They share frozen scene/trace inputs, representations, eligible quality sets,
mandatory base delivery, physical byte budgets and the existing native solver.
Actual image-quality evaluation, network simulation and playback are outside
this implementation.

## Recovered methods and fidelity

Let `tau_t` be normalized GOP timestamp weights, `P_it` the projected AABB area
from the historical `paper_discard` rule, and `A_it` the cached post-occlusion
alpha contribution. Define `I_it=P_it/sum_j P_jt` and
`J_it=A_it/sum_j A_jt`, with all-zero weights when the corresponding sum is zero.
`Q0` is the profile's declared mandatory lowest quality.

| Method ID | Utility kept intact | Reused implementation / convention |
| --- | --- | --- |
| `NAVA` | `sum_t tau_t I_it PSNR_i(q)` | `nava_mse.geometry.aabb_area(..., method="paper_discard")`, `policies.projected_aabb_importance`, audited original-PLY image-PSNR metadata. Arithmetic exactly matches historical NAVA utility construction in tests. |
| `VISIBILITY_PSNR` | `sum_t tau_t J_it PSNR_i(q)` | Normalized Visibility Analyzer contribution × finite original-reference metadata PSNR, as in `src_final/client/abr/planner.py` and earlier ABR comparisons. Normalize each timestamp before GOP integration. |
| `VISIBILITY_MSE` | `sum_t tau_t J_it [MSE_i(Q0)-MSE_i(q)]` | Same normalized alpha-aware contribution as `VISIBILITY_PSNR`; retains fixed-reference pooled MSE reduction, including signed gains. |
| `SOLO_SSE` | `Dbar_i(Q0)-Dbar_i(q)` | Same signed SSE gains as `sse_profile_comparison/policies.py:sse_gains` and visibility-guided P0; delegates to the existing proposed builder at unit weight. |
| `VISIBILITY_WEIGHTED_SSE` | `[sum_t tau_t A_it / Abar_solo_i] [Dbar_i(Q0)-Dbar_i(q)]` | Calls existing `visibility_weighted_sse.profiles.build_utilities` and its existing formula-guarded `allocation.allocate`. No utility or trace-processing rewrite. |

The five primary IDs remain unchanged. Both Visibility × quality methods now
normalize the **same** Visibility Analyzer `A_visible` across objects at each
timestamp, then apply the same normalized GOP temporal integration. All-zero
samples contribute zero; GOP-averaged alpha is not normalized afterward.
Neither alpha baseline requires AABB bounds.

Historical `nava_mse.policies.AVERAGE_MSE` is separately accessible as
`AABB_MSE_HISTORICAL`: `sum_t tau_t I_it [MSE_i(Q0)-MSE_i(q)]`, preserving
`paper_discard` geometry and exact historical utility arithmetic. It is an
explicit reproducibility option, outside the five default primary methods.
`METHODS` remains the five primary IDs; `SUPPORTED_METHODS` additionally accepts
the historical ID in the Python API and both configuration-driven CLIs.
NAVA and proposed Visibility-Weighted SSE were not changed by this correction.

The [NAVA root-cause audit](NAVA_MSE_ROOT_CAUSE_AUDIT.md) found zero projected
importance in 15/24 visibly rendered camera/object pairs because `paper_discard`
drops triangles containing any exterior vertex. This is retained and explicitly
diagnosed as `historical_aabb_zero_with_visible_alpha`. The clipped-box
counterfactual is not substituted. NAVA therefore remains explicitly adapted:
its original geometric PSNR is replaced by the existing finite
`original_ply_synthetic_rgb` image-PSNR proxy, as documented in
[`NAVA_METHOD.md`](../experiments/nava_mse/references/NAVA_METHOD.md).

PSNR is the existing arithmetic mean in dB over the declared original-PLY source
frames/views, including source training and held-out views. It is neither
geometric PSNR nor photographic truth. Infinite CRF20 self-reference PSNR is
rejected rather than capped. MSE is pooled `sum(SSE)/(3*sum(foreground pixels))`
against each object's fixed CRF20 reference-alpha mask `>0.01`, with clipped
`[0,1]` RGB. SSE uses the same compression-relative reference but retains its
camera-mean, timestamp-summed distortion convention. Their quality references,
normalizations and units are exported separately. Signed measured gains remain
signed. No extra area, Gaussian-count or visibility multiplier is introduced.

The source-control smoke reuses the proposed method's existing offline sampling
population, which includes the recorded control camera. It is not a new
leave-one-camera-out study or evidence of unseen-view generalization. The
earlier SSE profile-comparison LOCO protocol remains a separate experiment.

## Architecture, reuse and changes

The common API is `AdaptationPlanner.plan(method, gop_id, budget_bytes)` in
[`interface.py`](../experiments/adaptation_baselines/interface.py). It returns
selected quality and selected utility per object, total required bytes, full
candidate utilities, physical dependency/cost diagnostics, status and timing.
`runner.load_experiment(config_path)["planner"]` supplies a validated instance.

Data flow: frozen configuration and hashes → existing trace provider and cached
visibility adapter → common GOP preparation → method utility adapter → existing
physical-resource allocator/native DP → budget verification → deterministic
decision exports and separate runtime evidence.

New files under `experiments/adaptation_baselines/`:

- `methods.py`: five primary objectives and explicit historical AABB-MSE, reusing existing normalization/AABB/SSE helpers.
- `metadata.py`: baseline reference, representation, revision, distribution and profile-SHA binding.
- `interface.py`, `runner.py`, `__main__.py`: common API, configuration, validation, CLI and exporters.
- `real_cached.py`: bounded, audit-checked scalar metadata importer calling the proposed converter once.
- `tests/`: objective recovery, shared solver, integration and actual-cache validation.
- `configs/`, `fixtures/`, `runs/`, `README.md`: frozen examples, synthetic metadata, verified outputs and instructions.

Two minimal compatibility changes were made in the isolated proposed module:

1. `visibility_weighted_sse/allocation.py` exposes its existing physical allocator
   as `allocate_utilities`. Its original `allocate` remains the guarded proposed
   entry point, with the same formula checks and zero-visibility base behavior.
   Both entry points accept an optional eligible-quality map. Full dependency
   metadata stays available even when an intermediate level is not a candidate.
2. `visibility_weighted_sse/runner.py` adds optional schema/input-name parameters
   to existing frozen-input helpers. Defaults preserve its previous CLI behavior.

No `src_final/` code, historical NAVA/MSE implementation, renderer, network
transport or publication format was changed. The existing calibrated Camera,
object/source-local transformations, complete trace validation, cached analyzer
outputs, profile accessors, resource hashes, optional MPD checking and native
`ProposedDPSolver` are reused.

All methods now use that **same physical-byte optimizer**. This deliberately
replaces historical NAVA graph DP and MSE/Solo rational sparse DP in the common
comparison. Their quality metrics are preserved; primary MSE now shares PSNR
alpha weighting as specified above. Historical optimizer runtime,
minimum-byte/lexicographic tie handling and rational near-tie behavior are not
claimed as reproduced. The common native DP uses doubles and strict candidate
index ties. This solver adaptation isolates objective differences under the
same representation/budget constraints; it is not a new adaptation algorithm.

## Configuration and comparison contract

`methods` selects any nonempty subset of the five primary IDs, with explicit
`AABB_MSE_HISTORICAL` also accepted for reproducibility. Profiles, trace and
contributions use the existing proposed schemas. NAVA/PSNR/MSE additionally need
a baseline metadata sidecar bound to the exact profile file SHA256, content
revision, object/instance/GOP, representation identities and quality references.
MSE's offline distribution ID must match the SSE population; their aggregations
remain explicitly different. Solo/Proposed can run without that sidecar.

Object count is unrestricted by method selection and comes from compatible
frozen scene instances; `expected_object_count` checks the intended count.
Changing a scene requires matching trace, quality and contribution artifacts.
Removing an occluder while retaining its old multi-object visibility cache is
not a valid smaller-scene experiment. Complete synthetic N=1/N=5 scenes are
tested, as well as the provided N=3 two-GOP fixture.

`quality_levels` optionally maps each instance to a nonempty subset of declared
quality IDs, always including Q0. The native implementation supports 1–4 levels
per object. For example, selecting candidates Q0/Q2 still charges Q1 if Q2
depends on it. Byte budgets can be global or per GOP. All methods use the same
full selected standalone/progressive resource closures, transitive dependencies,
SHA256 sharing and declared resident-cache snapshot. There is no SKIP state.
Insufficient base budget returns an explicit infeasible result; cell/sharing
limits return incomplete without an uncertified selection.

Trace prediction remains `ORACLE_CAMERA_TRACE`: all actual upcoming frozen
geometry is supplied at GOP start, with no trained predictor. Timestamp-derived
left-hold integration covers the complete GOP. No joint result, future quality
target, renderer or evaluation callback enters the policy API.

Exports are `decisions.json`, `decisions.csv`, `selections.csv` and
`utilities.csv`. Runtime measurements are separate in `performance.json`,
referenced by each decision. Candidate/selected utility units are method-specific
and must not be compared as measured viewport quality. Actual quality evaluation
is explicitly unavailable.

## Focused correction validation and compatibility

**394 tests and 22 subtests passed:** 112 baseline, 186 streaming, 84 proposed,
and 12 existing network tests. Only small synthetic correctness cases and
read-only cached-metadata checks ran. No experiment, benchmark, rendering,
decoding or image-quality measurement was executed for this correction.

New regressions check identical alpha weights for PSNR/MSE under changing
visibility, partial occlusion and invisible intervals; unchanged raw PSNR and
signed MSE gains; MSE-only loading without AABBs; exact historical AABB-MSE
utilities and explicit selection through both interfaces. Existing NAVA
false-zero and direct proposed-allocation equality tests remain passing.
Streaming regressions cover common Q0 delivery for all five methods, persistent
stalls/loops, estimate recovery, partial service, shared base hashes, standalone
and progressive costs, candidate masks and original solver infeasibility.

Existing `VISIBILITY_MSE` configurations now select alpha-weighted MSE.
To reproduce pre-correction MSE weighting, select `AABB_MSE_HISTORICAL` instead.
Previously saved outputs retain their original IDs and bytes; their old
`VISIBILITY_MSE` records describe the historical AABB baseline, **not** the
corrected primary method. They were not regenerated. Inspect exported
`method_diagnostics.visibility_source` when interpreting older artifacts.

The adaptation-only planner still returns no selection for `infeasible_base`.
The streaming layer now supplies actual Q0 delivery separately while preserving
that infeasible allocation; see the streaming implementation report.

## Archived Task-1 validation and example results (before correction)

The following records are retained historical evidence, not measurements or
validation rerun for this correction. In these records, old `VISIBILITY_MSE`
means `AABB_MSE_HISTORICAL` under the current interface.


| Check | Result |
| --- | --- |
| New baseline tests | 106 passed |
| Existing proposed tests | All 84 passed; direct allocation equality checked in 8 fixture conditions |
| Combined baseline + proposed command | 190 passed in 5.50 s |
| Wider NAVA/SSE/visibility-related regression | 609 passed, 328 subtests passed in 28.85 s |
| Existing `tests_final` suite | 122 passed in 6.40 s, using host CPU fixtures and permitted localhost sockets |
| Actual audited-cache smoke | 15/15 decisions optimal and physically feasible |
| Repeat fixture and actual-cache runs | Byte-identical JSON and all three decision CSV files |

Tests cover historical utility reconstruction, per-timestamp normalization,
the NAVA false-zero case, signed gains, unbounded proposed weights, missing and
incompatible metadata, direct-API quality-reference rejection, full trace and
GOP ordering, variable N/quality subsets, progressive/shared/resident bytes,
mandatory-base infeasibility, resource limits, exhaustive small-case solver
agreement, deterministic exports and CLI execution with GPU/renderer/network
imports forbidden. A native bridge initialization test verifies initialization
runs once before all compared method calls and is timed separately.

The actual converter verifies **279 source files**, including the preserved NAVA
audit inventory, 384 scalar samples reconstructing pooled AverageMSE, original-PLY
PSNR metadata, full-GOP CRF20 bounds over all 60 presentation frames, and the
proposed converter's trace/contribution/physical inputs. No RGB payload or joint
evaluation result is read, and no new rendering or decoding is invoked.

At a common **20,000,000-byte** budget, the cached recorded-source control gives:

| Method | longdress | loot | soldier | Required bytes |
| --- | --- | --- | --- | ---: |
| NAVA | CRF40 | CRF40 | CRF20 | 18,758,016 |
| Visibility × PSNR | CRF28 | CRF20 | CRF36 | 16,359,202 |
| Historical AABB × MSE | CRF40 | CRF40 | CRF20 | 18,758,016 |
| Solo SSE | CRF20 | CRF28 | CRF36 | 17,600,085 |
| Proposed | CRF20 | CRF28 | CRF36 | 17,600,085 |

The same smoke also covers 10 MB and 30 MB byte budgets. All use the same
7,234,456-byte base resource union. Identical physical SHA resources are charged
once, consistent with `src_final`; this differs from older independent-package
cost totals. No quality-superiority conclusion follows from these selections.
The synthetic two-GOP fixture produces 40 records: 30 optimal and 10 explicitly
base-infeasible. Synthetic values are not empirical measurements.

Full evidence is under
[`runs/cached_recorded_source_control/`](../experiments/adaptation_baselines/runs/cached_recorded_source_control/)
and [`runs/fixture/`](../experiments/adaptation_baselines/runs/fixture/).
Each includes `determinism.json` with repeat-run artifact hashes.

One observed cached-input CPU replay at the 20 MB budget, milliseconds:

| Method | Utility construction | Allocation adapter | Native solve within adapter |
| --- | ---: | ---: | ---: |
| NAVA | 6.6802 | 260.0728 | 259.7400 |
| Visibility × PSNR | 0.0872 | 273.7705 | 273.4569 |
| Historical AABB × MSE | 6.9188 | 265.4764 | 265.1641 |
| Solo SSE | 0.0442 | 264.5057 | 264.0485 |
| Proposed | 0.0445 | 263.9327 | 263.5490 |

Common input initialization was 9.3357 ms, native bridge initialization
0.2718 ms, and cached GOP aggregation/preparation 0.0373 ms. The process had
already run the fixture and the native library was cached. Bridge build/loading
is separated from compared method calls. Total-with-shared-inputs is a shared
pipeline cost, including visibility preparation even for Solo; utility and
allocation stages are the method-specific runtime evidence. This replay does
not measure acquisition of new visibility or establish real-time scalability.
Native solve cost depends on the exact reduced byte budget, not only N.

## Execution

From `MultiObj3dgsVstream/`, choose the environment containing the CPU dependencies:

```bash
PYTHON=/home/fil/miniconda3/envs/Hoang/bin/python

$PYTHON -m experiments.adaptation_baselines methods
$PYTHON -m experiments.adaptation_baselines run --config experiments/adaptation_baselines/configs/fixture.frozen.json --output /tmp/baselines_fixture

$PYTHON -m experiments.adaptation_baselines.real_cached --source experiments/visibility_guided_sse/runs/pilot_v1 --output /tmp/baselines_inputs
$PYTHON -m experiments.adaptation_baselines run --config /tmp/baselines_inputs/config.json --output /tmp/baselines_cached

$PYTHON -m pytest experiments/adaptation_baselines/tests experiments/visibility_weighted_sse/tests -q
$PYTHON -m pytest experiments/adaptation_baselines/tests experiments/visibility_weighted_sse/tests experiments/nava_mse/tests experiments/sse_profile_comparison/tests experiments/visibility_guided_sse/tests -q
$PYTHON -m pytest tests_final -q
```

`validate --config ...` checks readiness without solving. For custom inputs,
copy `configs/fixture.json`, select methods/qualities/budgets and compatible
scene files, then use `freeze --config my_config.json --output my_config.frozen.json`.
The final regression command needs local socket permission for its existing
HTTP fixture tests; the new planning pipeline needs no server or network.

The module is comparison-ready within its declared input conventions and native
solver limits. Broader frozen scenes require compatible cached inputs. Actual
viewport quality, generalization, network behavior and streaming performance
remain for the next experimental stage.

# One-interval bandwidth estimation and quality allocation

This implementation is the bounded four-quality prototype agreed on 16 October
2026. It allocates exactly one target Q0–Q3 to every object using floating-point
utility and exact remaining request bytes. It does not schedule requests or
guarantee playback deadlines. The broader research proposals in
[the system design](MULTI_OBJECT_3DGS_SYSTEM_DESIGN.md) are future work where they
differ from this prototype.

## Bandwidth and transfer accounting

`BandwidthEstimator` consumes the existing immutable `BandwidthSample` from
`src/client/request_api.py`; it does not depend on curl or a transport backend.
`accept_transfer_sample()`, `estimate_bandwidth()`, `get_sample_history()`, and
`reset()` expose sample acceptance, estimates, immutable history, and network
context reset. The default window contains five valid observations. With `m`
fresh valid samples:

\[
\widehat R=\frac{m}{\sum_{i=1}^{m}1/R_i}.
\]

Use `body_goodput_bps` consistently. Reject unsuccessful/cancelled transfers,
empty payloads, missing observations, and nonfinite or nonpositive durations or
rates. Rejected samples do not evict valid history. Warmup uses the valid samples
already available. The default maximum age is 10 seconds, measured from local
monotonic receipt; it is provisional and requires trace-based sensitivity
evaluation. Unix sample timestamps are not elapsed-time clocks. With no fresh
samples, return an explicitly configured bootstrap prior or unavailable, with
stale-history/source status visible to the caller. Reset on a known media-origin
or network-path change. Do not discard a sample merely for having a low rate.

The application installs one completion observer with
`NetworkClient.subscribe()`, accounts for the result, filters contextual
eligibility, and passes `to_bandwidth_sample(result)` to the estimator. Tag
requests through the existing `RequestSpec.metadata`. Base and enhancement
traffic are eligible; metadata/profile and explicitly unrepresentative transfers
are excluded from estimation. Their actual bytes and elapsed time are still
accounted. Do not also ingest values returned by `fetch()`. Equal samples cannot
be safely deduplicated without request identity. Retain the unsubscribe callback
for application shutdown. Initial integration requires concurrency=1; individual
overlapping transfer rates are not aggregate link capacity.

```python
from src.client.bandwidth import BandwidthEstimator, TransferObserver

estimator = BandwidthEstimator(window_size=5, max_sample_age_s=10.0,
                               startup_bandwidth_bps=None)
observer = TransferObserver(estimator, concurrency=1)
# After constructing an existing NetworkClient:
# unsubscribe = client.subscribe(observer)
# Tag RequestSpec.metadata with traffic_kind="base" or "enhancement".
# Set bandwidth_eligible=False for explicitly unrepresentative transfers.
```

`TransferObserver.snapshot()` returns actual byte records for metadata, profiles,
Base, enhancements, and other traffic. Those categories are disjoint. Its
`failed_bytes` is an overlapping diagnostic subtotal and must not be added to
`total_bytes` again. Planned remaining bytes belong in the plan result, outside
this actual ledger. All received HTTP body bytes
count, whether or not eligible for estimation. Container headers are included;
HTTP/TLS overhead is not. Application goodput (received bits per transfer second)
is distinct from content bitrate (encoded bits per media second).

## Base-first orchestration and planning contracts

Future decoded Q0 is an acquired prerequisite. External application orchestration:

1. Loads MPD/index metadata and quality information.
2. Resolves the required Base closure for the upcoming interval's objects/frames.
3. Downloads and decodes those states using existing transport and a future state
   adapter, retaining verified compatible Q0 states.
4. Runs fixed-Q0 visibility for supplied GT camera samples; every occluder also
   uses decoded Q0.
5. Freezes availability and supplies the remaining transfer time after this work.
6. Calls the Planner to allocate enhancement qualities.

Missing or failed Base decode/visibility blocks this policy. It does not imply
free future data or permit another visibility representation. The Planner itself
does not fetch, parse, decode, render, predict, or implement orchestration.
Standalone utility/cost/DP entrypoints remain independently testable.

`collect_inputs()`, `calculate_utility()`, `calculate_cost()`,
`discretize_cost()`, `solve_dp()`, `backtrack()`, and `plan()` keep these
responsibilities separate. Contracts use immutable snapshots:

| Input | Required information |
|---|---|
| Content index | Content revision, stable object ordering, Q0–Q3 targets, covered media frames/timestamps, payload dependency graph, request units, and profile location. |
| Request unit | Namespaced identity, exact complete bytes, request URL/range, and expected checksum. |
| Payload | Object, frame/time, epoch, layer, containing request unit, codec/version, state hash, and compatible parent hashes. |
| Availability | Verified complete cached request units and explicitly retained compatible decoded states; no inferred scalar `current_quality`. |
| View sample | GT camera, timestamp, object transforms, output resolution, fixed-Q0 `VisibilityResult`, profile coordinates, and optional temporal weight. |
| Density provider | Supported normalization/reference metadata, calibrated coordinate lookup, finite nonnegative density or explicit unavailability. |

Density capability metadata must declare `decoded_Q0` visibility,
`decoded_Q3` RGB reference, `decoded_Q0_solo_alpha_area` normalization, and
`full_image` support. Content/decode provenance, renderer/RGB/background
conventions, coordinate calibration, dimensions, and area threshold must be
explicit. Q3-area normalization is a future ablation, not a compatible initial
provider. Missing density is not silently interpreted as legacy MSE or zero gain.

`UtilityTable` and `CostTable` keep numerical `U[N][4]`, integer-byte `C[N][4]`, validity
masks, and invalid-option reasons aligned with stable object order. A successful
`PlanResult` returns the selected quality vector, total utility, exact required
bytes, budget, missing request-unit IDs, and snapshot identity. Failure statuses
distinguish unavailable inputs/bandwidth, invalid options, true byte
infeasibility, and infeasibility introduced by the discretization grid. A failed
allocation never returns a partial quality vector or invents a skip quality.

Use `PlanningContext(index, availability, frames_by_object, view_samples,
density_provider, transfer_budget_seconds)` as the input boundary.
`SegmentViewSample` snapshots an existing visibility result without requiring
renderer imports. `ObjectViewContext` carries the frame, calibrated
`ProfileCoordinates`, transform, and Q0 state hash. Application adapters must
record matched rendering provenance; supplying a calibrated name alone does not
validate an unmeasured profile. The provider returns `DensityValue` with both
the numerical density and its Q0 solo support. `density_from_mse()` is an
explicit scalar conversion helper after provenance validation; it does not
validate or enrich legacy profile files.

The [synthetic integration fixtures](../tests/test_adaptation_integration.py)
show complete index, retained Base state, view/provenance, and density-provider
construction, including a call to `plan()` and separate actual-traffic accounting.

MPD/index adapters and actual decoder availability adapters are separate work.
The existing MPD `bandwidth` is descriptive cumulative rate metadata; it must not
be treated as an enhancement layer's incremental request cost.

## Budget, utility, and exact remaining cost

For raw estimated goodput in bits/s and caller-supplied remaining transfer time
`T` in seconds:

\[
B=\left\lfloor\eta\widehat R_{\mathrm{bps}}T/8\right\rfloor,
\qquad\eta=0.85\text{ by default}.
\]

Apply the safety factor once. Recompute `T` after startup/Base transfer, decode,
and visibility work; it is not automatically the media segment duration. Do not
subtract completed Base/startup traffic from this new post-Base budget again.
Record those bytes as actual traffic. Body timing excludes setup/TTFB under the
handler's documented convention and excludes decoding. Byte feasibility does
not establish deadline feasibility: request latency, processing, contention,
and forecast errors remain outside DP. Report actual deadline outcomes only
once delivery integration exists.

Budget flooring uses the decimal representations of the supplied parameters,
so the default `0.85 * 8000 bps * 1 second / 8` gives exactly 850 bytes. This
avoids losing a byte by expanding the safety factor's binary float fraction.

The unchanged utility is:

\[
U_{n,k,s}=\sum_j\omega_j\frac{A^{vis}_{n,s,j}}P
\left[d_{n,0}(z_{n,s,j})-d_{n,k}(z_{n,s,j})\right].
\]

Default to uniform normalized temporal weights across the whole interval,
including occluded samples. Preserve floating-point signed gains and
`U[n][0]=0`. Use fixed-Q0 `A_visible` exactly once. It already includes projected
area and inter-object suppression, so do not multiply by another distance,
occlusion, or projected-area factor. Profile view/scale/time are lookup
coordinates, not additional utility weights. An exactly zero contribution needs
no density lookup; a tiny positive contribution still does. `out_of_view` alone
does not establish exact zero.

The required density uses full-image RGB differences against decoded Q3 and the
same decoded Q0 solo denominator for all qualities:

\[
A_{ref}(z)=\sum_p\alpha^{solo}_{Q0}(p),\qquad
d_k(z)=\frac{\sum_{p,c}(I_{Qk,c}(p)-I_{Q3,c}(p))^2}{3A_{ref}(z)}.
\]

Camera/projection, media time, resolution, object transform, renderer, and RGB/
background conventions must match. Preserve numerator energy outside the Q0
silhouette. Compatible existing full-image MSE gives
`d_k = MSE_k * P / A_ref`; no Q1–Q3 rerender is needed. Construct density per raw
context before nearest lookup or later interpolation. Dividing separately
averaged MSE and area produces a different quantity.

The default alpha-area guard `1e-6` is a provisional numerical guard in
alpha-weighted pixels, not an empirically calibrated support threshold. Area at
or below it makes density unavailable. Nonfinite/invalid measurements are also
unavailable. Do not clamp denominators, use Q3 area instead, or invent zero
density. A positive contribution with unavailable density invalidates affected
upgrade options. Fixed Q0 can miss geometry/silhouette that exists only in Q3;
the scalar density model approximates actual composite distortion improvement.

`RequirementsResolver.required_units()` traverses the target payload's closure,
stopping only at an explicitly retained state compatible with content, frame,
epoch, codec, state hash, and parent identity. Map required payloads to whole
requestable containers, deduplicate within a candidate, remove verified cached
containers, and sum missing exact bytes. Cached raw E2 still needs compatible
parents; retained Q1 does not establish a separate retained Q0 state. Handle
different layer segment boundaries by media coverage rather than segment ordinal.
Whole-container headers and unrelated members cost bytes without forcing decode
of unrelated members. Partial/in-flight/corrupt data receives no credit. Freeze
availability across all candidates. Missing shared units across objects are
unsupported by this additive prototype. Never infer playback availability from
an analysis asset or use average bitrate as exact cost.

## DP semantics and synthetic evaluation

For `B > 0`, configurable `K=1024` by default:

\[
\delta=B/K,\qquad\widehat C_{n,k}=\lceil C_{n,k}K/B\rceil.
\]

Compute bins with integer arithmetic: `(cost * K + B - 1) // B`. Utilities remain
floating-point. `DP[n,b]` means exactly `b` consumed bins after exactly `n`
objects. Initialize only `DP[0,0]=0`, all others to negative infinity; initialize
each fresh row unreachable. The recurrence considers every valid Q0–Q3 target
from the prior row. Select the best final state over all `0 <= b <= K`, then
backtrack one object per step, including zero-cost options. At `B=0`, permit
only zero-cost options without division.

Tie order: greater utility, then fewer actual bytes, then first candidate under
stable object/bin/quality iteration. Use exact comparisons, no utility epsilon.
Verify backtracking arrives at `(0,0)` and recomputed actual bytes fit the budget.
The DP uses `O(NQK)` time, rolling `O(K)` score storage, and compact `O(NK)`
backpointers. Upward rounding preserves bytes but can exclude feasible vectors;
artificial cost inflation is below `N*delta` without a general relative utility
guarantee. Doubling K produces nested feasible grids; arbitrary increases need
not do so.

The standalone solver needs no parser, profiles, GPU, or network:

```python
from src.client.adaptation_planner import AdaptationPlanner

planner = AdaptationPlanner(budget_bins=1024, safety_factor=0.85)
result = planner.solve_dp(
    utility=[[0.0, 0.4, 0.7, 0.9], [0.0, 0.5, 0.8, 1.0]],
    cost=[[0, 100, 200, 300], [0, 100, 200, 300]],
    budget_bytes=300,
)
```

Run a small benchmark or the complete N/K grid:

```bash
python tools/benchmark_adaptation_planner.py --objects 3 10 --bins 128 1024 --repetitions 3 --bounds
python tools/benchmark_adaptation_planner.py --bounds --output /tmp/adaptation-benchmark.json
```

The default grid is N=3,10,20,50,100; K=128,256,512,1024,2048,4096; budget
fractions 0.2,0.5,0.8. Tables use fixed-seed heterogeneous cumulative remaining
costs, zero-cost Q0, cached zero-byte enhancements, and signed floating gains.
Compare original-byte exhaustive optima for N<=8, and evaluate runtime, Python
traced peak memory, feasibility, utilization, and approximation as N/K vary.
Memory is a separate tracemalloc pass, not process RSS; neither oracles nor bounds
are in solver timings. Default output is stdout with no file writes.

`evaluate_bounds()` is evaluation-only. Upward costs give a feasible lower
utility; downward costs give an optimistic upper utility under the same tables
and masks. A floor-rounded allocation may overspend and is never executable.
Report the absolute gap and relative gap only with valid sign/denominator
conditions. If the upward grid is infeasible, another verified feasible vector
can provide a lower bound; all-Q0 normally does after Base-first. Floating-point
bounds are numerical estimates, not formally certified arithmetic bounds.

## Separate real-content and quality-model gates

Passing synthetic tests establishes component readiness only. Track these gates
separately: actual full-run validation, producer density readiness, real MPD/index
and decoded-state adapters, and real-content/model evaluation.

The configured `output/content_preparation/longdress_full` is not present in the
current checkout's mount. The six previously inspected contexts do not prove
compatibility of the actual completed 300-frame Longdress output. Locate that
artifact root instead of restarting its completed pipeline. The current full
configuration has 300 content frames but 21 sampled profile times, eight
azimuths, three elevations, and three scales: 1,512 Q0 contexts and 6,048 quality
rows. Validate the actual run's configuration and completion records; do not
assume those counts for a different run.

The separate producer follow-up must:

1. Verify content coverage, declared profiling grid, completion records, artifact
   hashes, delivery mode, decoded-state provenance, row/reference matching, and
   existing MSE/image consistency using existing prepared-content validation.
2. Inventory each required Q0 alpha archive and compare its map and float64 area
   sum with Visibility Analyzer solo diagnostics/`A_solo` under identical camera,
   projection/clipping, transform, resolution, scaling, decoded hash, and pinned
   rasterizer. Require map max absolute error <=1e-6 and
   `abs(A_saved-A_solo) <= 1e-5 + 1e-6*max(A_saved,A_solo)` for each publishable
   context. These are numerical parity tolerances, not support calibration.
   Export clamps saved alpha whereas the analyzer reduces the solo map directly;
   report failures and block publication rather than silently loosening limits.
3. Reuse compatible RGB/MSE. Sum valid archived Q0 alpha in float64; perform
   Q0-only parity rendering as validation and measure alpha where compatible
   archives are missing. Do not rerender Q1–Q3 or rewrite completed journals,
   MSE fields, images, or manifests.
4. Publish a versioned density sidecar and validation report referencing source
   hashes, with a new publication revision for later parser integration.
5. Sweep area guards and projected scale; report Q0-support distributions,
   density extremes, undefined rates, and utility/ranking sensitivity. Include
   Q0-zero/Q3-positive geometry and high-error energy outside Q0 support. Masks
   used to diagnose those regions do not change the full-image numerator.

A separate quality-model experiment renders every decoded quality combination
for N=1,2,3 and compares predicted summed utility to:

\[
G_{true}(\mathbf k)=MSE(I_{allQ0},I_{allQ3})-MSE(I_{\mathbf k},I_{allQ3}).
\]

Use matched contexts, separated/partly/fully occluded objects, depth swaps,
silhouette stress, and paired scenes with similar visible area but different
error regions hidden. Report correlation/ranking inversions, gain error, and
original-byte budget-feasible regret. Use exhaustive original-byte allocation to
separate quality-model error from solver rounding. Reuse existing scene/RGB/MSE
evaluation; freeze profiles, lookup, and support settings before held-out tests.
Q3-area normalization may be a later ablation only.

No real-content or empirical quality claim follows from synthetic component
tests. Missing actual profiles, parity validation, decoder/Base orchestration,
or MPD/index adapters leaves those integration gates open.

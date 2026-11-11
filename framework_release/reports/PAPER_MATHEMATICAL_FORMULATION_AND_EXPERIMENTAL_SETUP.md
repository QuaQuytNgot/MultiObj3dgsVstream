# Mathematical Formulation and Experimental Setup of MultiObj3dgsVstream

This report documents the algorithms and evaluation as implemented in the repository. Equations below describe the code paths and checked-in run artifacts, not an idealized ABR system. The principal method comparison is NAVA, Visibility × PSNR, and Visibility × MSE. Visibility × SSE and Exhaustive Search are retained as supporting methods and validation baselines.

## 1. Scope, evidence, and reproducibility

The latest full evaluation is `experiments/streaming_evaluation/runs/streaming_http2_dp015_v1/`, configured by `experiments/streaming_evaluation/configs/streaming_http2_dp015.json` and summarized in [`reports/STREAMING_HTTP2_DP015_EVALUATION.md`](STREAMING_HTTP2_DP015_EVALUATION.md). The requested `reports/PIPELINE_MATHEMATICS.md` is absent; the related document is [`docs/PIPELINE_MATHEMATICS.md`](../docs/PIPELINE_MATHEMATICS.md). The requested `reports/STREAMING_EXPERIMENT_RESULTS(2).md` is also absent. The closest file, [`reports/STREAMING_EXPERIMENT_RESULTS.md`](STREAMING_EXPERIMENT_RESULTS.md), documents an earlier, different experiment matrix and transport. Its results are not pooled with the current evaluation.

The current matrix is 7 object counts × 3 scenes × 3 durations × 5 methods = 315 cases. It uses real HTTP/2 and includes Exhaustive Search. The historical report describes 7 different object counts (including N=8 rather than N=20), earlier methods (including a solo-SSE baseline), and serial virtual FIFO transport. Its “315/315” refers to that historical matrix only.

Implementation references use repository-relative paths and function names so they remain portable. Principal sources are `experiments/adaptation_baselines/{methods.py,interface.py,metadata.py,exhaustive.py,real_cached.py}`, `experiments/visibility_weighted_sse/{profiles.py,visibility.py,allocation.py,real_cached.py,trace.py}`, and `experiments/streaming_evaluation/{runtime_scenes.py,runtime_matrix.py,inputs.py,runtime_adapter.py,network.py,simulator.py,playback.py,timeline.py,real_http2.py,http2_server.py,quality.py,metrics.py,runtime_native.py,two_phase.py,http2_dp015_evaluation.py}`. The native DP implementation is `src_final/multiobj3dgs/client/abr/native/proposed_dp.cpp`.

## 2. Notation and shared optimization model

Let $i\in\{1,\ldots,N\}$ index object instances, $q\in Q=\{40,36,28,20\}$ index the available CRF representations, and $g$ index a global 2-second media GOP. Smaller CRF is higher quality. The packaged choice is one representation per object and GOP. Let $x_{iqg}\in\{0,1\}$ indicate that instance $i$ uses representation $q$ in GOP $g$. In the standalone quality-profile order, Q0 is CRF40, the mandatory lowest-quality representation.

The four DP-based adapters maximize a method-specific additive utility:

\[
\max_{x}\quad \sum_{i=1}^{N}\sum_{q\in Q} U_{iqg}x_{iqg},
\qquad \sum_{q\in Q}x_{iqg}=1\quad\forall i.
\tag{1}
\]

The resource constraint is evaluated against physical package bytes, not an estimate from nominal bitrate. If $C_{iqg}$ is the set of physical resources needed by instance $i$, $s_h$ is the byte length of resource $h$, and $\mathcal H(x)=\bigcup_{i,q:x_{iqg}=1}C_{iqg}$, then:

\[
\sum_{h\in\mathcal H(x)}s_h\le B_g.
\tag{2}
\]

In the current standalone run, each of the 17 files belonging to a selected object/representation is requested independently. The configured `resource_sharing` is `per_instance`, there are no cross-instance shared resources or declared warm resource cache, and dependency lists are empty. Thus the union in (2) reduces to a sum over the selected per-instance packages for this run. The general allocator still supports common mandatory resources, SHA-based shared-resource unions, dependency activation patterns, and cache-resident resources. It accounts for these before solving and verifies the selected physical-byte union afterward (`experiments/visibility_weighted_sse/allocation.py`, `plan_resources`, exact-cost reconstruction). A repeated six-second source loop does not reuse the downloaded media package: the runtime requests the next GOP's resources again.

The methods share the same physical allocator and native DP solver. Their principal difference is the construction of $U_{iqg}$, plus the visibility weights used to construct it. They optimize the same one-choice-per-instance form with a common byte budget, but not a common perceptual metric.

### 2.1 Visibility and object importance

Visibility traces contain a joint alpha-contribution area $A^{vis}_{it}$ for instance $i$ at camera sample $t$, and a solo alpha-contribution area $A^{solo}_{it}$ from rendering that instance alone. These are sums of alpha/transmittance contribution over pixels, in alpha-weighted pixel units; they are not binary bounding-box pixel counts. For ordered Gaussians along a pixel ray, the underlying contribution is:

\[
T_{p,j}=\prod_{\ell<j}(1-\alpha_{p,\ell}),\qquad
c^{joint}_{i,p}=\sum_{j\in G_i}T_{p,j}\alpha_{p,j},\qquad
A^{vis}_i=\sum_p c^{joint}_{i,p}.
\tag{3}
\]

The solo counterpart omits other objects' occluders. The trace reports the ratio-style occlusion diagnostic $1-A^{vis}_i/A^{solo}_i$ when $A^{solo}_i>0$. In the implementation, aggregate visibility is a time-weighted mean across samples within the GOP. Samples use left-held values over half-open frame intervals; with 60 samples in a 2-second GOP at 30 fps, each sample has \(\tau_t=1/60\). See `src_final/multiobj3dgs/client/visibility/analyzer.py` (`render_joint`, `render_solo`, `A_visible`) and `experiments/visibility_weighted_sse/visibility.py` (`validate_trace`, `aggregate`), with temporal weights in `experiments/visibility_weighted_sse/trace.py`.

Visibility × PSNR and Visibility × MSE normalize the joint visible area across objects at each sample:

\[
I^{vis}_{it}=\begin{cases}
 A^{vis}_{it}/\sum_j A^{vis}_{jt},&\sum_j A^{vis}_{jt}>0,\\
 0,&\text{otherwise},
\end{cases}
\qquad w_i^{vis}=\sum_t\tau_t I^{vis}_{it}.
\tag{4}
\]

The per-sample normalization makes the weights sum to one when at least one object contributes visible alpha. A fully occluded or empty sample contributes zero weight. Visibility input validation enforces nonnegative areas, $A^{vis}_{it}\le A^{solo}_{it}$, and the configured image-area limit.

NAVA uses a different, projected-AABB proxy. experiments/nava_mse/geometry.py:aabb_area transforms the eight source-local bounding-box corners, projects the 12 triangles making up six faces, and sums positive signed projected triangle areas, scaled by $WH/4$. A triangle is discarded if any vertex has nonpositive homogeneous $w$ or lies outside the clip bounds; the code does not clip triangles against the frustum. This is geometric projected area, not alpha visibility, and can assign zero importance to alpha-visible content when the box projection is rejected. The normalized weight is:

\[
I^{AABB}_{it}=\begin{cases}
a_{it}/\sum_j a_{jt},&\sum_j a_{jt}>0,\\
0,&\text{otherwise},
\end{cases}
\tag{5}
\]

where $a_{it}$ is the projected AABB area. NAVA's per-object temporal weight is $w_i^{AABB}=\sum_t\tau_t I^{AABB}_{it}$. See `experiments/adaptation_baselines/methods.py:_aabb_samples`, `projected_aabb_importance`, and the geometry function above.

### 2.2 Method-specific utility profiles

All profile utilities are computed per object and representation from offline profiles; none uses joint-scene rendered quality as an online input. $P_{iq}$ is the arithmetic mean of per-view, per-frame RGB PSNR values stored in the NAVA/PSNR profile, in dB. Its population is 24 camera views × 60 frames = 1,440 samples per GOP/CRF. The source profile is built from RGB PSNR against that object's CRF20 representation, with the profile's recorded full-frame RGB protocol. `experiments/adaptation_baselines/real_cached.py` verifies that the stored summary agrees with the arithmetic mean of the row-level PSNR values.

**NAVA.** The implementation's utility is the weighted absolute profile PSNR:

\[
U^{NAVA}_{iqg}=\sum_t\tau_t I^{AABB}_{it}P_{iq}.
\tag{6}
\]

This is the implemented NAVA-like objective in `experiments/adaptation_baselines/methods.py`; its quality values come from `adaptation_baselines/real_cached.py` and its AABB weights from `nava_mse/geometry.py`. It is not the same as the alpha-visibility-weighted PSNR objective used by the proposed baseline. Since each object must choose exactly one quality, adding a choice-independent constant to all of one object's $P_{iq}$ values would not change the optimizer's choice when its weight is fixed. The material NAVA/Visibility × PSNR distinction is the AABB-versus-alpha weight, together with any profile-source differences, rather than the dB origin alone.

**Visibility × PSNR.**

\[
U^{V\times PSNR}_{iqg}=\sum_t\tau_t I^{vis}_{it}P_{iq}.
\tag{7}
\]

It uses the same stored PSNR profile as NAVA, but alpha-derived joint visibility weights. It is implemented by `experiments/adaptation_baselines/methods.py` and `adaptation_baselines/real_cached.py`.

**Visibility × MSE.** The offline $MSE_{iq}$ is pooled SSE divided by the number of foreground RGB scalar samples. The foreground mask is the object's CRF20 alpha greater than 0.01; RGB values are clipped to [0,1], with three channels and data range 1. The profile aggregates 8 camera IDs × 4 timestamps per GOP by pooling SSE and scalar count, not by averaging per-image MSE values. CRF40 (Q0) supplies the method's reference quality. Its implemented gain and utility are:

\[
\Delta MSE_{iq}=MSE_{i,Q0}-MSE_{iq},\qquad
U^{V\times MSE}_{iqg}=\sum_t\tau_t I^{vis}_{it}\Delta MSE_{iq}.
\tag{8}
\]

Values are signed; there is no clipping of a negative gain to zero. $MSE$ has squared normalized-RGB units. See `experiments/adaptation_baselines/metadata.py` for the profile aggregation/reference contract, `experiments/adaptation_baselines/methods.py` for utility construction, and `experiments/nava_mse/metrics.py` for the underlying RGB metric.

**Visibility × SSE (supporting baseline).** This method uses the visibility-trace GOP mean $A^{vis}_{ig}$ divided by a separately computed mean CRF40 solo alpha area $\bar A^{solo,offline}_i$ as its weight:

\[
w^{SSE}_{ig}=\begin{cases}
A^{vis}_{ig}/\bar A^{solo,offline}_i,&\bar A^{solo,offline}_i>0,\\
0,&\text{otherwise},
\end{cases}
\quad
U^{V\times SSE}_{iqg}=w^{SSE}_{ig}\bigl(SSE_{i,Q0}-SSE_{iq}\bigr).
\tag{9}
\]

Here each $SSE_{iq}$ is the equal-camera mean of the sum over four timestamp SSE values from 8 fixed camera IDs. The per-view SSE is solo RGB SSE against the fixed CRF20 reference alpha mask (threshold 0.01), with clipped RGB and black background. Thus this is not the same sample aggregation as MSE: SSE sums timestamps inside each camera before camera averaging, while MSE pools foreground scalar errors and counts over the samples. A zero visibility weight restricts the allocator to Q0 for that object. See `experiments/visibility_weighted_sse/profiles.py:build_utilities`, `visibility_weighted_sse/real_cached.py`, and `visibility_weighted_sse/allocation.py`.


For camera $c\in\{1,\ldots,8\}$ and timestamp $r\in\{1,\ldots,4\}$, the stored aggregation contracts are:

\[
D_{iq}=\frac{1}{8}\sum_{c=1}^{8}\sum_{r=1}^{4}SSE_{iqcr},\qquad
MSE_{iq}=\frac{\sum_{c,r}SSE^{fg}_{iqcr}}{\sum_{c,r}n^{fg}_{icr}},
\tag{9a}
\]

$SSE^{fg}$ includes only RGB scalars whose reference-alpha mask passes the configured threshold; $n^{fg}$ is its scalar count. The offline MSE and SSE profiles use distinct stored aggregation contracts.
For a checked Longdress GOP0 profile, the SSE values include CRF40=13,547.629, CRF36=9,554.944, CRF28=1,676.234, CRF20=0, with mean solo-area denominator about 23,673.559 alpha-weighted pixels. These values illustrate the actual SSE scale; they are not universal constants.

| Method | Instance weight | Per-quality signal | Implemented utility | Profile / interpretation |
|---|---|---|---|---|
| NAVA | $I^{AABB}_{it}$, normalized projected box area | $P_{iq}$, arithmetic-mean RGB PSNR in dB | $\sum_t\tau_t I^{AABB}_{it}P_{iq}$ | Geometric proxy; does not measure alpha occlusion |
| Visibility × PSNR | $I^{vis}_{it}$, normalized joint alpha contribution | Same $P_{iq}$ as NAVA | $\sum_t\tau_t I^{vis}_{it}P_{iq}$ | Same PSNR profile; different importance weights |
| Visibility × MSE | $I^{vis}_{it}$ | $MSE_{i,Q0}-MSE_{iq}$, pooled foreground squared RGB error | $\sum_t\tau_t I^{vis}_{it}(MSE_{i,Q0}-MSE_{iq})$ | Signed gain; 8 views × 4 timestamps |
| Visibility × SSE | $A^{vis}_{ig}/\bar A^{solo,offline}_i$ | $D_{i,Q0}-D_{iq}$, camera-mean timestamp-summed SSE | $w^{SSE}_{ig}(D_{i,Q0}-D_{iq})$ | Supporting method; separate normalization and sampling |

The $P$, $MSE$, and $D$ values have different units/scales. Their raw objective values are not cross-method comparable.


### 2.3 Profile comparability and quality representation

All methods use the same four CRFs but their offline profile populations differ. NAVA and Visibility × PSNR use 1,440 samples/GOP/CRF; MSE and SSE use 32 samples/GOP/CRF. MSE uses pooled foreground RGB squared error, while SSE uses camera-averaged timestamp-summed solo error. Therefore raw utility magnitudes are not cross-method comparable. Even MSE and SSE weights are not the same visibility normalization. The five utilities, profile builders, and quality representations are unchanged by the evaluation runner.

No algorithm uses future Phase B joint-scene PSNR in its objective. That metric is computed only after the selected playback frames have been recorded. The current configuration sets `joint_quality_used=false`.

## 3. Resource costs, bandwidth budgets, and selection

Each representation package contains 17 independent resources per object/GOP, as recorded by the package manifest and requested separately by the request path. Costs are physical byte lengths of those selected assets and manifest resources. Nominal representation bitrate is not substituted for package size. The optimizer receives a byte budget derived from the causal body-rate estimate and safety/deadline allowance. The actual byte constraint is checked after allocation against selected physical resources.

### 3.1 Conservative discretized DP

The main experiment uses GOP duration $d_g=2$ s and configurable bitrate grid $\Delta=150{,}000$ bps. For a residual resource byte cost $c$, the adapter first expresses it as a GOP bitrate with integer ceiling:

\[
r(c)=\left\lceil\frac{8c}{d_g}\right\rceil\;\text{bps},\qquad
\hat c=\left\lceil\frac{r(c)}{\Delta}\right\rceil,
\qquad
\hat B=\left\lfloor\frac{\lfloor 8B/d_g\rfloor}{\Delta}\right\rfloor.
\tag{10}
\]

Mandatory bytes are reserved exactly before the residual optional choices are discretized. The selected physical cost is reconstituted and checked against $B$ after the native solve. Thus a selected result cannot exceed the byte budget because of rounding. Conservative quantization can, however, reject an exact-byte-feasible selection. The effective grid corresponds to 37,500 bytes per 150 kbps step for a 2-second GOP, subject to the intermediate integer-ceiling conversions. See `experiments/visibility_weighted_sse/allocation.py` and `src_final/multiobj3dgs/client/abr/native/proposed_dp.cpp`.

The C++ solver rounds each optional cost up to integer units and floors the budget. It preserves mandatory Q0 cost and accepts precomputed utilities. Strict-improvement tie handling makes equal-score outcomes follow candidate order. The adapter cap is 2,000,000 DP cells. For $N$ objects, $K\le4$ choices, and integer budget $B_u$, runtime is $O(NK(B_u+1))$; memory is $O(B_u+NB_u)$ in the parent representation. Optional shared-resource activation can multiply solver calls by up to $2^s$ for $s$ shared resources; the current per-instance packages do not exercise that case.

The current full-run observed maximum DP cells were approximately 15,160 (NAVA), 20,508 (Visibility × PSNR), 19,500 (Visibility × MSE), and 20,010 (Visibility × SSE), far below the 2-million cap. Solver times are measured per plan in run artifacts. These observed values depend on each budget/cost profile and are not worst-case bounds.

The diagnostic DP comparison in `reports/HTTP2_DP_DIAGNOSTICS.md` and `reports/DISCRETIZED_DP_REAL_HTTP2.md` is bounded: grids of 0.05, 0.10, 0.15, and 0.25 Mbps were compared to exact-byte references on frozen cases. In the nine newer tight-budget cases, steps up through 0.15 Mbps had zero observed utility gap; 0.25 Mbps had a maximum 11.5437% gap and changed six selections. Maximum allocated cells in that set were 3,615, 1,815, 1,215, and 735 respectively. Earlier 15-case comparisons reported zero gap at each step. These results motivate the 0.15 Mbps choice but are not a proof of zero loss on all 315 workload cases. Exact-byte baselines can require very large DP capacity; this report does not imply billions of cells were allocated in the current full run.

### 3.2 Exhaustive Search (supporting baseline)

Exhaustive Search enumerates all $K^N=4^N$ vectors without DP, pruning, or partial-result incumbent. It uses Visibility × PSNR utility and exact physical-byte feasibility, retaining the best complete feasible vector. Limits are 5 seconds/GOP, 64 MiB additional RSS, 4 GiB absolute RSS, with checks every 128 candidates. Timeout/memory-limit rows remain TIMEOUT with unavailable metrics blank; if no vector satisfies exact cost the status is INFEASIBLE. It never substitutes Q0 or DP. Complexity is $O(K^N C)$ candidate evaluation and $O(N)$ extra memory.

### 3.3 Causal bandwidth estimator and GOP budget

The selected trace provides aggregate service capacity in bits per second. A causal estimator starts both body-rate and application-goodput predictors at 80,000,000 bps, maintains up to five recent observations, and treats measurements older than 30 seconds as stale. For a completed GOP request batch, body throughput is measured over the union of request body intervals, while application goodput uses the union of request start-to-completion intervals:

\[
\widehat C_{body}=\frac{8\sum_r bytes_r}{\left|\bigcup_r[ready_r,end_r]\right|},\qquad
\widehat G_{app}=\frac{8\sum_r bytes_r}{\left|\bigcup_r[start_r,end_r]\right|}.
\tag{11}
\]

Rates are in bit/s; times are seconds and byte counts are bytes. The implementation does not sum per-request rates, which would overcount concurrent transfers. Incomplete GOP batches do not update the estimator. The predictors combine valid samples with a harmonic estimate. See `experiments/streaming_evaluation/network.py:CausalBandwidthEstimator` and `simulator.py` completion processing. For fresh GOP-batch rates $r_j$, the predictor is $\widehat C=n/(\sum_{j=1}^{n}1/r_j)$; with no fresh sample it returns the configured 80 Mbps bootstrap.

For $K_g$ requested resources and stream limit $M$, latency is reserved as $L_{res}=L$ when $K_g\le M$, otherwise $L(1+K_g/M)$, with per-request latency $L=0.02$ s. With safety factor $\eta=0.9$, estimated transferable bytes are:

\[
B_g=\left\lfloor\frac{\widehat C_{body}}{8}\cdot\max\left(0,\eta d_g-L_{res}\right)\right\rfloor.
\tag{12}
\]

Here $B_g$ is bytes, $\widehat C_{body}$ is bit/s, and $d_g$ is seconds. Candidate completion uses the causal estimate and shared scheduler; it is considered deadline-feasible only within $\eta d_g$. The constant-rate predictor does not deduct adaptation-computation wall time from the budget or transfer-duration prediction; elapsed compute still affects later buffer advancement.

Trace values are service capacity, not achieved application rate. Service is the integral of configured capacity over time. If the aggregate rate is $C(t)$ Mbps and each trace sample spans one second, service volume is $\int C(t)dt$ Mbit before application and host limitations.

## 4. Experimental configuration

### 4.1 Workload, scenes, and cameras

The active configuration is `experiments/streaming_evaluation/configs/streaming_http2_dp015.json`; scene generation is in `runtime_scenes.py`, and matrix expansion/validation is in `http2_dp015_evaluation.py` and `runtime_matrix.py`.

| Setting | Exact current value | Evidence / interpretation |
|---|---|---|
| Source contents | Longdress, Soldier | 6-second source set and current config; package assets under `experiments/nava_mse/runs/gop0_v1/assets` |
| Representations | Independent CRF20, CRF28, CRF36, CRF40 packages | Same source GOPs and 17-resource package layout per representation; no progressive layer sharing is declared for this matrix |
| Object counts | N = 3, 5, 7, 10, 12, 15, 20 | Current config; counts include both alternating source identities |
| Scene setups | `camera_rotation`, `long_line`, `random_spatial` | Runtime scene builder |
| Playback lengths | 10, 20, 30 seconds | 5, 10, 15 global GOPs at 2 seconds/GOP |
| Methods | NAVA, Visibility × PSNR, Visibility × SSE, Visibility × MSE, Exhaustive Search | IDs: `NAVA`, `VISIBILITY_PSNR`, `VISIBILITY_WEIGHTED_SSE`, `VISIBILITY_MSE`, `EXHAUSTIVE_SEARCH` |
| Random seed | 238019 | For scene placement / deterministic generated states |
| Frame rate / source duration | 30 fps / 6 seconds | 180 source frames per source loop |
| GOP | 2 seconds | 60 nominal frames; playback rounds up to whole GOP count and final GOP remains a full package |
| Object timeline | One shared global playback timeline and source phase modulo 6 s | Every object uses source frame $f=tick\bmod180$ on that timeline; Longdress/Soldier identities differ, but they do not have independently phased clocks. Requests/adaptation are barriered per global GOP |
| Camera resolution | 640 × 480 | Used for visibility and Phase B rendered metric |
| Camera intrinsics | $f_x=f_y=772.5483399593904$, $c_x=319.5$, $c_y=239.5$ | Saved camera matrices / runtime scene provider |
| Base camera pose | Center (0,0,3.8) m; world-to-camera rotation rows [1,0,0], [0,−0.98480775,−0.17364818], [0,0.17364818,−0.98480775] | Saved camera matrix; camera position stays fixed while orientation changes |
| Renderer protocol | TF32 disabled; at most 8 resident models; 1 image writer and at most 2 pending writes | Native renderer settings in current config |
| Execution bounds | 172,800 s wall time; 86,400 s native time; 100 GiB storage; at least 20 GiB free space | Runner configuration limits; not an observed run duration |
| Camera path | Scene-specific yaw amplitudes 40° / 65° / 48° for camera_rotation / long_line / random_spatial; period 180 frames; random_spatial adds 5° pitch | `runtime_scenes.py`; frame is source-loop frame `tick mod 180` |
| Camera trajectory policy | `ORACLE_CAMERA_TRACE`; visibility for all GOPs is prepared before playback clock | Perfect pre-known camera path, but not future network bandwidth; predictor is disabled |
| Resource sharing | `per_instance` | No cross-object deduplication in the current matrix |
| Prepared profile inputs | `experiments/offline_profiles/runs/full_source_v1/prepared/profiles.json` and `baseline_metadata.json` | Offline profiles are precomputed and are not generated by the streaming run |

camera_rotation places instance $i$ at $(2\cos(2\pi i/N),0,1.2\sin(2\pi i/N))$ m. Its camera yaw amplitude is 40°. long_line places instances at $x_i=1.25(i-(N-1)/2)$, $y_i=z_i=0$ m, spanning −11.875 m to +11.875 m at N=20; yaw amplitude is 65°. random_spatial uses PCG64 seed 238019, samples $x\in[-3.5,3.5]$, $y\in[-0.9,0.9]$, $z\in[-1.2,1.2]$ m and rejects points less than 0.8 m apart; yaw amplitude is 48°. For frame $f\in[0,179]$, yaw is $A\sin(2\pi f/180)$. Only random_spatial adds pitch $5^\circ\sin(4\pi f/180)$.

```text
0 ( 0.005867,  0.344699, -0.314734)   10 ( 1.721657, -0.074317,  0.698572)
1 (-3.094234, -0.106862,  0.934498)   11 (-1.128004,  0.163585, -0.583456)
2 ( 1.178515,  0.592271, -0.772508)   12 ( 2.553270, -0.690024,  1.133012)
3 (-0.439471, -0.761893, -0.594238)   13 ( 2.141100,  0.691075,  0.558834)
4 (-3.444954,  0.597441, -0.232614)   14 ( 3.326331, -0.290840,  0.550288)
5 (-0.502308,  0.051251,  0.419440)   15 (-1.131513,  0.688509,  0.755827)
6 (-2.218320, -0.095174, -0.500669)   16 ( 2.762323,  0.555548, -1.178296)
7 (-2.684841,  0.739646, -0.664606)   17 ( 0.407601, -0.370801,  0.032926)
8 (-3.073620, -0.430897, -1.152409)   18 ( 1.491074, -0.823550,  0.114826)
9 ( 3.055508, -0.121164, -0.577376)   19 ( 1.968128, -0.022308, -0.822864)
```

The runtime scene builder applies camera-center-preserving object transforms to the source-local model poses and verifies the recorded camera center. The exact camera/instance matrices for displayed frames are persisted in Phase A frame records; this report does not approximate them with placement coordinates.

### 4.2 Network trace and transport

| Setting | Value in the current matrix | Notes |
|---|---:|---|
| Trace | Lumos5G-v1.0 dataset; processed HV.csv; selection metadata under experiments/abr_network_evaluation/inputs/lumos5g/ | Raw catalog has 68,118 rows across 118 runs; selected run 84 (walking/CW), sequence 39–158 |
| Trace offset | 18 seconds | Rebased so original trace time 18 is runtime time zero; 102 one-second samples remain through original time 119 |
| Original selected-run summary | Mean 913.15 Mbps, median 1,180 Mbps, min 0, max 1,740 Mbps, std 586.045 Mbps | 120 original one-second samples; not throughput achieved by the app |
| Rebased current-trace summary | Mean 905.5 Mbps, median 1,165 Mbps, min 0, max 1,740 Mbps, std 591.786 Mbps | 102 samples, integrated service 92,361 Mbps·s = 92.361 Gbit, or 11.545 GB at full service utilization |
| Per-request server latency | 20 ms | Configured request latency; overlapping requests share one aggregate trace service |
| Maximum concurrent streams | 16 | One TLS connection, HTTP/2, `httpx.AsyncClient(http2=True, http1=False)` and Hypercorn with TLS/ALPN `h2`; fallback fails validation |
| HTTP connection | One persistent connection per case | Connection reuse and HTTP/2 stream IDs are logged and validated; independent resources remain independent GET requests |
| Link shaping | Shared aggregate byte grants; max-min fair allocation over active body transfers; 2 ms service ticks, up to 1 MiB body grant per active stream per tick | Capacity is conserved across streams; it is not multiplied by the number of requests |
| Startup estimate | 80 Mbps body and goodput bootstrap | Used until completed GOP observations are available |
| Estimator history | 5 samples, stale after 30 s | Causal completed-GOP updates |
| Adaptation safety factor | 0.9 | Budget uses at most 90% of GOP duration minus latency reserve |

The processed trace begins at the raw sample at or before offset 18 and rebases later samples; no smoothing or interpolation is applied. There is no second bandwidth scenario in the current 315-case matrix. Separate bandwidth targets (100–1,500 Mbps and unshaped) in `reports/HTTP2_DP_DIAGNOSTICS.md` are diagnostic controls and must not be described as current matrix conditions.

`real_http2.py` performs actual HTTP requests and wall-clock timing. The one connection has a bounded stream pool; server-side aggregate shaping meters response bodies against the trace integral. The simulator also retains a deterministic virtual transport as a separate control implementation, but the current matrix selects `real_http2`. This is simulated aggregate bandwidth shaping around real HTTP/2 requests; it is not radio/network-interface shaping. HTTP/TLS/TCP framing overhead is not charged against the payload byte ledger. The server, client, and shaper run on the same host, and diagnostics show a significant host/event-loop ceiling: an unshaped 16-stream sustained diagnostic had a median body rate around 670 Mbps with one CPU core near saturation. Thus trace capacity is an upper service envelope, not a guarantee of realized payload throughput.

### 4.3 Phase A: streaming and adaptation

The runtime first prepares visibility for the complete known camera trajectory, before media playback. It does not reveal future bandwidth to the method. Per global GOP it forms visibility weights, estimates the causal budget, calls the selected utility adapter and common allocator, requests each selected object's 17 resources, and waits at the whole-GOP barrier before playback. After a completed GOP batch it records request measurements and updates the estimator. The player has a 2-second startup buffer and a 6-second maximum buffer. Rebuffering is measured from actual playback starvation.

Phase A stores a record for every actually displayed frame, including playback/media timestamps, camera intrinsics and extrinsics, per-instance transforms, source frame/GOP and source-loop identity, and the selected CRF vector. It also records requests, transferred bytes, request/queue/TTFB/body/completion timing, stream identifiers, estimator samples, bandwidth budget, chosen qualities, utility, solver time/memory, startup and rebuffering. The frame-record protocol is versioned and checksummed by the run manifests. The source content loops every six seconds; the common playback clock does not restart at the loop boundary.

One confirmed fallback detail affects interpretation: when the selected solver reports that mandatory Q0 is over budget (`infeasible_base`), the simulator still requests and plays the exact Q0 package. It records the allocation as infeasible and records the byte overrun. This is a delivery fallback for continuity, not a feasible optimizer solution. Q0 fallback QoE/PSNR must therefore be labeled as delivered fallback output rather than budget-feasible quality selection. Exhaustive Search does not use this fallback.

The timing model has one important limitation. Transfer estimates use a shared constant-rate fluid predictor and a concurrency-adjusted latency reserve. The real request completion path is wall-clock measured, but time spent computing the adaptation decision advances the wall clock and is not included in the candidate's expected transfer duration or byte budget. It is reflected later by playback-buffer advancement. Report predicted completion and actual startup/stalls separately.

### 4.4 Phase B: rendered joint-scene PSNR

Phase B runs only after Phase A and replays the exact logged displayed frame/camera/transform/quality states. The manifest records 162,000 displayed-frame occurrences across 270 eligible runs, deduplicated to 36,060 unique selected/reference render jobs and 38,700 unique image identities. For each unique selected joint scene state, the native StreamSTGS renderer produces the selected joint-scene RGB image and its corresponding all-CRF20 joint reference. It does not enumerate all unselected quality combinations. Cache identity includes scene/frame/camera/transforms/quality/protocol; cached images are reused only after checksum verification. A single GPU owner is used, at most 8 models are resident, and the configured image output queue is bounded. Phase B output does not feed back into adaptation.

For candidate RGB image $C_f$ and CRF20 reference $R_f$, each clipped to $[0,1]$, the full-viewport metric is:

\[
SSE_f=\sum_{x=1}^{W}\sum_{y=1}^{H}\sum_{c=1}^{3}
  (\operatorname{clip}(R_f)_{xyc}-\operatorname{clip}(C_f)_{xyc})^2,
\quad MSE_f=\frac{SSE_f}{3WH},
\quad PSNR_f=10\log_{10}\frac{1}{MSE_f}.
\tag{13}
\]

At $W=640,H=480$, denominator is 921,600 RGB scalar values per frame. The aggregate run PSNR pools squared error and samples before applying the logarithm:

\[
PSNR_{pool}=10\log_{10}\left(\frac{3WHF}{\sum_{f=1}^{F}SSE_f}\right).
\tag{14}
\]

An exact match to the CRF20 reference has zero SSE and infinite PSNR. This is full-viewport RGB PSNR, including black/background pixels; it is not the foreground-only MSE used by the adaptation profile. See `experiments/streaming_evaluation/quality.py`, `runtime_native.py`, `two_phase.py`, and `experiments/nava_mse/metrics.py`.

## 5. Matrix size, outcomes, and observed limits

| Dimension | Values | Count |
|---|---|---:|
| Object counts | 3, 5, 7, 10, 12, 15, 20 | 7 |
| Scene setups | camera_rotation, long_line, random_spatial | 3 |
| Playback durations | 10, 20, 30 s | 3 |
| Methods | NAVA, Visibility × PSNR, Visibility × SSE, Visibility × MSE, Exhaustive Search | 5 |
| Total Phase A cases | Cartesian product | 315 |

The completed run has 270 successful Phase A cases, 18 Exhaustive Search `INFEASIBLE` cases, and 27 Exhaustive Search `TIMEOUT` cases. The 18 infeasible and 27 timeout cases are all Exhaustive Search at N≥7; the four DP methods have 63 successful cases each. Phase B completed for the 270 successful Phase A cases; 45 Phase B result rows are unavailable because Exhaustive Search did not yield a streaming selection in those cases. No full 315-case rerun is warranted to reproduce this report; the recorded manifests, summaries, and run command are in the versioned output directory.

The solver columns below are computed across successful cases: solve time is the median over cases of each case’s median GOP solve time; RSS is the median/max case process high-water RSS, including the experiment process and renderer rather than solver-only memory; DP cells are the maximum recorded for a GOP. Exhaustive Search has no DP cells. Across the 270 successful cases, per-method descriptive means from `summary.csv` are:

| Method | Success | Body Mbps | Goodput Mbps | Startup s | Rebuffer s | Rendered PSNR dB | Q0 fallback GOPs | Median solver ms | RSS median / max MiB | Max DP cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NAVA | 63 | 372.55 | 352.93 | 0.983 | 0 | 23.51 | 68 | 0.152 | 1,282.95 / 1,415.99 | 15,160 |
| Visibility × PSNR | 63 | 443.83 | 428.74 | 0.769 | 0 | 33.74 | 65 | 0.170 | 1,282.95 / 1,425.22 | 20,508 |
| Visibility × MSE | 63 | 445.31 | 430.27 | 0.762 | 0 | 34.03 | 64 | 0.171 | 1,282.95 / 1,425.22 | 19,500 |
| Visibility × SSE | 63 | 445.35 | 430.07 | 0.767 | 0 | 34.04 | 60 | 0.190 | 1,282.95 / 1,425.22 | 20,010 |
| Exhaustive Search | 18 | 504.48 | 478.58 | 0.356 | 0 | 43.66 | 0 | 1.726 | 1,282.95 / 1,282.95 | 0 |

These are means over each method's successful rows, not a common-coverage comparison: Exhaustive Search contributes only N=3 and N=5, while DP-method means include N through 20. For the common N=3/N=5 coverage (18 cases per method), the recorded means are:

| Method | Body rate (Mbps) | Goodput (Mbps) | Startup (s) | Rebuffer (s) | Rendered PSNR (dB) |
|---|---:|---:|---:|---:|---:|
| NAVA | 387.63 | 352.87 | 0.406 | 0 | 26.53 |
| Visibility × PSNR | 501.65 | 474.98 | 0.360 | 0 | 43.66 |
| Visibility × MSE | 501.67 | 475.90 | 0.355 | 0 | 44.21 |
| Visibility × SSE | 502.41 | 476.14 | 0.363 | 0 | 44.21 |
| Exhaustive Search | 504.48 | 478.58 | 0.356 | 0 | 43.66 |

These results show the expected low-object exhaustive validation coverage and the high-N feasibility/timeout boundary. They do not imply the four utility scales are comparable, nor that every reported DP fallback met the budget: all four DP methods use at least one Q0 fallback in every N≥7 matrix slice, and fallback GOPs can transmit more bytes than the estimated budget. Completed playback and zero rebuffer in this finite run are not proofs of feasible allocations or of sustainable performance under arbitrary traces.

Real HTTP/2 diagnostic runs outside the matrix found that a sustained host-side unshaped 16-stream rate is materially below the selected trace's mean capacity. The current matrix's body rates around 373–445 Mbps are not evidence that the trace itself was only that fast. They reflect actual host and request-path limits plus application behavior. Application goodput is lower than body throughput because it includes request latency and scheduling gaps. The separate HTTP/2 diagnostic report contains target-rate, concurrency, CPU, and shaper measurements; do not present those diagnostic conditions as part of the 315-case matrix.

### 5.1 Interpretation and limitations

1. **Different adaptation proxies.** NAVA's AABB proxy can diverge from actual alpha visibility, especially under occlusion or conservative projection rejection. Its lower rendered PSNR and lower byte use are consistent with that objective/weight mismatch; the metric does not establish an optimizer bug by itself.
2. **Profile population mismatch.** PSNR and MSE/SSE profiles are built from different views/times and use different error aggregation. This is a real fairness limitation for a paper comparison and should be disclosed or addressed with a harmonized profile set in future work; the current report does not claim one was used.
3. **Budget versus fallback.** Q0 is mandatory in the optimization, but when even its physical bytes exceed the GOP budget, the simulator delivers it anyway to maintain playback and marks infeasibility. Such cases must not be represented as optimizer-feasible.
4. **Real transport scope.** HTTP/2 is negotiated and multiplexed, but bandwidth is application-level aggregate shaping in a same-host setup, not a physical Lumos5G access network. Host CPU and Python scheduling limit measured rates. Trace capacity and application throughput are distinct quantities.
5. **Causal budget timing.** The estimator is updated only after completed request batches and measures union intervals correctly. The constant-rate expected-completion estimate omits adaptation compute wall time, so it can be optimistic relative to actual media deadlines.
6. **Camera knowledge.** The method sees exact precomputed visibility for the known camera trace (`ORACLE_CAMERA_TRACE`). This is an offline/oracle trajectory assumption, not a deployed camera predictor result.
7. **Exhaustive coverage.** Exhaustive Search is a useful exact reference only at low N in this setup. Timeout and infeasible rows have no substituted quality metrics and are not directly comparable to completed methods at high N.
8. **Phase B metric.** Full-viewport rendered RGB PSNR is measured against joint CRF20 output; the adaptation utilities use per-object offline profiles. Those numbers answer different questions and should be presented separately.
9. **Historical results.** `STREAMING_EXPERIMENT_RESULTS.md` belongs to a previous matrix and virtual serial transport. The missing requested “(2)” file should not be cited as if it exists.

## 6. Suggested paper organization

**System Model.** Define $N$ object instances, source-loop frame, common GOP playback barrier, camera pose $v_t$, independent CRF representation $q_i$, package-resource sets, physical bytes, and aggregate time-varying service $C(t)$. State that resources use one persistent HTTP/2 connection with at most 16 streams and that trace shaping is application-level. Define startup/rebuffering and the causal estimator separately.

**Problem Formulation.** Introduce $x_{iqg}$, one representation per object, exact byte constraint, and method-specific additive objective. Present (6)–(9) and the formula comparison table. Explain the AABB/alpha distinction, profile differences, Q0 feasibility caveat, conservative discretization, and bounded low-N Exhaustive Search.

**Experimental Setup.** Report the source sequences, GOP/CRF package construction, profile provenance, scene placement/camera path, exact object counts and durations, trace selection/offset, HTTP/2/shaper configuration, playback policy, and two-phase protocol. Separate Phase A QoE and request measurements from Phase B joint-render quality. Give matrix coverage/status and common-subset comparisons, and avoid interpreting per-method success-only means as a paired algorithm comparison.

## 7. Source and artifact map

| Topic | Primary implementation / artifact |
|---|---|
| Adaptation method objectives | `experiments/adaptation_baselines/methods.py`; `experiments/adaptation_baselines/real_cached.py` |
| NAVA projected geometry | `experiments/nava_mse/geometry.py` |
| Offline PSNR/MSE profile contract | `experiments/adaptation_baselines/metadata.py`; `experiments/nava_mse/metrics.py`; `experiments/offline_profiles/runs/full_source_v1/prepared/` |
| Visibility and temporal aggregation | `src_final/multiobj3dgs/client/visibility/analyzer.py`; `experiments/visibility_weighted_sse/visibility.py`; `experiments/visibility_weighted_sse/trace.py` |
| SSE objective and profiles | `experiments/visibility_weighted_sse/profiles.py`; `experiments/visibility_weighted_sse/real_cached.py` |
| Resource allocator and discretization | `experiments/visibility_weighted_sse/allocation.py`; `src_final/multiobj3dgs/client/abr/native/proposed_dp.cpp` |
| Exhaustive Search | `experiments/adaptation_baselines/exhaustive.py` |
| Scene generation and matrix | `experiments/streaming_evaluation/runtime_scenes.py`; `runtime_matrix.py`; `http2_dp015_evaluation.py`; `configs/streaming_http2_dp015.json` |
| Network estimator / simulator / playback | `experiments/streaming_evaluation/network.py`; `simulator.py`; `playback.py`; `timeline.py` |
| Real HTTP/2 path | `experiments/streaming_evaluation/real_http2.py`; `http2_server.py` |
| Phase A / Phase B / metric | `experiments/streaming_evaluation/two_phase.py`; `quality.py`; `runtime_native.py`; `metrics.py` |
| Current results | `experiments/streaming_evaluation/runs/streaming_http2_dp015_v1/summary.csv`; `phase_b_jobs.json`; per-case manifests and logs |
| Current run report | `reports/STREAMING_HTTP2_DP015_EVALUATION.md` |
| Transport and DP diagnostics | `reports/HTTP2_DP_DIAGNOSTICS.md`; `reports/DISCRETIZED_DP_REAL_HTTP2.md` |
| Older, non-current experiment | `reports/STREAMING_EXPERIMENT_RESULTS.md` |


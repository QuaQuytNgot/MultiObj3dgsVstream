# NAVA provenance, fidelity, and StreamSTGS adaptation

Original source inspected on 2026-10-29: Quang et al., *NAVA: A Network-Adaptive
View-Aware Volumetric Video Streaming Framework*, IEEE Access 13 (2025),
25223–25238, [DOI 10.1109/ACCESS.2025.3538802](https://doi.org/10.1109/ACCESS.2025.3538802).
The [author-uploaded original full text](https://www.researchgate.net/publication/388716331_NAVA_A_Network-Adaptive_View-Aware_Volumetric_Video_Streaming_Framework)
was available through web browsing. No secondary description substitutes for it.

The relevant original procedure is Section III-E, equations 3–4; Section III-G,
equations 7–10; and Algorithm 1, printed pages 25227–25230. A GOP AABB is an
outward triangle mesh. After MVP transformation, the paper discards triangles
having an exterior clip-space vertex and sums positive signed projected areas.
Object importance is its area divided by all object areas; all-zero areas give
all-zero weights. The DP objective is importance-weighted **geometry PSNR**,
using RMS reconstruction error and original-box diagonal. Graph layers represent
objects and graph nodes represent versions. Recursive pulse traversal uses
dominance, minimum-resource feasibility, and maximum-value bounds to prune.
The paper also presents a separate Lagrange method; this experiment implements
the graph DP. [Original paper](https://doi.org/10.1109/ACCESS.2025.3538802)

The prior repository implementation inspected was
`experiments/abr_solver_processing_time/cpp/nava.cpp`, with its
`references/NAVA_METHOD.md`, and `experiments/abr_network_evaluation/native/nava.cpp`.
`experiments/abr_network_evaluation/nava_visibility.py` uses homogeneous polygon
clipping: this is an existing improvement that differs from the paper's
triangle-discard rule. The new geometry module records its selected rule.

The implementation is named **NAVA_STREAMSTGS_ADAPTED**. The following are
explicit deviations and implementation choices:

* Versions are independent StreamSTGS CRF20/28/36/40 packages, replacing the
  source paper's subsampled V-PCC versions. Costs are physical encoded GOP bytes.
  For one fixed GOP duration, a byte constraint and its equivalent bitrate
  constraint have identical feasible sets. No MPD average or rounded budget
  grid replaces those costs.
* No verified, comparable StreamSTGS geometry-PSNR profile was found in the
  audited repository inputs. Existing quality profiles are measured image
  quality against `original_ply_synthetic_rgb` renders. Their finite per-GOP
  PSNR summaries are accepted as an explicitly labelled proxy. These existing
  summaries average dB across their recorded views and frames: they are metadata
  utilities, **not pooled viewport PSNR**, geometry quality, photographic truth,
  or a claimed new measurement. CRF20 self-reference PSNR is forbidden; infinite
  and NaN utilities fail rather than being capped to invented quality.
* The metadata proxy is camera independent. It can contain recorded training
  and held-out source views; it supplies no joint-rendered evaluation results.
  Its provenance and original reference must remain attached to the input.
* The offline experiment supplies exact recorded camera/time conditions and a
  frozen budget; it does not implement NAVA's viewport/network prediction,
  download scheduling, device processing constraint, or online player.
* Multiple sampled timestamps are combined with declared normalized temporal
  weights before selecting a single CRF per object for the entire GOP. The
  weighted per-object proxy uses each sample's normalized projected area.
* `solvers.nava_graph_dp` instantiates each graph node with a dictionary of
  nondominated `(prefix bytes, prefix utility)` labels. It checks dominance at
  the same node only. Resource suffixes sum the cheapest remaining versions;
  value suffixes sum each remaining layer's best utility. These bounds remain
  valid even for nonmonotonic costs or quality. No Lagrange fallback runs.
* Finite float utilities become exact rational numbers for comparisons and
  sums. This is a numerical implementation choice: safe pruning needs no
  tolerance, discretization, or assumption about dB ordering. Equal objectives
  choose fewer bytes, then lexicographic input-order choices. Node/label caps
  explicitly return `incomplete`; an incumbent is not a certified optimum.

The two MSE policies deliberately use a different exact sparse integer-byte DP,
shared between them. For object `i`, sample `t`, representation `q`, their common
benefit is `sum_t temporal_weight[t] * importance[t,i] * (D[t,i,40]-D[t,i,q])`.
Average MSE repeats its camera-independent pooled profile at each sample; exact
MSE uses only exact matching per-object camera/time profiles. MSE is normalized
by foreground RGB scalar count. Normalized AABB importance supplies the area
factor once; no foreground-pixel or Gaussian-visibility multiplier is applied.
This surrogate estimates area-weighted distortion reduction and makes no claim
to equal composited joint SSE under occlusion. Negative measured reductions
remain negative. Missing exact entries are unavailable, never interpolated.

The policies import only their explicit input schema and solvers. Their input
type has no oracle, renderer, joint SSE, or evaluation callback. The exact
per-object view profile is intentionally an in-sample information reference;
the independently rendered joint-SSE oracle is evaluation only.

`tests/test_policies.py` independently enumerates Cartesian assignments for
both algorithms, covers every NAVA pruning category, exact-byte boundaries,
negative utilities, numerical near ties, all-invisible geometry, finite proxy
enforcement, incomplete searches, shared MSE solver/area weights, whole-GOP
decisions, missing exact entries, and the oracle input boundary. These are
CPU-only synthetic correctness fixtures, not scientific measurements.

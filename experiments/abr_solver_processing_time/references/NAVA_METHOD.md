# NAVA method provenance and adaptation

Source: Nguyen Long Quang, Truong Thu Huong, Duc Nguyen, *NAVA: A
Network-Adaptive View-Aware Volumetric Video Streaming Framework*, IEEE
Access 13 (2025), pp. 25223–25238,
[DOI 10.1109/ACCESS.2025.3538802](https://doi.org/10.1109/ACCESS.2025.3538802).
Read Section III-G1 and Algorithm 1 (printed pp. 25229–25230) from the
[author-uploaded full paper](https://www.researchgate.net/publication/388716331_NAVA_A_Network-Adaptive_View-Aware_Volumetric_Video_Streaming_Framework)
on 2026-10-25. The page exposes the full paper text; direct PDF download
returned HTTP 403 and the publisher returned HTTP 418, so no PDF is bundled
here. DOI metadata is retained in `nava_crossref.json`; its publisher
record is [IEEE article 10870276](https://ieeexplore.ieee.org/document/10870276/).

## What is inherited

NAVA represents each sequence by a graph layer containing its versions.
Entering a version node contributes its bitrate and visibility-weighted
geometry PSNR. Algorithm 1 initializes resource/utility suffix bounds and
recursively propagates `pulse()`. `checkDominance`, `checkFeasibility`, and
`checkBounds` reject prefixes using dominance, the minimum remaining
resource, and the maximum remaining value. Its incumbent is the best full
path discovered. This is NAVA's DP-based solution, separate from its
Lagrange-multiplier alternative. [Section III-G1](https://doi.org/10.1109/ACCESS.2025.3538802)

## This implementation's concrete choices

`cpp/nava.cpp` implements that layered graph implicitly: the input matrices
contain node-entry costs and values and the next layer's four successors
are visited recursively in CRF array order. No production module is used.
Graph-bound generation, allocation, label management, traversal and selection
reconstruction all happen inside the timed solver invocation.

The node representation is different: a layer is a StreamSTGS object
instance and its four nodes are independent CRF20/28/36/40 encodings.
The value is measured synthetic-RGB-reference PSNR times the supplied
visibility weight, replacing point-cloud geometry PSNR. The rate is the
audited GOP cost. The effective budget is supplied directly; this experiment
does not run NAVA's network/viewpoint prediction or visibility pipeline.

The publication specifies the pruning categories but does not give their
data structures or floating-point policy. The following definitions make
this standalone instantiation reviewable:

* `checkDominance`: each graph node has an ordered Pareto map of prefix
  `(cost, utility)` labels. A label dominates another at the same node if its
  cost is no greater and utility no smaller. Utilities strictly increase
  with stored costs after dominated labels are erased. The map is not a
  discretized bandwidth DP.
* `checkFeasibility`: the current prefix cost plus the sum of minimum costs
  of the remaining object layers must fit the original integer budget.
* `checkBounds`: prefix utility plus the sum of maximum utilities in the
  remaining layers must reach the incumbent. This is an unconstrained
  admissible upper bound; it does not assume monotonic bitrate or PSNR.
  Long-double bound additions round toward positive infinity and a
  conservative arithmetic margin retains near-equal paths.

There is no Lagrangian bound, heuristic fallback, cheapest-representation
shortcut, global object reordering, or approximation. Equal labels retain
the first visited prefix. Equal complete utilities retain the first
selection, giving deterministic results without requiring identical
selections across solvers.

## Why completed searches are exact

Every feasible assignment corresponds to a path through all layers. A
dominant prefix ending at the same node can use every suffix available to
the dominated prefix, at no greater total cost and no smaller utility.
Minimum-suffix infeasibility cannot discard a feasible completion. A sum
of independent layer maxima is at least the utility of every possible
suffix, so the objective bound cannot discard an improving completion.
Consequently every discarded path has a certificate and the best remaining
complete path is optimal when traversal finishes. Small-instance tests
compare objectives with literal exhaustive enumeration; representation
ties are permitted.

The frontier can grow exponentially and recursion can explore exponentially
many graph paths: this is not an O(NKB) discretized DP. With ordered maps,
each insertion/query is O(log L) plus dominated labels erased, with L live
labels, and memory is O(NK + L). The suffix setup is O(NK).

Both a wall-clock limit and a maximum live-label count are configurable.
Exceeding either returns `timeout`; any attached feasible selection is an
unproven incumbent and must not become an exact oracle. The solver never
reports a capped search as `optimal` or silently replaces it with another
algorithm. Timed invocations also include periodic clock-check overhead.

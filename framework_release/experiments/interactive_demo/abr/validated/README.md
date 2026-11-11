These small Python modules were copied for this isolated implementation:

- `common.py`, `profiles.py`, `allocation.py`: `experiments/visibility_weighted_sse/`.
- `baseline_metadata.py`: `experiments/adaptation_baselines/metadata.py`.

Imports point to the reusable production metadata types and isolated solver.
The copied allocator keeps its exact-byte and residual-bitrate preprocessing,
strict candidate ordering, native four-choice padding, useful-budget cap and
explicit resource limits. Its only accounting change is `_resource_key`: runtime
resources may supply an admission `owner_key` separately from their true payload
SHA. Thus identical downloaded bytes cannot collapse two independent rows.
The owner key is confined to numerical allocation metadata and never replaces
the digest used for integrity or an HTTP request.

`../objectives.py` copies the MSE branch's multiplication/accumulation order from
`adaptation_baselines.methods.construct_utilities` and the normalization kernel
from `nava_mse.policies.projected_aabb_importance`. Focused tests execute the
original helpers read-only to check frozen numerical parity.

The C++ recurrence and ABI are unchanged and read directly from
`src_final/multiobj3dgs/client/abr/native/`; the compiled library is placed only
under `experiments/interactive_demo/output/solver/`.

# Adaptation implementation provenance

This release packages the repository's existing implementations without changing source bytes. It does not distribute original-author NAVA code or implement missing paper algorithms. The copied-file inventory and hashes are in [COPIED_FILES.json](COPIED_FILES.json) and [SOURCE_SHA256SUMS](SOURCE_SHA256SUMS).

| Method | Existing implementation and interpretation |
| --- | --- |
| `NAVA` | Adapted reproduction: paper-discard projected AABB area, normalized at each camera sample and integrated over the GOP, multiplied by the supplied finite image-PSNR profile. StreamSTGS CRF packages replace the paper's V-PCC versions, and image PSNR replaces geometry PSNR. |
| `VISIBILITY_PSNR` | Repository comparison method: normalized alpha contribution at each sample × supplied offline PSNR, integrated over the GOP. |
| `VISIBILITY_MSE` | Repository comparison method: the same normalized alpha contribution × signed offline MSE reduction relative to Q0. |
| `VISIBILITY_WEIGHTED_SSE` | Repository method: measured visible contribution divided by offline mean solo contribution × signed solo-SSE reduction relative to Q0. |
| `SOLO_SSE` | Existing supporting comparison method using the unweighted solo-SSE reduction. |
| `AABB_MSE_HISTORICAL` | Explicit historical method retaining projected-AABB weighting for MSE. It is separate from the primary alpha-weighted `VISIBILITY_MSE`. |
| `EXHAUSTIVE_SEARCH` | Existing bounded Cartesian enumeration, using Visibility × PSNR utility and exact physical-byte feasibility. Timeout/infeasible results remain unavailable; no replacement selection is invented. |

The common Phase A comparison uses the unchanged native Proposed DP through the existing physical-resource adapter for its DP methods, including NAVA. This is a solver adaptation: the historical NAVA graph DP is not the solver used for that comparison. Exhaustive Search retains its own implementation. The interactive demo uses its existing validated copies and its Visibility × MSE planner; it is a distinct existing entry point.

Input metadata carries the quality reference and sample population. Preserve it when comparing methods: image PSNR, foreground MSE/SSE and rendered full-viewport PSNR describe different measurements. A Phase A run does not render or establish Phase B PSNR.

## Source mapping

- Objectives and metadata contracts: `experiments/adaptation_baselines/{methods,metadata,interface}.py`.
- Exhaustive enumeration: `experiments/adaptation_baselines/exhaustive.py`.
- Visibility/SSE profiles and physical-resource allocation: `experiments/visibility_weighted_sse/{profiles,trace,visibility,allocation}.py`.
- NAVA projected AABB geometry and normalization: `experiments/nava_mse/{geometry,policies}.py`. Their unchanged import closure also includes `solvers.py` and `experiments/abr_network_evaluation/{nava_visibility,visibility}.py`.
- Native production DP bridge: `src_final/multiobj3dgs/client/abr/solver.py` and `native/{solver_bridge.cpp,proposed_dp.cpp,common.hpp}`.
- Interactive demo: `experiments/interactive_demo/abr/`, `experiments/interactive_demo/runtime/` and the production codec, renderer, viewer and metadata modules they import.

Preserve the original module locations: experiments import both `src_final.multiobj3dgs.*` and `multiobj3dgs.*`. The README supplies the required release-local Python paths.

## Original attribution and limitations

The repository's NAVA provenance identifies Quang et al., *NAVA: A Network-Adaptive View-Aware Volumetric Video Streaming Framework*, IEEE Access 13 (2025), 25223–25238, DOI [10.1109/ACCESS.2025.3538802](https://doi.org/10.1109/ACCESS.2025.3538802). Its detailed fidelity and adaptation notes are preserved in [NAVA_METHOD.md](../experiments/nava_mse/references/NAVA_METHOD.md). The common-method changes and profile limitations are documented in [ADAPTATION_BASELINES_IMPLEMENTATION.md](../reports/ADAPTATION_BASELINES_IMPLEMENTATION.md) and [the mathematical formulation report](../reports/PAPER_MATHEMATICAL_FORMULATION_AND_EXPERIMENTAL_SETUP.md).

StreamSTGS inference sources are pinned by `third_party/streamstgs/manifest.json` to commit `ca8602516352efd83e9e69e665b2e26ed03cb5cc`. Their existing patched bytes, upstream hashes and license notices are preserved. Consult [third_party/README.md](../third_party/README.md), the StreamSTGS and rasterizer `LICENSE.md` files, FAST's `license`, and GLM's `copying.txt`; these components carry their own terms, including research/noncommercial restrictions for StreamSTGS/Graphdeco code.

No top-level license for the repository authors' own code was present in the source checkout. Packaging does not invent a license or grant additional rights. The authors must supply their chosen license before public distribution; third-party terms remain applicable independently.

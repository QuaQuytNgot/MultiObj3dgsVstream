# Content preparation migration

On 2026-10-05 the project-owned offline preparation pipeline and experiment tools
were copied from the sibling `dynamic-lapis-gs` workspace into this repository.
The original workspace is retained. This repository owns the pipeline code;
external trainer, Gaussian model and renderer source remain in pinned submodules.

## Ownership and layout

| Location | Purpose |
| --- | --- |
| `tools/content_preparation/` | Nine-stage preparation pipeline, codec/state/package helpers, profiles, proxies, journals, self-tests and backend path resolution |
| `tools/*.py` | Dataset acquisition, composability/correction/temporal experiments, quality-ladder training wrappers and validation/reporting CLIs |
| `tools/content_trial_viewer/` | Static offline PNG/metadata viewer; it does not render 3DGS in the browser |
| `scripts/` | Development/smoke utilities and project-owned publication inventory |
| `configs/content_prepare_*.yaml` | Smoke, raw preparation and historical checkpoint-import configurations |
| `docs/CONTENT_PREPARATION_*.md` | Preparation design, codec boundary, run guide and status |
| `docs/validation/*_summary.json` | Unchanged compact evidence from the source workspace |
| `third_party/dynamic-lapis-gs/` | Original training/preprocessing/Gaussian/renderer backend |
| `third_party/dynamic-lapis-gs/submodules/` | Backend's pinned `diff-gaussian-rasterization`, `simple-knn` and nested GLM sources |
| `third_party/draco/` | Pinned codec source for future work; current preparation codec is `gaussian_attribute_zlib` |

The Dynamic-LapisGS dependency already supplies the Gaussian model and 3DGS CUDA
extensions used here. A second copy of the complete `gaussian-splatting`
repository is not required for this preparation pipeline. See
[dependencies](DEPENDENCIES.md) for pins and optional installation requirements.

Project paths resolve relative to this repository. Upstream Python imports and
training subprocesses resolve against `third_party/dynamic-lapis-gs`, through
`tools/content_preparation/paths.py`; they do not require the sibling checkout or
an absolute workstation path. CUDA extension binaries must match that backend.

The project launcher `tools/content_preparation/backend.py` executes the pinned
`train.py`/`render.py`/`metrics.py` source with unchanged arguments. Its process-local
NeRF callback retains the old workspace's unsigned RGB/Pillow compatibility fix.
`compatibility.py` also delegates the original preprocessing function with the
obsolete Open3D constructor keyword removed. Model/renderer/loss source remains
unchanged; provenance records adapter and launcher hashes explicitly.

## Installation and first checks

Run from the project root:

```bash
git submodule update --init --recursive
conda activate Hoang
python -m pip install -r requirements.txt
python -m pip install -r requirements-content-preparation.txt
python tools/content_preparation/prepare_content.py \
  --config configs/content_prepare_smoke.yaml --dry-run
python tools/content_preparation/self_test.py
```

The transport requirements remain separate from the optional preparation stack.
Use the existing compatible PyTorch/CUDA environment; recursive submodule
initialization downloads source without building it. Follow
[dependencies](DEPENDENCIES.md) before a CUDA build or GPU pipeline run.
`scripts/smoke_test.py` exercises upstream preprocessing/training/rendering on tiny
fixtures and needs the GPU/backend stack; it is not a prerequisite for reading a
config or using the network client.

The bounded self-test reports optional browser skips without failing core readiness.
Use `--strict-optional` to require that coverage too. Missing core dependencies or
cached LPIPS weights still fail readiness. Codec characterization scripts also
require a `zstd` command-line executable on `PATH`; browser verification additionally
requires separately installed Playwright and Chromium. Neither is needed for the
core network client or a preparation dry-run.

This migration does not run full training, rebuild CUDA extensions, acquire a
dataset or regenerate the Longdress media/profile/viewer artifacts. Validation
in the destination consists of bounded tests, import/CLI/config checks and
whitespace checks. The historical PASS counts in the copied reports are evidence
for the source workspace, rather than results of those destination checks.

Destination checks on 2026-10-05 in conda `Hoang` completed with **184 pytest tests
passed, 1 skipped and 121 subtests passed**. The preparation self-test ran 105 tests
with the same optional Playwright/browser skip; its CPU pipeline exercised cached
pretrained VGG LPIPS. Compilation, 18 CLI help checks, all six config dry-runs,
two legacy codec self-tests and Git whitespace checks passed. These checks do not
establish CUDA preprocessing, training or rendering compatibility on a fresh clone.

## Inputs and artifacts

Generated data belongs under the ignored `output/` root, or explicit external
storage selected through CLI/config paths. Keep raw point clouds, prepared images,
poses, checkpoints, CUDA binaries, caches, media, profiles and preview PNGs out of
Git. Their original paths in the historical summaries are intentionally retained:
changing paths or hashes in evidence would misrepresent the original experiment.

A fresh checkout contains the code/configs/docs, not the old run's `output/` tree.
Start with a new output root and produce raw/prepared inputs, or supply a verified
checkpoint manifest and its checkpoint files to an `existing` config. Historical
four-level configs expect `output/longdress_4level_training/manifest.json`; the
two-level configs expect `output/progressive_gap_real/manifest.json`. Dry-run does
not manufacture these assets or training lineage.

Pipeline implementation/backend paths change during migration, so old journals
and guarded environment signatures cannot be treated as valid resume state in
the new tree. Preserve the original run for historical audit. For a new migrated
run, use a new work/output root and create fingerprints in the current checkout.
Do not rewrite original hashes to make a resume check pass.

The preparation output index and profile schema are offline artifacts. This
migration does not add a runtime MPD parser, scheduler, bandwidth estimator, LoD
selection or renderer adapter. The transport block continues to assume a static
HTTP server supporting H2/H3 and Range. Python's optional `http.server` command
in the run guide serves offline PNG previews locally over HTTP/1.x; it is not a
streaming-protocol integration test or a custom project server.

## Preserved experiment record

The historical Longdress trial covers five frames (1051–1055), four trained
qualities Q0–Q3, f32/f16 variants and independent/progressive delivery. It does not
cover full-sequence convergence. The measured tables, counts and limitations are
retained in [readiness](CONTENT_PREPARATION_READINESS.md),
[the two-level trial report](LONGDRESS_ENCODING_TRIAL.md), progressive/correction/
temporal experiment reports and the three unchanged JSON summaries. Follow
[the run guide](CONTENT_PREPARATION_STEP_BY_STEP.md) to reproduce them with inputs.

Publish project source through normal Git review; see
[publishing](PUBLISHING.md). The source workspace's private backup instructions
were not copied as this project's publishing workflow.

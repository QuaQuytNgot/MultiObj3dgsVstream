# Multi-object adaptive streaming of dynamic 3D Gaussian Splatting

The current implementation provides an async network transport block and offline
content preparation. Runtime MPD parsing, LoD selection, visibility, bandwidth
prediction, payload decoding and multi-object renderer integration are reserved
for subsequent work.

## Install

```bash
git clone --recursive https://github.com/QuaQuytNgot/MultiObj3dgsVstream.git
cd MultiObj3dgsVstream
conda activate Hoang
python -m pip install -r requirements.txt
```

For an existing checkout whose submodules have not been initialized:

```bash
git submodule update --init --recursive
```

The project is its own Git repository. Run project Git commands from
`MultiObj3dgsVstream/`, even when it is stored inside another workspace repository.
External source repositories live in `third_party/` and are pinned by Git
submodules. Initialization downloads source; it does not build Dynamic-LapisGS or
Draco. Dependency decisions and pinned commits are recorded in
[docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

## Offline content preparation

Project-owned preparation tools, scripts, configs and experiment documentation
have been migrated from the sibling `dynamic-lapis-gs` workspace. The pipeline in
`tools/content_preparation/` follows preprocess → train/import → export → encode
→ package → decode → profile → manifest. Proxy generation can be excluded with
`proxy.enabled: false`; an existing external descriptor may be referenced.
Original Gaussian/trainer/renderer
source comes from `third_party/dynamic-lapis-gs`, including its pinned nested
3DGS extensions; it is not copied into the project source.

Start with [the migration note](docs/CONTENT_PREPARATION_MIGRATION.md) and
[the step-by-step guide](docs/CONTENT_PREPARATION_STEP_BY_STEP.md). Optional
preparation dependencies stay separate from the networking requirements:

```bash
python -m pip install -r requirements-content-preparation.txt
python tools/content_preparation/prepare_content.py \
  --config configs/content_prepare_smoke.yaml --dry-run
python tools/content_preparation/self_test.py
```

Use the existing compatible PyTorch/CUDA environment; see the dependency note for
backend setup. This migration validates source/CLI/configs and bounded tests. It
does not build CUDA extensions, train or regenerate the full Longdress pipeline.
Generated datasets, checkpoints, binaries and media stay under ignored `output/`
or external storage. A fresh clone contains no historical run outputs.

The preserved source-workspace trial covers **Longdress 1051–1055, four trained
qualities Q0–Q3**, f32/f16 and independent/progressive variants. Its sampled f32
decoded matching-GT diagnostic is retained as historical evidence:

| Quality | Training images | Mean MSE | Mean PSNR (dB) | Mean LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Q0 / res8 | 128×128 | 0.00260485 | 25.84 | 0.068016 |
| Q1 / res4 | 256×256 | 0.00122150 | 29.13 | 0.053845 |
| Q2 / res2 | 512×512 | 0.00051612 | 32.91 | 0.041993 |
| Q3 / res1 | 1024×1024 | 0.00042250 | 33.85 | 0.038619 |

Each row contains four comparisons (frames 1051/1055 and test cameras 5/10),
full-image RGB on black background and pretrained VGG LPIPS with batch 1.
Training budgets differ across qualities; the five-frame trial does not establish
full-sequence convergence. Adaptation profiles use the highest decoded quality in
each configured run as their reference. That historical trial used
`gaussian_attribute_zlib` 1.0.0. The new first batch also provides
`gaussian_attribute_draco_byteplanes` 1.0.0: a lossless project adapter over pinned
public Draco 1.5.7, with shared-state refinements. It is not the unpublished LTS
encoder. See
[readiness and limits](docs/CONTENT_PREPARATION_READINESS.md),
[design](docs/CONTENT_PREPARATION_DESIGN.md),
[codec boundary](docs/CONTENT_PREPARATION_CODEC.md) and
[the unchanged historical summary](docs/validation/longdress_4level_trial_summary.json).

The bounded Draco batch adds a validator, experimental MPD writer and decoded
Gaussian inspection in the existing viewer. Follow
[CONTENT_PREPARATION_BATCH1.md](docs/CONTENT_PREPARATION_BATCH1.md) for exact build,
test and two-frame smoke commands. That config imports existing Q0/Q1 checkpoints,
disables proxy work and uses the original CUDA renderer at 128×128. It does not
start training or full Longdress preparation. Viewer selection supports arbitrary
quality counts, including Q0–Q3 when those decoded assets are available.

First-batch validation: **139 self-tests passed**, actual Q0/Q1 two-frame
encode/decode/profile/MPD smoke passed, and **4/4 decoded Gaussian browser checks
passed**. Resume executes zero tasks. Open the
[local smoke viewer](http://127.0.0.1:8766/?object=longdress&quality=Q1&display=gaussians)
after starting its server; current evidence is in
[the batch summary](docs/validation/content_preparation_batch1_smoke_summary.json).

[Publishing instructions](docs/PUBLISHING.md) describe reviewing and pushing this
project's source while keeping generated artifacts out of Git.

## Request Handler

See [docs/REQUEST_HANDLER.md](docs/REQUEST_HANDLER.md) for API examples, protocol
configuration, Range validation, metrics, error handling, and integration points.
Application modules use the stable public interface in `src/client/request_api.py`:
`NetworkClient`, `TransferClient`, `RequestSpec`, `TransferResult`, and
`to_bandwidth_sample`. `request_handler.py` remains the low-level transport.
The project assumes an existing static HTTP server supporting H2/H3 and Range
requests; this phase does not add a custom web server.
[configs/client.yaml](configs/client.yaml) is an example configuration; the handler
does not load YAML itself.

Run validation from the project root:

```bash
python -m py_compile src/client/request_api.py src/client/request_handler.py
python -m pytest
python -c "from src.client.request_api import NetworkClient, RequestSpec, TransferClient"
git diff --check
```

Default tests use a local HTTP/1.1 server and require no Internet. HTTP/2 and
HTTP/3 build capability checks are separate from successful protocol negotiation
with a real server.

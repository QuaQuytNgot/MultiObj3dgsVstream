# StreamSTGS DASH/native implementation

`src_final/multiobj3dgs` is the new installable implementation. Original `src`,
`tools`, datasets and codec trials remain available for regression. No solver,
progressive CRF dependency or device-specific quality cap is introduced.

## Setup

Use a dedicated Python/CUDA environment. Install the PyTorch build appropriate
for the GPU/driver separately, then install the package and optional dependencies:

```bash
python -m pip install -e '.[network,content,streamstgs,native,test]'
python scripts/setup_streamstgs.py
python -m pip install --no-build-isolation --no-deps \
  third_party/streamstgs/submodules/diff-gaussian-rasterization
```

The source-only pinned vendor tree is included and its manifest records hashes,
commit and patches. `setup_streamstgs.py --source /path/to/StreamSTGS` reproduces it
offline from the pinned upstream Git objects. FAST is imported from that pinned
source tree by the adapter. Its dependencies include PyOpenGL, PyGLM, rich and
cuda-python with the pre-13 `cudart` API. FFmpeg/ffprobe with HEVC and imagecodecs
with JPEG-XL are required. Do not copy compiled extensions between Torch/CUDA/GPU
architectures. See `third_party/streamstgs/manifest.json`, `patches/README.md` and
`src_final/multiobj3dgs/codecs/streamstgs/README.md` for provenance and runtime setup.

An installed wheel keeps optional vendor sources external. Set
`MULTIOBJ3DGS_VENDOR_ROOT=/absolute/path/to/third_party/streamstgs` outside a source
checkout. Metadata parsing and the development HTTP origin need no CUDA/Torch.

## Publish existing content

```bash
python -m multiobj3dgs.content publish \
  --archive data_stream/streamstgs_soldier_536_835_crf20_28_36_40.tar.gz \
  --archive data_stream/streamstgs_longdress_1051_1350_crf20_28_36_40.tar.gz \
  --output output/published/scene-n2 --instances 2
python -m multiobj3dgs.content validate output/published/scene-n2
python -m multiobj3dgs.server --root output/published/scene-n2 --port 8080
```

Publication validates both archive and member hashes, preserves codec bytes,
deduplicates shared assets by SHA256, and publishes immutable CPSEG GOP bundles,
a normalized index and static MPD. Repeating identical publication is idempotent;
a conflicting existing publication is rejected. Choose any positive instance
count or use `--scene configs/final/scene_four_objects.json`; runtime add/remove is
outside v1. CRF representation IDs are `crf-20`, `crf-28`, `crf-36`, `crf-40`.

The project DASH profile requires this Gaussian client: its CPSEG resources are
not ordinary video segments for a generic browser DASH player. Each content has
an independent timeline/representation set, while scene instances own transform,
media offset and decoder state. Shared assets vary by GOP and are not one global
Initialization segment. The index owns actual sizes, hashes and asset closure;
codec JSON owns model/atlas/lookahead details.

## Native and headless playback

On the Linux/NVIDIA machine with a local display:

```bash
python -m multiobj3dgs.client --mpd http://127.0.0.1:8080/manifest.mpd \
  --mode native --report output/player/native-run.json
```

Native controls: left drag orbit, right drag pan, wheel zoom; Space play/pause;
arrows seek, Home restart, L loop; F fly mode, WASD/E/C move; Tab selects an
instance, Q cycles its representation. The imgui overlay exposes timeline,
per-instance quality selection, buffers, FPS and VRAM. Quality changes apply at
GOP boundaries; the pending selection is visible in diagnostics.

For SSH/server validation:

```bash
python -m multiobj3dgs.client --mpd output/published/scene-n2/manifest.mpd \
  --mode headless --frames 300 --prefetch 1 --workers 1 \
  --width 512 --height 512 --report output/player/egl-run.json
```

Headless EGL and native context bootstraps are separate. Do not force
`PYOPENGL_PLATFORM=egl` for native mode. The player renders one joint Gaussian
scene; no per-object image compositing or mandatory CPU image readback is used.
N is scene-configured without a fixed cap. An insufficient active memory budget
produces an explicit resource error. Adjust `--host-mib`, `--raw-mib`, `--gpu-mib`,
`--gpu-transient-mib`, `--cache-mib` for the actual machine. Defaults are startup
settings, not automatic quality caps or an FPS guarantee.

Network and CPU preparation run ahead of playback. GPU realization,
reconstruction, contribution analysis and GL interop share one owner thread.
Current+next is a bounded admission target, not a promise of deadline feasibility.
The default uses two persistent CPU workers, two FFmpeg decode threads, and a
2 ms upload-owner allowance per pump (scaled with active instances, capped at
four). CPU preparation publishes canonical data early; each frame waits for its
exact three-feature temporal window. A dedicated CUDA stream and two reusable
8 MiB pinned buffers transfer into independent model storage. Complete GOP bytes
are reserved before admission. `--upload-budget-ms 0` selects full synchronous
preparation/upload for comparison; `--fast-jit-specialization` enables FAST's
profiling specialization for ablation. Native idle presentations reuse a clean
scene cache; camera input still reconstructs and renders the displayed timestamp.
The whole scene stalls on missing due temporal data. No frames are silently
dropped to improve a reported FPS. Seek invalidates generations and drains stale
jobs. Scene time, source-frame time and codec model time remain distinct.
Presentation timestamps use integer frame steps anchored at the seek position;
local timeline conversion normalizes only binary floating-point roundoff. Manual
quality activates on the first presentation sample of a new local GOP, including
off-grid seek phases. Natural scene loops retain valid prefetched GOP handles;
explicit seek/restart still invalidates stale asynchronous jobs.

See [startup/GOP profiles and reproduction commands](STREAMSTGS_STARTUP_GOP_OPTIMIZATION.md)
for launch-to-first-swap, actual 300-frame native playback, memory tradeoffs and
remaining compute limits at N=4. These results use bounded current+next buffering.

`--representation instance-0=crf-28` sets initial manual quality.
`--control-trace controls.json` replays explicit manual actions by fresh frame
number for headless seek/switch regression. Records use `after_frame`, `action`
(`quality`, `seek`, `pause`, `play`, `loop`) and the corresponding instance/quality,
seconds, pause duration or loop-enabled fields. They do not run an adaptation solver.
`--visibility-every 30` samples the existing RGB-basis contribution algorithm via
its StreamSTGS CUDA tensor backend; this adds real GPU work. It does not use FAST
alpha or claim exact FAST visibility. Camera prediction is explicitly current-pose
hold. Planner contracts are available; the runtime selector is manual, with no
DP/Lagrangian implementation.

## Measurements and comparisons

```bash
python -m pytest -q tests_final
python tests_final/gpu/validate_publication.py \
  --mpd output/published/scene-n2/manifest.mpd \
  --report output/validation/all-packages.json --render
python tests_final/gpu/validate_switching.py \
  --mpd output/published/scene-n2/manifest.mpd \
  --report output/validation/switching.json
python tests_final/gpu/validate_composition.py \
  --report output/validation/composition.json
python benchmarks/streaming/run_matrix.py \
  --archive data_stream/streamstgs_soldier_536_835_crf20_28_36_40.tar.gz \
  --archive data_stream/streamstgs_longdress_1051_1350_crf20_28_36_40.tar.gz \
  --output output/benchmarks/new-run --objects 1 2 4 \
  --mbps 20 80 --workers 1 2 --prefetch 0 1 --frames 300
```

The server also supports `--mbps 80 --latency-ms 20` or a JSON `--trace` list of
`{at_s,bps,latency_s}` points. This is shared application-body pacing, not a kernel
network emulator: it does not model packet loss, QUIC congestion or TCP internals.
Concurrent request rates are not summed as bandwidth. Completed busy groups feed
an explicitly aggregate goodput observation; metadata and failed bytes are kept
in the ledger.

Reports separate cold startup, rebuffer, stage latencies, actual transferred bytes,
compute time, wall completion deadlines and unique scene frames. Frame-swap or EGL
completion is not physical display measurement. Reproduce runs using fixed scene,
camera, quality, cache and pacing settings before comparing frameworks. Codec
interfaces and shared event schemas allow later adapters; legacy codecs remain
separate comparison implementations.

Preparation provides explicit stage argv/journal/resume and calibrated camera
conversion/validation utilities. Full 8i training requires raw PLY/synthetic
multiview inputs and external pinned training source; deployment archives exclude
those inputs. No full retraining was launched by this implementation.

## Validation record and remaining checks

Run artifacts are retained under `output/src_final_validation/`; the source tree
contains tests/harnesses rather than generated media or copied environments.
A baseline hash ledger verifies preserved legacy source. The GPU tests use the
available GTX 1660, Python 3.10/Torch 1.13.1 runtime, plus isolated new dependencies.
Other GPU architectures and Windows require their own build/smoke qualification.

Native physical display interaction remains a separate on-site check when a
DISPLAY/Wayland session is available. Ground-truth visual seams and measured RD
utility cannot be certified from deployment archives alone. No universal 30 FPS,
novelty or cross-codec quality claim follows from these functional checks.

Functional results and their scope are recorded in [SRC_FINAL_VALIDATION.md](SRC_FINAL_VALIDATION.md).

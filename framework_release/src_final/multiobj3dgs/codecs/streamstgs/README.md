# StreamSTGS inference runtime

This adapter reuses the pinned StreamSTGS model, normalization, temporal feature
window, and MLP reconstruction. Atlas tiles follow the encoded picture's row-major
layout, validated against `atlas_rows`/`atlas_columns` when declared. Both 4×3 and
3×4 tile layouts retain their spatial pixel order; the entire picture must not be
transposed to fit the upstream model's default layout.
It consumes a verified materialized GOP, independent of publication layout.

The historical Soldier trial transposed 3×4 pictures, corrupting channel and
Gaussian identity. Matching that trial was not a correctness oracle. See
`docs/NATIVE_VIEWER_FIX.md` for independent picture/reference checks and native
framebuffer regression results from 2026-10-24.

## Setup

Use Linux with a working NVIDIA driver and CUDA-capable PyTorch. Install PyTorch
for the local driver/toolkit in an isolated environment first, then from the
repository root:

```sh
python -m pip install -e '.[network,content,streamstgs,native,test]'
python scripts/setup_streamstgs.py
python -m pip install --no-deps --no-build-isolation third_party/streamstgs/submodules/fast-gaussian-rasterization
# Required for CUDA contribution visibility, separate from FAST presentation:
python -m pip install --no-deps --no-build-isolation third_party/streamstgs/submodules/diff-gaussian-rasterization
```

The CUDA extension build needs a compatible CUDA toolkit (`nvcc`), C++ compiler,
and PyTorch headers. Choose `TORCH_CUDA_ARCH_LIST` for the GPU(s) to deploy; the
project does not force SM 7.5. The installed Torch build, toolkit, driver, and
`cuda-python` must be compatible. FAST currently uses `from cuda import cudart`;
`cuda-python` is constrained below 13. FFmpeg/ffprobe with HEVC decode and
imagecodecs with JPEG XL support are required. Native controls additionally use
GLFW, imgui, PyOpenGL, PyGLM and Rich.

For an offline source setup:

```sh
python scripts/setup_streamstgs.py --source /path/to/StreamSTGS-git-clone
```

For a wheel installed outside the checkout, set
`MULTIOBJ3DGS_VENDOR_ROOT=/absolute/path/to/third_party/streamstgs`, or supply
`vendor_root` to the adapter and renderer. Vendor sources intentionally are not
embedded in the wheel; their upstream licenses and notices must accompany them.
See repository `third_party/README.md` and `patches/README.md`.

## Ownership and stages

- `estimate_memory(package)` returns conservative raw/host/GPU byte reservations.
- `prepare_cpu(package, cancel=None)` uses CPU HEVC/JXL decode, exact released
  normalization/tile ordering, and CPU-only weight deserialization. It never
  constructs a CUDA model. A worker must reserve memory before calling it.
- `realize_gpu(prepared)` constructs a fresh model per instance/GOP, uploads
  tensors and binds weights. It records initialization, tensor allocation/upload,
  weight upload and total latency without resetting global GPU peak counters.
- `start_cpu(package, cancel, pool)` publishes canonical/weight host state while
  the persistent CPU worker fills temporal features. Reserve the complete host
  payload and retain its package lease until the producer future stops.
- `begin_gpu(prepared)` / `poll_gpu(pending, budget_s)` transfer bounded chunks on
  one copy stream through two reusable 8 MiB pinned buffers. The first handle is
  published after its first three feature copies complete; `frame_ready(handle,
  model_time)` gates each later window on CUDA completion events. Partial readiness
  retains full-GOP byte admission and independent GPU storage for every instance.
- `reconstruct(handle, model_time, camera)` returns activated tensors. Model time
  comes from explicit presentation mappings, not the number of temporal slots.
- `cancel_gpu(pending)` drains in-flight transfer work before dropping a model.
- `release(handle)` synchronizes its current render stream and drops the model.
  Call only after joint rendering has completed. No per-GOP `empty_cache()`.

GPU calls and FAST presentation must stay on one owner thread. Each CPU worker
enters an explicit inference context. Shared compressed assets may be cached;
mutable GPU models remain independent. Camera conditioning occurs in local
object coordinates; Gaussian transforms and the single joint draw occur after
reconstruction. Positive similarity transforms are supported; no device quality
caps or automatic quality reductions are imposed.

## Native versus headless

Use separate processes. Headless mode selects EGL before importing OpenGL.
Native mode requires an existing desktop GL context and `OpenGL.GL` imported
before FAST. Native rendering writes a clean scene FBO, blits to the native
backbuffer and returns a receipt without CPU image readback. `present_cached()`
restores this scene for idle UI swaps; camera/size changes require a fresh draw.
FAST uses scoped generic TorchScript execution to avoid shape-specialization
pauses when Gaussian counts change. A CUDA/GL device check rejects a context on
another GPU. FAST owns a process-wide singleton; a second renderer/context session
requires a new process. `clear()` handles scene intervals with no active objects.

## Validation evidence

The integration environment used Python 3.10, Torch 1.13.1+cu116, NumPy 1.26.4,
cuda-python 11.8.5 and the existing GTX 1660 development environment. Its historical
library paths were supplied only to validation subprocesses; production imports
use new source exclusively. Other hardware/software combinations require their
own build and capability check.

`tests_final/gpu/validate_streamstgs.py` runs legacy/new GOP parity in separate
processes. It checks exact SHA256 agreement of all decoded canonical and temporal
tensors, reconstructed state tolerance `1e-6`, and image tolerance `1/255 + 1e-6`
(the FAST framebuffer is uint8). Camera float32 rounding can shift a quantization
boundary. `tests_final/gpu/validate_publication.py` exercises all representations
and GOPs at first/middle/last presentation frames. Real validation artifacts are
under `output/src_final_validation/gpu/`, outside production source.

These are decode/offscreen correctness checks. They do not establish physical
native display FPS or guarantee scene deadlines for arbitrary object counts.

Current progressive/full-path GPU parity and native startup/playback measurements
on Torch 2.3.1+cu118 are documented in
`docs/STREAMSTGS_STARTUP_GOP_OPTIMIZATION.md`; the older environment above describes
historical validation. The native report includes cached-framebuffer regression,
per-instance timeline checks and bounded-memory tradeoffs.

# Renderer contribution and occlusion analysis

The Visibility Analyzer measures alpha-weighted contribution using the unchanged
Dynamic-LapisGS CUDA rasterizer. The RGB renderer, proxy generation, quality
estimation, LoD selection, bandwidth prediction, and scheduling are unchanged.

## Measurement and renderer semantics

For front-to-back splats at pixel `p`, contribution from object `m` is:

```text
T_i(p)       = product_(j<i) (1 - alpha_j(p))
W_m(p)       = sum_(i belonging to m) T_i(p) * alpha_i(p)
A_visible[m] = sum_p W_joint[m,p]
A_solo[m]    = sum_p W_solo[m,p]
occlusion[m] = 1 - A_visible[m] / A_solo[m]
```

Areas have alpha-weighted pixel units. They are neither binary coverage counts
nor distance-weighted scores. The solo render already includes self-occlusion;
the ratio measures suppression by the other objects. Resolution changes area
units, so compare joint and solo at the same resolution.

The upstream compositing kernel computes `color[channel] += color_i[channel] *
alpha_i * T`. In the analysis path, Gaussian colors are replaced with an
object's RGB basis vector and the background is black. The resulting channels
are object contribution maps accumulated directly by CUDA. No CUDA source or
extension rebuild is required and no CPU rasterizer is used.

Up to three objects need one joint pass. More objects use consecutive groups of
three channels. Every joint pass includes all scene geometry and opacity;
objects outside its measured group have zero color but still occlude other
objects. Each nonempty object also needs one solo pass with red basis colors.

Projection, Gaussian footprints, opacity, camera transforms, and splat depth
ordering use the normal renderer. Contributions inherit its numerical policy:

- Sort splats by Gaussian-center camera depth inside tiles.
- Clamp per-pixel alpha to `0.99` and skip alpha below `1/255`.
- Stop when the next transmittance would be below `1e-4`.
- Discard that threshold-crossing splat before accumulation, matching the
  existing kernel's early-exit behavior.

These renderer cutoffs truncate the mathematical infinite-tail contribution.
The analysis intentionally measures the compositing performed by that renderer.
Near-plane/frustum handling likewise comes from the rasterizer rather than a
separate CPU estimate.

## Runtime API and inputs

```python
import numpy as np
from src.client.visibility import analyze_visibility
from src.renderer.analysis import VisibilityObject
from tools.content_preparation.proxy import load_proxy

timestamp = 0.0
objects = [VisibilityObject(
    object_id="object_a",
    proxy=load_proxy("output/object_a/proxy/proxy.json", frame=0),
    timestamp=timestamp,
    object_to_world=np.eye(4),
)]
camera = {
    "world_view": np.eye(4).tolist(),
    "width": 256, "height": 256,
    "fovx": np.pi / 2, "fovy": np.pi / 2,
    "near": 0.01, "far": 100.0,
}
result = analyze_visibility(objects, camera, timestamp)
visibility = result.per_object["object_a"]
print(visibility.A_visible, visibility.A_solo,
      visibility.occlusion_ratio, visibility.out_of_view)
```

The example uses existing asset paths; substitute actual proxy descriptors.
Resolve the correct media frame before calling the runtime API. It accepts
timestamp-resolved `GaussianState` snapshots and does not load assets, infer
frame mappings, or evaluate animation. Each snapshot timestamp must match the
requested finite timestamp within `1e-9` seconds. Object IDs must be unique.
Gaussian IDs remain local to each object and may repeat between objects.

`GaussianState` uses raw `wxyz` rotations, log scales, logit opacities, and
`[N,3,(degree+1)^2]` SH attributes. The analyzer validates and copies the
snapshots, applies transforms once, and reuses the exact prepared tensors for
joint and solo passes. Transforms support translation, proper rotation, and
positive uniform scale. Shear, reflection, nonuniform scaling, singular
transforms, and invalid/nonfinite inputs are rejected.

`world_view` and `object_to_world` are ordinary nontransposed 4×4 matrices
acting on column vectors. Camera `+Z` points forward and image `+Y` downward.
The upstream adapter transposes camera matrices once at the boundary. FOV
values are radians. Object identity transforms default to `np.eye(4)`.

```python
analyze_visibility(
    objects, camera, timestamp, *,
    epsilon=1e-6,
    diagnostics=False,
    renderer_settings=None,
) -> VisibilityResult
```

The result contains `timestamp`, `resolution`, `epsilon`, `per_object`, optional
`diagnostics`, and backend/pass-count `metadata`. Each object result contains
`object_id`, `A_visible`, `A_solo`, `occlusion_ratio`, and `out_of_view`.

When `A_solo <= epsilon`, the object has no measurable contribution: set
`out_of_view=True` and `occlusion_ratio=None`. This can include an object outside
the view, an empty proxy, or opacity below the renderer cutoff. It is not 100%
occlusion. An object with positive solo contribution and negligible joint
contribution instead has high occlusion with `out_of_view=False`.

The client validates finite nonnegative areas and the bound `A_visible <=
A_solo`, with tolerance `1e-6 + 1e-5*A_solo`. Only numerical excursions within
that tolerance are clamped. Larger violations raise an error.

Normal execution reduces FP32 maps on the GPU using FP64 sums and transfers
only compact scalar measurements to the client. Temporary RGB-sized rasterizer
buffers are released after reduction. `diagnostics=True` additionally exports
CPU-owned joint and solo FP32 maps keyed by object ID; it is intended for
validation and is disabled online. Access them through
`result.diagnostics.joint_maps[object_id]` and
`result.diagnostics.solo_maps[object_id]`; each map has shape `(height, width)`.
Result `resolution` is `(width, height)`. There is no depth-map output in this version.
A scene containing no Gaussians returns zero areas without CUDA, including an
empty scene. Scenes with Gaussians require the configured CUDA backend, with no
automatic approximate fallback.

`renderer_settings` accepts an `extension_path` for the installed rasterizer,
`debug`, `scaling_modifier` (default `1.0`), and `backend` (`upstream_cuda` only).
`VISIBILITY_EXTENSION_PATH` can
select the installed package for tests and the benchmark. Keep the binary built
from the pinned source identified in [DEPENDENCIES.md](DEPENDENCIES.md).

The public API in `src/client/visibility.py` computes ratio/status only.
`src/renderer/analysis.py` performs joint/solo orchestration and GPU reductions;
`src/renderer/upstream.py` prepares camera/models and calls the upstream renderer
under the existing GPU serialization guard.

## Correctness validation

CPU contract tests cover epsilon/status handling, invalid inputs, snapshot
consistency, unsupported transforms, diagnostics, and explicit backend failure.
CUDA fixtures check the actual compositing backend:

| Scene | Required behavior |
|---|---|
| One object with self-overlap | Joint equals solo; occlusion approximately zero. |
| Two disjoint footprints | Both objects have approximately zero occlusion. |
| High-alpha foreground covering a rear object | Rear occlusion exceeds `0.95`, with positive solo area. |
| Partial overlap | Rear occlusion lies between `0.05` and `0.9`. |
| Outside image / behind camera / negligible alpha | Solo area is below epsilon; ratio is undefined. |
| Exchanged near/far placement | Contributions and occlusion follow the new depth order. |
| More than three objects | Off-group foreground objects still suppress measured objects. |
| Interleaved splat depths | Ordering is per splat, not per object. |
| Translation, rotation, uniform scale | Joint and solo share the transformed footprint. |

Additional CUDA checks cover repeated local IDs, identity-transform parity,
alpha clamp/cutoffs, and early exit. Constant per-object RGB fixtures reconstruct
the normal composite, including a nonblack background:

```text
RGB(p) = sum_m color[m] * W_joint[m,p]
         + background * (1 - sum_m W_joint[m,p])
```

Compare raw float maps before PNG conversion or clipping using initial
`atol=2e-6`, `rtol=2e-5`; zero-occlusion tolerance is `2e-5`. Sum joint maps and
compare with white-on-black alpha rendering and matching rasterizer radii.
Constant object colors are deliberate: object-level maps do not reconstruct
arbitrary per-Gaussian varying colors.

Run CPU checks from the project root, then explicitly enable GPU tests:

```bash
python -m pytest -q tests/test_visibility.py tests/test_benchmark_visibility.py tests/renderer/test_visibility_contribution.py
VISIBILITY_CUDA_TESTS=1 bash scripts/run_gpu0.sh python -m pytest -q tests/renderer/test_visibility_contribution.py
```

If the extension lives outside the environment's normal package directory, set
`VISIBILITY_EXTENSION_PATH=/absolute/directory/containing/package` before the
second command. CPU checks and skipped CUDA tests do not establish renderer
correctness.

## Performance baseline

`tools/benchmark_visibility.py` reports JSON and a readable table for RGB,
each joint group and the total, each solo pass and the total, reductions plus
scalar transfer, and complete API execution. RGB and analysis use identical
proxy geometry and camera/timestamp/transforms. This comparison does not imply a
full display-quality RGB scene has the same cost.

Defaults are 10 warmups and 100 samples, object counts `1,3,6`, and square
resolutions `128,256,512`. Pass timing uses CUDA events. Complete API timing uses
synchronized wall time and includes validation, snapshot preparation, GPU upload,
joint/solo passes, reductions, and scalar transfer. Optional diagnostic export
is measured separately via `--diagnostics`.

```bash
bash scripts/run_gpu0.sh python tools/benchmark_visibility.py --output output/visibility_benchmark.json
bash scripts/run_gpu0.sh python tools/benchmark_visibility.py --scene /absolute/path/visibility_scene.json --output output/visibility_with_assets.json
```

Each existing-proxy scene JSON provides explicit frame, timestamp, transform,
and camera mappings. Proxy paths resolve relative to the scene JSON. Example:

```json
{
  "name": "existing_proxies",
  "timestamp": 0.0,
  "camera": {
    "world_view": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    "width": 256, "height": 256,
    "fovx": 1.5707963267948966, "fovy": 1.5707963267948966,
    "near": 0.01, "far": 100.0
  },
  "objects": [{
    "object_id": "object_a",
    "proxy_path": "object_a/proxy/proxy.json",
    "frame": 0,
    "timestamp": 0.0,
    "object_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
  }]
}
```

Add the other scene objects with their actual transforms. Benchmark counts use
the first `N` supplied objects; counts exceeding available objects are recorded
as skipped. Synthetic fixtures always cover the requested counts. Use
`--no-synthetic --objects 1 3` to measure only supplied three-object assets.
Resolutions override camera image dimensions while preserving its view matrix
and FOV; every pass within a case uses that same camera.

The report contains mean/median/nearest-rank p95 milliseconds, object and
Gaussian counts, resolution, actual pass counts, prepared-input memory baseline,
phase peak allocated/reserved bytes and incremental peaks, GPU name, runtime
versions, source commits, and rasterizer wrapper/binary paths and SHA-256 hashes.
Peak allocated/reserved memory is measured through PyTorch's CUDA allocator and
does not represent total device memory used by unrelated processes.

The isolated reduction measurement stages maps solely to separate reduction and
transfer timing from rasterization. Its phase memory baseline includes those
maps; normal analysis reduces and releases one pass at a time. Output references
are released between phases, and prepared GPU models are released before public
API measurements. The benchmark does not call `empty_cache` inside timing loops;
reserved memory therefore includes allocator caches from earlier phases.

GPU correctness and timing results are pending. On the implementation machine,
NVIDIA reports a driver/library mismatch (`NVML 595.99`) and the configured
PyTorch environment reports CUDA unavailable. The benchmark fails explicitly
without producing measurements in that condition. Run the CUDA checks and the
synthetic/existing-asset benchmark in a working pinned CUDA environment before
accepting a performance baseline; no timing or GPU-memory values are inferred.

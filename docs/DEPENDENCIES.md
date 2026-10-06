# Dependency and third-party setup

## Project Python dependencies

The Request Handler uses `curl_cffi` for reusable asynchronous HTTP sessions,
HTTP/2, HTTP/3, Range headers, and transport timing. Tests use `pytest` and
`pytest-asyncio`. `requirements.txt` lists these three direct dependencies only;
it is not a dump of the existing experiment environment.

The environment was inspected before installation on 2026-10-05:

| Component | Initial state in conda `Hoang` | Tested version |
| --- | --- | --- |
| Python | Installed | 3.11.16 |
| curl_cffi | Absent | 0.16.3 |
| pytest | Absent | 9.1.1 |
| pytest-asyncio | Absent | 1.4.0 |
| PyYAML | Installed | 6.0.3; not required by current transport |
| lxml | Absent | Not installed; MPD parsing is out of scope |

The pins record the versions installed for this implementation after checking
the actual environment and dependency resolver. No existing packages were
uninstalled or upgraded. The installation added the small transitive packages
`cffi==2.1.1`, `pycparser==3.0`, `iniconfig==2.3.0`, and `pluggy==1.6.0`.
Existing `certifi`, `packaging`, `pygments`, and `typing-extensions` satisfied
the remaining requirements. The transport's config example is documentation;
it does not introduce a YAML parser dependency. Add XML/config dependencies
when project code actually consumes them.

```bash
conda activate Hoang
python -m pip install -r requirements.txt
python -m pytest
```

The inspected interpreter is
`/home/fil/miniconda3/envs/Hoang/bin/python`. No Dynamic-LapisGS Python/CUDA
packages or Draco build tools were installed by this setup.

## HTTP capability of the tested wheel

The installed `curl_cffi` wheel reports:

```text
libcurl/8.21.0-IMPERSONATE BoringSSL zlib/1.3.1 brotli/1.2.0
zstd/1.5.7 libidn2/2.3.7 nghttp2/1.63.0 ngtcp2/1.20.0 nghttp3/1.15.0
```

`nghttp2` provides the compiled HTTP/2 backend. `ngtcp2` and `nghttp3`
provide the compiled HTTP/3 backend. `CurlHttpVersion` exposes `V2_0`,
`V2TLS`, `V2_PRIOR_KNOWLEDGE`, `V3`, and `V3ONLY`. This confirms local
backend availability; it does not claim that the offline HTTP/1.1 test server
negotiates HTTP/2 or HTTP/3. HTTP/3 transfers also require a suitable HTTPS
server and working UDP/QUIC connectivity.

To inspect a different environment without making a network request:

```bash
python -c 'from curl_cffi import Curl; c = Curl(); print(c.version().decode()); c.close()'
```

See [REQUEST_HANDLER.md](REQUEST_HANDLER.md) for protocol configuration,
negotiated protocol reporting, and transfer validation.

## Git submodules

The project directory initially had no `.git`, and the parent workspace's
unrelated `360vViaMoQ` repository reported the entire project as untracked.
An independent project Git repository was initialized on branch `main`;
the parent repository, its existing user changes, and sibling codebases were
left untouched. `third_party/` was empty before adding the submodules.

| Path | Upstream | Pinned Git commit | Intended future use |
| --- | --- | --- | --- |
| `third_party/dynamic-lapis-gs` | https://github.com/nus-vv-streams/dynamic-lapis-gs | `da8efaa42a9d021b5ff2958c0a87eba6ed7a47c4` | Gaussian representation, original renderer, checkpoint/model compatibility |
| `third_party/draco` | https://github.com/google/draco | `15bdb3a4f15a7a8d77489ac348a7a50d93de17a3` | Codec backend and content preparation |

The project Git index records these commits as gitlinks, and each checkout is
detached at its recorded commit. The initial clones use `--depth 1` to avoid
downloading unnecessary upstream history. No upstream source was modified and
neither project was built. During the content-preparation migration,
Dynamic-LapisGS's existing nested rasterizer, simple-knn, and GLM submodules
were initialized at their upstream gitlink pins; their source was left clean.

After cloning a committed project checkout, initialize everything needed for
future renderer work with:

```bash
git submodule update --init --recursive
git submodule status
```

This uses the committed gitlink pins; do not use `git submodule update --remote`
when reproducing an experiment. An upstream bump should update the gitlink and
this note together. If full upstream history is needed, run `git fetch
--unshallow` inside the relevant submodule; the current commit pin remains
unchanged until explicitly updated.

## Offline content preparation

The project owns its preparation tools, scripts, and configs. The external
Gaussian model, original renderer, training entry point, and LPIPS implementation
come from the pinned `third_party/dynamic-lapis-gs` checkout. A second whole
`gaussian-splatting` repository is unnecessary: Dynamic-LapisGS already carries
that model/renderer API and its required 3DGS extensions. Using its exact nested
gitlinks keeps the CUDA extension source aligned with the backend.

| Nested dependency | Upstream | Pinned Git commit |
| --- | --- | --- |
| `third_party/dynamic-lapis-gs/submodules/diff-gaussian-rasterization` | https://github.com/graphdeco-inria/diff-gaussian-rasterization | `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d` |
| `third_party/dynamic-lapis-gs/submodules/simple-knn` | https://gitlab.inria.fr/bkerbl/simple-knn | `86710c2d4b46680c02301765dd79e465819c8f19` |
| `third_party/dynamic-lapis-gs/submodules/diff-gaussian-rasterization/third_party/glm` | https://github.com/g-truc/glm | `5c46b9c07008ae65cb81ab79cd677ecc1934b903` |

These nested pins are already committed by their containing upstream
repositories; the top-level Dynamic-LapisGS gitlink did not change. To fetch only
the backend needed by the offline tools:

```bash
git submodule update --init --recursive third_party/dynamic-lapis-gs
git submodule status --recursive third_party/dynamic-lapis-gs
```

`requirements-content-preparation.txt` is optional and contains only direct
non-PyTorch dependencies used by the migrated tools or the upstream entry points.
It is separate from the transport requirements and is not the old
`requirements-hoang.txt` environment dump. Versions below were read from installed
distribution metadata in `Hoang` on 2026-10-05. No package was installed, upgraded,
or removed for this migration.

| Direct dependency | Inspected version | Use |
| --- | --- | --- |
| NumPy | 2.4.6 | Gaussian arrays, payloads, numerical checks |
| PyYAML | 6.0.3 | Preparation configs |
| Pillow | 11.3.0 | Image loading/resizing and reports |
| plyfile | 0.8.1 | PLY checkpoints and Gaussian data |
| SciPy | 1.17.1 | Rotation conversion in coding experiments |
| Open3D | 0.19.0 | Raw point-cloud preprocessing and offscreen rendering |
| opencv-python-headless | 5.0.0.93 | `cv2` imported by upstream `train.py` |
| tqdm | 4.70.1 | Upstream training progress |
| PyTorch | 2.3.1+cu118 | Separately managed training, rendering, and metrics |
| torchvision | 0.18.1+cu118 | Separately managed pretrained LPIPS backbones |

```bash
conda activate Hoang
python -m pip install -r requirements-content-preparation.txt
```

PyTorch/torchvision are deliberately documented rather than installed by that
file. For a new GPU machine, provision a matching PyTorch/CUDA/toolchain environment
before building the pinned extension source in an **isolated environment**.
The inspected host uses Python 3.11.16 and CUDA 11.8; the upstream
`environment.yml` still describes an older Python 3.7/PyTorch 1.12 stack and is
not the tested `Hoang` setup. Choose `TORCH_CUDA_ARCH_LIST` for the target GPU;
the local GTX 1660 uses `7.5`. For an isolated environment that already has its
chosen compatible PyTorch and CUDA compiler, the extension build commands are:

```bash
python -m pip install --no-build-isolation --no-deps \
  ./third_party/dynamic-lapis-gs/submodules/simple-knn \
  ./third_party/dynamic-lapis-gs/submodules/diff-gaussian-rasterization
```

No build was run during migration. Installed `Hoang` extension metadata points
to previously built sibling Scaffold-GS extensions (`diff_gaussian_rasterization`
and `simple_knn`, each version `0.0.0`). They remain untouched. Reinstalling the
upstream rasterizer in that shared environment would remove Scaffold-GS's extra
`visible_filter` API; use an isolated environment for a new build. Metadata
confirms installation and provenance, not GPU execution or binary compatibility
on another machine. Offline CPU fixture tests do not validate CUDA training,
rendering, Open3D/EGL availability, or shared extension compatibility.

LPIPS is imported from upstream `lpipsPyTorch`, using torchvision and the official
backbone/linear weight cache under `torch.hub.get_dir()/checkpoints`. The separate
PyPI `lpips` package is not needed. By default missing weights produce a clear
error; downloads require the explicit metrics config opt-in. Dataset archives,
model checkpoints, pretrained weight caches, generated payloads, and compiled
CUDA binaries are runtime artifacts, not vendored source or files to push.

The sibling checkout had two Python compatibility fixes beyond the upstream pin:
Open3D's removed `headless=False` constructor argument and the NeRF image reader's
signed `np.byte` conversion (the local fix uses `np.uint8`). The migrated
project-owned compatibility adapter preserves these fixes without editing the
pinned submodule. Preparation provenance must identify the adapter as well as
the upstream source; a clean submodule does not imply unmodified execution.

Draco remains a separately pinned future codec backend. The migrated tools'
current payload adapter uses NumPy/zlib and does not import or build Draco.
Its nested build/test dependencies can stay uninitialized until a Draco build
is actually needed.

StreamSTGS inference sources are pinned at `ca8602516352efd83e9e69e665b2e26ed03cb5cc`.
`streamstgs/manifest.json` records original upstream bytes, nested source archive
hashes, and patched file hashes. Reproduce them with:

```sh
python scripts/setup_streamstgs.py
# Or use an existing upstream Git clone without network access:
python scripts/setup_streamstgs.py --source /path/to/StreamSTGS
```

This is an inference-only subset, not a training distribution. The GaussianModel,
TimeFormer, camera/math helpers, FAST, and CUDA rasterization sources retain their
upstream licenses. StreamSTGS/Graphdeco sources carry non-commercial research
restrictions; consult `streamstgs/LICENSE.md` and the nested rasterizer licenses
before redistribution or other use. FAST's license is in its source directory;
GLM retains its `copying.txt`. No dataset, model weights, environment or compiled
extension is included.

The historical atlas-layout and FAST-background patches are preserved under
`patches/`. The import patch isolates upstream `scene`/`utils` in a private
namespace and defers training-only dependencies. It does not change inference
math. Native and EGL bootstrap must run in separate processes. CUDA extensions
must be built for the target GPU architecture; no fixed architecture is supplied.

Core/network/content parsing do not import this vendor tree. A packaged wheel can
use `MULTIOBJ3DGS_VENDOR_ROOT=/absolute/path/to/streamstgs` to locate source outside
the editable checkout. Install the viewer dependency group and build the optional
CUDA rasterizer using the commands printed by the setup script.

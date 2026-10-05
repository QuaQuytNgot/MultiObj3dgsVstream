# Multi-object adaptive streaming of dynamic 3D Gaussian Splatting

The current implementation provides an async network transport block. MPD parsing,
LoD selection, visibility, bandwidth prediction, codecs, and renderer integration
are reserved for subsequent work.

## Install

Replace `<PROJECT_REPOSITORY_URL>` with this project's Git URL:

```bash
git clone --recursive <PROJECT_REPOSITORY_URL> MultiObj3dgsVstream
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

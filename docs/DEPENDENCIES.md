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
neither project was built. Dynamic-LapisGS's nested rasterizer and simple-knn
submodules are intentionally uninitialized in this transport-only checkout.

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

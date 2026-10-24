# StreamSTGS content publication

Publish existing archives into a **new** directory. Repeat `--archive` to include
independent contents; `--instances` creates any positive number of scene instances
using those contents in round-robin order. Install the repository package first,
or set `PYTHONPATH=src_final` when running from the checkout.

```bash
python -m multiobj3dgs.content audit data_stream/*.tar.gz
python -m multiobj3dgs.content publish \
  --archive data_stream/streamstgs_soldier_536_835_crf20_28_36_40.tar.gz \
  --archive data_stream/streamstgs_longdress_1051_1350_crf20_28_36_40.tar.gz \
  --instances 4 --output output/publication-v1
python -m multiobj3dgs.content validate output/publication-v1
```

Every source archive requires its `.sha256` sidecar. The importer checks every
file recorded in `stream_package_manifest.json`, rejects links/traversal/duplicate
members, and preserves codec files byte for byte. Publishing uses a sibling
staging directory and exposes the final directory only after validation. A retry
with identical input verifies and reuses the immutable publication; different
content or scene settings require a different output directory. Source archives,
trial directories and existing publications are never modified.

For custom transforms or timeline offsets, use `--scene scene.json` instead of
`--instances`. The scene declares `duration`, an `instances` list, and optional
`pivots` mapping content IDs to stable XYZ pivots. Each instance accepts `id`,
`content_id`, `translation`, normalized WXYZ `rotation`, positive uniform `scale`,
`scene_start`, `media_offset`, `active_duration` and `loop`. The default pivot is
the object-local origin; measure and specify a stable pivot when content requires
one. A nonlooping active interval cannot extend beyond available media. There is
no object-count or quality cap.

`manifest.mpd` contains one AdaptationSet per content and independent CRF
representations. `index.json` is the normalized runtime catalog, checksummed from
the MPD and carrying its own deterministic revision. Codec-specific details remain
in the original reconstruction JSON and associated GOP assets. Playback assets
shared by actual hash are stored under `assets/`; representation-specific temporal
video, reconstruction JSON and local manifest are in deterministic CPSEG1 GOP
bundles. This project DASH profile requires the StreamSTGS client adapter and is
not generic HEVC DASH playback. No progressive representation dependency is used.

`bandwidth` describes the ceiling of the peak cold resource-closure rate (including
container overhead). It is not a utility score. Runtime missing-byte cost must
union cached resources by SHA256; independent instances reuse compressed bytes
without sharing mutable decoder models.

```python
from pathlib import Path
from multiobj3dgs.client.metadata import MPDParser
from multiobj3dgs.content.streamstgs import materialize_gop

catalog = MPDParser.from_local('output/publication-v1/manifest.mpd')
content = catalog.contents['soldier']
segment = content.segment_at('crf-28', 2.0)
frame = segment.frame_at(2.1)  # explicit source frame and model_time
package = materialize_gop(
    segment,
    {ref.id: Path(ref.url) for ref in segment.resources},
    Path('/tmp/decoder-package-not-yet-existing'),
)
```

`await MPDParser.from_http(url, transfer_client)` uses the existing transport and
resolves publication-relative URLs. The parser checks sidecar hash/revision and
exact MPD/index semantic agreement. Core imports are stdlib-only. The optional
offline XSD check uses `xmllint --nonet` and packaged, checksummed MPEG schemas:

```python
from multiobj3dgs.content.publication.schema import validate_mpd_schema
validate_mpd_schema('output/publication-v1/manifest.mpd')
```

Run `python -m pytest tests_final/content -q`. The original progressive packager
and schemas remain intact. CPSEG framing is extracted from the existing writer;
this new profile supplies GOP semantics rather than modifying the old profile.

## External preparation stages and calibration

The existing Dynamic-Lapis preparation commands remain available. For a portable
StreamSTGS workflow, use the explicit stage runner without importing either
trainer into playback:

```bash
python -m multiobj3dgs.content.preparation run stages.json --dry-run
python -m multiobj3dgs.content.preparation run stages.json --resume
python -m multiobj3dgs.content.preparation calibration cameras.json \
  --poses-bounds /new/scene/poses_bounds.npy
```

`stages.json` has schema `multiobj3dgs.preparation.v1`, a `journal` path, and an
ordered `stages` list. Paths are relative to the config file. A stage supplies
`name`, `argv` (a list, never shell text), `cwd`, `inputs`, `outputs`, optional
`env`, `pinned_inputs` (path→SHA256) and `training` boolean. Training stages must
pin their invoked Python entrypoint. Declare the trainer checkout, calibration,
input data and dependency lockfiles as inputs when those affect reproducibility.
The journal records command/interpreter hashes, input/output hashes, status,
return code and a separate log for each execution. Environment overrides are
fingerprinted without exposing their values in the journal. Resume reuses only
matching, hash-verified completed outputs. Existing changed/partial outputs are
preserved; use new output paths for recomputation. Dry-run does not execute
commands or create files and labels inputs supplied by future stages as pending.
The runner does not supply an unverified training recipe or launch a full train.

`cameras.json` contains `cameras`, `near_fars`, and `observations`. Each camera
requires explicit `convention: "opencv"`, `width`, `height`, full 3×3 `K`, and 4×4
`w2c`. OpenCV camera axes are +X right, +Y down, +Z forward, with transforms acting
on column vectors. Each observation names a zero-based `camera` index and matching
`world_points`/`image_points`. The optional `tolerance_pixels` defaults to 0.01.
All cameras must have measured reprojection observations before CLI export.
The validator supports full pinhole intrinsics and checks rigid/right-handed
transforms and positive point depths.

The pinned StreamSTGS NDC loader supports a narrower calibration profile: shared
image dimensions/focal length, centered principal points, square pixels and zero
skew. The poses-bounds converter explicitly rejects cameras outside that profile
instead of dropping their calibration. It emits LLFF rows with columns
`[C2W_y, C2W_x, -C2W_z, camera_center, H/W/focal]` plus explicit near/far values;
a regression executes the pinned loader's actual pose code to check the recovered
OpenCV rotation/translation. `blender_to_opencv()` converts Blender c2w transforms
by flipping camera Y/Z while preserving world axes. No per-frame recentering or
implicit scene scaling occurs. Camera order must match `camNNN.mp4` naming.
Video encoding, multiview generation and training remain explicit external stages
that must be configured and validated for the target dataset.

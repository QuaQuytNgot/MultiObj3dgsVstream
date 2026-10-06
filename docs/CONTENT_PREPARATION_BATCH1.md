# Content preparation — first implementation batch

Scope: public Draco bridge, codec dispatch, proxy-off orchestration, artifact
validator, static MPD and decoded Gaussian viewer. No native training pilot or
full Longdress run belongs to this batch. No proxy generation, proxy validation,
refresh ablation, online adaptation or networking implementation is added.

Validated on **2026-10-06**: 139 self-tests passed without skips; actual two-frame
Longdress smoke, offline MPD/XSD, decoded PLY equality and 4/4 WebGL conditions
passed. Pipeline/MPD/viewer/validator resumes executed zero tasks. See
[machine-readable evidence](validation/content_preparation_batch1_smoke_summary.json)
and [readiness](CONTENT_PREPARATION_READINESS.md). No full run was started.

## Reused infrastructure

`tools/content_preparation/prepare_content.py` remains the pipeline entry point.
It reuses existing export, sparse absolute refinement, CPSEG packaging,
segment-member decoding, original CUDA rendering, real pretrained LPIPS,
view-conditioned profiles, gains, manifests and per-task journals.

`tools/content_trial_viewer/index.html` is extended in place. Historical PNG
catalog v1 remains readable; catalog v2 adds object selection and a free-camera
Gaussian viewport. The browser loads decoded PLY, not training checkpoints.

## Codec boundary

`gaussian_attribute_draco_byteplanes` 1.0.0 uses public Draco 1.5.7 at commit
`15bdb3a4f15a7a8d77489ac348a7a50d93de17a3`. Project native bridge
`tools/content_preparation/native/draco_byte_codec.cpp` carries exact Gaussian
attribute/ID bytes using sequential `GENERIC DT_UINT8` attributes, at most 64
components per attribute, no prediction, quantization or deduplication.

The outer project packet is `CPDRAC01`; its JSON header describes shapes,
stream hashes, input/decoded/parent state hashes, codec parameters and native
binary/compiler/source identity. f32 is lossless. The legacy `CPGAUS01` zlib
codec remains supported. Actual packet and complete CPSEG file sizes include
all metadata and identity bytes.

Enhancements are sparse absolute replacements against the **decoded** previous
quality of the same frame. They include changed shared attributes, new and
deleted Gaussian IDs, row order and SH schema changes. They are not append-only.
There is no temporal prediction. Every Base frame is self-contained; every
enhancement requires its same-frame decoded parent. GoF/refresh labels and
network segment cuts do not create video prediction dependencies.

This is **new project code using verified public Draco**. It is not the enhanced
Draco encoder of LTS, whose implementation source is unavailable. Reference:
[official LTS repository](https://github.com/AIINS-NTHU/LTS-DASH-Streaming-System-for-3DGS).

## Exact bounded execution order

Run from the project root in the environment from
[the clean-machine setup](DEPENDENCIES.md#clean-machine-setup-rtx-5060-ti). The
smoke config imports Q0/Q1 checkpoints for frames 1051–1052 from
`output/imports/longdress_4level_training/manifest.json`. This bundle is not in
Git: transfer the manifest and every referenced checkpoint from the previous
machine, preserve its frame/quality mapping, update checkpoint file paths to the
copied relative paths if they were absolute, and keep the original `commands`
and SHA-256 provenance unchanged. The four-quality visual config uses the same
import location. `runtime.extension_path: null` uses the pinned upstream CUDA
extensions installed in the active environment; do not copy GTX 1660 binaries.
Without the checkpoint bundle, run the self-test and dry-run only, or complete
the separately documented native pilot. Do not replace missing smoke inputs by
starting full training.

1. Build the CPU-only native bridge from the pinned submodule:

   ```bash
   cmake -S tools/content_preparation/native \
     -B output/build/content_preparation -DCMAKE_BUILD_TYPE=Release
   cmake --build output/build/content_preparation --target draco_byte_codec -j2
   output/build/content_preparation/draco_byte_codec --version
   ```

   Check reported Draco commit/version and sequential byte-attribute policy.

2. Run relevant tests. Chromium/Playwright are optional local browser-test
   dependencies; strict coverage requires them and local socket permission.

   ```bash
   OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
     python tools/content_preparation/self_test.py --strict-optional
   python tools/content_trial_viewer/vendor_dependencies.py \
     --lock tools/content_trial_viewer/dependencies.lock.json
   python tools/content_trial_viewer/vendor_dependencies.py --verify
   python tools/content_preparation/prepare_content.py \
     --config configs/content_preparation/smoke.yaml --dry-run
   ```

   Dry-run must show backend `existing`, only progressive mode, two qualities,
   two frames and no proxy stage. Do not substitute a full-run config.

3. Optional bounded preflight before creating the run:

   ```bash
   OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
     python tools/validate_prepared_content.py \
     --config configs/content_preparation/smoke.yaml \
     --stage environment --resume
   ```

   This checks native identity, a tiny configured-renderer render and real
   same-image LPIPS. Preflight reports outside an unowned run directory, so it
   does not interfere with pipeline ownership protection.

4. Execute only the bounded checkpoint-import smoke:

   ```bash
   OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
     python tools/content_preparation/prepare_content.py \
     --config configs/content_preparation/smoke.yaml --stage all --resume
   ```

   Expected: 4 exported/encoded/decoded states, 4 segments, 12 rendered metric
   samples and 6 transition gains. Reference is decoded Q1 in this two-level
   smoke. The train stage records existing checkpoints and original commands;
   it does not run a trainer. There is no `longdress/proxy/` generation.

5. Produce and validate MPD/sidecars from actual server manifests:

   ```bash
   python tools/export_content_mpd.py \
     --prepared-root output/content_preparation/smoke --resume
   ```

   Expected: `manifest.mpd`, `content_index.json`, `mpd_validation.json`.
   MPD has object AdaptationSets, layer Representations, transitive `dependencyId`,
   actual SegmentLists/SegmentTimelines and cumulative peak bandwidth. Detailed
   exact bytes/access/provenance stay in the JSON sidecar. It is an experimental
   Gaussian/CPSEG profile using `application/octet-stream`; ordinary DASH video
   players cannot play it. Validation uses pinned offline MPEG MPD XSD with
   `xmllint --nonet`, plus checks against actual files and deterministic sources.

   MPD `bandwidth` is a uint32 field. If the exact cumulative rate exceeds its
   maximum, the standard field is explicitly saturated and a mandatory versioned
   `extended-bandwidth-bits-per-second` EssentialProperty carries the exact rate.
   JSON records both values and `bandwidth_saturated`; consumers of this project
   profile must honor the extended rate. This does not change actual media bytes,
   FPS or representation quality and is not ordinary DASH interoperability.

6. Export decoded Gaussian viewer assets:

   ```bash
   python tools/export_decoded_viewer_assets.py \
     --prepared-root output/content_preparation/smoke \
     --mode progressive --frames all --resume
   ```

   Expected: 4 binary little-endian Gaussian PLYs in
   `viewer_assets/longdress/{Q0,Q1}/{1051,1052}.ply`, asset index, catalog,
   local UI/dependencies and 12 native-render PNG anchors. PLY numeric hashes
   must equal decoded NPZ hashes. Display PLY bytes are excluded from media rate.

7. Launch the existing viewer through a simple local server:

   ```bash
   python -m http.server 8766 --bind 127.0.0.1 \
     --directory output/content_preparation/smoke
   ```

   Open `http://127.0.0.1:8766/?object=longdress&quality=Q1&display=gaussians`.
   Select each quality and frame; rotate, pan and zoom. Confirm the displayed
   object/quality/frame/count/hash, camera preservation when switching,
   upright content and no blank/missing assets. PNG mode shows the objective
   upstream-renderer samples for comparison.

8. With that server running, record real browser evidence:

   ```bash
   python tools/validate_trial_viewer.py \
     --url 'http://127.0.0.1:8766/?display=gaussians' \
     --display gaussians --require-webgl --timeout-ms 120000 \
     --output output/content_preparation/smoke/viewer_validation.json
   ```

   Require all four decoded assets checked, nonblack Gaussian pixels, no
   JS/HTTP errors, working navigation, one resident mesh and matching hashes.
   The smoke browser gate is automated; personal visual acceptance still
   requires opening the viewer.

9. Audit all artifacts, including independently re-decoding real segment bytes:

   ```bash
   OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
     python tools/validate_prepared_content.py \
     --config configs/content_preparation/smoke.yaml --stage all --resume
   ```

   Expected: `checks/all.json` passes every stage. Individual `--stage` checks
   support environment/dataset/preprocess/checkpoints/export/encoding/package/
   decode/profile/final. Final requires successful browser evidence tied to the
   current catalog, deployed UI/vendor hashes, PLY equality and valid MPD.

10. Repeat steps 4–6 and 9 with `--resume`. All task execution counts must be zero.
    **Stop here. Do not start a native pilot or full Longdress run.**

## Resume and provenance

Pipeline journal remains `.preparation/journal.json`, with individual quality,
frame and view tasks. Validator task keys use a separate namespace there. MPD
and viewer have their own journals and output ownership checks. Journals hash
input/output files, relevant config, core code, pinned repositories, native
sources and binary, and compatible extension binaries. Changed or corrupted
completed outputs require a new run root or explicit `--overwrite`; resume does
not silently trust stale artifacts.

No artifact is silently reinterpreted as LTS temporal coding. Historical
checkpoint training budgets/provenance remain recorded separately from the
current decode/render runtime. LPIPS batch stays 1, CPU by default. One object,
quality, frame and view is processed at a time; GPU guards and process locks
prevent concurrent model rendering. No config is silently reduced on OOM.

## Viewer precision and dependencies

Mandatory path is CPSEG → project decoder → decoded NPZ → checked Gaussian PLY →
the existing web shell. Browser-side Draco/WASM reconstruction is deferred.

Pinned local modules are SparkJS 2.2.0, Three.js 0.180.0 and OrbitControls. Lock
file includes archive and deployed-file SHA256; vendored licenses are included.
`vendor_dependencies.py --verify` performs an offline check; fetching missing
dependencies is an explicit separate operation.

Spark's packed attributes/SH introduce browser display precision differences.
The viewport is for qualitative inspection. Objective profiles always use the
original Dynamic-LapisGS renderer on decoded payload states. Switching unloads
the old mesh before loading the next; stale async loads are disposed.

## Deliberately deferred

Full/native-pilot configuration, original 30k training, Q0–Q3 full-sequence
execution, grouped profile summaries, progressive GT diagnostics, refresh-duration
ablation and browser-side packet decoding remain later work. Existing unrelated
networking code and pinned upstream source are left unchanged.

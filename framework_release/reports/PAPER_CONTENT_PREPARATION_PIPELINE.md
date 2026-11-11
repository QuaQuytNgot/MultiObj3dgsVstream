# Content Preparation Pipeline for the Streaming System

## Scope and artifact distinction

The streaming evaluation consumes the standalone StreamSTGS packages published under `data_stream/streamstgs_8i_quality_metadata/` and materialized under `experiments/nava_mse/runs/gop0_v1/assets/`. Those packages contain four independently decodable CRF choices for each object and GOP. They are not progressive layers: the quality profiles declare empty dependency lists, and each CRF manifest lists its complete component set.

The repository also contains a newer, general-purpose preparation tool in `tools/content_preparation/`. It can emit independent or progressive `CPSEG` packages, and its current Longdress configuration selects progressive delivery. That is a separate preparation path and codec configuration; it is not the origin of the CRF20/28/36/40 StreamSTGS packages used by the streaming matrix. In particular, `Q0` in the ABR metadata means CRF40, while `Q0` in the generic Longdress YAML means the lowest-resolution training operating point. The paper should use CRF labels when discussing the StreamSTGS experiment.

## Numbered pipeline for the StreamSTGS streaming inputs

1. **Source sequence and camera provenance.** The source is the dynamic 8i point-cloud sequence for each object. The checked-in capture manifests identify 300 consecutive source frames at 30 fps: Longdress 1051–1350, Soldier 536–835, Loot 1000–1299 and Redandblack 1450–1749. Their recorded reference-capture recipe uses 24 cameras, 512×512 images, 40,000 initialization points and `software-zbuffer-3x3-v1`; camera intrinsics, extrinsics, normalization and per-frame PLY hashes are recorded in `reference/`. Purpose: bind model states, synthetic RGB references and later reconstruction checks to the same sequence and camera geometry. The raw PLY files and trained checkpoints are not included in the published quality bundle.

2. **Dynamic StreamSTGS model/checkpoint.** The archived reconstruction metadata identifies the StreamSTGS implementation and upstream commit, checkpoint SHA-256 and iteration, temporal frame mapping, Gaussian count, feature dimensions, atlas layout, SH degree, normalization and per-CRF encoding recipe. For example, Longdress GOP 0 records 81,796 Gaussians, feature dimension 12, a 4×3 atlas, SH degree 3 and checkpoint iteration 12,000. The model is dynamic: its media package includes canonical Gaussian attributes, a temporally varying point-feature video and the MLP weights used to reconstruct time-conditioned Gaussian attributes. `third_party/streamstgs/scene/streamoff_gaussian_model.py` contains the vendored model; `src_final/multiobj3dgs/codecs/streamstgs/adapter.py` documents project decoding and reconstruction. The source-to-checkpoint training procedure itself cannot be reconstructed completely from this checkout; see “Missing and uncertain details.”

3. **GOP segmentation and time mapping.** Every sequence is split into five 60-frame GOPs. At 30 fps, each GOP spans 2 seconds. The GOP manifest records its source frame range, exact timestamps, GOP index, media origin and model-time mapping; GOP 0 of Longdress covers frames 1051–1110. The metadata records two lookahead frames as well. The 8i MPD uses `timescale=30`, `S d=60`, and one segment URL per GOP. This fixed temporal partition provides the unit independently selected, costed and requested by the ABR planner.

4. **Four standalone CRF representations per GOP.** The available alternatives are CRF20, CRF28, CRF36 and CRF40. Each `gop_NNN/encoded/crf_CC/` directory is a full GOP representation, with its own `manifest.json`, reconstruction sidecar, camera/normalization data, codec side information, Gaussian streams and MLP weights. The adaptation ordering treats CRF40 as Q0 (lowest quality/base choice), followed by CRF36, CRF28 and CRF20. No CRF representation requires another CRF representation to decode. Identical component hashes across some CRFs do not create a declared dependency; each candidate manifest still describes its complete files.

   The per-representation manifest lists 16 components: `_xyz.jxl`, `_features_dc.jxl`, `_scaling.jxl`, `_rotation.jxl`, `_opacity.jxl`, `_point_feats.mp4`, five `mlp_*.pth` weights, `cameras.json`, `normalization.npz`, `compression_config.yaml`, `compression_info.csv` and `reconstruction.json`. The representation manifest itself is the seventeenth resource. The observed Longdress GOP0 manifests report 9,958,744 transfer bytes for CRF20, 4,432,589 for CRF28, 2,138,660 for CRF36 and 2,072,587 for CRF40, including each representation manifest. In that GOP, the five JXL attributes and five MLP weights have identical hashes across CRFs, while `_point_feats.mp4` varies substantially. This is artifact evidence for where those particular size differences lie; it is not a general claim that all CRF settings alter only that stream.

5. **Offline full-view PSNR/MSE quality bank.** The published `quality-index.json` contains 4 objects × 5 GOPs × 4 CRFs = 80 quality profiles. Each profile has 60 frames × 24 camera views at scale 1.0, or 1,440 rows per object/GOP/CRF and 115,200 rows overall. Its recorded reference is `original_ply_synthetic_rgb`, not a camera-captured photograph and not CRF20. For each row, RGB quality metrics include MSE, PSNR, SSIM and LPIPS; the profile summary stores the arithmetic mean of per-sample PSNR in dB, with pooled-mean-MSE PSNR retained separately. The profile metadata specifies stored RGB/sRGB values without linearization, clipping to [0,1], no mask (the full image including white background), unit PSNR peak, and Pillow LANCZOS when downsampling the reference. Formally, for a normalized RGB image \(x_{q,f,v}\) at CRF \(q\), frame \(f\), and view \(v\),

   \[
   \mathrm{MSE}_{qfv}=\frac{1}{3WH}\sum_{i=1}^{W}\sum_{j=1}^{H}\sum_{k=1}^{3}
   (x_{qfv}(i,j,k)-x_{\mathrm{ref},fv}(i,j,k))^2,
   \qquad \mathrm{PSNR}_{qfv}=10\log_{10}\frac{1}{\mathrm{MSE}_{qfv}}.
   \]

   Here \(W=H=512\); RGB is normalized to [0,1], so the peak value is one. The stored reported score is \(\overline{P}_q=(1/1440)\sum_{f,v}\mathrm{PSNR}_{qfv}\), while pooled MSE is \(\overline{M}_q=(1/1440)\sum_{f,v}\mathrm{MSE}_{qfv}\) and its corresponding PSNR is \(10\log_{10}(1/\overline{M}_q)\). These are separate statistics. `src_final/multiobj3dgs/client/abr/quality_metadata.py:load_dataset` validates profile identity, coverage, stored PSNR means, representation-manifest hashes, resource byte counts and MPD cross-links before emitting normalized per-GOP inputs.

6. **Planner-specific offline MSE/SSE profiles.** `experiments/offline_profiles/core.py` verifies the stream package and generates the profiles consumed by the adaptation planner. It samples each object/GOP at 640×480, camera IDs 0, 1, 6, 7, 12, 13, 18 and 19, and local GOP times 0, 0.5, 1.0 and 1.5 seconds: 32 camera-time samples per quality. Each candidate is rendered alone with the native StreamSTGS renderer and compared with that object's own CRF20 decoded render. RGB is clipped to [0,1]; the fixed foreground mask comes from CRF20 reference alpha greater than 0.01. For sample \(s\), \(SSE_{qs}=\sum_{p\in M_s}\sum_{k=1}^3(I_{qsk}-I_{20sk})^2\). The stored pooled MSE is

   \[
   MSE_q=\frac{\sum_s SSE_{qs}}{3\sum_s |M_s|},
   \]

   where \(|M_s|\) counts foreground pixels. A separate SSE summary is the equal-camera mean of each camera's four time-summed SSEs: \(\overline{SSE}_q=\frac{1}{8}\sum_{c=1}^{8}\sum_{t\in T}SSE_{qct}\), where \(T=\{0,0.5,1.0,1.5\}\) seconds. These MSE/SSE numbers use a different reference, viewport and sample grid from the original-PLY PSNR bank and must not be substituted for it. At CRF40, the pipeline also records mean solo alpha contribution over the 32 samples; the cached SSE provenance labels its proxy representation `standalone_CRF40`.

7. **Resource cost, checksums and publication metadata.** The object `stream_package_manifest.json` inventories each package file with path, byte length and SHA-256. Each CRF/GOP `manifest.json` lists 16 payload/metadata components with byte counts and hashes; `representation_cost()` adds the actual manifest length, so the full standalone cost is `payload_bytes + manifest_bytes`. For a two-second GOP, the equivalent rate is exactly \(8B/2=4B\) bit/s when \(B\) is the total transfer bytes. The ABR planner uses these per-GOP physical costs; it does not replace them with the MPD's representation-average bandwidth. `publication.json`, per-object `quality-index.json`, root `quality-index.json`, `content-links.json` and `manifest.mpd` bind object/GOP/CRF profiles to package and archive hashes. The MPD maps each CRF representation to five independently addressed GOP manifests. `experiments/offline_profiles/core.py:prepare` verifies the metadata inventory and archive links, reuses an already materialized package when its manifest matches, or calls `extract_verified()` to safely extract the archived package and verify every file.

8. **Runtime consumption (separate from preparation).** The streaming runner loads the measured profiles and representation costs, then the Adaptation Planner chooses one standalone CRF per object for the upcoming GOP under its byte budget. The request path requests that representation's declared resources; the decoder uses its manifest, temporal feature video, Gaussian attributes, MLP weights, normalization and frame/time mapping to reconstruct Gaussian states. In the two-phase evaluation, selected states are rendered jointly only after adaptation. `VisibilityAnalyzer` computes continuous alpha contribution from caller-supplied reconstructed Gaussian states using joint and solo rasterization; it does not choose a CRF or read the offline PSNR bank. For the cached visibility/SSE preparation, the recorded Q0 proxy is specifically CRF40. Therefore a paper can say that the offline contribution proxy uses the Q0/CRF40 decoded state; it should not claim that `VisibilityAnalyzer` internally forces CRF40 for every runtime call.

## Concise stage table

| Stage | Inputs | Outputs consumed downstream | Main implementation/evidence |
|---|---|---|---|
| Source capture | 300 8i PLY frames, 30 fps, 24 calibrated cameras | Capture manifest, camera/normalization metadata, source hashes | `data_stream/streamstgs_8i_quality_metadata/<object>/reference/`; `capture-manifest.json` |
| Dynamic model and GOP encode | StreamSTGS checkpoint; temporal features; GOP frame/time range; CRF | Five GOPs × four full standalone representations per object | Vendored `third_party/streamstgs/`; package `reconstruction.json`; `stream_package_manifest.json` |
| Offline PSNR profile | Decoded CRF package, synthetic RGB reference, 24 views × 60 frames | Per-sample and mean-dB PSNR/MSE/SSIM/LPIPS JSON | `quality-index.json`, `quality/profiles/*.json`, `quality_metadata.py:load_dataset` |
| Planner MSE/SSE profile | Decoded CRF packages, CRF20 reference RGB/alpha, 8 cameras × 4 times | Pooled MSE, camera-mean SSE and CRF40 solo-alpha denominator | `experiments/offline_profiles/core.py:generate`; `prepared/profiles.json`, `baseline_metadata.json` |
| Cost and publication metadata | Representation manifests, profile index, checksums | Exact physical bytes, hashes, GOP/CRF links, MPD | `nava_mse/costs.py:representation_cost`; `publication.json`; `manifest.mpd` |
| Runtime adaptation and rendering | Profiles, visibility, trace/budget, selected standalone package | Requests, decoded state, joint rendered frames | `streaming_evaluation/`, `StreamSTGSRenderer`, `VisibilityAnalyzer` |

## What the metadata files look like

There is not one monolithic metadata file. The checked-in Longdress bundle separates source provenance, one-representation resource manifests, decoder/reconstruction instructions, quality samples, whole-package inventory, and the MPD used to describe the GOP timeline. The excerpts below are from GOP 0 / CRF20; fields or entries omitted from the excerpts remain present in the original files.

```text
longdress/
├── reference/capture-manifest.json
├── content/stream_package_manifest.json
├── content/gop_000_crf_20/
│   ├── manifest.json
│   └── reconstruction.json
├── quality/quality-index.json
├── quality/profiles/gop_000_crf_20.json
├── publication.json
└── manifest.mpd
```

`capture-manifest.json` records the source frame names and hashes, camera and normalization asset hashes, and the capture recipe (frame range, frame rate, view count, resolution and reference renderer). It establishes provenance; it is not a list of streaming requests.

A representation `manifest.json` is the request-level component inventory. It has 16 component entries, each with its byte length and SHA-256. For example, this excerpt shows two real entries:

```json
{
  "schema": 1,
  "crf": 20,
  "actual_payload_bytes": 9956476,
  "files": {
    "_point_feats.mp4": {
      "bytes": 7909157,
      "sha256": "e14e02aa1135e2871b1762ac37a81abf03deb19bdb319445ce1f7c45f9ed72c1"
    },
    "_xyz.jxl": {
      "bytes": 675356,
      "sha256": "1e6a04f5b6813d08ae96d8a9f8fc60eb7d1614cdf69e9472fa286096dc5cc220"
    }
  }
}
```

The real manifest contains the other 14 components as well. `actual_payload_bytes` is their total (9,956,476 bytes here); this JSON manifest is itself a separate requestable resource (2,268 bytes), so the physical GOP transfer total is 9,958,744 bytes. The manifest has no inter-CRF `dependencies` field: the component list describes the standalone representation. See [`manifest.json`](../data_stream/streamstgs_8i_quality_metadata/longdress/content/gop_000_crf_20/manifest.json).

The paired `reconstruction.json` records provenance and decoder state rather than transfer hashes alone. A shortened excerpt illustrates its fields:

```json
{
  "schema": 1,
  "implementation": "kkkzh/StreamSTGS",
  "upstream_commit": "ca8602516352efd83e9e69e665b2e26ed03cb5cc",
  "crf": 20,
  "checkpoint_iteration": 12000,
  "recipe": {
    "start_frame": 1051,
    "end_frame": 1350,
    "fps": 30,
    "views": 24,
    "resolution": 512,
    "renderer": "software-zbuffer-3x3-v1"
  },
  "model": {
    "feat_dim": 12,
    "num_gaussians": 81796,
    "atlas_columns": 4,
    "atlas_rows": 3,
    "active_sh_degree": 3
  },
  "segment": {
    "gop_index": 0,
    "gop_size": 60,
    "gop_start_frame": 1051,
    "gop_end_frame": 1110,
    "lookahead_frames": [1111, 1112]
  }
}
```

The complete sidecar also stores checkpoint identity, detailed time/frame mappings, timestamps and codec/model settings. These fields let the decoder map a selected GOP's temporal feature stream back to the correct dynamic Gaussian state. See [`reconstruction.json`](../data_stream/streamstgs_8i_quality_metadata/longdress/content/gop_000_crf_20/reconstruction.json).

The whole-object `stream_package_manifest.json` inventories package paths, bytes and hashes and summarizes the dataset-level layout. For this object its summary says frames `[1051, 1350]`, 30 fps, five 60-frame GOPs, CRFs `[20, 28, 36, 40]`, 342 inventoried files and 85,986,665 total bytes. Each item in its `files` array has the form `{"path": "...", "bytes": ..., "sha256": "..."}`. This is the integrity inventory for the package, while each GOP/CRF manifest is the narrower request manifest. See [`stream_package_manifest.json`](../data_stream/streamstgs_8i_quality_metadata/longdress/content/stream_package_manifest.json).

A quality profile is larger because it stores its measurement protocol and per-frame/per-view rows in addition to its aggregate. A compact excerpt of the actual CRF20 profile is:

```json
{
  "schema": "streamstgs.quality-profile.v1",
  "metadata": {
    "representation": {
      "object": "longdress",
      "gop": 0,
      "crf": 20,
      "payload_bytes": 9956476,
      "manifest_sha256": "705fe3e69e65c1ee7d9d823ad3f1c7b68859cee04c9302996a209dbf02b1fcda"
    },
    "scales": [1.0],
    "metrics": {
      "color_space": "stored RGB/sRGB values; no linearization",
      "mask": "none; full image including white background",
      "rgb_range": [0, 1],
      "psnr_peak": 1
    }
  },
  "rows": [{
    "object": "longdress",
    "gop": 0,
    "crf": 20,
    "frame": 1051,
    "view": 0,
    "timestamp": 0.0,
    "width": 512,
    "height": 512,
    "metrics": {
      "mse": 0.0007633583092522231,
      "psnr": 31.172715626946584,
      "ssim": 0.9706422686576843,
      "lpips": 0.0341641865670681
    }
  }],
  "summary": {
    "sample_count": 1440,
    "aggregation": "uniform_sample_mean; psnr is mean dB; psnr_from_mean_mse is pooled",
    "metrics": {
      "mse": 0.0009095991860032545,
      "psnr": 30.509402155551605,
      "psnr_from_mean_mse": 30.411499369685767,
      "ssim": 0.9639878444787529,
      "lpips": 0.04223858301476058
    }
  }
}
```

The excerpt shows one selected member of the actual `rows` array; the real profile contains all 1,440 frame/view samples and the full camera, reference and runtime metadata. The object `quality-index.json` points to each profile by URI and SHA-256, and repeats its object/GOP/CRF identity, representation-manifest hash and summary. `publication.json` then binds the object to the MPD, package inventory hash and quality-index hash. Together these files make it possible to discover a profile and verify it belongs to the exact streamed representation. See [`profile`](../data_stream/streamstgs_8i_quality_metadata/longdress/quality/profiles/gop_000_crf_20.json), [`quality-index.json`](../data_stream/streamstgs_8i_quality_metadata/longdress/quality/quality-index.json) and [`publication.json`](../data_stream/streamstgs_8i_quality_metadata/longdress/publication.json).

The MPD is XML rather than JSON. It declares one representation per CRF and maps 60-frame timeline entries to GOP component manifests. For example, CRF20 is declared with average `bandwidth="36861531"`, `timescale="30"`, five timeline entries of duration 60, and segment URLs `gop_000/encoded/crf_20/manifest.json` through `gop_004/encoded/crf_20/manifest.json`. The client uses those independent GOP manifests to locate the selected representation's components; the planner's physical byte cost comes from the files, not this average MPD bandwidth field. See [`manifest.mpd`](../data_stream/streamstgs_8i_quality_metadata/longdress/manifest.mpd).

For the planner's visibility/SSE input, a separate prepared JSON records compatibility and proxy semantics. For example, [`profiles.json`](../experiments/offline_profiles/runs/full_source_v1/prepared/profiles.json) declares the proxy as `standalone_CRF40`, resolution `[640, 480]`, and the contribution/time aggregation conventions. The cached [`offline_contribution.json`](../experiments/visibility_guided_sse/runs/pilot_v1/prepared/offline_contribution.json) stores `quality: 40`, a convention string, sample identities and per-object aggregate values. Those values are visibility contribution proxies computed from offline CRF40 Gaussian renders; the JSON is not the Gaussian attribute payload itself and does not mean the runtime analyzer always forces CRF40.

## Separate generic preparation tool in this repository

The in-repository `tools/content_preparation/prepare_content.py` orchestrates nine configurable stages: `preprocess → train → export → encode → package → decode → profile → proxy → manifest`. The raw-data path preprocesses source PLYs with the pinned Dynamic-LapisGS upstream scripts; the checkpoint path imports and verifies existing checkpoint provenance. Training can run upstream Dynamic-LapisGS or import verified checkpoints. Export converts checkpoints to canonical Gaussian-state NPZ files. Encoding runs the selected project codec and immediately decodes each payload for reconstruction-hash verification. Packaging writes actual requestable segment containers and dependency metadata; decode reconstructs from those containers. Profile renders decoded states against the highest-quality decoded delivery, then writes sample profiles and transition gains. The optional proxy stage freezes Gaussian IDs from the highest decoded representation. Finally the manifest binds frame/time maps, package index, profile/proxy assets, actual bytes, hashes and provenance; `mpd.py` exports/validates the project-specific MPD.

This tool is important implementation context, but its Longdress configuration is not the CRF experiment: `configs/content_preparation/longdress.yaml` selects progressive delivery, Q0–Q3 resolution scales 8/4/2/1, 30,000 initial and dynamic training iterations, a project `gaussian_attribute_draco_byteplanes` f32 codec, 180-frame GOPs, 15-frame segments, and Base/E1/E2/E3 refresh periods 90/60/30/15 frames. Those Q levels, containers and dependencies are distinct from the actual 60-frame StreamSTGS CRF20/28/36/40 resources. The StreamSTGS CRF representations should therefore be described as independent alternatives in the paper.

## Missing and uncertain details

- The raw PLY files, StreamSTGS checkpoints and original training logs are excluded from the published media bundle. `capture-manifest.json` preserves raw-source hashes and training-related sidecars preserve checkpoint identity, but the full training recipe and checkpoint bytes are not available for independent re-execution.
- `reports/commands.json` records historical `tools/profile_streamstgs.sh evaluate ...` invocations, and `profiling-run.json` records the old package/dataset paths and runtime. The referenced `tools/profile_streamstgs.sh` and `tools/streamstgs_quality/pipeline.py` are absent from this checkout. Thus the saved manifests, decode/profile metadata and validation artifacts establish the outputs and measured protocol, but not every original encoder command or source RGB-generation instruction.
- The checked-in StreamSTGS representation sidecar records JXL quantization/levels and decoded temporal-feature metadata. It does not preserve the complete historical CRF-to-FFmpeg encoder command. Do not invent its rate-control settings from the `.mp4` suffix or from the CRF labels.
- The full-view original-PLY PSNR profile and planner-specific own-CRF20 foreground MSE/SSE profile are different datasets. State their reference, resolution and sampling distribution whenever reporting one.
- The generic `tools/content_preparation` path has its own packaging and dependency semantics. Its progressive configuration is not evidence of progressive dependencies in the current StreamSTGS CRF packages.

## Paper-ready subsection (English, 500–700 words)

### Offline content preparation

We prepare dynamic 3D Gaussian Splatting (3DGS) content from four 8i point-cloud sequences: Longdress, Soldier, Loot and Redandblack. Each source contains 300 consecutive frames at 30 frames/s. The preparation metadata records the source frame range and hashes, a 24-view camera calibration, the normalization transform and the reference-render recipe (512×512, 40,000 initialization points, software z-buffer splatting). The source PLYs and original trained checkpoints are not part of the published media bundle; consequently, the released package supports integrity checks and decoding, while complete retraining requires the original source assets and training provenance.

Each sequence is represented with StreamSTGS as a time-conditioned Gaussian model. Its GOP reconstruction sidecar records the checkpoint identity, frame-to-media/model-time mapping, Gaussian and feature dimensions, atlas layout, SH degree, normalization and codec configuration. A GOP contains 60 frames, corresponding to two seconds at 30 frames/s; five GOPs cover each ten-second source sequence. A GOP package combines canonical Gaussian attributes, a temporally varying point-feature video, the StreamSTGS MLP weights, cameras and normalization data, and reconstruction/codec side information. The client reconstructs time-specific Gaussian states from these components using the StreamSTGS decoder.

For each object and GOP, we provide four standalone representations at CRF20, CRF28, CRF36 and CRF40. Every CRF has its own complete representation manifest and component resources. CRF40 is the lowest-quality Q0 choice in the adaptation metadata, while CRF20 is the highest-quality representation. The alternatives are independently addressable and have no declared inter-CRF dependencies; they are not progressive refinement layers. An object package manifest records file sizes and SHA-256 hashes, and each representation manifest lists 16 component files plus the manifest resource itself. We compute adaptation costs from the exact physical size of these files, including the representation manifest, rather than from average MPD bandwidth.

Offline quality metadata is prepared separately from adaptation. The full-view StreamSTGS profile bank covers every CRF, GOP, source frame and recorded view: 1,440 rows per object/GOP/CRF (60 frames × 24 views), 80 profiles and 115,200 rendered/reference pairs. Its reference is synthetic RGB rendered from the original PLY, rather than a photograph or the CRF20 representation. Per-row metrics include MSE, PSNR, SSIM and LPIPS. The published summary reports the arithmetic mean of per-row PSNR in dB; it also retains PSNR computed from pooled mean MSE as a distinct statistic. The root and per-object quality indexes bind those profile rows to representation manifests, while the MPD identifies each CRF as a representation and maps its five GOPs to separate component manifests.

The adaptation planner also consumes a second offline profile bank generated at 640×480 using eight fixed camera IDs and four timestamps per GOP. For each candidate, the system compares a solo decoded render against that object's decoded CRF20 render. RGB is clipped to [0,1], and the scoring mask is the CRF20 reference alpha greater than 0.01. Pooled MSE divides the sum of squared foreground RGB error by the number of foreground RGB scalar samples; SSE is aggregated as the equal-camera mean of four timestamp-summed errors. CRF40 provides the Q0 solo-alpha contribution proxy recorded for the visibility/SSE preparation. The runtime Visibility Analyzer separately computes joint alpha contribution and solo contribution from the reconstructed Gaussian states supplied by its caller; it does not select a representation or consume the PSNR profile.

Before use, the preparation/import code verifies checksums, object/GOP/CRF identities, component byte counts, profile coverage, frame/timestamp mappings and cross-links between package manifests, quality indexes and MPDs. The streaming client then selects a standalone CRF per object and GOP, requests its declared resources, decodes the selected time-conditioned state, and renders the joint scene. These offline measurements and manifests are inputs to runtime adaptation; they do not use future streaming or rendered Phase-B results.

## Implementation references

- Actual CRF bundle and profile validation: [quality_metadata.py](../src_final/multiobj3dgs/client/abr/quality_metadata.py), [streamstgs_8i_quality_metadata README](../data_stream/streamstgs_8i_quality_metadata/README.md), [Longdress MPD](../data_stream/streamstgs_8i_quality_metadata/longdress/manifest.mpd), [Longdress package manifest](../experiments/nava_mse/runs/gop0_v1/assets/longdress/stream_package_manifest.json).
- Planner profile creation and integrity binding: [offline_profiles/core.py](../experiments/offline_profiles/core.py), [offline_profiles/README.md](../experiments/offline_profiles/README.md), [nava_mse/metrics.py](../experiments/nava_mse/metrics.py), [nava_mse/costs.py](../experiments/nava_mse/costs.py).
- Visibility input contract: [VisibilityAnalyzer](../src_final/multiobj3dgs/client/visibility/analyzer.py), [visibility-weighted profile metadata](../experiments/visibility_guided_sse/runs/pilot_v1/prepared/offline_contribution.json), [visibility-weighted configuration](../experiments/visibility_weighted_sse/configs/cached_recorded_source_control/config.json).
- Separate general-purpose pipeline: [prepare_content.py](../tools/content_preparation/prepare_content.py), [pipeline.py](../tools/content_preparation/pipeline.py), [packaging.py](../tools/content_preparation/packaging.py), [manifest.py](../tools/content_preparation/manifest.py), [MPD exporter](../tools/content_preparation/mpd.py), [Longdress config](../configs/content_preparation/longdress.yaml), [step-by-step preparation note](../docs/CONTENT_PREPARATION_STEP_BY_STEP.md).
- Overall streaming configuration and paper context: [streaming matrix configuration](../experiments/streaming_evaluation/configs/streaming_http2_dp015.json), [mathematical formulation and setup](PAPER_MATHEMATICAL_FORMULATION_AND_EXPERIMENTAL_SETUP.md).

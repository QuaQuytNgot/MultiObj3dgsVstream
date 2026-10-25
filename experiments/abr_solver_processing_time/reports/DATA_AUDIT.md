# Measured metadata audit

Source: `/home/fil/Hoang/MultiObj3dgsVstream/data_stream/streamstgs_8i_quality_metadata`.

The bundle contains Longdress, Soldier, Loot and RedAndBlack; each has five consecutive 60-frame GOPs at 30 fps, independently encoded as CRF20/28/36/40. All 80 object × GOP × CRF profiles are present. No quality or bitrate values are synthesized.

PSNR is the profile's arithmetic mean in dB over 60 frames × 24 views (1,440 measurements per profile; 115,200 total). It is not PSNR computed from pooled MSE. All 24 views, including training and held-out views, are used at scale 1.0. The reference is original-PLY synthetic RGB rendered by the offline point-splat capture; it is not captured photography. The observed profile mean range is 23.790314–35.777103 dB.

Each quality profile identifies the representation manifest by SHA256 and repeats its payload byte count. The copied manifests reside under `<object>/content/gop_XXX_crf_YY/manifest.json`; the original publication paths in the package inventory are `gop_XXX/encoded/crf_YY/manifest.json`. The loader explicitly reconciles these layouts.

The cost counts the sum of all manifest-listed component bytes plus the manifest itself. This includes compressed media, MLP parameters and auxiliary files listed by the publication. No cross-object sharing, caching, HTTP headers or startup cost is assumed. The MPD timeline declares timescale 30 and duration 60: each GOP is exactly two seconds. Therefore `cost_bps = 4 × transfer_bytes` is an exact integer without bitrate rounding. Observed GOP costs range from 5,246,772 to 60,156,248 bit/s.

All 16 MPD Representation bandwidths equal the ceiling of the measured ten-second average including component manifests. These averages are checked for provenance; the solver uses each actual GOP's size, so no GOP variation is replaced by the average.

Validation checks the bundle SHA256 inventory for every consumed file, publication → object quality index → profile → representation/reconstruction hashes, package inventory file counts/bytes, all frame-view cells, and the mean PSNR recomputed from raw metric rows. Payload media, PLY and RGB images are not loaded, extracted, decoded or rehashed. Their recorded hashes and sizes are cross-checked across metadata.

Missing or inconsistent metadata causes a hard error. `--cost-root` can point to an external metadata bundle or extracted object publication if per-GOP cost manifests are relocated; their hashes and package inventory must still match. There is no fabricated fallback.

The complete consumed-file SHA256 list, normalized profile values and MPD checks are in `inputs/dataset_audit.json` and `inputs/measured_profiles.json`.

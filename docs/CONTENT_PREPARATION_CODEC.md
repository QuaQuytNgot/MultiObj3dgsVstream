# Codec inspection and implementation

## Current addition — 2026-10-06

The first batch adds `gaussian_attribute_draco_byteplanes` 1.0.0 / `CPDRAC01`.
The project C++ bridge uses verified public Draco 1.5.7 sequential byte attributes
to preserve exact f32 state/IDs and sparse shared-state replacements. It has no
temporal prediction and does not reproduce the unpublished enhanced LTS encoder.
The legacy zlib packet remains unchanged. Per-quality codec dispatch and an
explicit local decoder runtime are tested, including decoding after relocating
the bridge. See [the batch runbook](CONTENT_PREPARATION_BATCH1.md) for the native
policy, actual smoke commands and codec limits, and
[current evidence](validation/content_preparation_batch1_smoke_summary.json).

The sections below preserve the inspected historical zlib baseline and source
boundary; their statements about missing Draco/DASH describe that earlier state.

> **Migration note (2026-10-05):** project-owned preparation code and this document
> were migrated from the sibling `dynamic-lapis-gs` workspace. Measured results,
> PASS counts and hardware observations below describe that historical workspace;
> they do not establish a completed run in `MultiObj3dgsVstream`. Dataset, models,
> vendor binaries and generated `output/` artifacts are not bundled. Commands now
> use this project root and the pinned backend in `third_party/dynamic-lapis-gs`.
> See [migration/status](CONTENT_PREPARATION_MIGRATION.md) before reproducing a run.

Inspected on 2026-10-04. The [official LTS repository](https://github.com/AIINS-NTHU/LTS-DASH-Streaming-System-for-3DGS) contains its README and assets, and states that the authors cannot release the implementation publicly. No exact LTS encoder implementation was found in this local repository or the inspected workspace files.

The server implementation described in the [LTS paper](https://people.cs.nycu.edu.tw/~chuang/pubs/pdf/2025mmsys.pdf) uses an enhanced Draco encoder in lossless mode for Gaussian tiles, followed by GoF fragmentation and DASH generation. It also estimates quality using rendered viewpoints. The publicly indexed paper text verifies these details; the browser could not open the complete 11 MB PDF. The unavailable Draco enhancement prevents a claim of bitstream compatibility or exact reproduction. Our pipeline preserves the content preparation sequence and layered representation concept. It does not implement LTS tiles, enhanced Draco or DASH.

## Existing local implementations reused as design evidence

- `scene/gaussian_model.py`: authoritative PLY layout and parameter domains. Exported scale is logarithmic, opacity is logit, rotation is raw wxyz, and SH fields are channel-major. The new reader/writer reproduces these field mappings on CPU.
- `tools/check_progressive_composability.py` and prior progressive-gap reports: shared Gaussian attributes can change between qualities; suffix-only enhancement is insufficient.
- `tools/evaluate_correction_coding.py`: attribute quantization, exact replacement controls, stable prefix correspondence checks, byte accounting and payload-only decoding.
- `tools/evaluate_full_payload_coding.py`: quantization/zstd characterization and closed-loop enhancement against decoded Base.
- `tools/evaluate_temporal_geometry.py`: distinct absolute access and predictive temporal states. The new baseline intentionally has no temporal prediction.

These older experiment scripts are coupled to their existing experiment directories. The new modules implement their generalizable asset/accounting and closed-loop principles without importing their experiment orchestration or changing upstream training/rendering.

## Exact codec supplied here

Adapter ID: `gaussian_attribute_zlib`, version `1.0.0`, packet magic `CPGAUS01`, Gaussian state version `gaussian-state-v1`.

The default stores every attribute as little-endian float32 and compresses one deterministic byte stream with zlib level 6. It is lossless for the numeric internal Gaussian state, including ID ordering. There are no hidden parent checkpoints, learned entropy models, Draco, video prediction or renderer changes. Compression method can explicitly be `none`, `zlib` or optional `zstd`; zstd requires the Python `zstandard` package. The recorded compressor version allows reproducibility within the same runtime.

Additional configurable precisions are float16 and signed uniform quantization at 8 through 16 bits. Quantized values are actually bit packed. Each attribute uses full-target maximum absolute value, a transmitted float64 step, nearest-even rounding and symmetric signed levels. Per-attribute precision overrides are supported. Every scale, index, header, ID, deletion and checksum is part of the charged packet file size. Float16 overflow and invalid zero decoded quaternions fail explicitly; the codec never silently clips or changes configuration.

This is a conventional baseline, adapted from the local experiments' approach. Lossy variants are additional approximations for measurement, not claims about LTS. Lossless zlib is operationally comparable to delivering compressed lossless states, but its algorithm and rates cannot stand in for enhanced Draco results.

## Progressive behavior

`encode(state, path, config, parent=decoded_parent)` first quantizes the entire target with the same quantizer that independent encoding would use. It then compares **all five attributes** bitwise with the decoded parent's corresponding stable IDs and sends sparse absolute replacements. Enhancements can therefore replace geometry, opacity, scale and SH, add/delete Gaussians, change SH degree and reorder output rows. Raw quaternion replacement avoids introducing an unverified quaternion residual transform.

The output of progressive decoding equals the output of independent decoding of that target with the same config. It does not assume that a target is a parent plus only new Gaussians. Quantizer recalibration can require many replacement rows; bandwidth savings are measured rather than guaranteed. Parent numeric hashes prevent applying an enhancement to another decoded state. Explicit stable IDs or externally verified lineage are required. Imported upstream PLY files have unverified row identity until orchestration verifies lineage; merely loading two PLYs never proves prefix correspondence.

## Formats and validation

Packets contain an 8-byte magic, little-endian uint64 JSON-header length, canonical JSON header, compressed attribute streams and a 32-byte SHA-256 of the entire preceding packet. The header records attribute offsets, lengths, shapes, quantizers, per-stream hashes, target and decoded hashes, parent hash, metadata, codec/config/compressor versions and ID accounting. `decode` reads this packet and an explicitly supplied decoded parent only. It checks packet/stream/state hashes, sparse coverage, ID accounting and the parent hash.

Internal NPZs use deterministic ZIP timestamps, uncompressed NPY members and a JSON metadata member represented as uint8; no pickle is loaded. Numeric state hashes cover IDs, row order, shapes and all float32 attributes; provenance metadata remains separately covered by the asset file hash. Gaussian PLYs preserve original upstream fields and add an ignored `uint gaussian_id` field plus a JSON metadata comment. These extended PLY IDs must fit uint32; NPZ supports signed int64 IDs. Original PLYs without this field remain supported.

The focused CPU self-tests cover deterministic files, lossless and lossy roundtrips, closed-loop refinement, every attribute, additions/deletions/reordering, SH degree changes, missing/wrong parents, unverified identities, corrupted bytes and exact actual file accounting. Rendering is tested by the main pipeline suite through its renderer adapter.

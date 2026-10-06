# Full-payload rate–distortion characterization — Longdress

> **Migration note (2026-10-05):** project-owned preparation code and this document
> were migrated from the sibling `dynamic-lapis-gs` workspace. Measured results,
> PASS counts and hardware observations below describe that historical workspace;
> they do not establish a completed run in `MultiObj3dgsVstream`. Dataset, models,
> vendor binaries and generated `output/` artifacts are not bundled. Commands now
> use this project root and the pinned backend in `third_party/dynamic-lapis-gs`.
> See [migration/status](CONTENT_PREPARATION_MIGRATION.md) before reproducing a run.

## Scope and reproducibility

Longdress 1051–1055, 41,700 shared + 36,721 new Gaussians, res8/res4 checkpoints; 16 original test cameras/frame (80 views). No retraining, training/model/renderer edits, or learned codec. This is a five-frame, single-sequence characterization, not a general codec result.

Run: `conda activate Hoang && python tools/evaluate_full_payload_coding.py`. Existing payloads: append `--render-only`; validation/report only: `--report-only`. Exact upstream renderer extension is loaded from the real experiment’s vendor directory. Source/checkpoint provenance is inherited and checked by `verify_lineage`; source hashes and environment are in summary.json.

## Method and accounting

- Static lifetime is measured bitwise, not assigned by attribute name. All opacity/scale/SH groups are static in these checkpoints; xyz/rotation groups are dynamic. Per-row lifetimes remain in lifetimes.json. Group-static reuse is used for all owners; partially static rows inside dynamic groups are not separately factored in this baseline.
- Base/new: raw quaternion float32 for exact controls; otherwise SO(3) principal rotation vectors, reconstructed to unit wxyz quaternions. Corrections: left-composed SO(3) rotation residual; xyz/opacity use arithmetic residuals. Scale is log-scale, opacity is logit, SH is the checkpoint coefficient representation.
- Closed-loop: correction is calculated against the **decoded quantized Base**, never the original Base hidden at the decoder. Only attributes actually differing between original qualities are refined; identical shared scale/SH inherit lossy Base. New state is independently quantized. Thus shared geometry can be more accurate than suffix geometry; all costs are included.
- Fixed-point signed q8/q10/q12/q16 uses actual packed bits, round-to-nearest-even and separate max-abs calibration per owner/attribute over this GoF. No clipping. Scales are transmitted in JSON. Float16 is IEEE binary16. Mixed config: Base/New xyz q16, rotation/scale/opacity q12, SH q8; correction q16/q12/q8. Mixed choice is a simple preset, not an exhaustive optimization.
- Every binary stream and JSON header is compressed independently with zstd level 6, --long=27, one thread; actual file sizes include zstd headers. GoF-wide groups same owner/lifetime/attribute across five frames. Frame-access groups each dynamic frame independently, plus shared static state/header. Both still use offline GoF calibration; this is **not causal acquisition**.
- Nonoverlapping shares: Base/New/Correction **value bytes** + Metadata (all JSON, quantizer scales, indices/masks). CSV also retains owner-inclusive *_bytes for comparison with prior work; do not add Metadata to those again. Static/dynamic per-owner raw and compressed columns are included. All numbers are logical transmitted sizes, independent of disk hardlinks. MB = 1,000,000 bytes. Cameras, filesystem, transport/container headers outside the specified binary format are excluded consistently.
- Renderer uses original float RGB, clamped [0,1]; original repo SSIM and LPIPS-VGG convention; equal-weight 80-view averages including background. Max/p95 losses are per-view, not per-pixel. Very small metric improvements are not evidence of a better method.

## Full rate-quality table

| Configuration | Base % | New % | Correction % | Metadata % | Total MB | PSNR GT | SSIM | LPIPS | Mean loss dB | p95 / worst dB | PSNR frontier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F0_exact_repeated | 45.55 | 39.12 | 15.33 | 0.005 | 29.529503 | 39.64798 | 0.9949313 | 0.0052865 | 0.00000 | 0.00000 / 0.00000 |  |
| F0_exact_reuse | 45.55 | 39.12 | 15.33 | 0.004 | 29.522414 | 39.64798 | 0.9949313 | 0.0052865 | 0.00000 | 0.00000 / 0.00000 | yes |
| F1_all_f16 | 44.78 | 38.45 | 16.76 | 0.009 | 14.100077 | 39.64146 | 0.9949192 | 0.0052888 | 0.00652 | 0.03942 / 0.05563 |  |
| F1_all_q16 | 45.47 | 39.32 | 15.20 | 0.013 | 12.906946 | 39.64434 | 0.9949233 | 0.0052861 | 0.00363 | 0.02308 / 0.04176 | yes |
| F1_all_q12 | 44.88 | 39.47 | 15.64 | 0.017 | 9.483280 | 39.63768 | 0.9949186 | 0.0052919 | 0.01029 | 0.03705 / 0.05528 |  |
| F1_all_q10 | 44.46 | 39.14 | 16.38 | 0.021 | 7.562941 | 39.52832 | 0.9947898 | 0.0053660 | 0.11965 | 0.30098 / 0.40148 |  |
| F1_all_q8 | 44.64 | 38.40 | 16.93 | 0.035 | 4.541833 | 38.02755 | 0.9926426 | 0.0068785 | 1.62042 | 2.82410 / 3.79144 | yes |
| F2_mixed_cq16 | 40.42 | 33.75 | 25.81 | 0.021 | 7.891003 | 39.61811 | 0.9948829 | 0.0053161 | 0.02987 | 0.05921 / 0.07850 | yes |
| F2_mixed_cq12 | 43.71 | 36.50 | 19.77 | 0.023 | 7.296466 | 39.61599 | 0.9948798 | 0.0053145 | 0.03199 | 0.06505 / 0.09277 | yes |
| F2_mixed_cq8 | 48.46 | 40.46 | 11.05 | 0.025 | 6.581517 | 39.57719 | 0.9948299 | 0.0053510 | 0.07079 | 0.13376 / 0.14464 |  |
| F3_q12_repeated | 44.88 | 39.47 | 15.63 | 0.019 | 9.486076 | 39.63768 | 0.9949186 | 0.0052919 | 0.01029 | 0.03705 / 0.05528 |  |
| F0_exact_reuse_frame_access | 44.00 | 37.78 | 18.21 | 0.004 | 30.564136 | 39.64798 | 0.9949313 | 0.0052865 | 0.00000 | 0.00000 / 0.00000 |  |
| F1_all_q12_frame_access | 44.88 | 39.47 | 15.63 | 0.017 | 9.482845 | 39.63768 | 0.9949186 | 0.0052919 | 0.01029 | 0.03705 / 0.05528 | yes |
| F1_all_q8_frame_access | 44.61 | 38.55 | 16.81 | 0.034 | 4.544898 | 38.02755 | 0.9926426 | 0.0068785 | 1.62042 | 2.82410 / 3.79144 |  |
| F2_mixed_cq8_frame_access | 48.49 | 40.49 | 11.01 | 0.025 | 6.578068 | 39.57719 | 0.9948299 | 0.0053510 | 0.07079 | 0.13376 / 0.14464 | yes |

Exact controls above are dense replacement, **not** the previous row-alias best-exact encoder (28.065 MB). The prior exact result is not overwritten and remains a stronger exact coding point. Residual float/quantized states are not claimed bit-exact.

## GoF-wide versus frame-access

| Configuration | Total delta B | Delta % | Base delta B | New delta B | Correction delta B | Correction share GoF → frame |
|---|---:|---:|---:|---:|---:|---|
| F0_exact_reuse | +1,041,722 | +3.5286 | +1,396 | -17 | +1,040,396 | 15.33% → 18.21% |
| F1_all_q12 | -435 | -0.0046 | +153 | +236 | -789 | 15.64% → 15.63% |
| F1_all_q8 | +3,065 | +0.0675 | +100 | +7,961 | -4,960 | 16.93% → 16.81% |
| F2_mixed_cq8 | -3,449 | -0.0524 | +102 | +118 | -3,624 | 11.05% → 11.01% |

Fresh decoder instances decode each requested frame in reverse order and assert no other frame’s dynamic stream is read. Static streams and global headers are required. This measures frame-dynamic accessibility **within an already initialized GoF**, not arbitrary refresh/rejoin across GoFs or quality switching latency. Negative deltas can occur because separate zstd frames choose different coding tables; they are measured, not clamped to zero.

## Operating point and research questions

Illustrative operating point: **F2_mixed_cq8_frame_access**, 6.578068 MB/GoF, GT PSNR 39.57719, mean loss 0.07079 dB, worst 0.14464 dB. Selection: minimum bytes subject to mean loss ≤0.1 dB, worst ≤0.25 dB, LPIPS increase ≤0.0002. This budget does not establish perceptual equivalence.

**A. Correction share:** 11.01% of the full coded payload (723,930 value bytes); owner-inclusive 724,405 B. This differs from the old ~2.8% figure with float32 Base/New.

**B. Dominant owner:** Base 48.49%, New 40.49%, Correction 11.01%; metadata 0.025%.

**C. Controlled gain comparisons:**

- Static reuse, exact: raw 99,218,655 → 33,301,872 B; zstd 29,529,503 → 29,522,414 B (0.02% saving).
- Static reuse, q12: raw 36,320,113 → 11,596,004 B; zstd 9,486,076 → 9,483,280 B (0.03% saving).
- Full-payload q12 versus dense exact (includes SO(3)/residual representation change): raw 33,301,872 → 11,596,004 B; zstd 29,522,414 → 9,483,280 B (67.88% saving).
- Quantization q16 to q12, same representation: raw 15,457,617 → 11,596,004 B; zstd 12,906,946 → 9,483,280 B (26.53% saving).
- Only correction precision, fixed mixed Base/New: raw 10,948,474 → 9,655,725 B; zstd 7,891,003 → 6,581,517 B (16.59% saving).

These gains are not additive. Static duplication is mostly already recovered by GoF-wide zstd; static reuse still matters for explicit lifetime/access semantics. Frame-access deltas above isolate the effect of joint temporal compression, not a video prediction codec.

**D. Perfect removal of correction:** optimistic ceiling 11.01% from correction values, at most 11.01% including its entire header. This assumes no added state-sharing costs and no quality loss; simply dropping residuals does not achieve this bound.

**E. Access penalty:** quantified per configuration above. A five-frame GoF and cached static payload do not establish refresh/rejoin as a bottleneck.

**F. Attribute breakdown at the operating point:**

| Attribute | Coded bytes, all owners | Total % |
|---|---:|---:|
| xyz | 2,472,372 | 37.59 |
| rotation | 2,064,280 | 31.38 |
| sh | 1,559,319 | 23.70 |
| scale | 329,983 | 5.02 |
| opacity | 150,502 | 2.29 |

**G. Recommendation: DEPRIORITIZE cross-quality representation redesign as the main rate contribution; investigate Base/New dynamic xyz/rotation coding first.** At this operating point Base+New account for 88.97% of total, while xyz+rotation across owners account for 68.97%. Correction is not negligible (~11%, and ~20% at mixed q12), but is secondary at the relaxed operating point. An ideal zero-cost correction removal cannot exceed the ceiling in D. A state-sharing method could remain a secondary refinement, or be justified by separately demonstrated access/quality benefits, but the measured frame-access penalty does not currently supply that justification. This recommendation is conditional on this sequence and quality budget, not a universal rejection.

The lossless versus lossy trade-off remains real: uniform q8 saves more bytes but loses 1.62 dB on average and 3.79 dB in the worst view. Mixed q8 correction is a substantially better tested compromise; q12 throughout supports a stricter ~0.01 dB mean-loss point. Tiny GoF/frame byte differences are compressor effects, not evidence that frame segmentation improves representation quality.

## Validation and limitations

All 15 configurations × 5 frames × 16 views rendered; raw/zstd decoders agree. Exact controls assert bitwise state equality and pixel equality. File sizes, hashes, decompression, source integrity, and independent frame access validated in validation.json. summary.csv contains raw/compressed owner/lifetime breakdown, GT metrics, vs-original-Q1 PSNR and maximum pixel error; per_view.csv/per_frame.csv preserve distributions; streams.csv provides attribute detail.

No optional attribute-video codec is included. ffmpeg is available, but an invertible SO(3)/signed-bit packing and codec pixel-format/precision protocol would add a separate lossy representation and validation axis. The requested main quantization+zstd characterization is complete; these results are not a claim about H.264/H.265 or a final streaming codec.

Only five frames, one subject, this camera subset/resolution and these trained checkpoints were evaluated. Group-level reuse leaves partial-row static reuse, alternative transforms/calibration and temporal predictors unexplored. Background-heavy mean metrics need caution. Future general claims require more sequences/GoFs, stronger coding baselines and foreground/tail-quality checks.

# Temporal geometry baseline — Longdress 1051–1055

> **Migration note (2026-10-05):** project-owned preparation code and this document
> were migrated from the sibling `dynamic-lapis-gs` workspace. Measured results,
> PASS counts and hardware observations below describe that historical workspace;
> they do not establish a completed run in `MultiObj3dgsVstream`. Dataset, models,
> vendor binaries and generated `output/` artifacts are not bundled. Commands now
> use this project root and the pinned backend in `third_party/dynamic-lapis-gs`.
> See [migration/status](CONTENT_PREPARATION_MIGRATION.md) before reproducing a run.

## Outcome

Both prefixes pass the specified quality budget. The selected temporal operating point is **T9_q10**, **5.530009 MB/GoF**: 15.93% smaller than the previous operating point and 9.74% smaller than the best tested absolute configuration satisfying the same two-prefix budget. Cold access to the last Q1 frame is 72.48% more expensive than that absolute comparator. Geometry still accounts for 63.08% of total coded bytes.

**Recommendation: CONTINUE a bounded investigation of temporal geometry rate versus access cost.** The motivation is the measured RD/access trade-off after the simple baseline, not non-composability alone. These five frames do not establish that a new codec/motion method is warranted or competitive with stronger codecs.

## Reproduce and scope

```bash
conda activate Hoang
python tools/evaluate_temporal_geometry.py --phase prefix
python tools/evaluate_temporal_geometry.py --phase temporal
# Existing measurements: validate and regenerate aggregates/report
python tools/evaluate_temporal_geometry.py --phase report
```

A fresh run requires a new output directory if temporal payloads already exist (`--output PATH`). The first stage must finish before temporal coding chooses its parent. No training/model/renderer source is changed. No learned codec, DASH or video codec is used. Existing Longdress checkpoints, prefix correspondence checks and exact upstream CUDA renderer are reused.

27 configurations: 10 existing absolute controls and 17 new temporal/absolute-recoding variants. 3,520 new view-pair evaluations (800 Q0-prefix checks + 2,720 temporal/control Q0/Q1 checks), plus 800 previously measured Q1 view pairs reused from full_payload_coding. The aggregate contains 4,320 view pairs and 270 frame/prefix rows.

Both Q0 and Q1 are rendered at the same **256×256 res4 test cameras/GT**, 16 views/frame. Q0 was trained at lower resolution; its official PSNR at these evaluation views is 29.612411 dB versus 39.647976 dB for official Q1. “Q0 quality preserved” means quantization adds little loss, not that Q0 has Q1-level absolute quality. No Q1 suffix/correction is loaded by the Q0 decoder.

PSNR loss means `official-prefix vs GT PSNR − decoded-prefix vs GT PSNR`; it is not the PSNR between the two rendered images. Positive is degradation. Direct decoded-vs-official PSNR/SSIM/LPIPS and max pixel errors are retained as `*_reference_*` and `*_global_max_abs_pixel` columns. Equal-weight full-image averages include background; small improvements can be incidental smoothing/metric noise.

## Phase 1: validate each prefix

Budget: mean GT PSNR loss ≤0.1 dB and worst-view loss ≤0.25 dB, **for each prefix**. SSIM/LPIPS are also measured, not implicitly assumed equal.

| Existing configuration | Q0 bytes | Full bytes | Q0 GT PSNR | Q0 loss | Q1 loss | Q0 worst | Q1 worst | Both pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| F0_exact_reuse | 13,448,417 | 29,522,414 | 29.612411 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | True |
| F1_all_f16 | 6,314,056 | 14,100,077 | 29.611972 | 0.000439 | 0.006519 | 0.004209 | 0.055625 | True |
| F1_all_q16 | 5,869,466 | 12,906,946 | 29.612461 | -0.000050 | 0.003632 | 0.000772 | 0.041764 | True |
| F1_all_q12 | 4,256,842 | 9,483,280 | 29.614529 | -0.002118 | 0.010293 | 0.006730 | 0.055280 | True |
| F1_all_q10 | 3,363,542 | 7,562,941 | 29.624179 | -0.011768 | 0.119652 | 0.013103 | 0.401477 | False |
| F1_all_q8 | 2,027,951 | 4,541,833 | 29.791936 | -0.179525 | 1.620425 | -0.074358 | 3.791443 | False |
| F2_mixed_cq16 | 3,190,002 | 7,891,003 | 29.611452 | 0.000959 | 0.029868 | 0.004422 | 0.078502 | True |
| F2_mixed_cq12 | 3,190,002 | 7,296,466 | 29.611452 | 0.000959 | 0.031988 | 0.004422 | 0.092770 | True |
| F2_mixed_cq8 | 3,190,002 | 6,581,517 | 29.611452 | 0.000959 | 0.070788 | 0.004422 | 0.144644 | True |
| F2_mixed_cq8_frame_access | 3,190,087 | 6,578,068 | 29.611452 | 0.000959 | 0.070788 | 0.004422 | 0.144644 | True |

**A1:** mixed cq8 passes both prefixes: Q0 mean loss 0.000959 dB, worst 0.004422 dB; Q1 mean 0.070788 dB, worst 0.144644 dB.

**A2:** no tested configuration has good Q1 but materially damaged Q0 under the specified GT-PSNR budget. In fact uniform q8 improves Q0 GT PSNR while greatly damaging Q1. That does not make it a faithful reconstruction of official Q0; see direct-reference metrics. Mixed cq16/cq12/cq8 produce identical Q0 metrics, confirming correction precision does not enter Q0 rendering.

**A3:** the existing frame-access mixed operating point is valid for both-prefix quality and the tested in-GoF access semantics. This does not validate an end-to-end adaptive streaming system, transport behavior, arbitrary GoF switching, or perceptual equivalence.

## Coding method and fair accounting

- Targets are the **already decoded** Base/New geometry streams from the selected parent, not original floating checkpoints. This is a controlled lossy transcode. Correction payloads, including their header, stay byte-identical. Other attributes stay byte-identical. Additional Base error is therefore not secretly repaired by newly optimized correction.
- First-frame absolute bytes and quantizer parameters are copied exactly and fully charged. For later frames, xyz residual is target minus decoded previous xyz. Rotation is `Log(R_target · inverse(R_decoded_previous))`, decoded by left multiplication `Exp(r) · R_previous`, with wxyz at the renderer boundary. No quaternion subtraction.
- Predictor state is always the previous **decoded float32** state. Encoder/decoder state hashes and previous-state hashes are checked. Principal SO(3) rotation vectors are three-dimensional in both absolute and temporal variants; there is no dimension-reduction advantage.
- T1–T9 use parent precision (xyz q16, rotation q12). T9_q16/q12/q10/q8 sweep the predicted streams after frame 0; static state, initialization and correction remain fixed. A_q* controls recode the same decoded targets with absolute values, the same bit depth, and the same per-frame max-abs scale policy, without prediction.
- Per-frame signed fixed-point max-abs scale is transmitted, nearest-even rounding, no clipping. Every value block is a separate zstd level-6 frame (`--long=27 -T1`). Global JSON contains all reference/scale descriptors. Metadata includes recursively represented predecessor descriptors; their actual bytes are counted.
- Full cost includes Base, New, Correction, frame-0/reference cost, static state once, scales, metadata and zstd headers. Existing cameras and transport/filesystem overhead are excluded consistently. MB is decimal. Owner-inclusive byte columns already contain each owner’s headers; component value bytes + separate metadata is the nonoverlapping breakdown. No cost is inferred from in-memory array sizes.

## Owner × attribute breakdown

| Owner | Attribute | Parent raw B | Parent zstd B | Parent total % | T9_q10 zstd B |
|---|---|---:|---:|---:|---:|
| base | xyz | 1,251,000 | 1,216,446 | 18.492 | 802,892 |
| base | rotation | 938,250 | 902,258 | 13.716 | 760,972 |
| base | sh | 2,001,600 | 834,706 | 12.689 | 834,706 |
| base | scale | 187,650 | 174,711 | 2.656 | 174,711 |
| base | opacity | 62,550 | 61,270 | 0.931 | 61,270 |
| new | xyz | 1,101,630 | 926,075 | 14.078 | 569,313 |
| new | rotation | 826,225 | 804,403 | 12.229 | 667,569 |
| new | sh | 1,762,608 | 724,613 | 11.016 | 724,613 |
| new | scale | 165,245 | 155,272 | 2.360 | 155,272 |
| new | opacity | 55,082 | 52,772 | 0.802 | 52,772 |
| correction | xyz | 625,500 | 329,851 | 5.014 | 329,851 |
| correction | rotation | 625,500 | 357,619 | 5.437 | 357,619 |
| correction | sh | 0 | 0 | 0.000 | 0 |
| correction | scale | 0 | 0 | 0.000 | 0 |
| correction | opacity | 41,700 | 36,460 | 0.554 | 36,460 |
| metadata | headers | 11,289 | 1,612 | 0.025 | 1,989 |

Before prediction: Base/New geometry is 3,849,182 B (58.515% total); correction geometry is 687,470 B (10.451%). Thus the old 68.97% geometry figure is mainly Base/New, not correction.

## Temporal ablations and precision sweep

T1 Base xyz; T2 Base rotation; T3 Base both; T4 New xyz; T5 New rotation; T6 New both; T7 Base+New xyz; T8 Base+New rotation; T9 all four streams. A_* is the no-prediction control.

| Configuration | Full MB | Q0 loss dB | Q0 worst | Q1 loss dB | Q1 p95 | Q1 worst | Both pass |
|---|---:|---:|---:|---:|---:|---:|---|
| F2_mixed_cq8_frame_access | 6.578068 | 0.000959 | 0.004422 | 0.070788 | 0.133764 | 0.144644 | True |
| T1_same_precision | 6.511734 | 0.000934 | 0.004251 | 0.070719 | 0.133687 | 0.144726 | True |
| T2_same_precision | 6.553273 | 0.000948 | 0.004356 | 0.070801 | 0.133713 | 0.144496 | True |
| T3_same_precision | 6.486931 | 0.000955 | 0.004272 | 0.070748 | 0.133714 | 0.144436 | True |
| T4_same_precision | 6.528042 | 0.000959 | 0.004422 | 0.070708 | 0.133883 | 0.144559 | True |
| T5_same_precision | 6.545198 | 0.000959 | 0.004422 | 0.070784 | 0.133584 | 0.144377 | True |
| T6_same_precision | 6.495164 | 0.000959 | 0.004422 | 0.070722 | 0.133805 | 0.144559 | True |
| T7_same_precision | 6.461708 | 0.000934 | 0.004251 | 0.070564 | 0.133699 | 0.144347 | True |
| T8_same_precision | 6.520403 | 0.000948 | 0.004356 | 0.070796 | 0.133537 | 0.144228 | True |
| T9_same_precision | 6.404027 | 0.000955 | 0.004272 | 0.070607 | 0.133853 | 0.144060 | True |
| T9_q16 | 6.842118 | 0.000938 | 0.004160 | 0.070574 | 0.134024 | 0.144374 | True |
| A_q16 | 7.025696 | 0.000924 | 0.003964 | 0.070875 | 0.131820 | 0.144255 | True |
| T9_q12 | 5.971267 | 0.000760 | 0.004633 | 0.070613 | 0.131341 | 0.149652 | True |
| A_q12 | 6.126432 | -0.000027 | 0.005406 | 0.077659 | 0.128292 | 0.161986 | True |
| T9_q10 | 5.530009 | -0.000086 | 0.007494 | 0.074038 | 0.128356 | 0.146368 | True |
| A_q10 | 5.692608 | -0.007755 | 0.015801 | 0.146071 | 0.254747 | 0.334988 | False |
| T9_q8 | 4.701602 | -0.008253 | 0.004794 | 0.106991 | 0.228328 | 0.258076 | False |
| A_q8 | 5.012652 | -0.143639 | 0.003632 | 1.300703 | 2.201192 | 2.401297 | False |

| Operating point | Prefix | GT PSNR | SSIM | LPIPS | p95 loss | Worst loss |
|---|---|---:|---:|---:|---:|---:|
| F2_mixed_cq8_frame_access | q0 | 29.611452 | 0.96358080 | 0.02347257 | 0.003072 | 0.004422 |
| F2_mixed_cq8_frame_access | q1 | 39.577189 | 0.99482993 | 0.00535099 | 0.133764 | 0.144644 |
| A_q12 | q0 | 29.612438 | 0.96358545 | 0.02347038 | 0.004204 | 0.005406 |
| A_q12 | q1 | 39.570317 | 0.99482271 | 0.00535717 | 0.128292 | 0.161986 |
| T9_q10 | q0 | 29.612498 | 0.96358577 | 0.02347688 | 0.004477 | 0.007494 |
| T9_q10 | q1 | 39.573938 | 0.99482616 | 0.00535255 | 0.128356 | 0.146368 |

### Quality-matched interpretation

At the original precision, T9 costs 6.404027 MB: only 2.65% less than the parent. With the same q12 per-frame calibration policy, T9_q12 versus A_q12 saves 2.53% total. Prediction alone at fixed bit depth is therefore a modest coding gain.

The useful gain is that residuals tolerate q10 while A_q10 fails the Q1 budget. T9_q10 costs 5,530,009 B versus A_q12 6,126,432 B, a **9.74%** saving at the same stated quality budget. This is not mathematically identical quality: Q1 mean/worst losses are 0.074038/0.146368 versus 0.077659/0.161986 dB; Q0 worst loss is 0.007494 versus 0.005406 dB, and Q0 LPIPS is slightly higher for temporal coding. No BD-rate is claimed from this sparse sweep.

A separate near-parent quality filter (each prefix: mean loss no more than parent +0.01 dB; worst no more than parent +0.02 dB) also selects T9_q10. These tolerances are an explicit comparison convention, not proof of visual equivalence. T9_q8 fails both Q1 bounds (0.106991 mean, 0.258076 worst); do not select it because its total is smaller.

## Temporal redundancy diagnostics

Measured on decoded parent states, before predictor quantization. xyz norms use the normalized scene coordinate system; rotation angles are radians.

| Owner | Attribute | Mean range over transitions | Median range | p95 range |
|---|---|---:|---:|---:|
| base | xyz | 0.02339–0.03199 | 0.01918–0.02954 | 0.05449–0.05930 |
| base | rotation | 0.21226–0.23293 | 0.18206–0.19627 | 0.48101–0.54134 |
| new | xyz | 0.01994–0.02910 | 0.01614–0.02712 | 0.04807–0.05308 |
| new | rotation | 0.21989–0.23848 | 0.18096–0.19742 | 0.52876–0.57916 |

| Owner / attribute | Parent raw/zstd | T9 same-precision raw/zstd |
|---|---:|---:|
| base / xyz | 1.0284 | 1.0878 |
| base / rotation | 1.0399 | 1.0694 |
| new / xyz | 1.1896 | 1.2576 |
| new / rotation | 1.0271 | 1.0710 |

Same-precision value raw bytes are unchanged. Smaller residual magnitude does not itself reduce a fixed-width symbol’s byte count; it improves precision at a given bit depth. zstd benefits are modest; most additional saving at T9_q10 comes from fewer bits enabled by prediction. Static payload and correction are unchanged, and both xyz/rotation remain 3D. No learned entropy or motion model is used.

## Sequential and cold random access

Cold access includes all required shared static/header bytes and the chosen frame’s enhancement. Temporal Base/New references recursively fetch the necessary frame0→t geometry; unrelated prior-frame correction blocks are not fetched. Depth is number of predecessor links. Sequential costs and startup headers are separately recorded in access_cost.csv; a fresh full-Q1 sequential decode accounts for every transmitted byte.

| Frame index | Absolute Q0 B | Temporal Q0 B | Absolute Q1 B | Temporal Q1 B | Temporal depth |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,494,131 | 1,494,227 | 2,693,285 | 2,693,478 | 0 |
| 1 | 1,435,815 | 1,786,677 | 2,879,239 | 3,392,480 | 1 |
| 2 | 1,435,083 | 2,077,071 | 2,896,129 | 3,950,690 | 2 |
| 3 | 1,436,140 | 2,354,403 | 2,907,980 | 4,485,653 | 3 |
| 4 | 1,436,297 | 2,635,437 | 2,916,199 | 5,029,877 | 4 |

Worst Q1 cold access is 5,029,877 B for temporal versus 2,916,199 B for A_q12 (+72.48%). Relative to the previous mixed parent’s 3,029,315 B it increases 66.04%. Worst Q0 cold access is 2,635,437 B temporal versus 1,494,131 B absolute (absolute worst is frame0). Static state is not free. Cached/warm access will differ; no GoF switching latency or network throughput is simulated.

## Answers B1–B10

**B1. Q0 quality:** preserved by the selected mixed parent and temporal q10 under the specified budget. Q0’s lower absolute GT quality remains a property of the original lower level.

**B2. Valid point for both prefixes:** original parent remains valid; T9_q10 is the smallest tested full payload passing both mean/worst limits. A_q12 is the best tested absolute comparator under those limits.

**B3. Owner geometry bytes:** see the exact owner×attribute table. Parent Base xyz/rotation = 1,216,446/902,258 B; New = 926,075/804,403 B; Correction = 329,851/357,619 B.

**B4. Geometry savings:** including unchanged correction, xyz 2,472,372 → 1,702,056 B (31.16%); rotation 2,064,280 → 1,786,160 B (13.47%); total 15.93%. These include both prediction and the successful lower precision, not prediction alone.

**B5. Matched-quality gain:** 9.74% versus the budget-matched A_q12, with small nonidentical metric differences explicitly given above. Near-parent-tolerance saving is 15.93%. Same-bit q12 gain is only ~2.53%.

**B6. Base versus New:** both benefit. At same precision, Base geometry saves 91,329 B (4.31%) and New 83,087 B (4.80%); Base saves slightly more absolute bytes, New slightly more proportionally. At T9_q10 the corresponding savings are 554,840 B (26.19%) and 493,596 B (28.52%).

**B7. xyz versus rotation:** xyz has the larger measured coding benefit: same-precision Base+New xyz saves 116,569 B (~5.44%) versus rotation 57,847 B (~3.39%). This is a rate/quality observation, not a comparison of scene units to radians.

**B8. Access penalty:** significant for late cold access: +72.48% Q1 bytes versus A_q12, depth four at the last frame, despite lower sequential total. All dependencies and initialization are paid.

**B9. Remaining bottleneck:** xyz+rotation still occupy 63.08% of total; Base/New geometry alone is 2,800,746 B (50.65%). Static SH is now a substantial secondary component. Remaining geometry share is not a measure of theoretically removable redundancy.

**B10. CONTINUE, narrowly:** a material geometry rate share remains after a legitimate closed-loop baseline, and the measured quality-preserving reduction comes with a large cold-access penalty. That is enough to investigate the rate/access frontier further. It is **not** evidence to launch a complex new motion/learned codec or claim novelty: simple prediction already captures the demonstrated gain, zstd ratios at fixed precision are near one, and no remaining entropy bound or stronger codec comparison was measured. Next evidence should come from longer GoFs/more sequences and stronger conventional prediction/quantization controls before redesign.

## Validation and limits

validation.json checks raw/zstd file hashes/roundtrip, encoder and decoder closed-loop state hashes, exact absolute initialization, unchanged correction bytes, sequential versus fresh random-access decoding, Q0-only reads, full byte accounting and protected source hashes. Randomized SO(3) composition/sign/identity tests are reused. Raw and compressed decoding agree for every rendered candidate. Prefix correspondence is rechecked against checkpoint provenance; no new matching is introduced.

Only five consecutive frames of one sequence and 16 background-inclusive views/frame. No motion prediction beyond the previous state, no temporal coding of correction, no random-access points beyond frame0, no integer-domain lossless temporal codec, no foreground-specific metric or user study. Quantizer scales/headers are computed offline even though predictor reconstruction is closed-loop; this is not a real-time causal encoder claim. The fixed parent’s quantization errors cannot be undone by this decoded-target transcode.

Artifacts: summary.json/summary.csv, prefix_quality.csv, per_frame.csv, per_view.csv, owner_attribute_breakdown.csv, temporal_stats.csv, streams.csv, access_cost.csv, payload_index.json (encoder traces), validation.json, source_validation.json and serialized raw/zstd payloads. Earlier experiments are unchanged.

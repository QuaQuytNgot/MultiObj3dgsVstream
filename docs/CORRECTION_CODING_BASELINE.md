# Efficient correction coding baseline — Longdress

> **Migration note (2026-10-05):** project-owned preparation code and this document
> were migrated from the sibling `dynamic-lapis-gs` workspace. Measured results,
> PASS counts and hardware observations below describe that historical workspace;
> they do not establish a completed run in `MultiObj3dgsVstream`. Dataset, models,
> vendor binaries and generated `output/` artifacts are not bundled. Commands now
> use this project root and the pinned backend in `third_party/dynamic-lapis-gs`.
> See [migration/status](CONTENT_PREPARATION_MIGRATION.md) before reproducing a run.

Chạy thực tế ngày 2026-10-03 (UTC+7), trên 5 cặp checkpoint Longdress **1051–1055** có sẵn. Không train lại, không đổi renderer, không learned entropy codec/DASH. **29 configurations**, 16 test views/frame, 2,320 variant-view pairs.

## Kết luận A–E

**A. Sau static-state reuse, overhead còn bao nhiêu?** Không có một tỷ lệ duy nhất độc lập với format. Giữ IDs như B0, static reuse giảm raw correction (gồm metadata) từ **7,666,010 xuống 6,330,597 B/GoF**, nhưng new payload cũng giảm từ **43,333,582 xuống 12,780,401 B**: ratio tăng từ **17.69% lên 49.53%**, không được giữ denominator cũ để tuyên bố tiết kiệm. Tổ chức exact tốt nhất đã thử — dense theo prefix, tách static rows và tham chiếu base frame đầu khi bitwise bằng nhau — còn **3,321,619 B raw / 3,068,138 B zstd**, tức **25.99% raw / 26.57% compressed** so với new payload cùng format. Đây là kết quả toàn GoF 5 frame, không so trực tiếp với ~21% chỉ của từng dynamic frame ở báo cáo trước.

**B. Dense hay sparse tốt hơn?** **Dense là mặc định tốt hơn trên đoạn này.** B1 dense exact nén còn 3.637 MB correction, thấp hơn B0 IDs 4.004 MB. Với residual q12 + static split, sparse nhẹ chỉ tiết kiệm **362 B/GoF** so với dense; threshold cao hơn đánh đổi quality rõ. Sparse aggressive q12 cần **0.847 MB**, nhưng mất **1.639 dB**; dense q8 vừa nhỏ hơn (**0.727 MB**) vừa giữ quality tốt hơn (**0.040 dB** mất).

**C. xyz/rotation residual compress tốt không?** **Không tự động tốt hơn ở float32.** xyz replacement 2,001,600 B → 1,420,216 B zstd, trong khi Δxyz f32 cùng kích thước → 1,801,460 B. Rotation SO(3) residual dùng 3 thay vì 4 thành phần nên raw nhỏ hơn, nhưng chỉ nén 2,001,600 → 1,859,641 B. Tổng f32 residual correction 3.813 MB còn lớn hơn dense replacement 3.637 MB. **Quantization** mới tạo phần lớn mức giảm thực tế.

**D. Có giảm mạnh correction mà mất rất ít quality không?** Có trên run này. So với exact baseline mạnh nhất đã thử:
- Dense q12: **1,432,286 B zstd**, giảm **53.3174% correction**, mean PSNR loss vs GT **0.000207 dB**, worst view **0.009834 dB**.
- Dense q8: **726,919 B zstd**, giảm **76.3075% correction**, mean loss **0.039941 dB**, worst view **0.107363 dB**.
- Tổng payload **base + new + correction** chỉ giảm tương ứng **5.8289% / 8.3422%**; không đánh đồng giảm correction với giảm toàn representation.

**E. Baseline đã đủ tốt chưa, hay còn trade-off để justify joint method?** Đây đã là **strong practical baseline**: riêng hiện tượng non-composable không đủ biện minh cho một phương pháp phức tạp. Muốn exact vẫn cần ~3.07 MB correction/GoF; chấp nhận lỗi rất nhỏ giảm xuống ~0.73–1.43 MB. Trade-off exactness–bytes còn rõ, nhưng một joint state-sharing/refinement method phải vượt baseline này ở **tổng bytes, cùng quality, cùng accounting**, hoặc chứng minh lợi ích latency/robustness khác. Experiment hiện tại chưa chứng minh cần redesign training hoặc joint method sẽ tốt hơn.

## 1. Lifetime xác minh từ checkpoint, không hard-code

Đã chạy lại prefix-lineage/checksum validation của phase real. Có 41,700 shared + 36,721 new identities/frame. Phân loại bằng so sánh **bitwise từng hàng qua cả năm frame**; số row static được lưu trong lifetimes.json (`output/correction_coding_baseline/lifetimes.json`, historical artifact).

| State | xyz static rows | rotation static rows | Opacity / scale / SH static |
|---|---:|---:|---|
| Q0 base | 17 / 41,700 | 17 / 41,700 | Toàn bộ |
| Q1 shared target | 13,654 / 41,700 | 13,654 / 41,700 | Toàn bộ |
| Q1 new suffix | 0 / 36,721 | 0 / 36,721 | Toàn bộ |
| Cross-quality residual | 15 / 41,700 | 15 / 41,700 | Toàn bộ |

**Q1 state static không đồng nghĩa correction residual static**: một Q1 row đứng yên trong khi Q0 row tương ứng di chuyển vẫn cần residual thay đổi. Scale/SH shared bằng Q0 chính xác nên không gửi correction cho chúng. Opacity override/residual không đổi theo thời gian trong dữ liệu này, gửi một lần khi bật split. Base/new áp dụng reuse theo cả nhóm thuộc tính; B4 exact row-static/base-reference còn khai thác 13,654 shared target rows static.

## 2. Representation và decoder

Script: [tools/evaluate_correction_coding.py](../tools/evaluate_correction_coding.py). Decoder chỉ đọc payload directory; **không nhận/checkpoint Q1 hoặc manifest huấn luyện**. Q0 cũng nằm trong payload/accounting, không là side information miễn phí.

- **B0:** uint32 IDs + float32 replacement cho từng shared row thay đổi. Value và ID đặt trong các stream riêng để dùng chung cách layout/nén với các baseline khác. Reinterleave các stream này **bằng từng byte toàn bộ correction records của phase cũ**; tổng raw values+IDs giữ nguyên 7,662,352 B, cộng riêng 3,658 B metadata mới. Vì đã tách channels, không gọi số nén B0 là kết quả nén nguyên file AoS cũ.
- **B1:** dense float32 replacement theo prefix order, không IDs; bỏ block nếu toàn bộ nhóm không đổi trong frame.
- **B2:** dense residual, thử f32/f16 và signed fixed-point q16/q12/q8. xyz/opacity dùng hiệu trong miền native (xyz, opacity **logit**); scale/SH không phát sinh correction ở GoF này.
- **B3:** sparse residual f16, chỉ giữ row vượt threshold; chọn uint32 IDs hoặc bitmask theo cái nào ít raw bytes hơn, tính đủ index/mask. Không dùng learned selection.
- **B4:** static/dynamic split cho **base, new và correction**, không chỉ correction numerator. Có exact IDs/dense, exact row-static, exact base-reference, dense residual năm precision và sparse residual f16/q12 bốn threshold.
- **Exact row-static:** target row nào không đổi trong GoF được lưu một lần, các row còn lại theo frame, mask lưu một lần. **Exact base-reference** bỏ cả static value nếu nó bằng bitwise Q0 frame đầu, và gửi chỉ dẫn copy từ base frame 1051 đã có trong payload. Điều kiện được kiểm chứng trên dữ liệu, không dự đoán/mượn target miễn phí.
- **Naive split:** base + suffix, không corrections; vẫn có correction header rỗng, nên 105 B compressed không phải correction values.

Rotation residual đúng hình học:
`d = Log(R1 · R0⁻¹)` (3-vector trục–góc, radians), decode `R̂1 = Exp(d̂) · R0`.
PLY wxyz được đổi sang xyzw khi gọi SciPy rồi đổi lại. Đã kiểm tra composition, identity và ±q; xem định nghĩa [rotation composition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.__mul__.html) và [rotation vector](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_rotvec.html).

**f32 residual không phải lossless bitwise**: phép trừ/cộng có rounding, rotation vector bỏ norm/sign dư thừa của raw quaternion. Chỉ replacement baselines được gắn exact; cả sáu bản exact qua bitwise state và float-render equality (PSNR ∞ / SSIM 1 / LPIPS 0). Residual f32 rất gần Q1 nhưng không được ghi là exact.

Fixed-point dùng một scale/attribute/GoF: `step=max_abs_residual/(2^(bits-1)-1)`; round-to-nearest-even, signed 8/12/16-bit **bit-packed thực sự** (q12 không bị tính như int16). Scale nằm trong metadata, không có silent clipping. q8 steps: xyz **0.00357225 scene units**, rotvec **0.02417430 rad**, opacity residual **0.09851657 logit**. Base/new luôn float32 exact; chỉ corrections được quantize.

Thresholds sparse được xác định trước khi đo image quality:

| Mức | ‖Δxyz‖ scene units | Rotation angle | abs(Δalpha) sau sigmoid |
|---|---:|---:|---:|
| mild | 0.001 | 0.5° | 0.001 |
| medium | 0.005 | 2° | 0.01 |
| coarse | 0.02 | 5° | 0.03 |
| aggressive | 0.05 | 15° | 0.1 |

Giữ row nếu magnitude **lớn hơn** threshold tương ứng; không hard-code phần trăm sparse.

## 3. Nén và accounting công bằng

Dùng **zstd CLI 1.5.7, level 6, `--long=27 -T1`**, không dictionary học. Mỗi owner/category/attribute có một stream xuyên GoF; mỗi stream và JSON header được nén riêng. Cửa sổ tối đa 128 MiB cho phép compressor nhận phần static lặp lại ngay cả ở B0/B1/B2/B3. Đây là offline GoF baseline, **không giả định latency streaming/frame-independent decoding**.

Tất cả số là **byte thực từ file serialize/compress**, không estimate entropy. Tính đủ:
`total = base + new + correction`;
mỗi owner gồm value streams, index/mask nếu có và JSON metadata của chính nó. Header cấu trúc chung nằm trong base. zstd frame headers được tính; không tính filesystem blocks, network/container vận chuyển, camera/renderer weights/GT hay báo cáo ngoài payload (những phần này chung cho mọi variant).

Các file chung được hardlink để tiết kiệm disk cục bộ; **vẫn tính đủ logical file bytes cho từng candidate**, không nhận miễn phí base/new vì hardlink. Xem streams.csv (`output/correction_coding_baseline/streams.csv`, historical artifact) và `payload_index.json` để audit từng file/hash/byte count.

Raw base/new value bytes, chưa metadata:

| Layout | Base static | Base dynamic | Base values tổng | New static | New dynamic | New values tổng |
|---|---:|---:|---:|---:|---:|---:|
| Không reuse (B0–B3) | 43,368,000 | 5,838,000 | 49,206,000 | 38,189,840 | 5,140,940 | 43,330,780 |
| Reuse (B4/naive split) | 8,673,600 | 5,838,000 | 14,511,600 | 7,637,968 | 5,140,940 | 12,778,908 |

B4 thêm base metadata 2,987 B và new metadata 1,493 B. Sau zstd, **base=13,448,421 B, new=11,548,140 B**, đã gồm metadata. Không đổi denominator này giữa các B4 candidates.

Breakdown correction, toàn GoF:

| Variant | Static values raw | Dynamic values raw | IDs/masks raw | Metadata raw | Tổng correction raw | Tổng correction zstd |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 833,960 | 4,662,336 | 2,166,056 | 3,658 | 7,666,010 | 4,003,683 |
| B4 exact IDs | 166,792 | 4,662,336 | 1,498,888 | 2,581 | 6,330,597 | 4,003,488 |
| B4 exact dense | 166,800 | 4,670,400 | 0 | 1,713 | 4,838,913 | 3,636,520 |
| B4 exact base-reference | 166,800 | 3,141,152 | 10,426 | 3,241 | 3,321,619 | 3,068,138 |
| B4 residual f16 | 83,400 | 2,001,600 | 0 | 1,700 | 2,086,700 | 1,934,252 |
| B4 residual q12 | 62,550 | 1,501,200 | 0 | 2,756 | 1,566,506 | 1,432,286 |
| B4 residual q8 | 41,700 | 1,000,800 | 0 | 2,723 | 1,045,223 | 726,919 |
| B4 sparse q12 coarse | 44,457 | 1,326,768 | 46,917 | 3,665 | 1,421,807 | 1,296,045 |

Static reuse **đơn thuần** (B0 → B4 exact IDs) chỉ giảm compressed correction **195 B**, bởi zstd long-window đã mã hóa hiệu quả việc lặp. Dense layout/row reuse/precision mới tạo giảm bytes đáng kể. Với B4 exact base-reference, correction/new = **25.9899% raw / 26.5682% zstd**; q12 = **12.2571% / 12.4027%**; q8 = **8.1783% / 6.2947%**. Không gọi các tỷ lệ này là bitrate codec hoặc cận dưới tối ưu.

## 4. Rate–quality frontier

Renderer/GT/cameras giống phase real: 16 held-out views/frame ở 256×256, SH degree 3, nền đen. Metric RGB float clamp [0,1], có background, trung bình đều trên 80 views; PSNR trung bình dB. SSIM/LPIPS-VGG implementation gốc; LPIPS nhận [0,1] như repo. Official Q1 vs GT: **PSNR 39.647976 dB, SSIM 0.994931, LPIPS 0.005287**.

Bảng dưới gồm các điểm frontier và control quan trọng; **MB = 1,000,000 B**, tất cả rates tính cho **toàn GoF 5 frame**, đã bao gồm metadata/index. Exactness là ràng buộc riêng, không loại exact baseline chỉ vì một cấu hình lossy có metric GT cao hơn vài phần triệu.

| Variant | Correction zstd MB | **Total** zstd MB | GT PSNR ↑ | GT SSIM ↑ | GT LPIPS ↓ | Mean PSNR loss dB ↓ |
|---|---:|---:|---:|---:|---:|---:|
| B0 raw IDs | 4.003683 | 29.007232 | 39.647976 | 0.994931 | 0.005287 | 0.000000 |
| B1 dense exact | 3.636619 | 28.640168 | 39.647976 | 0.994931 | 0.005287 | 0.000000 |
| B2 dense residual f32 | 3.813338 | 28.816887 | 39.647977 | 0.994931 | 0.005287 | -0.000000 |
| B3 sparse f16 mild | 1.931789 | 26.935338 | 39.648020 | 0.994931 | 0.005286 | -0.000043 |
| B3 sparse f16 coarse | 1.713332 | 26.716881 | 39.101947 | 0.993780 | 0.006181 | 0.546029 |
| B4 exact + IDs | 4.003488 | 29.000049 | 39.647976 | 0.994931 | 0.005287 | 0.000000 |
| B4 exact dense | 3.636520 | 28.633081 | 39.647976 | 0.994931 | 0.005287 | 0.000000 |
| B4 exact row-static | 3.422726 | 28.419287 | 39.647976 | 0.994931 | 0.005287 | 0.000000 |
| B4 exact base-reference | 3.068138 | 28.064699 | 39.647976 | 0.994931 | 0.005287 | 0.000000 |
| B4 dense residual f32 | 3.813237 | 28.809798 | 39.647977 | 0.994931 | 0.005287 | -0.000000 |
| B4 dense residual f16 | 1.934252 | 26.930813 | 39.647948 | 0.994931 | 0.005287 | 0.000028 |
| B4 dense residual q16 | 1.913130 | 26.909691 | 39.647913 | 0.994931 | 0.005286 | 0.000064 |
| B4 dense residual q12 | 1.432286 | 26.428847 | 39.647770 | 0.994931 | 0.005285 | 0.000207 |
| B4 dense residual q8 | 0.726919 | 25.723480 | 39.608035 | 0.994880 | 0.005325 | 0.039941 |
| B4 sparse q12 mild | 1.431924 | 26.428485 | 39.647857 | 0.994931 | 0.005285 | 0.000119 |
| B4 sparse q12 medium | 1.423665 | 26.420226 | 39.631978 | 0.994819 | 0.005343 | 0.015998 |
| B4 sparse q12 coarse | 1.296045 | 26.292606 | 39.101792 | 0.993779 | 0.006179 | 0.546185 |
| B4 sparse q12 aggressive | 0.846709 | 25.843270 | 38.008734 | 0.991683 | 0.008101 | 1.639242 |
| Naive, no corrections | 0.000105 | 24.996666 | 37.856629 | 0.992860 | 0.007268 | 1.791347 |

`summary.csv` có đủ 29 configurations, raw/compressed breakdown và cờ Pareto. Có hai frontier: (total zstd bytes, GT PSNR) và thêm SSIM/LPIPS. Đây là **strict numerical Pareto**, không phải bằng chứng thống kê cho những gain cực nhỏ do quantization; không diễn giải +0.00004 dB là cải thiện chắc chắn. `per_frame.csv`/`per_view.csv` giữ chi tiết, kể cả metric so Q1.

Kiểm tra quality không chỉ nhìn mean:

| Dense B4 precision | Mean GT PSNR loss | p95 view loss | Worst view loss | Mean PSNR vs Q1 | Max abs(pixel−Q1) toàn 80 views |
|---|---:|---:|---:|---:|---:|
| f16 | 0.000028 | 0.001198 | 0.005545 | 91.342 | 0.075829 |
| q16 | 0.000064 | 0.001258 | 0.006274 | 93.362 | 0.062167 |
| q12 | 0.000207 | 0.004216 | 0.009834 | 77.458 | 0.117247 |
| q8 | 0.039941 | 0.095584 | 0.107363 | 62.021 | 0.221172 |

Vì vậy “rất ít quality loss” là kết quả theo metric/view ở đây, **không phải đảm bảo mỗi pixel gần Q1**. q12/q8 vẫn có isolated max RGB error 0.117/0.221; SSIM/LPIPS và worst-view PSNR cũng được cung cấp để tránh che lỗi bằng trung bình/background.

### Residual có thực sự nén tốt?

Cùng stream layout, window và prefix order:

| Stream | Raw B | zstd B | zstd/raw |
|---|---:|---:|---:|
| xyz replacement f32 (B1) | 2,001,600 | 1,420,216 | 70.95% |
| Δxyz f32 (B4) | 2,001,600 | 1,801,460 | 90.00% |
| Raw quaternion replacement f32 (B1) | 2,668,800 | 2,068,875 | 77.52% |
| Rotvec residual f32 (B4) | 2,001,600 | 1,859,641 | 92.91% |
| Δxyz q12 (B4) | 750,600 | 677,427 | 90.25% |
| Rotvec q12 (B4) | 750,600 | 693,367 | 92.38% |
| Δxyz q8 (B4) | 500,400 | 332,298 | 66.41% |
| Rotvec q8 (B4) | 500,400 | 357,727 | 71.49% |

Reduction trước nén do bỏ IDs, reuse hoặc giảm số component/precision phải được phân biệt với hiệu quả lossless compressor. Không có bảo đảm rằng residual có entropy thấp chỉ vì magnitude nhỏ.

## 5. Validation, giới hạn và tái chạy

Đã qua validation.json (`output/correction_coding_baseline/validation.json`, historical artifact):

- Prefix/checkpoint lineage; B0 reinterleaved record equality và raw counts khớp experiment trước.
- Signed bit-packing q8/q12/q16, quantization half-step bounds, SO(3) composition/±q/identity, zstd roundtrip.
- Decoder compressed-only chạy được với directory chỉ có `.zst`, không raw files/manifest/checkpoints; compressed/raw decode equality cho mọi variant/frame.
- **6 exact variants × 80 views = 480 exact float renders**, toàn bộ original renderer attributes bitwise bằng Q1. 29×5 = 145 frame-variant pairs, **2,320 view-variant rows**.
- Naive regression khớp phase real; từng payload file size/hash kiểm tra lại; source training/renderer hash không đổi; py_compile và git diff checks qua.
- Các payload trước khi thêm base-reference được render lại với decoder cuối; results giữ nguyên.

Giới hạn: một Longdress GoF ngắn đã train rút ngắn ở phase trước, không retrain/Soldier/multiple seeds; đây là ablation exploratory trên cùng test cameras, không có validation set riêng để chọn threshold. Các calibration max/lifetime dùng đầy đủ GoF tại encoder. Không báo con số này như tối ưu entropy, codec chuẩn, network bitrate hay kết quả về latency. Base/new vẫn float32, chưa có temporal predictor chuyên dụng, semantic visibility filtering, hay các phép tổ chức/nén khác; joint method mới phải so với baseline mạnh và giữ accounting toàn representation.

```bash
cd MultiObj3dgsVstream
conda activate Hoang
python tools/evaluate_correction_coding.py --self-test

# Dùng payload hiện có, đo lại bằng renderer gốc:
python tools/evaluate_correction_coding.py --render-only

# Tạo một run mới, không overwrite kết quả này:
python tools/evaluate_correction_coding.py \
  --manifest output/progressive_gap_real/manifest.json \
  --output output/correction_coding_baseline_repeat
```

Không cài thêm package vào Hoang: dùng zstd CLI đã có, SciPy 1.17.1, Python 3.11.16, torch 2.3.1+cu118 và renderer vendor của phase real. Muốn decode độc lập trên CPU: `Decoder(payload_directory, compressed=True).frame(frame_index)` trả về structured Gaussian array; frame index 0…4. Thông số để dựng Gaussian đều nằm trong payload.

Artifacts: summary.json (`output/correction_coding_baseline/summary.json`, historical artifact), summary.csv (`output/correction_coding_baseline/summary.csv`, historical artifact), `per_frame.csv`, `per_view.csv`, `lifetimes.json`, `streams.csv`, `payload_index.json`, `validation.json`, `payloads/<variant>/` (raw + zstd + headers), `renders/<variant>/<frame>/` (view 0 để xem nhanh). `blob_pool/` chỉ là backing store hardlink, không phải bytes gửi thêm. `output/` vẫn gitignore; không publish dữ liệu/ảnh/model 8i.

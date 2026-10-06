# Longdress: raw download, encoding trial and local viewer

> **Migration note (2026-10-05):** project-owned preparation code and this document
> were migrated from the sibling `dynamic-lapis-gs` workspace. Measured results,
> PASS counts and hardware observations below describe that historical workspace;
> they do not establish a completed run in `MultiObj3dgsVstream`. Dataset, models,
> vendor binaries and generated `output/` artifacts are not bundled. Commands now
> use this project root and the pinned backend in `third_party/dynamic-lapis-gs`.
> See [migration/status](CONTENT_PREPARATION_MIGRATION.md) before reproducing a run.

Ngày chạy: 2026-10-05. Object: Longdress, 8i Voxelized Full Bodies v2. [Mô tả chính thức của JPEG](https://plenodb.jpeg.org/pc/8ilabs) ghi 30 FPS và khoảng 10 giây mỗi sequence.

## Phạm vi thực tế

- **Đã tải đủ** raw frame **1051–1350**, 300 frame, **5,685,631,637 bytes (~5.69 GB)**. Download có inventory, SHA-256, ZIP CRC, ETag, atomic writes, resume và file license đi kèm. Inventory: `output/datasets/8i/longdress/inventory.json`, `complete_for_selection: true`. ZIP nguồn có 1,635,191,404 bytes; downloader lấy selected PLY members, không giữ toàn ZIP.
- Encode thử **1051–1055**, từ 10 checkpoint Q0/Q1 đã có trong `output/progressive_gap_real/manifest.json`. Không chạy thêm optimization/training. Q0 có 41,700 Gaussians; Q1 có 78,421 Gaussians.
- Hai biến thể **f32 lossless** và **f16 lossy**, cả independent Q0/Q1 và progressive Base/E1. Codec chính xác là `gaussian_attribute_zlib` **1.0.0**, zlib level 6; đây là baseline attribute codec, **không phải encoder LTS enhanced Draco**. Xem [codec verification](CONTENT_PREPARATION_CODEC.md).
- Mỗi biến thể chạy đủ 9 stages, 256 tasks; lần resume sau đó chạy 0 và skip 256. Stage `train` ở đây chỉ kiểm tra/import checkpoint bằng backend `existing`.
- Decode chỉ từ requestable CPSEG1 containers; render bằng `gaussian_renderer.render` gốc, orbit Z-up, 256×256, 3 views × 2 scales × 3 sampled frames. Có 72 profile rows và 90 GPU view operations mỗi biến thể. LPIPS VGG pretrained chạy CPU, batch 1.
- Proxy cố định 256 stable IDs, 5 frames; có alpha/depth/silhouette validation. Đây là baseline proxy chưa tối ưu.

## Duration và bytes

Raw sequence: **300/30 = 10 giây**. Clip đã encode: **5/30 = 0.166666667 giây**. Timestamp span clip là **4/30 = 0.133333333 giây**. Không thể dùng 300 raw frames đã download để tuyên bố có 10 giây 3DGS encoded: hiện chỉ 5 frame có checkpoint tương ứng.

Các số dưới đây là **bytes của đầy đủ requestable segments cho toàn clip 5 frame**, gồm headers:

| Biến thể | Independent Q0 | Independent Q1 | Progressive Base | Progressive E1 riêng | Progressive Q1 = Base + E1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| f32 lossless | 47,870,778 | 89,774,787 | 47,870,788 | 48,170,269 | 96,041,057 |
| f16 | 25,010,627 | 46,850,121 | 25,010,637 | 25,542,000 | 50,552,637 |

Progressive highest quality lớn hơn independent Q1 khi tải từ đầu: khoảng **6.98% với f32**, **7.90% với f16**. Chi phí upgrade khi client đã có Base là E1, không phải Base + E1. Tổng storage của catalog chứa mọi quality là một phép tính khác. `catalog.json` và từng `run_report.json` giữ riêng các phép accounting này, cả cold access tại mỗi frame và complete segment request bytes.

Tổng encode-task wall time được giữ trong journal: f32 **27.849402 s**, f16 **20.734312 s**. Task bao gồm CPU load, encode, verification decode và ghi encoder cache; không bao gồm package/profile và không phải pure codec kernel benchmark. Latest stage wall time sau resume chủ yếu là kiểm tra/skip, nên không dùng nó thay thời gian encode ban đầu.

## Validation

- **76 self-tests PASS**, 9.595 s; Python syntax/import/CLI checks PASS. Web được kiểm tra bằng Chromium thật ở đủ 144 tổ hợp selector, links/assets hợp lệ, không có page errors, mobile không tràn ngang và missing-catalog error được xử lý.
- Rehash lại **đủ 300** raw files: SHA-256, ZIP CRC và size hợp lệ; mọi PLY header có vertex count dương. Open3D đọc frame 1051/1200/1350, lần lượt 765,821 / 800,259 / 806,806 points; coordinates/colors finite. License hash hợp lệ, không còn `.part`/`.tmp`. Resume thật verify đủ 300 files và download **0 member bytes**, chỉ 65,607 bytes ZIP metadata.
- f32: checkpoint → exported state → decoded independent/progressive **bit-exact** tại tất cả 10 quality/frame states.
- f16: decoded state đúng từng giá trị của phép `float32 → float16 → float32`; hai delivery modes reconstruct giống nhau.
- E1 có additions và shared-state replacements đủ mọi thuộc tính thay đổi, verified decoded-parent hashes; không giả định chỉ có Gaussians mới.
- Profile so với highest decoded Q1 của từng biến thể; Q1 self-reference MSE=0, PSNR=`"inf"`, LPIPS=0. Ảnh profile đều có foreground, alpha/depth hợp lệ; raw condition rows được giữ lại.
- Đánh giá codec riêng: 36 matching view/scale/time conditions so f16 với f32 decoded **cùng quality**, không phải GT. Q0 mean MSE=3.18686e-8, mean LPIPS=3.81842e-6; Q1 mean MSE=1.60564e-7, mean LPIPS=7.99785e-6. Xem `codec_quality_vs_f32.json` cho toàn bộ rows.
- GPU guard ghi `peak_active=1`, `active=0`, 90 view operations mỗi biến thể. Peak PyTorch CUDA allocation **62,297,600 bytes (~59.41 MiB)** trong trial này; không đo driver/context hoặc full training.
- Upstream training/loss/renderer/CUDA source giữ nguyên; input checkpoint hashes và original training provenance được xác thực. Báo cáo: `validation_summary.json`, `audit_f32.json`, `audit_f16.json`.

Dataset/browser evidence: `dataset_validation.json`, `dataset_resume_validation.json`, `viewer_validation.json` và screenshots desktop/mobile trong trial output. Compact backup summary được lưu trong Git tại [docs/validation/longdress_encoding_trial_summary.json](validation/longdress_encoding_trial_summary.json).

## Local viewer

```bash
cd MultiObj3dgsVstream
python -m http.server 8765 --bind 127.0.0.1 \
  --directory output/content_prepare_longdress_trial
```

Mở **http://127.0.0.1:8765/**. Viewer chọn f32/f16, mode, quality, sampled frame, view, scale; hiển thị duration, byte costs, dependencies, Gaussian counts và links manifest/profile/proxy. Có 144 PNG previews được xuất từ decoded renderer caches. Preview hiện có **3 sampled keyframes 1051/1053/1055**, không phải animation 10 giây hoặc browser 3DGS renderer. Web không chạy training/GPU; không có framework, npm hoặc build step.

## Lệnh reproducible

```bash
conda activate Hoang

# Whole raw sequence; repeat the same command to resume/verify.
python tools/download_8i_object.py --object longdress --all-frames \
  --output output/datasets/8i --cache output/progressive_gap_real/raw \
  --allow-http-fallback --download-workers 4

# Existing checkpoints only: these configurations do not retrain.
python tools/content_preparation/prepare_content.py \
  --config configs/content_prepare_longdress_trial_f32.yaml --resume
python tools/content_preparation/prepare_content.py \
  --config configs/content_prepare_longdress_trial_f16.yaml --resume

# Read-only state/media audit (CPU), then regenerate the catalog/PNGs/web.
python tools/validate_8i_trial.py \
  --root output/content_prepare_longdress_trial/f32_lossless/longdress \
  --source-manifest output/progressive_gap_real/manifest.json --precision f32 \
  --report output/content_prepare_longdress_trial/audit_f32.json
python tools/validate_8i_trial.py \
  --root output/content_prepare_longdress_trial/f16/longdress \
  --source-manifest output/progressive_gap_real/manifest.json --precision f16 \
  --report output/content_prepare_longdress_trial/audit_f16.json
python tools/summarize_8i_trial.py \
  --trial-root output/content_prepare_longdress_trial \
  --inventory output/datasets/8i/longdress/inventory.json
python tools/content_preparation/self_test.py
```

HTTPS của official archive hiện lỗi certificate verification trong môi trường này, kể cả system CA bundle. Downloader giữ TLS verification bật và chỉ dùng HTTP cùng hostname khi có `--allow-http-fallback`. Inventory ghi rõ transport này. ZIP CRC kiểm tra archive member; SHA-256 phục vụ resume/reproducibility. Chúng không phải publisher signature. License được giữ nguyên tại `output/datasets/8i/license.pdf`.

Default download xử lý một member tại một thời điểm, giữ compressed cache tối đa 8 MiB. Trial dùng `--download-workers 4` chỉ cho network/CPU, mỗi worker có reader/ZIP riêng, main thread ghi inventory; memory cache tối đa 8 MiB/worker. Download utilities không dùng GPU. Pipeline training/render/profiling luôn xử lý một object/quality/frame/view, LPIPS batch 1.

Config preparation tiếp theo `configs/content_prepare_longdress.yaml` đã trỏ raw input vào thư mục download và dùng đúng Z-up. Config này vẫn chọn **30 frame 1051–1080**, 3 qualities; chưa chạy. Muốn full 300 frame ở phase sau, sửa `frames.end: 1350` và chọn GoF/view/time sampling phù hợp trước khi chạy. Không có full Longdress training/preparation trong trial này.

Raw data, checkpoints và generated outputs được Git ignore. Push source/config/docs/tests; giữ dataset/media outputs trên storage riêng.

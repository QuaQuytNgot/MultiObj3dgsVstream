# Progressive gap trên real volumetric data — Longdress

> **Migration note (2026-10-05):** project-owned preparation code and this document
> were migrated from the sibling `dynamic-lapis-gs` workspace. Measured results,
> PASS counts and hardware observations below describe that historical workspace;
> they do not establish a completed run in `MultiObj3dgsVstream`. Dataset, models,
> vendor binaries and generated `output/` artifacts are not bundled. Commands now
> use this project root and the pinned backend in `third_party/dynamic-lapis-gs`.
> See [migration/status](CONTENT_PREPARATION_MIGRATION.md) before reproducing a run.

Ngày thực hiện: 2026-10-03 (UTC+7). **Đã chạy end-to-end**, không dùng kết quả synthetic để điền bảng real-data.

## Trả lời A–E

**A. Q1 có thể tái tạo bằng Q0 + only-new-Gaussians không?** Không, trên cả 5 frame Longdress 1051–1055. Prefix identity vẫn hợp lệ nhưng shared state không bất biến: 41,698/41,700 shared Gaussians (**99.9952%**) khác ít nhất một thuộc tính. Naive khác official Q1 trên **80/80 test views**; thêm shared-state corrections tái tạo đúng từng bit renderer attributes và đúng từng pixel float.

**B. Shared state nào gây mismatch nhiều nhất?** **Opacity** ở frame đầu và lớn nhất theo ablation cô lập trên bốn frame động của run này; **xyz** cũng đóng góp đáng kể. Rotation thay đổi nhưng tác động ảnh nhỏ hơn hai nhóm trên; scale/SH bằng nhau chính xác. Không so trực tiếp magnitude khác đơn vị để xếp hạng. Rotation lại chiếm nhiều **byte correction** nhất ở frame động, khoảng 45.4–45.5%, do phải ghi quaternion 4 thành phần.

**C. Mismatch tồn tại qua nhiều frame hay chỉ một vài frame?** Cả **5/5 frame liên tiếp**, không frame nào composable append-only. Naive-vs-Q1 PSNR lần lượt **45.6357, 42.3025, 41.6316, 41.1847, 40.6690 dB**. Trong đoạn này gap ảnh tăng theo thời gian, nhưng năm frame ngắn không chứng minh drift dài hạn hoặc tính đơn điệu trên sequence khác.

**D. Raw correction overhead lớn đến đâu?** New-Gaussian payload cố định **8,666,156 B/frame**. Correction **333,584 B** ở frame đầu (**3.8493%** new), và **1,830,644–1,834,244 B/frame** ở frame động (**21.1241–21.1656%** new), chưa tính metadata nhỏ. Đây là **raw representation overhead**, tuyệt đối không phải bitrate codec hay cận dưới nén.

**E. Đủ mạnh để tiếp tục nghiên cứu prefix-consistent dynamic layered 3DGS không?** **Có, để biện minh cho bước nghiên cứu tiếp theo.** Counterexample đã lặp trên point cloud người thật, đủ 100 camera train của repo, khởi tạo ngẫu nhiên 100,000 điểm mặc định, và multiview preprocessing gốc; do đó hiện tượng không chỉ là artifact của hai ellipsoid synthetic/512 điểm. Việc loại bỏ corrections còn giảm PSNR so với ground truth khoảng **1.11–2.13 dB** ở đây. Tuy nhiên chưa chứng minh một phương pháp prefix-consistent mới sẽ có rate–distortion tốt hơn: chỉ một sequence, hai resolutions, một seed, năm frame và lịch train rút ngắn; chưa chạy Soldier, full 30k iterations, nhiều seed hoặc đánh giá nén.

## Dữ liệu, provenance và thiết lập

- Dataset: **8i Voxelized Full Bodies v2, Longdress**, frame gốc **1051–1055**, 765,821–806,026 điểm RGB/frame. Đây là dữ liệu volumetric người thật; ảnh supervision được render từ point cloud theo preprocessing của repo, **không phải ảnh camera RGB gốc của hệ capture**.
- Nguồn: [JPEG Pleno / 8i Labs](http://plenodb.jpeg.org/pc/8ilabs), archive [longdress.zip](http://plenodb.jpeg.org/pc/8ilabs/longdress.zip). Dùng HTTP Range để lấy đúng 5 PLY: **23,731,912 B truyền** thay vì tải toàn archive 1,635,191,404 B. Ghi ZIP CRC32, ETag, SHA256 từng PLY trong `raw/provenance.json`. HTTPS gặp lỗi certificate chain tại máy; dùng endpoint HTTP chính thức, không tắt xác minh TLS toàn cục. CRC/SHA256 cục bộ không phải chữ ký xác thực của publisher.
- Đã lưu giấy phép đi kèm (`output/progressive_gap_real/raw/license.pdf`, historical artifact); dữ liệu/ảnh/model nằm trong `output/` bị gitignore, không commit hay publish dataset. Credit: Eugene d’Eon, Bob Harrison, Taos Myers, Philip A. Chou, *8i Voxelized Full Bodies – A Voxelized Point Cloud Dataset*, MPEG/JPEG WG11M40059/WG1M74006, January 2017; owner 8i Corporation.
- Repo commit `da8efaa42a9d021b5ff2958c0a87eba6ed7a47c4`. Hoang: Python 3.11.16, torch 2.3.1+cu118, CUDA 11.8, GTX 1660 6 GB. Dùng lại binary submodule chính thức đã build ở phase synthetic qua symlink `vendor`; hash binary ghi trong `extensions.json`, không đổi package chung của Hoang.
- **Giữ 100/100 train camera** từ `transforms_train.json`. Chọn 16/200 test camera bằng index `floor(i*200/16)`, i=0…15, cố định giữa các frame. Chỉ rút gọn evaluation views; không giảm train cameras, không có train/test pose trùng nhau.
- Gọi trực tiếp **`dataset_prepare.render_2d_image` gốc**: voxel downsample 1.7, center theo AABB mỗi frame, chia tọa độ cho 480, dịch y −0.1, xoay trục theo code; unlit point size 2, ảnh RGBA 1024×1024, background đen. Sau đó **`rescale_image` gốc** downsample Lanczos: **Q0=res8 (128×128), Q1=res4 (256×256)**. Giữ nguyên chính sách centering theo frame; không bổ sung alignment/motion preprocessing.
- 5 frame là đoạn ngắn nhất trong phạm vi yêu cầu; dùng res8/res4 ngay từ đầu như cấu hình nhỏ được cho phép. **Không gặp OOM**, không phải giảm camera hoặc đổi training method để chạy.
- Dùng **train.py nguyên bản**: frame đầu 6,000 iterations/level, bốn frame sau 1,500/level; tổng **24,000 iterations**. Training RNG=0 theo `safe_state`; không tuyên bố deterministic giữa các lần train CUDA.
- Frame đầu giữ defaults densification (from 500, until 15,000, interval 100), opacity reset interval 3,000, learning-rate schedules; không rescale learning rate theo số iterations rút ngắn. SH degree 3, có khoảng 3,000 iterations sau khi bật bậc 3. `lambda_dssim=0.8`, `--eval` để giữ test views ngoài training. Q1 dùng `--foundation_gs_path Q0 --dynamic_opacity`.
- Frame sau: mỗi quality kế thừa **chính nó ở frame trước**; `--dynamic_lapis`, chỉ optimize xyz/rotation, `--densify_until_iter 0`, densify-from/reset interval lớn hơn iterations. Tất cả argv/checkpoint SHA256 có trong `manifest.json`.
- Không dùng wrapper `train_full_pipeline.py` vì wrapper hardcode `iteration_30000`; orchestration gọi cùng chế độ train trực tiếp với đường dẫn checkpoint thực. Không sửa logic train, Gaussian model hay renderer.
- Lưu ý log upstream: `gs_merge` thay biến `gaussians` ở [train.py](../third_party/dynamic-lapis-gs/train.py) dòng 63, nhưng `training_report` dòng 244 render `scene.gaussians`. Vì vậy log validation Q1 ở frame đầu không đại diện merged checkpoint. Báo cáo này **chỉ dùng metric render lại từ PLY đã lưu**, không dùng số trong log ấy; không sửa bug ngoài phạm vi.

## Prefix correspondence và shared-state differences

Không có explicit ID trong PLY. Dựa trên `gs_merge` nối foundation trước, pruning bảo vệ prefix, densification append suffix và temporal training không prune/reorder, dùng **row i ↔ row i**. “New” là identity bổ sung theo quality, không phải Gaussian mới sinh ở mỗi frame thời gian; clone/split tạo hàng con cũng được tính là new identity.

Sau **mỗi cặp frame Q0/Q1**, script xác minh checksum, argv lineage, count, các trường bị freeze, rồi mới train frame tiếp theo. Frame đầu: xyz/rotation/scale/SH prefix bằng nhau chính xác. Qua thời gian: trong từng quality, count và scale/opacity/SH giữ nguyên theo hàng. **PASS cả 5 frame**, không dùng nearest-neighbor hay matching thay thế. Assumption vẫn là PLY chưa bị reorder bởi công cụ ngoài pipeline. Negative test đổi chỗ hai hàng prefix (kèm cập nhật checksum) bị từ chối.

Mỗi frame: **41,700 shared + 36,721 new = 78,421 Q1 Gaussians**. Thống kê dưới đây tính trên toàn bộ shared rows, không chỉ hàng thay đổi; xyz theo đơn vị scene, rotation dùng quaternion normalize và bất biến ±q.

| Frame | Đổi ít nhất 1 thuộc tính | xyz đổi | rotation đổi | ‖Δxyz‖ mean / p95 | Δrotation mean / p95 (độ) |
|---|---:|---:|---:|---:|---:|
| 1051 | 99.9952% | 0.0000% | 0.0000% | 0.00000 / 0.00000 | 0.000 / 0.000 |
| 1052 | 99.9952% | 99.7242% | 99.7242% | 0.03160 / 0.07199 | 15.640 / 35.256 |
| 1053 | 99.9952% | 99.7962% | 99.7962% | 0.04616 / 0.10245 | 24.141 / 53.843 |
| 1054 | 99.9952% | 99.8249% | 99.8249% | 0.05686 / 0.12668 | 31.247 / 69.172 |
| 1055 | 99.9952% | 99.9640% | 99.9640% | 0.06745 / 0.14661 | 36.498 / 80.578 |

Opacity đổi **99.9952%** ở mọi frame; mean / p95 `|Δlogit|` = **2.73284 / 6.45993**. Sau sigmoid, mean / p95 / max `|Δalpha|` = **0.192306 / 0.810475 / 0.993644**. Delta opacity cố định theo thời gian vì hai level đã khác từ frame đầu và sau đó opacity bị freeze.

**Scale và SH/color coefficients: 0% thay đổi, magnitude=0**, cả log-scale/scale sau exp và toàn bộ 48 SH coefficients. Màu pixel vẫn có thể thay đổi khi geometry/opacity thay đổi. Percent theo float32 chính xác và ngưỡng tuyệt đối 1e-6/component trùng nhau trong run này. Mean/median/p95/max, raw MAE/RMS và physical metrics chi tiết trong JSON/CSV.

## Render reconstruction và ablation

`naive = Q0 + Q1_suffix`; `correct = apply(shared replacements, Q0) + Q1_suffix`.
Decoder chỉ nhận Q0 và payload đã ghi, không nhận Q1. Corrected attributes được so bitwise trước render. Dùng `GaussianModel.load_ply` và `gaussian_renderer.render` gốc cho tất cả variants, cùng 16 camera test ở **256×256**, SH degree 3 và nền đen.

Metrics trên RGB float clamp [0,1], trước PNG quantization, trung bình theo view (trung bình dB cho PSNR). SSIM và LPIPS-VGG dùng implementation của repo; LPIPS nhận **[0,1] theo metrics.py**, không đổi sang [-1,1]. Có background trong metric toàn ảnh, không phải foreground-only. `inf` lưu dạng chuỗi trong JSON.

| Frame | Naive vs Q1 PSNR ↑ | SSIM ↑ | LPIPS ↓ | Correct vs Q1 PSNR / SSIM / LPIPS |
|---|---:|---:|---:|---|
| 1051 | 45.6357 | 0.998773 | 0.001231 | ∞ / 1 / 0 |
| 1052 | 42.3025 | 0.997432 | 0.002400 | ∞ / 1 / 0 |
| 1053 | 41.6316 | 0.996988 | 0.002772 | ∞ / 1 / 0 |
| 1054 | 41.1847 | 0.996665 | 0.003098 | ∞ / 1 / 0 |
| 1055 | 40.6690 | 0.996164 | 0.003480 | ∞ / 1 / 0 |

Corrected có **max absolute float pixel error=0 trên 80/80 views**. Naive khác Q1 trên **80/80 views**, không chỉ trung bình theo frame.

Ablation: bắt đầu từ trạng thái đúng, bỏ correction của đúng một nhóm; PSNR so Q1 càng thấp nghĩa là nhóm bị bỏ ảnh hưởng ảnh lớn hơn trong điều kiện này.

| Frame | Bỏ opacity (`motion_only`) | Bỏ xyz (`without_xyz`) | Bỏ rotation (`without_rotation`) |
|---|---:|---:|---:|
| 1051 | 45.6357 | ∞ | ∞ |
| 1052 | 38.0974 | 42.9254 | 53.2623 |
| 1053 | 35.5985 | 41.0853 | 50.1850 |
| 1054 | 34.3874 | 40.0438 | 47.8575 |
| 1055 | 33.0874 | 39.1513 | 46.2354 |

Opacity có ảnh hưởng lớn nhất theo phép thử này; xyz đứng sau, rotation nhỏ hơn. Không cộng lỗi từng nhóm hoặc suy ra xếp hạng phổ quát: alpha compositing có tương tác; sửa riêng xyz mà chưa sửa opacity có thể làm ảnh **xấu hơn naive**. Các variants chỉ sửa một nhóm cũng có trong CSV.

Sanity check so ground truth Open3D, tất cả ở cùng test camera 256×256:

| Frame | Q0 PSNR vs GT | Official Q1 = corrected PSNR vs GT | Naive PSNR vs GT | PSNR mất khi bỏ corrections |
|---|---:|---:|---:|---:|
| 1051 | 29.6258 | 40.0961 | 38.9882 | 1.1079 |
| 1052 | 29.6064 | 39.6797 | 38.0192 | 1.6605 |
| 1053 | 29.6090 | 39.7093 | 37.7427 | 1.9666 |
| 1054 | 29.5988 | 39.5916 | 37.5023 | 2.0893 |
| 1055 | 29.6221 | 39.1632 | 37.0307 | 2.1324 |

Q1 không phải model ngẫu nhiên/không học được cảnh: PSNR held-out **39.16–40.10 dB** ở cấu hình này. GT SSIM/LPIPS đầy đủ cũng nằm trong JSON/CSV. Chưa coi đây là đánh giá hội tụ cuối cùng.

## Raw representation overhead

Giữ đúng format đo phase synthetic: new = **59 float32 = 236 B/Gaussian** (xyz 3, rotation 4, scale 3, opacity 1, SH 48); bỏ normals bằng 0 mà renderer không dùng. Correction là **uint32 row ID + toàn bộ float32 của nhóm thuộc tính** cho mỗi shared row thay đổi bitwise: opacity 8 B/row, xyz 16 B/row, rotation 20 B/row. Đây là replacement chính xác, không phải entropy/delta codec; không quantize.

| Frame | New bytes | Opacity corrections | xyz corrections | Rotation corrections | Total corrections | Correction / new |
|---|---:|---:|---:|---:|---:|---:|
| 1051 | 8,666,156 | 333,584 | 0 | 0 | 333,584 | 3.8493% |
| 1052 | 8,666,156 | 333,584 | 665,360 | 831,700 | 1,830,644 | 21.1241% |
| 1053 | 8,666,156 | 333,584 | 665,840 | 832,300 | 1,831,724 | 21.1365% |
| 1054 | 8,666,156 | 333,584 | 666,032 | 832,540 | 1,832,156 | 21.1415% |
| 1055 | 8,666,156 | 333,584 | 666,960 | 833,700 | 1,834,244 | 21.1656% |

Scale/SH correction = **0 B**. Metadata JSON thêm **2,415 B** frame đầu, **2,423 B/frame** sau. Base Q0 không tính vào enhancement. Full Q1 renderer attributes = **18,507,356 B/frame**; không so nhầm với PLY còn header/normals.

Tổng năm frame, tính độc lập: **43,330,780 B new + 7,662,352 B correction**, ratio **17.6834%** trước metadata. Rotation là overhead lớn nhất ở frame động dù không gây mismatch ảnh lớn nhất. Opacity corrections không đổi nhưng được ghi lại mỗi frame; representation có trạng thái thời gian có thể tái dùng chúng. Không suy ra network bitrate hoặc compression gain từ các số thô này.

## Kiểm thử, artifacts và tái chạy

Đã kiểm chứng:

- 1,160 PNG train/test ở hai levels đúng dimensions/RGBA; năm PLY và ảnh cùng-view theo thời gian đều khác nhau; train/test poses tách biệt.
- Self-test exact roundtrip, byte count, identity/no-new, quaternion ±q; checksum và prefix-reorder negative tests.
- Correspondence kiểm tra ngay sau từng frame; corrected attributes bitwise, 80 float renders exact; 10 variants × 16 views × 5 frames = **800 hàng per-view**, 25 hàng attribute.
- Chạy **render.py nguyên bản** cho cả năm Q1 checkpoint: **80/80 PNG giống hệt** official render của measurement script (`original_cli_validation.json`).
- Hash source training/model/renderer/preprocessing trước–sau không đổi (`source_integrity_before.json`, `measurement_validation.json`); compile và `git diff --check` qua. Hai sửa compatibility ở `dataset_prepare.py`/`scene/dataset_readers.py` đã có từ setup trước, không thêm sửa logic ở phase này.
- Không thêm V³, H.264/H.265, DASH, không redesign method.

Code: mở rộng [check_progressive_composability.py](../tools/check_progressive_composability.py) bằng `--train-data`, early correspondence check và correction/new ratio; thêm helper [prepare_progressive_real.py](../tools/prepare_progressive_real.py) chỉ download/preprocess có giới hạn. Output real (`output/progressive_gap_real`, historical artifact) gồm `summary.json`, `summary.csv`, `attributes.csv`, `per_view.csv`, provenance/license, data spec, argv/logs/checkpoints, reconstruction PLY/binary payload và ảnh.

```bash
cd MultiObj3dgsVstream
conda activate Hoang
python tools/check_progressive_composability.py --self-test

# Đo lại model đã train, không chạy lại training:
python tools/check_progressive_composability.py \
  --manifest output/progressive_gap_real/manifest.json \
  --output output/progressive_gap_real

# Một run mới, giữ nguyên run đã đo:
python tools/prepare_progressive_real.py --download --prepare \
  --output output/progressive_gap_real_repeat --start 1051 --frames 5 --test-views 16
ln -s ../progressive_gap/vendor output/progressive_gap_real_repeat/vendor
python tools/check_progressive_composability.py \
  --train-data output/progressive_gap_real_repeat/data_spec.json \
  --output output/progressive_gap_real_repeat \
  --initial-iterations 6000 --dynamic-iterations 1500
```

Helper Open3D đặt `EGL_PLATFORM=surfaceless`; cần GPU/device access. Nếu binary vendor của phase synthetic không còn, build lại hai submodule vào vendor riêng theo [hướng dẫn phase trước](PROGRESSIVE_GAP_CHECK.md), không lấy nhầm renderer Scaffold-GS. `--train-data` từ chối overwrite model/manifest cũ; nếu correspondence fail sẽ ghi `CORRESPONDENCE_FAILED.json` rồi dừng, không tự rematch.

Kết luận nghiên cứu: implementation hiện tại có **nested identities nhưng không có immutable shared state**. Corrections làm enhancement có thể cập nhật để tái tạo Q1, không biến nó thành phép append Gaussian bất biến. Real-data counterexample này củng cố động lực nghiên cứu prefix consistency; lợi ích của một thiết kế mới vẫn cần thực nghiệm riêng.

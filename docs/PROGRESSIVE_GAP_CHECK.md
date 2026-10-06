# Progressive composability check — Dynamic-LapisGS

> **Migration note (2026-10-05):** project-owned preparation code and this document
> were migrated from the sibling `dynamic-lapis-gs` workspace. Measured results,
> PASS counts and hardware observations below describe that historical workspace;
> they do not establish a completed run in `MultiObj3dgsVstream`. Dataset, models,
> vendor binaries and generated `output/` artifacts are not bundled. Commands now
> use this project root and the pinned backend in `third_party/dynamic-lapis-gs`.
> See [migration/status](CONTENT_PREPARATION_MIGRATION.md) before reproducing a run.

Ngày chạy: 2026-10-02. Đây là **experiment đã chạy trên GPU**, không chỉ phân tích code.

## Kết luận A / B / C

**A. Không phải hierarchy append-only của Gaussian bất biến.** Có nested **identity/prefix theo thứ tự hàng**, nhưng giá trị các hàng chung khác nhau giữa hai quality levels. `Q0 + chỉ Gaussian mới của Q1` không tái tạo đúng Q1 trên cả 4 frame. Bổ sung cập nhật thuộc tính thì tái tạo Q1 chính xác; đó là enhancement có thao tác cập nhật, không còn đơn thuần nối thêm Gaussian.

**B. Frame đầu: opacity. Frame sau: thêm xyz và rotation.** Scale và SH/color coefficients của Gaussian chung không đổi trong experiment này. Ablation cô lập cho thấy xyz và opacity ảnh hưởng ảnh nhiều hơn rotation ở các frame động; không thể cộng tuyến tính đóng góp lỗi do alpha compositing tương tác. Rotation là nhóm tốn byte nhất, không phải nhóm gây sai khác ảnh lớn nhất.

**C. Correction thô: 253,944 B ở frame đầu; 1,412,028–1,416,708 B/frame động**, ngoài 10,660,592 B Gaussian mới. Tương đương thêm **2.38% / 13.25–13.29%** so với payload chỉ-Gaussian-mới. Đây là chi phí của cách ghi float32 + uint32 ID được đo, không phải bitrate tối ưu hay cận dưới nén.

Giới hạn: một sequence **tổng hợp có kiểm soát**, hai mức phân giải, bốn frame, lịch train rút ngắn. Đủ cung cấp counterexample cho tính composable tuyệt đối của implementation; không suy rộng các tỷ lệ byte/metric sang 8i hoặc mọi sequence.

## 1. Cơ chế trong code hiện tại

Checkout: `da8efaa42a9d021b5ff2958c0a87eba6ed7a47c4`.

- [train_full_pipeline.py](../third_party/dynamic-lapis-gs/train_full_pipeline.py), dòng 36–85: các resolution scales `[8,4,2,1]`. Frame đầu train mức thấp trước; mức cao dùng checkpoint mức trước làm `--foundation_gs_path --dynamic_opacity`. Frame sau, **mỗi level tự kế thừa chính level đó ở frame trước**, dùng `--dynamic_lapis --initial_gs_path`, tắt densification và opacity reset.
- [train.py](../third_party/dynamic-lapis-gs/train.py), dòng 62–79, 164–168: prefix foundation chỉ được cập nhật opacity; frame động chỉ cập nhật xyz và rotation. Gaussian mới ở frame đầu được tối ưu tất cả thuộc tính.
- [train.py](../third_party/dynamic-lapis-gs/train.py), dòng 175–200: `gs_merge` nối `foundation` trước, Gaussian mới sau cho mọi tensor.
- [scene/gaussian_model.py](../third_party/dynamic-lapis-gs/scene/gaussian_model.py), dòng 323–341, 377–390, 415–428: freeze gradient, bảo vệ prefix khỏi pruning, append Gaussian khi densify. Clone/split có thể tạo con của Gaussian cũ, nhưng các hàng con vẫn là **identity mới** ở suffix.
- PLY save/load, dòng 208–265: không có explicit Gaussian ID; lưu xyz, SH DC/rest, opacity logit, log-scale, quaternion và normals bằng 0. Thứ tự hàng được giữ.
- Loss đang thực thi là L1 + DSSIM ([train.py](../third_party/dynamic-lapis-gs/train.py), dòng 123–127); không giả định có thêm rigidity/isometry loss từ mô tả phương pháp.

**Matching:** hàng `i < N(Q0, frame 0)` của Q0 ↔ hàng `i` của Q1. Script kiểm chứng argv lineage, SHA256 checkpoint, bốn nhóm frozen ở frame đầu giống chính xác; các frame sau giữ nguyên số hàng và scale/opacity/SH trong từng level. Không dùng nearest-neighbor, không suy đoán correspondence từ khoảng cách sau motion. Assumption: checkpoint do pipeline này sinh ra và chưa reorder/prune ngoài pipeline; không áp dụng matching này cho hai mô hình train độc lập hoặc PLY đã bị sắp xếp lại.

## 2. Thiết lập thực nghiệm

- Conda `Hoang`: Python 3.11.16, PyTorch 2.3.1+cu118, CUDA 11.8, NVIDIA GTX 1660 6 GB.
- Build riêng đúng submodule renderer `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d` và simple-knn `86710c2d4b46680c02301765dd79e465819c8f19` vào `output/progressive_gap/vendor`; không thay extension dùng chung của Scaffold-GS.
- Không có dataset động/checkpoint sẵn được tìm thấy trong các thư mục dữ liệu đã kiểm tra. Script tạo hai ellipsoid có texture, chuyển động tịnh tiến/quay khác nhau, 4 frame liên tiếp; ground truth ray-trace hình học độc lập, không lấy từ Gaussian renderer.
- 12 camera train + 4 camera test tách biệt; camera giữ nguyên giữa các frame. Q0 = 64×64, Q1 = 128×128, Q0 ảnh downsample Lanczos từ Q1. Cả hai khởi tạo từ cùng PLY 512 điểm bề mặt ở frame đầu.
- Gọi **train.py nguyên bản**: 3,000 iterations/level ở frame đầu, 800/level ở ba frame sau; tổng 10,800 iterations. `lambda_dssim=0.8`, `--eval`; densification frame đầu dừng ở 2,800, không opacity reset trong run này. Các learning rate/schedule còn lại giữ mặc định, không rescale về lịch ngắn. PLY chứa SH degree 3; bậc 3 được bật tại iteration 3,000, chưa có thời gian tối ưu bậc mới ở bước cuối. Đây không phải train đến hội tụ.
- Seed tạo point cloud = 7; `safe_state` của train.py đặt training RNG = 0. CUDA training không được tuyên bố bitwise deterministic giữa các lần train.
- Không gọi wrapper `train_full_pipeline.py` vì wrapper hardcode đường dẫn `iteration_30000`; script gọi cùng hai chế độ train trực tiếp và trỏ đúng checkpoint thực tế. Không sửa training/renderer/model gốc.

## 3. Gaussian chung và magnitude thay đổi

Mỗi frame: **32,428 shared + 45,172 new = 77,600 Q1 Gaussians**. “New” nghĩa là suffix bổ sung theo quality, không phải Gaussian sinh mới ở mỗi frame động. Các thống kê magnitude dưới đây tính trên **toàn bộ shared rows**, kể cả hàng không đổi; xyz dùng đơn vị scene, rotation là góc giữa quaternion đã normalize, bất biến với dấu ±q.

| Frame | Shared đổi ≥1 thuộc tính | xyz đổi | rotation đổi | opacity đổi | Mean / p95 ‖Δxyz‖ | Mean / p95 Δrotation (độ) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 97.8876% | 0% | 0% | 97.8876% | 0 / 0 | 0 / 0 |
| 1 | 99.6515% | 99.2013% | 99.2013% | 97.8876% | 0.03636 / 0.07815 | 14.547 / 35.764 |
| 2 | 99.7656% | 99.4418% | 99.4418% | 97.8876% | 0.05481 / 0.11520 | 21.495 / 49.809 |
| 3 | 99.8427% | 99.6022% | 99.6022% | 97.8876% | 0.06957 / 0.14371 | 26.821 / 61.606 |

- Opacity: 31,743 hàng đổi; mean / p95 / max `|Δlogit|` = **2.74034 / 7.03347 / 19.70994**; sau sigmoid, `|Δalpha|` = **0.25886 / 0.91527 / 0.99525**. Giống nhau ở cả 4 frame vì opacity bị đóng băng trong temporal training của mỗi level, nhưng giá trị giữa hai level đã khác từ frame 0.
- Scale: **0%**, chênh lệch log-scale và scale sau exp đều 0.
- SH/color coefficients: **0%**, toàn bộ 48 hệ số/row có chênh lệch 0. Điều này không có nghĩa màu pixel không đổi khi Gaussian di chuyển hoặc alpha thay đổi.
- Tỷ lệ đổi theo so sánh float32 chính xác và theo ngưỡng tuyệt đối `1e-6`/component trùng nhau trong run này. JSON chứa mean/median/p95/max, RMS/MAE thô và magnitude vật lý; CSV attribute chứa thống kê thô.

## 4. Reconstruction và metrics

`naive = Q0 + Q1_suffix`. `correct = apply(shared replacements, Q0) + Q1_suffix`.
Decoder chỉ nhận Q0 và các file enhancement, **không nhận Q1**. Sau decode, kiểm tra renderer attributes bitwise bằng Q1, rồi ghi PLY và load bằng `GaussianModel.load_ply` gốc.

Tất cả variants được render bằng `gaussian_renderer.render` gốc, cùng 4 camera test Q1 128×128, nền đen, SH degree 3. PSNR/SSIM/LPIPS tính trên RGB float clamp [0,1], trước lượng tử PNG; lấy trung bình metric theo view (PSNR trung bình dB). SSIM và LPIPS-VGG từ repo; LPIPS nhận **[0,1] theo metrics.py của repo**, không chuyển sang quy ước [-1,1]. Không so trực tiếp số LPIPS này với benchmark khác quy ước.

| Frame | Naive vs official Q1: PSNR ↑ | SSIM ↑ | LPIPS ↓ | Correct vs official Q1: PSNR / SSIM / LPIPS |
|---|---:|---:|---:|---|
| 0 | 43.7632 | 0.998638 | 0.001229 | ∞ / 1 / 0 |
| 1 | 36.5013 | 0.993856 | 0.004698 | ∞ / 1 / 0 |
| 2 | 34.9302 | 0.991244 | 0.006660 | ∞ / 1 / 0 |
| 3 | 35.1764 | 0.990522 | 0.007034 | ∞ / 1 / 0 |

Correct có max absolute pixel error = **0 trên cả 16 view**, không chỉ xấp xỉ. `inf` được ghi thành chuỗi trong JSON hợp lệ. Mismatch naive tồn tại ở mọi frame, tăng rõ sau frame đầu nhưng **không tăng đơn điệu** theo PSNR; không suy luận ổn định dài hạn từ bốn frame.

Sanity check chất lượng so với ground truth (PSNR / SSIM / LPIPS):

| Frame | Official Q1 = Correct vs GT | Naive vs GT | Q0 PSNR vs GT ở cùng camera 128×128 |
|---|---|---|---:|
| 0 | 25.7348 / 0.952753 / 0.035849 | 25.7320 / 0.952195 / 0.036943 | 22.2366 |
| 1 | 26.2483 / 0.955046 / 0.038463 | 25.9795 / 0.952032 / 0.042619 | 22.7315 |
| 2 | 26.2827 / 0.953052 / 0.045659 | 25.8714 / 0.949346 / 0.049080 | 22.6173 |
| 3 | 25.8263 / 0.944376 / 0.048123 | 25.6482 / 0.942488 / 0.053501 | 22.6801 |

**Ablation cô lập:** bắt đầu từ reconstruction đúng, bỏ correction của đúng một nhóm. PSNR so với Q1 càng thấp thì ảnh hưởng của nhóm bị bỏ càng lớn trong cấu hình đó:

| Frame | Bỏ xyz (`without_xyz`) | Bỏ rotation (`without_rotation`) | Bỏ opacity (`motion_only`) |
|---|---:|---:|---:|
| 0 | ∞ | ∞ | 43.7632 |
| 1 | 36.1694 | 45.1067 | 37.5663 |
| 2 | 34.4114 | 42.4148 | 34.7696 |
| 3 | 34.4814 | 41.0707 | 33.8480 |

xyz trội ở frame 1–2 theo phép thử này, opacity trội nhẹ ở frame 3; rotation ít ảnh hưởng ảnh hơn. Các ablation chỉ sửa một nhóm từ naive cũng có trong output. Sửa một phần có thể làm metric xấu đi do tương tác, không phải lỗi decoder.

## 5. Chi phí enhancement đo từ file thực tế

Format đo đơn giản, không nén/quantize: Gaussian mới = 59 float32 (xyz 3 + rotation 4 + scale 3 + opacity 1 + SH 48) = **236 B**; bỏ 3 normals bằng 0 mà renderer không dùng. Mỗi nhóm correction ghi `uint32 row_id + full float32 replacement group` cho từng shared row thay đổi bitwise. Vì vậy xyz = 16 B/row, rotation = 20 B/row, opacity = 8 B/row. Không giả vờ đây là byte của file PLY gốc: PLY còn normals và header.

| Frame | New bytes | xyz correction | rotation correction | opacity correction | Total correction | Correction / new | Correction / full Q1 attributes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10,660,592 | 0 | 0 | 253,944 | 253,944 | 2.3821% | 1.3866% |
| 1 | 10,660,592 | 514,704 | 643,380 | 253,944 | 1,412,028 | 13.2453% | 7.7103% |
| 2 | 10,660,592 | 515,952 | 644,940 | 253,944 | 1,414,836 | 13.2716% | 7.7256% |
| 3 | 10,660,592 | 516,784 | 645,980 | 253,944 | 1,416,708 | 13.2892% | 7.7358% |

Scale/SH corrections = 0 B. Full Q1 renderer attributes = 18,313,600 B/frame; Q0 base không tính vào enhancement. Metadata JSON thêm 2,415 B frame 0, 2,423 B/frame sau. Tổng bốn frame: **42,642,368 B new + 4,497,516 B correction**, chưa tính metadata. Rotation chiếm khoảng **45.6% correction frame động** và là nhóm lớn nhất vì 4 thành phần quaternion thay vì xyz 3/opacity 1.

Đây là phép đo **mỗi frame độc lập**: opacity correction cố định được ghi lại mỗi frame. Một representation có trạng thái thời gian có thể tái dùng nó; delta/entropy coding, chia sẻ index hoặc quantization cũng có thể giảm byte. Experiment này không triển khai hay ước lượng những phần đó, không đưa vào V³/H.264/H.265/DASH.

## 6. Kiểm chứng và tái chạy

Đã qua: self-test roundtrip, byte-count, identity/no-new controls, quaternion ±q; SHA256/lineage/frozen attributes; reconstruction bitwise; corrected render float equality trên 16 view. Chạy thêm **render.py nguyên bản cho cả 4 checkpoint Q1**: toàn bộ 16 PNG giống chính xác với ảnh official từ script đo (`original_cli_validation.json`). Các lần evaluate lại giữ nguyên metric được báo cáo.

Negative tests cũng qua: từ chối checksum sai; từ chối PLY bị đảo prefix ngay cả khi cập nhật checksum; từ chối chạy đè trước khi chạm vào dữ liệu cũ (`measurement_validation.json`). Kiểm tra cuối: 10 variants × 4 views × 4 frames = 160 hàng per-view, 20 hàng attribute; compile và `git diff --check` không lỗi.

Script độc lập: [tools/check_progressive_composability.py](../tools/check_progressive_composability.py). Không sửa `train.py`, `render.py`, Gaussian model hoặc renderer. Các thay đổi compatibility `dataset_prepare.py`/`scene/dataset_readers.py` đã có từ bước setup trước experiment; build tạo thêm artifact dưới submodule, không sửa source CUDA.

```bash
cd MultiObj3dgsVstream
conda activate Hoang
python tools/check_progressive_composability.py --self-test

# Đo lại checkpoint đã train, không train lại:
python tools/check_progressive_composability.py \
  --manifest output/progressive_gap/manifest.json --output output/progressive_gap

# Chạy mới khi muốn build; chọn CUDA toolchain/compiler/compute capability
# phù hợp với môi trường PyTorch/GPU. Không chạy build trong migration này.
# Dùng output mới, build đúng extension của checkout vào vendor riêng.
mkdir -p output/progressive_gap_repeat/vendor
CUDA_HOME="$CONDA_PREFIX" CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="<GPU_COMPUTE_CAPABILITY>" \
  python -m pip install --no-build-isolation --no-deps \
  --target "$PWD/output/progressive_gap_repeat/vendor" \
  ./third_party/dynamic-lapis-gs/submodules/diff-gaussian-rasterization \
  ./third_party/dynamic-lapis-gs/submodules/simple-knn
python tools/check_progressive_composability.py --run-demo \
  --output output/progressive_gap_repeat
```

`--run-demo` từ chối overwrite dữ liệu/model cũ. `--manifest` nhận schema như manifest đã sinh: `frames[{frame,Q0,Q1,eval_source}]` và `commands[{frame,level,argv,sha256,...}]`; evaluator cần các camera test dạng `transforms_test.json`. Không nhận hai checkpoint bất kỳ mà bỏ qua lineage. Nếu output khác thư mục đã build vendor, cần build vendor ở output mới hoặc đưa đúng vendor vào PYTHONPATH.

Kết quả tại output/progressive_gap (`output/progressive_gap`, historical artifact): `summary.json`, `summary.csv`, `attributes.csv`, `per_view.csv`, `manifest.json` (argv + checksum), `logs/`, `models/`, dữ liệu nguồn `data/`, và `analysis/<frame>/` gồm PLY reconstruction, binary payload, metadata và ảnh render. `output/` bị gitignore; cần giữ/copy thư mục này riêng nếu chuyển experiment sang máy khác.

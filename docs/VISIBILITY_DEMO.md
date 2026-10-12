# Demo trực quan contribution và occlusion

Chạy từ thư mục `MultiObj3dgsVstream`:

```bash
bash scripts/run_visibility_demo.sh --serve
```

Mở `http://127.0.0.1:8767/index.html`. Lệnh render, đo thời gian, xuất báo cáo
rồi phục vụ báo cáo trên localhost. Dừng server bằng Ctrl+C. Bỏ `--serve` nếu
chỉ muốn xuất file. Report mặc định nằm tại `output/visibility_demo/index.html`.

Wrapper sử dụng Python trong môi trường `Hoang` nếu có, và chọn physical GPU 0.
Để sử dụng môi trường CUDA khác đã được cấu hình đúng:

```bash
VISIBILITY_DEMO_PYTHON=/path/to/environment/bin/python bash scripts/run_visibility_demo.sh --serve
```

Demo cần CUDA và các extension có sẵn; không cài package, train model hay tạo
proxy. Config chỉ định extension đã build tại
`../dynamic-lapis-gs/output/progressive_gap_real/vendor`, vì extension trong
shared environment có thể thuộc một rasterizer khác.

## Object và camera mặc định

Config: [`configs/visibility/demo_longdress.json`](../configs/visibility/demo_longdress.json).
Demo dùng checkpoint đã train trong project:

```text
output/content_preparation/longdress_native_pilot/longdress/models/Q0/1051/
    point_cloud/iteration_30000/point_cloud.ply
```

Đây là Gaussian PLY với 34.944 splats, SH degree 3, frame 1051, media timestamp
0.0 s. Ba instance `longdress_A/B/C` có ID và transform riêng, dùng lại cùng
checkpoint. Tổng cộng 104.832 Gaussians tại resolution 512×512. Demo hiện dùng
các instance Longdress vì Soldier trong project output mới có preprocessing,
chưa có checkpoint Gaussian đã train.

Representation được cố định cho toàn bộ run. Cả ảnh RGB và các pass phân tích
đều dùng chính checkpoint này; không chọn subset hoặc thay LoD. Điều này giúp
kiểm chứng trực tiếp cùng geometry, opacity, footprint và depth ordering.

Camera theo convention hiện có, world +Z up, nhìn từ phía +Y. Object được đặt
vào cùng frustum bằng translation và positive uniform scale. Khi đổi chiều
sâu, scale thay đổi theo khoảng cách camera để giữ kích thước chiếu; các map
trong case aligned/depth-swapped vì thế có cùng footprint nhưng khác suppression.

## Bốn bố trí được kiểm chứng

| Bố trí | Quan sát mong đợi |
| --- | --- |
| `separated` | Ba object tách rời trong cùng ảnh; joint và solo giống nhau, occlusion ≈ 0. |
| `partial_overlap` | Silhouette giao nhau; object phía sau có joint contribution giảm. |
| `aligned` | Các instance có silhouette trùng nhau; object phía sau bị che mạnh. |
| `depth_swapped` | Đảo thứ tự trước/sau; contribution đổi theo object ID. |

Mỗi bố trí hiển thị ảnh RGB thật từ SH, bản đồ object identity, tổng alpha,
solo/joint contribution của từng object và bảng `A_visible`, `A_solo`,
`occlusion_ratio`, `out_of_view`. Map luôn dùng cùng thang 0–1; map bị che
không được tự tăng độ sáng. Có thể chuyển giữa màu object và grayscale.

Ảnh constant-color reference được rasterize bình thường, rồi so với:

```text
reconstruction = Σ_m color[m] * W_joint[m] + background * (1 - Σ_m W_joint[m])
```

Kiểm chứng này dùng constant-color reference. Ảnh RGB từ SH được hiển thị riêng,
vì object-level map không tái tạo được màu thay đổi theo từng Gaussian. Report
cũng kiểm tra sum contribution bằng alpha render trắng trên đen, radii giống
nhau, area/ratio hợp lệ và map sums khớp API. Các check sai khiến run trả exit 1.

## Thời gian thực hiện

Mặc định chạy 3 warmups và 20 samples cho mỗi phase. Report chứa mean, median
và p95, đơn vị ms. Thời gian PNG/HTML export và diagnostic export không nằm
trong vòng đo render.

- `normal_rgb`, `joint_total`, `solo_*`, `solo_total`: CUDA events trên scene
  đã được upload; bao gồm lời gọi render và tạo identity colors khi cần.
- `reductions_scalar_transfer`: đo riêng GPU reduction và transfer scalar;
  map được giữ tạm trong phase đo này để tách chi phí rasterization.
- `complete_analysis`: synchronized wall time của `analyze_visibility`, gồm
  copy/validate snapshot, transform, upload, joint/solo, reduction và transfer;
  diagnostics tắt trong vòng đo này.

Run đã kiểm chứng trên GTX 1660, PyTorch 2.3.1+cu118, với checkpoint trên:

| 3 object, 512×512 | RGB median | Joint median | Tổng solo median | API median |
| --- | ---: | ---: | ---: | ---: |
| separated | 3,626 ms | 3,015 ms | 4,913 ms | 62,115 ms |
| partial_overlap | 3,359 ms | 2,902 ms | 4,855 ms | 62,004 ms |
| aligned | 4,495 ms | 4,130 ms | 4,881 ms | 63,235 ms |
| depth_swapped | 4,499 ms | 4,119 ms | 4,847 ms | 63,549 ms |

Run trả `PASSED`. Occlusion trong aligned là A=0%, B≈95,7247%, C≈97,9158%;
khi đảo chiều sâu, C=0% và A≈97,9158%. Sai số RGB reconstruction tối đa của
bốn bố trí dưới `8.4e-7`. Peak allocated memory của complete API khoảng 87 MiB;
JSON có baseline/peak allocated và reserved chi tiết. Các giá trị timing phụ
thuộc GPU/load hiện tại; mỗi run ghi lại số đo của chính nó.

## Hai object và cấu hình riêng

```bash
bash scripts/run_visibility_demo.sh --objects 2 --output output/visibility_demo_2 --serve
bash scripts/run_visibility_demo.sh --resolution 768 --warmup 5 --samples 50 --serve
```

Chỉ chạy một server trên mỗi port; dùng `--port 8768` nếu server 8767 đang mở.
Để đọc asset và kiểm tra placement mà không render:

```bash
bash scripts/run_visibility_demo.sh --prepare-only --output output/visibility_demo_prepared
```

Có thể tạo config khác rồi truyền `--config`. Mỗi source chỉ định `object_id`,
`path` (relative project root hoặc absolute), `kind`, `frame`, `timestamp`,
`up_axis`, `representation`. `kind` hỗ trợ `checkpoint`, `gaussian_state` và
`proxy_descriptor`; loại cuối gọi loader proxy hiện có với frame được chỉ định.
Các source phải cùng media timestamp. Không tự chọn frame gần nhất.

## Output và kiểm thử

Report gồm `index.html`, `report.json`, và các thư mục `scenario-000..003`.
Mỗi scenario có PNG gốc, `montage.png`, `maps.npz` chứa float maps và `maps.json`
để ánh xạ các key. Các artifact nằm trong ignored `output/`, không sửa asset nguồn.

```bash
python -m pytest -q tests/test_visibility_demo_scene.py tests/test_run_visibility_demo.py tests/test_visibility_demo_report.py
VISIBILITY_CUDA_TESTS=1 VISIBILITY_EXTENSION_PATH=../dynamic-lapis-gs/output/progressive_gap_real/vendor bash scripts/run_gpu0.sh python -m pytest -q tests/renderer/test_visibility_contribution.py
```

Bộ kiểm thử renderer đã chạy thực trên CUDA: 60 tests passed, gồm 20 case GPU.
Nếu CUDA không truy cập được, demo ghi trạng thái `BLOCKED`, không tạo ảnh giả
hay suy đoán thời gian GPU. Chuẩn bị-only ghi `PREPARED`, không phải kết quả
correctness của rasterization.

# SERVER-SIDE CONTENT PREPARATION — Runbook cho MultiObj3dgsVstream

> **Status:** planning snapshot written before the first implementation batch.
> The Draco bridge, codec integration, validator, MPD writer and decoded
> Gaussian viewer now exist. Do not follow this snapshot's `TO IMPLEMENT` labels
> or workstation-specific commands as an executable runbook. For a clean machine
> use [DEPENDENCIES.md](DEPENDENCIES.md#clean-machine-setup-rtx-5060-ti), for the
> bounded smoke/viewer use [CONTENT_PREPARATION_BATCH1.md](CONTENT_PREPARATION_BATCH1.md),
> and for the current native pilot/full Longdress sequence use
> [CONTENT_PREPARATION_STEP_BY_STEP.md](CONTENT_PREPARATION_STEP_BY_STEP.md#14-new-machine-execution-order-visual-gate-native-pilot-full-longdress).
> The runnable full config is now `configs/content_preparation/longdress.yaml`.

**Tài liệu mục tiêu:** `docs/CONTENT_PREPARATION_PLAN.md`

**Trạng thái của turn này:** đã inspect repo hiện tại và nguồn public; chưa tạo/sửa file, build, training hoặc chạy experiment. Nội dung dưới đây là bản kế hoạch để coding agent thực hiện sau.

## 1. Quyết định và tiêu chí hoàn thành

Triển khai từ project root, tái sử dụng pipeline đã migrate và backend đã pin tại `third_party/dynamic-lapis-gs`.

Các quyết định đã chốt:

- Chỉ chạy **progressive delivery**: Base/E1/E2/E3. Giữ khả năng arbitrary N đã có.
- Tích hợp **Draco công khai**, dùng **zlib float32 lossless** hiện có làm đối chứng.
- Giữ representation **không có dependency giữa các frame** trong phase này.
- Mở rộng viewer hiện tại bằng viewport Gaussian có free camera; Python/C++ decode offline là đường bắt buộc đầu tiên.
- Full Longdress là **1051–1350, 300 frame, 30 fps, 10 giây**.
- Proxy generation, proxy validation, proxy research và networking nằm ngoài phạm vi. Manifest chỉ tham chiếu proxy đã chuẩn bị bên ngoài nếu được cung cấp; nếu không, ghi `null`.
- Không thay training, loss, Gaussian model hoặc renderer objective của Dynamic-LapisGS.

Phase hoàn thành khi:

1. Actual packaged payload decode đúng thành Gaussian state ở mọi quality/frame.
2. Objective profiles dùng reconstructed state từ payload, reference là highest decoded quality.
3. Q0–Q3 xem được trong browser, chọn object/frame/quality và xoay/pan/zoom.
4. Có MPD, JSON sidecars, exact accounting và provenance.
5. Resume không làm lại checkpoint/view/package đã hoàn thành và còn hợp lệ.
6. Refresh ablation report đúng giới hạn của codec không có temporal dependency.

## 2. Current repo inventory và phần tái sử dụng

Repo hiện tại có nhiều thay đổi migration đã stage và thay đổi networking riêng. Coding agent phải giữ nguyên các thay đổi này khi làm content preparation.

Các dependency đã inspect:

| Thành phần | Pin hiện tại | Trạng thái |
|---|---|---|
| Project HEAD | `8ce7ba76a5942556d776eb243389bdc0978167db` | Working tree còn thay đổi; HEAD chưa mô tả toàn bộ code migrate |
| Dynamic-LapisGS | `da8efaa42a9d021b5ff2958c0a87eba6ed7a47c4` | Source backend hiện có |
| Draco | `15bdb3a4f15a7a8d77489ac348a7a50d93de17a3` | Source version 1.5.7; chưa tích hợp codec pipeline |
| Gaussian rasterizer | `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d` | Nested submodule đã pin |
| simple-knn | `86710c2d4b46680c02301765dd79e465819c8f19` | Nested submodule đã pin |

Inventory code:

| Thành phần hiện có | File/module | Khả năng tái sử dụng |
|---|---|---|
| CLI và orchestration | `tools/content_preparation/prepare_content.py`, `pipeline.py` | Chín stage, dry-run, resume, overwrite protection |
| Config | `config.py`, `configs/content_prepare_*.yaml` | Progressive-only và arbitrary N qualities đã hỗ trợ |
| Backend launcher | `backend.py`, `paths.py`, `upstream.py`, `compatibility.py` | Gọi backend đã pin; compatibility adapters không sửa source upstream |
| Gaussian state | `assets.py` | NPZ deterministic, PLY read/write, IDs, numeric state hash |
| Codec baseline | `codec_adapter.py` | Float32 lossless; refinement gồm mọi attribute |
| Packaging | `packaging.py` | CPSEG containers, epochs, refresh labels, network cuts |
| Decode | `pipeline.py::stage_decode` | Đọc member từ actual segment container |
| Metrics/profile | `metrics.py`, `quality_profile.py` | MSE/PSNR/LPIPS, conditioned rows, query/interpolation, gains |
| JSON manifests | `manifest.py`, `run_report.py` | Object/server index, exact bytes, hashes/provenance |
| Checkpointing | `checkpoint.py` | Per-task journal, fingerprints, locks |
| Dataset acquisition | `tools/download_8i_object.py` | Official archive, inventory, SHA/CRC và verified-cache reuse |
| Trial viewer | `tools/content_trial_viewer/index.html` | PNG và metadata viewer |
| Reporting/visual tests | `summarize_8i_trial.py`, `validate_trial_viewer.py` | PNG generation/catalog và browser checks |
| GT diagnostic | `evaluate_decoded_gt.py` | Matching-camera GT evaluation; cần bỏ hard-code independent |
| Tests | `tests/content_preparation/` | Existing coverage cho codec, packaging, pipeline, migration và resume |

**Không xây lại các thành phần trong bảng này.**

### Artifacts thực tế

Tại thời điểm viết plan, repo mới **chưa có `output/`**; artifacts cũ nằm trong
workspace sibling. Các đường dẫn sau chỉ là provenance lịch sử, không phải input
path cho checkout mới. Current configs use project-local `output/datasets/` and
`output/imports/`; transfer those assets there or download the dataset again.

```text
../dynamic-lapis-gs/output/datasets/8i/longdress/inventory.json
../dynamic-lapis-gs/output/datasets/8i/longdress/Ply/
../dynamic-lapis-gs/output/longdress_4level_training/manifest.json
../dynamic-lapis-gs/output/progressive_gap_real/vendor/
```

Raw inventory có đủ 300 frame. Manifest checkpoint có Q0–Q3 trên năm frame 1051–1055.

Các reports đã migrate là **historical evidence**, không phải kết quả GPU/full run tại repo mới. Tài liệu migration ghi destination checks trước đây: 184 pytest tests passed, 1 skipped; preparation self-test 105 tests với optional browser skip. Turn này không chạy lại các checks đó.

## 3. Missing components — chỉ bổ sung phần thiếu

| Khoảng trống | Thay đổi tối thiểu |
|---|---|
| Draco chưa được dùng | Thêm adapter và native bridge; đăng ký với codec dispatch hiện có |
| Refresh dễ bị hiểu thành temporal access | Giữ semantics hiện có, bổ sung ablation/accounting và giải thích rõ |
| `--stage all` vẫn chạy proxy | Cho phép `proxy.enabled: false`; loại proxy khỏi selected stages |
| Manifest hard-require proxy files | Cho phép `proxy_path=None` hoặc descriptor có sẵn; không gọi proxy code |
| Chưa có MPD XML writer | Thêm writer từ package/manifest hiện có |
| Viewer chỉ xem PNG | Thêm Gaussian viewport trong cùng HTML và object selector |
| GT tool hard-code independent | Thêm `--mode progressive` |
| Trial auditor hard-code hai modes/proxy | Thêm validator tổng quát, reuse validators hiện có |
| Aggregates chỉ nhóm object/mode/quality | Thêm grouping theo time, scale và view region |
| Native GPU backend chưa được xác minh sau migration | Preflight bằng explicit verified extension path |
| Chưa có full progressive Longdress config | Tạo config riêng, không sửa historical configs |

Không chuyển hoặc viết client MPD parser. File `src/client/mpd_parser.py` hiện rỗng nhưng nằm ngoài phạm vi.

## 4. VERIFIED / REPRODUCTION / NEW PROJECT CODE

### VERIFIED

- Dynamic-LapisGS source có preprocessing, trainer, Gaussian model và original CUDA renderer.
- `train_full_pipeline.py` dùng resolutions **8/4/2/1**, default **30,000 iterations mỗi quality/frame**, `lambda_dssim=0.8`.
- Preprocessing upstream tạo nguồn 1024×1024 rồi downsample.
- Public Draco cung cấp point-cloud encoding và generic attributes.
- Codec project hiện tại thực sự hỗ trợ shared-state replacements, additions/deletions và decoded-parent verification.

### REPRODUCTION

LTS paper mô tả enhanced Draco lossless, GoF fragmentation và DASH generation. Tuy nhiên public repository thông báo source implementation không được phát hành. Không có cơ sở claim exact enhanced-Draco hoặc bitstream compatibility. [Official LTS repository](https://github.com/AIINS-NTHU/LTS-DASH-Streaming-System-for-3DGS).

Một khác biệt cần ghi rõ: experiment trong LTS paper dùng nguồn 600×600 và ba delivery layers; ladder bốn mức 128/256/512/1024 ở kế hoạch này đến từ **Dynamic-LapisGS source**, không phải exact reproduction của bảng cấu hình LTS paper. Paper cũng mô tả GoF bằng I/P-frame terminology, nhưng public code không đủ để xác minh encoder dependency tương ứng. [LTS paper, §5.1–5.2](https://people.cs.nycu.edu.tw/~chuang/pubs/pdf/2025mmsys.pdf).

### NEW PROJECT CODE

- Draco adapter cho toàn bộ Gaussian attributes.
- Progressive container/refinement integration.
- Experimental Gaussian MPD signaling.
- Viewer export/catalog extension.
- Refresh/access measurement và reporting.

Tên codec dự kiến:

```text
gaussian_attribute_draco_byteplanes / 1.0.0
```

Tên này phải xuất hiện trong config/provenance/report; không dùng tên “LTS codec”.

## 5. Quality levels và training lineage

| Quality | Source configuration | Training images | Training dependency | Delivered output |
|---|---|---:|---|---|
| Q0 | Upstream resolution scale 8 | 128×128 | First frame train từ initialization | Base |
| Q1 | Scale 4 | 256×256 | First frame dùng Q0 foundation | Base + E1 |
| Q2 | Scale 2 | 512×512 | First frame dùng Q1 foundation | Base + E1 + E2 |
| Q3 | Scale 1 | 1024×1024 | First frame dùng Q2 foundation | Base + E1 + E2 + E3 |

Full-run training settings:

```yaml
initial_iterations: 30000
dynamic_iterations: 30000
lambda_dssim: 0.8
sh_degree: 3
extra_args: ["-r", "1"]
```

Wrapper giữ ảnh training trên CPU. Đây là orchestration để giới hạn VRAM, không thay loss hoặc optimization.

Training lineage:

- First frame của Q1 trở lên: `--dynamic_opacity --foundation_gs_path`.
- Frame tiếp theo: previous checkpoint của **cùng quality**, `--dynamic_lapis --initial_gs_path`.
- Dynamic frames giữ upstream policy không densification/opacity reset.
- Không retrain tại mỗi packaging GoF. Training timeline và representation epoch là hai khái niệm riêng.

Q4 chưa có source-backed operating point được chọn. Cấu trúc hiện có đã hỗ trợ N levels; sau này thêm checkpoint/config riêng và E4. Không tạo Q4 bằng cách đổi nhãn Q3 hoặc upscale input.

Các budgets 6,000/10,000/1,500 trong trial cũ chỉ dùng để kiểm tra pipeline, không dùng làm full-run defaults.

## 6. Progressive representation thực tế

Giữ `GaussianState` hiện có:

```text
ids: int64[N]
xyz: float32[N,3]
rotation: float32[N,4]        raw wxyz
scale: float32[N,3]           log-scale
opacity: float32[N,1]         logit
sh: float32[N,3,(degree+1)^2]
```

Không activate opacity, exponentiate scale hoặc normalize quaternion tại export/codec.

Với frame `t`:

```text
Base(t) → reconstructed Q0(t)

E1(t) + decoded Q0(t) → reconstructed Q1(t)
E2(t) + decoded Q1(t) → reconstructed Q2(t)
E3(t) + decoded Q2(t) → reconstructed Q3(t)
```

Enhancement giữ semantics đã triển khai:

- Target IDs và row ordering.
- New/deleted IDs.
- Sparse **absolute replacements** cho xyz, rotation, scale, opacity và SH.
- Metadata, target shape/SH schema.
- Exact decoded-parent state hash.
- Target reconstructed-state hash.

Không giả định prefix chung bất biến giữa qualities ở các dynamic frames. Không thay refinements bằng only-new-Gaussians.

Dependency trực tiếp là lower reconstructed quality cùng frame. Dependency closure để đạt Q3 là Base/E1/E2/E3 cùng frame; không có shortcut trong phiên bản đầu.

## 7. Codec integration

### Code cần thêm

```text
tools/content_preparation/draco_adapter.py
tools/content_preparation/native/draco_byte_codec.cpp
tools/content_preparation/native/CMakeLists.txt
```

Public interface giữ tương thích với pipeline:

```python
encode(input_state, path, config, parent=None) -> payload_metadata
decode(path, parent=None) -> GaussianState
```

Native CLI — **TO IMPLEMENT**:

```bash
draco_byte_codec encode --input FILE --output FILE --rows N --row-bytes B
draco_byte_codec decode --input FILE --output FILE --rows N --row-bytes B
draco_byte_codec --version
```

### Phương pháp đã chọn

Dùng Draco sequential point-cloud encoding cho **byte attributes**:

1. Chuyển mỗi row của stream sang các byte little-endian nguyên bản.
2. Chia thành generic `DT_UINT8` attributes, tối đa 64 byte components mỗi attribute.
3. Disable point deduplication.
4. Force sequential encoding, no attribute prediction, no quantization.
5. Encoder/decoder speed 3/3; bật built-in attribute compression.
6. Decoder dùng unique attribute IDs và point mappings để ghép lại exact bytes.
7. Empty streams được biểu diễn rõ trong outer packet, không gọi Draco.

Cách này tránh các lỗi đã xác định trong pinned source:

- Stock Draco PLY reader không giữ đầy đủ Gaussian fields.
- Raw float bits dưới dạng uint32 có thể vượt conversion range.
- Full-word signed integer symbols có thể gây entropy frequency allocation rất lớn.
- Byte attributes giữ symbol range nhỏ và bảo toàn mọi float32 bit pattern hợp lệ.

Đây là baseline lossless có thể kiểm tra từ public Draco, chưa có bằng chứng compression efficiency tương đương enhanced Draco của LTS.

### Container và compatibility

- Giữ CPSEG packaging hiện có.
- Draco packet dùng magic riêng `CPDRAC01`, schema/version riêng.
- `inspect_payload()` và codec dispatch nhận cả legacy `CPGAUS01` lẫn Draco packet.
- Giữ legacy zlib decode compatibility.
- Reuse logic xây/apply sparse replacement; chỉ tách helper nhỏ nếu cần chia sẻ giữa hai adapters.
- Mọi ID/index/attribute stream và metadata đều được truyền/account; không lấy side information từ checkpoint bên ngoài decoder.

Packet metadata phải có codec version, Draco commit, binary SHA, stream offsets/bytes/hashes, parent hash và decoded-state hash.

## 8. GoF / refresh / segment semantics

| Khái niệm | Quy ước trong phase này |
|---|---|
| Representation GoF/epoch | Outer grouping/reset của package metadata |
| Configured refresh interval | Boundary/policy label theo layer |
| Network segment duration | Khoảng frame được chứa trong requestable CPSEG file |
| Temporal prediction GOP | **Không tồn tại trong production codec được chọn** |

Mỗi Base frame tự chứa. Mỗi enhancement frame cần lower decoded quality **cùng frame**.

Do đó:

- Không giữ Base tại 0 s để dùng thay Base tại 5 s.
- Không cần lịch sử frame trước.
- Không phải chờ refresh tương lai để nâng quality.
- “Full reset” tại epoch boundary cho phép clear decoder/cache state; không tạo một loại temporal I-frame mới.

### Initial hypothesis

Tại 30 fps:

```text
Outer epoch: 6.0 s = 180 frames
Network segment: 0.5 s = 15 frames

Base: 3.0 s = 90 frames
E1:   2.0 s = 60 frames
E2:   1.0 s = 30 frames
E3:   0.5 s = 15 frames
```

Schedule restart tại mỗi outer epoch. Segment boundaries là union của:

```text
epoch boundaries ∪ layer refresh boundaries ∪ network cuts
```

Timeline:

```text
Time        0       1       2       3       4       5       6 s
Epoch       R-----------------------------------------------R
Base        R                       R                       R
E1          R               R               R               R
E2          R       R       R       R       R       R       R
E3          R   R   R   R   R   R   R   R   R   R   R   R   R
Network     |---|---|---|---|---|---|---|---|---|---|---|---|
            mỗi cut = 0.5 s
```

Base 3 s và E1 2 s chỉ có common refresh tại 0/6 s. Tuy nhiên mọi boundary đều nằm trên grid 0.5 s, nên initial package vẫn có segment alignment chung.

Alternative aligned candidate: **2/1/1/0.5 s**, cùng outer epoch 6 s. Giữ initial 3/2/1/0.5 làm hypothesis để so sánh.

### Activate Q3 tại 5.3 s

```text
frame = 1051 + 5.3 × 30 = 1210
target segment interval = [5.0, 5.5)
frames = 1201–1215
```

Cold access:

1. Tải Base/E1/E2/E3 segment chứa frame 1210.
2. Extract member của frame 1210.
3. Decode Base1210 → E1₁₂₁₀ → E2₁₂₁₀ → E3₁₂₁₀.

Warm access nếu decoded Q1(1210) hợp lệ:

- Chỉ cần các E2/E3 segment chưa có trong cache.
- Không tải Base(3 s), E1(4 s) hoặc frame history.

Với network segment 2 s, boundaries có thể khác giữa layers. Request closure phải tìm theo **frame/time membership**, không dùng cùng segment number cho mọi layer.

## 9. REFRESH_DURATION_ABLATION

### Mục tiêu và giới hạn

Tách hai câu hỏi:

1. Configured refresh tự nó có làm thay đổi bytes/access không?
2. Container granularity làm thay đổi storage overhead và lượng frame không cần thiết phải tải bao nhiêu?

Không thêm temporal predictor để tạo ra trade-off giả.

### Search space

| Profile | Base/E1/E2/E3, giây | Lý do |
|---|---|---|
| A | 3 / 2 / 1 / 0.5 | Hypothesis ban đầu |
| B | 2 / 2 / 1 / 1 | Ít cut ở enhancement cao |
| C | 2 / 1 / 1 / 0.5 | Aligned candidate |
| Common | 0.5 / 1 / 2 / 3 / 4, mỗi lần dùng chung cho mọi layer | Control/sensitivity |

Outer epoch giữ 6 s. Interval 4 s sẽ bị cắt ngắn tại epoch reset; report phải ghi điều này.

Execution:

- Arm 1: A/B/C và năm common profiles, network 0.5 s.
- Arm 2: A với network 1 s và 2 s.
- Arm 3: B/C/common-1/common-2 với network 2 s.

Tổng **14 configurations**, thay vì brute-force `5⁴` combinations.

Dùng first 180 frames `[0,6)` của full trained sequence để giới hạn storage. Kiểm tra epoch reset 6 s riêng từ full package.

### Kết quả null phải kiểm tra

Ở Arm 1, mọi refresh boundary đã nằm trên network cuts. CPSEG header hiện không serialize refresh flags.

Vì vậy dự kiến:

- Media files byte-identical giữa profiles.
- Codec payload bytes không đổi.
- Temporal history = 0.
- Required dependency depth chỉ theo quality.
- Quality và activation access không đổi.
- Sidecar metadata có thể khác.

Đây là một kết quả có ý nghĩa; không diễn giải thành “short refresh cải thiện temporal coding”.

### Metrics bắt buộc

Lưu per profile và per layer:

- Actual payload bytes và complete segment bytes.
- Bytes/s và bits/s, dùng full media duration.
- Container/header overhead.
- Refresh-induced grouping overhead so với network-cut-only control.
- Sidecar bytes và retained storage riêng.
- Cold/warm whole-file access bytes.
- Member-byte lower bound, report riêng với whole-file access.
- Time-to-decodable Q0/Q1/Q2/Q3.
- Dependency depth và history frames.
- Decode correctness và state hashes.
- Quality difference so với canonical decoded states.

Temporal prediction efficiency/residual bytes/temporal chain length ghi **N/A**, không ghi số 0 để ngụ ý đã đo predictor.

### Activation model

Offline measurement, không viết network client/scheduler:

```text
transfer time = 8 × missing complete-file bytes / fixed bandwidth
activation time = transfer time + measured sequential decode/merge time
```

Report sensitivity tại 5/10/20 Mbps; thêm RTT scenarios 0/40 ms theo số file requests tuần tự. Đây là analytical model, không phải network measurement.

Cold nghĩa là logical asset cache rỗng; không reset OS page cache toàn hệ thống.

Đo mọi frame trong đoạn ablation. Decode lại representative closures tại segment/refresh boundaries, 5.3 s, và epoch reset; verify extracted member hashes trên mọi container.

### Tiêu chí lựa chọn

- Loại mọi profile không decode exact hoặc có dependency lỗi.
- Tạo Pareto frontier của **media rate/storage** với **p95 cold/warm access cost**.
- Report sensitivity theo các overhead budgets 1/2/5/10/20%; không chọn một X làm constraint mặc định.
- Nếu frontier có ít nhất ba điểm khác nhau, report knee bằng khoảng cách đến chord giữa hai extreme points sau chuẩn hóa hai axes. Report riêng cho cold và warm.
- Nếu profiles tương đương, report **“refresh optimum not identifiable with this codec”**.

Release đầu giữ **A + network 0.5 s** làm reproducible baseline, không gọi là optimum. Pareto recommendation là research output; không tự thay config hoặc retrain.

### Code — TO IMPLEMENT

```text
tools/content_preparation/refresh_ablation.py
tools/run_refresh_ablation.py
configs/content_preparation/refresh_ablation.yaml
```

CLI:

```bash
python tools/run_refresh_ablation.py \
  --prepared-root "$FULL_ROOT" \
  --config configs/content_preparation/refresh_ablation.yaml \
  --output "$ABLATION_ROOT" \
  --resume
```

Reuse `package_object()`, dependency validators, segment extraction và `Journal`. Mỗi profile có output/journal riêng. Hard-link immutable inputs và byte-identical containers khi cùng filesystem; report logical bytes và physical retained storage riêng.

## 10. Directory layout và config

### Layout

Giữ cấu trúc pipeline hiện có:

```text
output/content_preparation/
├── smoke/
├── longdress_visual/
├── longdress_native_pilot/
├── longdress_full/
│   ├── .preparation/
│   ├── manifest.json                  server index
│   ├── manifest.mpd
│   ├── content_index.json
│   ├── catalog.json
│   ├── index.html                     existing viewer shell, extended
│   ├── gaussian_viewport.js
│   ├── vendor/
│   ├── viewer_assets/
│   │   ├── index.json
│   │   └── longdress/Q*/<frame>.ply
│   ├── checks/
│   └── longdress/
│       ├── input/<frame>/res*/
│       ├── models/Q*/<frame>/point_cloud/iteration_30000/
│       ├── qualities/Q*/<frame>.npz
│       ├── encoded/progressive/{Base,E1,E2,E3}/
│       ├── .work/encoder_decoded/
│       ├── media/progressive/<layer>/gof_*/
│       ├── decoded/progressive/<layer>/
│       ├── profiles/
│       │   ├── images/
│       │   ├── samples/
│       │   ├── profile.json
│       │   ├── aggregates.json
│       │   └── gains.json
│       ├── logs/
│       ├── preprocess.json
│       ├── training.json
│       ├── export.json
│       ├── encoding.json
│       ├── package.json
│       ├── package_index.json
│       ├── decoding.json
│       └── manifest.json
├── longdress_zlib_control/
└── refresh_ablation/
```

Phân biệt:

- Raw PLY: source volumetric point cloud.
- Checkpoint PLY: output trainer.
- `qualities/*.npz`: exported, uncompressed numeric states.
- `.cpgs`: codec packets.
- `.cpseg`: requestable containers.
- `decoded/*.npz`: reconstructed states từ packaged payload.
- Profile NPZ/PNG: rendered images.
- Viewer PLY: decoded diagnostic assets; không tính vào codec bitrate.

Không overwrite giữa các loại artifacts hoặc dùng old journals để resume ở repo mới.

### Config chính — TO IMPLEMENT

Giữ schema v1 hiện có, không tạo một hệ thống config song song:

```yaml
version: 1
output: output/content_preparation/longdress_full
delivery_modes: [progressive]

objects:
  - id: longdress
    dataset:
      kind: raw
      raw_template: output/datasets/8i/longdress/Ply/longdress_vox10_{frame:04d}.ply
      inventory: output/datasets/8i/longdress/inventory.json
      train_poses: third_party/dynamic-lapis-gs/transforms_train.json
      test_poses: third_party/dynamic-lapis-gs/transforms_test.json
      width: 1024
    frames: {start: 1051, end: 1350, step: 1}
    fps: 30
    qualities:
      - {id: Q0, resolution_scale: 8}
      - {id: Q1, resolution_scale: 4}
      - {id: Q2, resolution_scale: 2}
      - {id: Q3, resolution_scale: 1}

training:
  backend: upstream
  initial_iterations: 30000
  dynamic_iterations: 30000
  lambda_dssim: 0.8
  sh_degree: 3
  extra_args: ["-r", "1"]

encoding:
  name: gaussian_attribute_draco_byteplanes
  precision: f32
  executable: output/build/content_preparation/draco_byte_codec
  encoder_speed: 3
  decoder_speed: 3

packaging:
  gof_frames: 180
  segment_frames: 15
  refresh_frames: {Base: 90, E1: 60, E2: 30, E3: 15}

sampling:
  azimuth: [-180, -135, -90, -45, 0, 45, 90, 135]
  elevation: [-20, 0, 20]
  scales: [0.5, 1.0, 1.5]
  distance: 4.0311287
  times: [1051, 1066, 1081, 1096, 1111, 1126, 1141,
          1156, 1171, 1186, 1201, 1216, 1231, 1246,
          1261, 1276, 1291, 1306, 1321, 1336, 1350]

renderer:
  backend: upstream_cuda
  up_axis: z
  width: 1024
  height: 1024
  fov_degrees: 39.5978
  center: [0, 0, -0.1]
  background: [0, 0, 0]

metrics:
  lpips: true
  lpips_net: vgg
  device: cpu
  batch_size: 1
  distortion: mse

proxy:
  enabled: false
  external_descriptor: null

runtime:
  seed: 0
  gpu_batch_size: 1
  extension_path: null

profiling:
  aggregate_groups:
    - [quality]
    - [quality, time_bin, scale_bin]
    - [quality, time_bin, scale_bin, view_region]

viewer:
  frames: all
  backend: spark
  spark_version: 2.2.0
  three_version: 0.180.0

publication:
  mpd: true
  profile: experimental_gaussian
```

Các fields `inventory`, `proxy.enabled`, `external_descriptor`, `profiling`, `viewer`, `publication` là **extensions cần implement**. Các fields còn lại tái sử dụng.

Field ownership:

| Fields | Stage chịu ảnh hưởng |
|---|---|
| Dataset/poses/frames/qualities | Discovery, preprocessing, training, lineage |
| Training | Training/checkpoints |
| Encoding | Encode/decode, byte accounting |
| Packaging | Containers/access/MPD, không retrain |
| Sampling/renderer/metrics | Profile |
| Aggregate groups | Aggregates/gains reporting |
| Viewer | Diagnostic export/catalog |
| Publication | MPD/content index |
| External proxy reference | Manifest reference בלבד; không gọi proxy pipeline |

### Configs cần tạo

| Config | Nội dung |
|---|---|
| `smoke.yaml` | Existing Q0/Q1 checkpoint, frames 1051–1052, 3 views, 1 scale, render 128² |
| `longdress_visual.yaml` | Existing Q0–Q3, five-frame import; profile 1051/1055, 3 views, 1 scale, render 1024² |
| `longdress_native_pilot.yaml` | Raw 1051–1052, Q0–Q3, original 30k/30k training, full poses |
| `longdress.yaml` | Full 300-frame run như trên |
| `longdress_zlib.yaml` | Import full `training.json`, f32/zlib, cùng packaging; không train lại |
| `refresh_ablation.yaml` | Profiles và scenarios ở mục 9 |

Tất cả nằm dưới `configs/content_preparation/`.

### Profiling grid

| Run | Qualities × times × views × scales | Total renders |
|---|---:|---:|
| Smoke | 2 × 2 × 3 × 1 | 12 |
| Historical visual import | 4 × 2 × 3 × 1 | 24 |
| Native pilot | 4 × 2 × 3 × 1 | 24 |
| Full adaptation profile | 4 × 21 × 24 × 3 | **6,048** |
| Zlib control sanity | 4 × 2 × 3 × 1 | 24 |
| Full sampled GT | 4 × 21 × 2 matching cameras | 168 |

Full objective work: **6,240 renders**, chưa tính smoke/pilots. Q3 reference đã nằm trong số renders, không render thêm một reference cho mỗi lower-quality comparison.

`scale` hiện là projected-size multiplier bằng `distance / scale`; giữ metadata này để query/interpolate đúng.

## 11. Stage-by-stage runbook

Quy ước:

```text
P = python tools/content_preparation/prepare_content.py
V = python tools/validate_prepared_content.py
C = config của run đang thực hiện
R = output root trong C
O = R/longdress
```

`P` là existing CLI. `V` là **TO IMPLEMENT**:

```bash
python tools/validate_prepared_content.py \
  --config CONFIG \
  --stage environment|dataset|preprocess|checkpoints|export|encoding|package|decode|profile|final \
  --report REPORT \
  --resume
```

Validator tái sử dụng Journal, lineage/state/hash/package validators; không kiểm tra proxy.

### Stage 0 — Environment và dataset discovery

- **Purpose:** xác minh backend, raw input, capacity và source provenance.
- **Existing code:** paths/upstream/compatibility helpers, downloader, inventory.
- **New code:** generic validator trên; environment GPU preflight dùng tiny fixture.
- **Input:** config, raw inventory/PLY, pinned sources, extension `.so`, LPIPS cache.
- **Command:** `V --config C --stage environment …`, sau đó `--stage dataset`.
- **Output:** `R/checks/environment.json`, `dataset.json`, resource budget.
- **Validation:** đúng 300 frame, FPS/frame map; all selected raw hashes/bytes/CRC; extension path/version/hash và tiny original-render test; poses có 100 train + 200 test views.
- **Failure/resume:** fail thiếu source/weights/extensions; không tự thay renderer hoặc cài đè shared environment.
- **VRAM:** tiny renderer/EGL preflight; không load full model.
- **Next:** preprocess.

Nếu raw inventory thiếu, downloader hiện có có thể acquisition ở phase execution:

```bash
python tools/download_8i_object.py \
  --object longdress --all-frames \
  --cache output/cache/8i/longdress \
  --output output/datasets/8i
```

HTTPS certificate verification remains enabled by default. Do not add the
historical workspace path to new configs or use an HTTP fallback implicitly.

### Stage 1 — Dynamic-LapisGS preprocessing

- **Purpose:** raw point clouds → source images/cameras/initialization.
- **Existing code:** `upstream.preprocess_worker`, upstream `dataset_prepare.py` functions.
- **New code:** không viết preprocessor mới; chỉ bổ sung validation.
- **Input:** raw PLY, full train/test poses, width 1024, scales 8/4/2/1.
- **Command:** `P --config C --stage preprocess --resume`.
- **Output:** `O/input/<frame>/res*/`, camera JSONs, `points3d.ply`, `canonical.json`, `preprocess.json`.
- **Validation:** dimensions 128/256/512/1024; nonempty foreground; camera counts; initialization seeded; per-frame normalization/hash.
- **Failure/resume:** restart riêng failed frame; completed frames được journal giữ lại.
- **VRAM:** Open3D offscreen/EGL dùng GPU dù phần I/O/downsample chủ yếu CPU; chạy từng frame/split.
- **Next:** train.

Giữ per-frame canonical transforms. Không biểu diễn mọi raw frame bằng transform của frame đầu.

### Stage 2 — Train Q0–Q3

- **Purpose:** tạo source-backed quality checkpoints.
- **Existing code:** `Pipeline.stage_train`, `build_training_command`, `backend.py`, upstream `train.py`.
- **New code:** không đổi trainer.
- **Input:** completed preprocessing, config source-backed, foundation/previous checkpoints.
- **Command:** `P --config C --stage train --resume`.
- **Output:** `O/models/Q*/<frame>/point_cloud/iteration_30000/point_cloud.ply`, logs, `training.json`.
- **Validation:** subprocess success, PLY đọc được, expected attributes, hash và argv.
- **Failure/resume:** failed Q2/frame chỉ restart task đó; giữ Q0/Q1 và completed Q2 tasks. Upstream wrapper hiện không resume optimizer giữa một failed frame.
- **VRAM:** rủi ro cao nhất ở densification/finer first-frame foundation merge; CPU image storage, một quality process.
- **Next:** checkpoint validation.

Không dùng PSNR logging của first-frame foundation branch làm gate: existing upstream reporting có thể render stale `Scene.gaussians`. Đánh giá saved/decoded checkpoint riêng.

### Stage 3 — Checkpoint validation

- **Purpose:** chứng minh lineage và representation schema trước encode.
- **Existing code:** `verify_checkpoint_lineage`, PLY/state helpers.
- **New code:** `V --stage checkpoints`.
- **Input:** `training.json`, mọi selected PLY, original argv/hashes.
- **Command:** `V --config C --stage checkpoints --report R/checks/checkpoints.json --resume`.
- **Output:** per-checkpoint records và validated lineage.
- **Validation:** đủ `N_quality × N_frame`; foundation prefix/static checks; previous-frame correspondence; không matching IDs bằng nearest position.
- **Failure/resume:** stop trước export khi lineage sai; repair đúng task/source.
- **VRAM:** CPU; chỉ giữ checkpoint pair cần so sánh.
- **Next:** export.

### Stage 4 — Quality export

- **Purpose:** checkpoint → canonical internal numeric state.
- **Existing code:** `stage_export`, `read_ply`, `save_state`.
- **New code:** validation, không đổi state format.
- **Input:** validated training manifest/checkpoints.
- **Command:** `P --config C --stage export --resume`.
- **Output:** `O/qualities/Q*/<frame>.npz`, `export.json`.
- **Validation:** attributes/IDs/SH preserved; numeric hash; checkpoint provenance.
- **Failure/resume:** per quality/frame.
- **VRAM:** CPU.
- **Next:** progressive construction/encode.

### Stage 5 — Progressive construction và encoding

Hai phần này chạy chung bằng existing `encode` stage; không invent CLI `--stage layers`.

- **Purpose:** tạo Base/refinements against decoded parent rồi encode.
- **Existing code:** `stage_encode`, sparse replacement semantics.
- **New code:** Draco adapter/native bridge/dispatch.
- **Input:** `export.json`, target states, decoded lower-quality encoder cache.
- **Command:** `P --config C --stage encode --resume`.
- **Output:** `.cpgs`, `encoding.json`, `.work/encoder_decoded/`.
- **Validation:** decode ngay mỗi packet; parent hash; exact state equality cho f32; actual bytes/hash; shared-state changes thực sự được truyền.
- **Failure/resume:** failed E2/frame không encode lại Base/E1 đã hợp lệ.
- **VRAM:** CPU Draco và CPU merge.
- **Next:** package.

### Stage 6 — GoF/refresh/network packaging

- **Purpose:** encoded packets → complete requestable files.
- **Existing code:** `package_object()`, CPSEG writer.
- **New code:** không đổi container cơ bản; ablation orchestrator dùng lại.
- **Input:** `encoding.json`, actual packets, epoch/refresh/network config.
- **Command:** `P --config C --stage package --resume`.
- **Output:** `media/…/*.cpseg`, `package.json`, `package_index.json`.
- **Validation:** serialized sizes/offsets/hashes; union boundaries; complete frame coverage; correct same-frame dependency graph.
- **Failure/resume:** riêng package task; ablation mỗi profile độc lập.
- **VRAM:** CPU/I/O.
- **Next:** decode.

### Stage 7 — Decode verification từ actual packaged payload

- **Purpose:** kiểm tra đường deliver thật.
- **Existing code:** `stage_decode`, `extract_segment_member`.
- **New code:** Draco dispatch và generic validation.
- **Input:** package index, CPSEG files, decoded parent cùng frame.
- **Command:** `P --config C --stage decode --resume`, rồi `V --stage decode`.
- **Output:** `decoded/progressive/<layer>/<frame>.npz`, `decoding.json`.
- **Validation:** `source = requestable segment containers only`; state hashes bằng target exported states; missing/wrong parent/corruption bị từ chối.
- **Failure/resume:** per layer/frame; không quay lại training.
- **VRAM:** CPU.
- **Next:** profiling và viewer export.

Smoke validator phải kiểm tra segment-only decode trong isolated fixture không có checkpoint/export/standalone packet làm decoder side information.

### Stage 8 — Decoded rendering, metrics và profile

Existing `profile` stage đã làm cả ba; giữ stage này.

- **Purpose:** deliverable distortion theo view/scale/time.
- **Existing code:** original renderer adapter, metrics engine, QualityProfile.
- **New code:** grouped aggregates theo time/scale/view region.
- **Input:** `decoding.json`, decoded states, sampling/config.
- **Command:** `P --config C --stage profile --resume`.
- **Output:** render caches, per-sample metrics, profile/aggregates/gains JSON.
- **Validation:** đủ expected grid; highest self-reference MSE=0/PSNR=`"inf"`/LPIPS≈0; native resolution; signed adjacent gains; query/interpolation.
- **Failure/resume:** failed view 35 không render lại completed views 1–34.
- **VRAM:** original renderer một view; LPIPS mặc định CPU/batch 1; reference cache trên disk.
- **Next:** JSON manifest, viewer, GT.

MSE là planner distortion. PSNR/LPIPS là reporting metrics. Không composite arbitrary weights.

Aggregates lưu mean/median/p95/worst; PSNR worst là minimum. Raw conditioned rows luôn giữ lại.

### Stage 9 — JSON manifest/index

- **Purpose:** tổng hợp server-side asset metadata.
- **Existing code:** `build_manifest`, writer, server index.
- **New code:** orchestration bỏ proxy hard dependency; thêm per-frame canonical/provenance references.
- **Input:** package/training/profile indices; optional external descriptor.
- **Command:** `P --config C --stage manifest --resume`.
- **Output:** object/server `manifest.json`, validation summary.
- **Validation:** relative paths, exact files/bytes/hashes, arbitrary N order, correct access requirements.
- **Failure/resume:** manifest riêng; không regenerate payload.
- **VRAM:** CPU.
- **Next:** viewer/MPD.

### Stage 10 — Web visual validation

- **Purpose:** người dùng xem reconstructed content, free camera và quality switching.
- **Existing code:** same viewer HTML, state PLY writer, catalog/report code, browser validator.
- **New code:** viewer exporter + viewport extension; chi tiết tại mục 12.
- **Input:** decoded NPZ/index và profile images.
- **Command:** exporter → static server → extended browser validator.
- **Output:** viewer PLY/index/catalog, screenshots, machine/manual evidence.
- **Validation:** Q0–Q3 đúng object/frame/state; rotate/pan/zoom; camera giữ nguyên; upright/non-mirrored/nonblack; loaded label đúng.
- **Failure/resume:** per asset; failed browser check không retrain.
- **VRAM:** browser WebGL, một selected mesh; đóng browser trước GPU pipeline tiếp theo.
- **Next:** GT/MPD/final acceptance.

### Stage 11 — Matching-GT diagnostic

- **Purpose:** report fidelity với source; tách khỏi adaptation profile.
- **Existing code:** `evaluate_decoded_gt.py`, pose validation.
- **New code:** thêm `--mode progressive`.
- **Input:** decoded states, generated res1 test images và matching pose JSON.
- **Command:** xem execution order.
- **Output:** `gt/report.json`, per-comparison metrics/images.
- **Validation:** exact camera/frame/resolution/background; không resize GT; reference type ghi rõ GT.
- **Failure/resume:** per comparison.
- **VRAM:** sequential original renderer; LPIPS batch 1.
- **Next:** final reports.

### Stage 12 — MPD/content index

- **Purpose:** machine-readable delivery description.
- **Existing code:** JSON package/object/server manifests.
- **New code:** `mpd.py`, `export_content_mpd.py`.
- **Input:** completed manifests/package/profiles.
- **Command:** xem mục 13.
- **Output:** MPD + content index/sidecars.
- **Validation:** XSD, URLs, timing, dependencies, bytes, access semantics.
- **Failure/resume:** XML/index export riêng.
- **VRAM:** CPU.
- **Next:** ablation/final validation.

### Stage 13 — Ablation và final validation

- **Purpose:** đo package/access frontier và kiểm tra toàn bộ deliverables.
- **Existing code:** package/dependency/state/hash helpers.
- **New code:** ablation CLI, generic final validator.
- **Input:** immutable encoded states, profiles, MPD, viewer evidence.
- **Command:** ablation CLI, rồi `V --stage final`.
- **Output:** measurements/Pareto plots/recommendations và final readiness.
- **Validation:** codec parity, expected counts, complete references, reproducibility/resume, no upstream changes.
- **Failure/resume:** giữ từng completed profile/check.
- **VRAM:** chủ yếu CPU; representative render chỉ chạy tuần tự nếu cần.
- **Next:** server assets ready.

## 12. Viewer extension — mandatory first path

### Hiện trạng

Viewer hiện tại chỉ hiển thị PNG và metadata. Nó:

- Chưa có Gaussian rendering/free camera.
- Chưa có object selector; dùng object đầu tiên.
- Chưa có browser Draco/project-container decode.

Draco JS/WASM hiện chỉ có trong third-party source, chưa được tích hợp vào Gaussian assembly.

### Đường bắt buộc

```text
CPSEG
→ project decoder
→ decoded Gaussian NPZ
→ standard Gaussian PLY
→ existing viewer + Gaussian viewport
```

**TO IMPLEMENT:**

```text
tools/export_decoded_viewer_assets.py
tools/content_trial_viewer/gaussian_viewport.js
tools/content_trial_viewer/dependencies.lock.json
tools/content_trial_viewer/vendor_dependencies.py
```

Extend existing:

```text
tools/content_trial_viewer/index.html
tools/validate_trial_viewer.py
tools/summarize_8i_trial.py / run_report helpers, nơi cần reuse catalog metadata
```

Dependency chọn: **SparkJS 2.2.0 + THREE.js 0.180.0**, OrbitControls. Dùng prebuilt modules vendored locally, pin package/release checksums/licenses; không cần tạo npm app mới. Public tagged loader hỗ trợ standard Gaussian PLY và SH degree 3. [Tagged package](https://raw.githubusercontent.com/sparkjsdev/spark/v2.2.0/package.json), [tagged PLY decoder](https://raw.githubusercontent.com/sparkjsdev/spark/v2.2.0/rust/spark-lib/src/ply.rs).

Exporter:

```bash
# TO IMPLEMENT
python tools/export_decoded_viewer_assets.py \
  --prepared-root "$RUN_ROOT" \
  --mode progressive \
  --frames all \
  --resume
```

Output:

```text
RUN_ROOT/viewer_assets/<object>/<quality>/<frame>.ply
RUN_ROOT/viewer_assets/index.json
RUN_ROOT/catalog.json                    v2
RUN_ROOT/index.html
RUN_ROOT/gaussian_viewport.js
RUN_ROOT/vendor/
```

PLY giữ raw fields từ decoded NPZ. Re-import PLY phải cho cùng numeric state hash. IDs không fit uint32 phải fail rõ theo writer hiện có, không tự remap.

Catalog ghi object, quality, frame/time, Gaussian count, SH degree, bytes/SHA, state hash, source decoding index và payload closure.

### UI và memory policy

- Object/quality/frame selector từ metadata, không hard-code Longdress hoặc bốn levels.
- Display mode: free-camera Gaussian hoặc sampled upstream PNG.
- Rotate/pan/zoom/reset camera.
- Giữ camera khi chuyển quality/frame.
- Chỉ một Gaussian mesh; dispose mesh cũ trước khi load mới.
- Bỏ kết quả async stale; loaded label chỉ cập nhật sau load thành công.
- Z-up/identity mesh theo canonical metadata; không copy coordinate flips từ demo.
- Tắt renderer-generated LoD.
- Chạy `extSplats` để giữ xyz float32.

Spark vẫn pack một số attributes/SH nội bộ. Vì vậy web viewport là qualitative diagnostic; objective metrics và PNG fidelity anchor luôn dùng original renderer. Metadata phải ghi browser backend/packing này. [Spark ExtSplats](https://sparkjs.dev/docs/ext-splats/).

Launch:

```bash
python -m http.server 8766 \
  --bind 127.0.0.1 \
  --directory "$RUN_ROOT"
```

URL:

```text
http://127.0.0.1:8766/?object=longdress&quality=Q3&display=gaussians
```

Browser test — **TO IMPLEMENT extended flags**:

```bash
python tools/validate_trial_viewer.py \
  --url 'http://127.0.0.1:8766/?object=longdress&quality=Q3&display=gaussians' \
  --display gaussians \
  --require-webgl \
  --output "$RUN_ROOT/checks/viewer.json"
```

Manual gate:

1. Mở từng Q0–Q3 tại cùng frame/camera.
2. Rotate quanh object; pan/zoom và reset.
3. Switch quality, giữ camera; thử switch nhanh.
4. Kiểm tra đúng loaded label/frame/count/hash.
5. Kiểm tra orientation, scale, opacity/color và nonblack foreground.
6. So với native upstream PNG presets.
7. Lưu screenshots và accepted/rejected evidence.

Browser-side Draco/WASM decode không nằm trong batch bắt buộc. Sau này nó còn cần outer-packet parser, byte-plane reconstruction và progressive merge; không chỉ load Draco decoder JS.

## 13. MPD và sidecar contract

**TO IMPLEMENT:**

```text
tools/content_preparation/mpd.py
tools/export_content_mpd.py
```

CLI:

```bash
python tools/export_content_mpd.py \
  --prepared-root "$RUN_ROOT" \
  --output "$RUN_ROOT/manifest.mpd" \
  --resume
```

MPD dùng static presentation, một Period cho clip và một AdaptationSet mỗi object. GoF epochs giữ trong sidecar; không bắt buộc chuyển thành Periods.

| MPD | JSON sidecars |
|---|---|
| Object/AdaptationSet ID | Detailed object/canonical/frame mapping |
| Base/E1/E2/E3 Representation IDs | Quality order, immediate parent và decoded-parent hash |
| `dependencyId` required closure | Per-frame payload DAG |
| SegmentList/Timeline và URLs | Exact bytes/hash/member offsets |
| Timescale 30, actual segment durations | Refresh flags/access points/full-reset metadata |
| Declared cumulative peak bandwidth | Own-layer/cumulative mean/peak byte rates |
| Experimental codec/container property | Codec/build/config/version provenance |
| Profile/sidecar reference | Full conditioned metric records/gains |
| Optional existing proxy reference | External descriptor reference hoặc `null` |

Quy ước:

- `dependencyId`: E1 cần Base; E2 cần Base/E1; E3 cần Base/E1/E2.
- Terminal frame interval được tính vào duration: 300/30 = 10 s, không dùng last timestamp 9.9667 s làm clip duration.
- `segmentAlignment` lấy từ actual boundary vectors.
- Không claim normative video SAP; Gaussian access points nằm trong sidecar.
- Không invent registered codec fourcc.
- MIME `application/octet-stream`, experimental project profile URI.
- CPSEG không phải `.m4s`/ISO-BMFF; MPD này không hứa ordinary DASH-video-player interoperability.

Writer và tests dùng public MPEG MPD XSD được pin bằng source hash. Local `xmllint` hiện có; validate XML schema và project semantics riêng. [MPEG DASH MPD schema](https://github.com/MPEGGroup/DASHSchema/blob/6th-Ed/DASH-MPD.xsd).

`content_index.json` là index tiện dụng cho later client, tái sử dụng object manifests; không implement runtime parser.

## 14. GPU, storage, resume và provenance

### GPU/RAM/storage

| Stage | Resource chính | Policy |
|---|---|---|
| Raw preprocessing | Open3D GPU/EGL + CPU I/O | Một frame/split |
| Training | CUDA | Một quality/frame; CPU image storage |
| Export/Draco/decode | CPU/RAM | Một state/pair |
| Profile rendering | CUDA | Một decoded model/view |
| LPIPS | CPU mặc định | Batch 1 |
| Viewer | Browser WebGL | Một selected mesh |
| Package/MPD/aggregate | CPU/I/O | Không load Gaussian vào GPU |

CUDA LPIPS đã fit trong historical trial, nhưng full config đầu tiên giữ CPU. Muốn dùng CUDA phải tạo explicit config variant sau preflight; không tự đổi device khi OOM.

Full preprocessing với original poses tạo:

- 90,000 native point-cloud camera renders.
- 360,000 PNG qua bốn resolutions.
- 1,200 initialization PLYs.

Q3 camera RGB tensors trên CPU có thể khoảng 3.52 GiB, chưa gồm PIL images và model structures.

Full training: **36,000,000 iterations**. Không đưa ETA cố định trước native pilot; extrapolate first-frame/dynamic-frame timings theo quality.

Native profile cache hiện khoảng 20 MiB/sample; 6,048 samples khoảng **126.8 GB** nếu giữ RGB/alpha/depth như hiện tại. Viewer PLY cả 300 frame có thể thêm hàng chục GB.

Disk hiện còn khoảng 719 GB. Capacity forecast phải cập nhật từ native pilot, vì 30k training có thể tạo nhiều Gaussian hơn historical trial. Thiếu capacity/OOM phải fail có diagnostic; không xóa intermediates hoặc giảm methodology âm thầm.

### Resume

Reuse existing Journal:

```text
task status
input hashes
resolved config signature
implementation/runtime signature
output paths/hashes
timestamps/commands/version
```

Lưu riêng:

- Pipeline journal.
- Viewer per-object/quality/frame journal.
- Validator per-stage/per-artifact journal.
- Ablation per-profile journal.

Code/config phải freeze trước full run. Hiện fingerprint bao gồm toàn bộ object definition và core module hashes; thêm quality/frame hoặc sửa code có thể invalidate nhiều tasks.

Không:

- Rewrite old historical fingerprints để resume.
- Chạy `--overwrite all` để vượt mismatch.
- Thay packaging config liên tục trong main output.

Packaging variants dùng output riêng với immutable source packets.

### Provenance

Mỗi run/artifact truy được:

- Project HEAD **và working-tree content hashes**.
- Dynamic-LapisGS/nested submodule commits.
- Draco commit/version, compiler/build flags/binary hash.
- Compatibility adapter/launcher hashes.
- Config file/resolved-config hash.
- Raw source inventory/license/hash/frame mapping.
- Checkpoint SHA/original argv/lineage.
- Codec/refinement/packaging settings.
- Decoded state hashes.
- Renderer/metric backend và weight identities.
- Viewer dependency versions/hashes.

Imported historical training provenance và current decode/render provenance phải được giữ riêng; không sửa historical argv/source hashes thành đường dẫn backend mới.

## 15. Smoke-test và test matrix

### Smoke bắt buộc

Một object, Q0/Q1, frames 1051–1052, ba views, một scale, 128².

Import manifest cũ bằng read-only path, output mới. Flow:

```text
verified checkpoint import
→ export
→ Draco encode
→ CPSEG package
→ segment-only decode
→ original renderer
→ metrics/profile
→ JSON manifest
→ MPD
→ decoded PLY
→ existing web viewer
```

Expected:

- 4 exported/decoded quality-frame states.
- 12 conditioned profile rows.
- Six adjacent transition records.
- Exact bit-preserving decode.
- Browser free camera hoạt động cho cả hai quality.
- Full resume zero recomputation.
- Không gọi proxy stage.

Sau đó visual import Q0–Q3 ở native 1024² trước khi training mới.

### Tests cần thêm/mở rộng

| Nhóm | Acceptance |
|---|---|
| Config | Progressive-only, arbitrary N, proxy disabled, invalid interval/missing fields |
| Draco | All attrs/SH, signed zero/subnormal/finite extremes, bounded byte symbols |
| Identity | Reorder, duplicate xyz/distinct IDs, additions/deletions, empty streams |
| Refinement | Shared/static/dynamic replacement, SH schema changes |
| Robustness | Wrong/missing parent, corruption/truncation, unsupported codec |
| Accounting | Actual packet/container bytes; headers/IDs đều tính |
| Decode isolation | Packaged segments đủ, không checkpoint side information |
| Renderer/metrics | Same-state images, MSE 0/PSNR inf/LPIPS≈0, no upstream modifications |
| Profiles | Conditional indexing/interpolation, grouped aggregation, signed gains |
| Access | Same-frame DAG, boundary membership, 5.3 s activation, epoch reset |
| Ablation | 0.5 s control files identical; deterministic profile isolation |
| MPD | XSD, URLs/timing/dependency closure/bandwidth policy |
| Viewer | PLY equality, SH3, free camera, selectors, stale loads, disposal |
| Resume | Crash/tamper/config mismatch; completed work giữ nguyên |

Các existing test cases liên quan proxy vẫn thuộc historical infrastructure; không mở rộng chúng trong batch này. Content phase test selection và new smoke configs phải chứng minh không chạy proxy stage.

## 16. EXACT EXECUTION ORDER

Các commands dưới đây là **runbook tương lai**, chưa chạy trong turn này. Commands với script/flag mới đều cần implementation batch trước.

### Step 1 — Hoàn thành batch code tối thiểu và freeze source

Expected: các missing modules/configs được implement; tests nhỏ pass; không sửa upstream/networking.

Record current commits/tree hashes và resolved configs trước khi chạy dài.

### Step 2 — Activate environment và đặt run paths

```bash
# Run from the repository root after following docs/DEPENDENCIES.md.
conda activate multiobj3dgs-cp

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

SMOKE_CFG=configs/content_preparation/smoke.yaml
VISUAL_CFG=configs/content_preparation/longdress_visual.yaml
PILOT_CFG=configs/content_preparation/longdress_native_pilot.yaml
FULL_CFG=configs/content_preparation/longdress.yaml
CONTROL_CFG=configs/content_preparation/longdress_zlib.yaml

SMOKE_ROOT=output/content_preparation/smoke
VISUAL_ROOT=output/content_preparation/longdress_visual
PILOT_ROOT=output/content_preparation/longdress_native_pilot
FULL_ROOT=output/content_preparation/longdress_full
CONTROL_ROOT=output/content_preparation/longdress_zlib_control
ABLATION_ROOT=output/content_preparation/refresh_ablation
```

Check: Python/torch versions đúng, explicit extension path, source inputs tồn tại.

### Step 3 — Build native Draco bridge

**TO IMPLEMENT source/target:**

```bash
cmake -S tools/content_preparation/native \
  -B output/build/content_preparation \
  -DCMAKE_BUILD_TYPE=Release \
  -DDRACO_SOURCE_DIR="$PWD/third_party/draco" \
  -DDRACO_TESTS=OFF \
  -DDRACO_BUILD_EXECUTABLES=OFF \
  -DDRACO_INSTALL=OFF

cmake --build output/build/content_preparation \
  --target draco_byte_codec --parallel 2

output/build/content_preparation/draco_byte_codec --version
```

Expected: binary/version/build manifest. Build ngoài submodule; không cài đè CUDA extensions.

### Step 4 — Vendoring viewer và bounded tests

**TO IMPLEMENT helper:**

```bash
python tools/content_trial_viewer/vendor_dependencies.py \
  --lock tools/content_trial_viewer/dependencies.lock.json
```

Expected: pinned prebuilt local modules, licenses và hashes. Không cần Node/npm trong run environment.

Run relevant existing/new content tests; optional browser coverage không được skip trong final viewer gate.

### Step 5 — Preflight và smoke dry-run

```bash
# TO IMPLEMENT validator
python tools/validate_prepared_content.py \
  --config "$SMOKE_CFG" --stage environment \
  --report "$SMOKE_ROOT/checks/environment.json" --resume

python tools/content_preparation/prepare_content.py \
  --config "$SMOKE_CFG" --dry-run
```

Check: progressive only, Q0/Q1, two frames, three views, proxy absent from selected stages.

### Step 6 — Smoke pipeline

```bash
for CP_STAGE in preprocess train export encode package decode profile manifest; do
  python tools/content_preparation/prepare_content.py \
    --config "$SMOKE_CFG" --stage "$CP_STAGE" --resume || break
done
```

Không chuyển bước nếu một command trả nonzero.

```bash
# TO IMPLEMENT
python tools/export_decoded_viewer_assets.py \
  --prepared-root "$SMOKE_ROOT" --mode progressive --frames all --resume

python tools/export_content_mpd.py \
  --prepared-root "$SMOKE_ROOT" \
  --output "$SMOKE_ROOT/manifest.mpd" --resume

python tools/validate_prepared_content.py \
  --config "$SMOKE_CFG" --stage final \
  --report "$SMOKE_ROOT/checks/final.json" --resume
```

Check: counts/bit-exact/segment-only decode/profile/MPD như mục 15.

### Step 7 — Smoke web gate và resume

```bash
python -m http.server 8766 \
  --bind 127.0.0.1 --directory "$SMOKE_ROOT"
```

Mở localhost 8766, chạy extended browser validator và kiểm tra Q0/Q1/free camera.

Rerun Step 6 với `--resume`; expected completed tasks được skip. Dừng server/đóng browser trước GPU work tiếp theo.

### Step 8 — Four-quality visual import

Lặp tám explicit stages của Step 6 với `$VISUAL_CFG`.

Expected: 20 quality-frame states và 24 native profile rows; xuất 20 viewer PLY assets.

Export viewer/MPD, mở browser với `$VISUAL_ROOT`. Kiểm tra đủ Q0–Q3. Đây là pipeline/visual gate, không chứng minh full convergence.

### Step 9 — Native training pilot

```bash
python tools/content_preparation/prepare_content.py \
  --config "$PILOT_CFG" --dry-run

python tools/content_preparation/prepare_content.py \
  --config "$PILOT_CFG" --stage preprocess --resume

python tools/content_preparation/prepare_content.py \
  --config "$PILOT_CFG" --stage train --resume

# TO IMPLEMENT
python tools/validate_prepared_content.py \
  --config "$PILOT_CFG" --stage checkpoints \
  --report "$PILOT_ROOT/checks/checkpoints.json" --resume
```

Expected: eight source-default checkpoints, 30k/30k, full train/test poses.

Tiếp tục export → encode → package → decode → profile → manifest và viewer/MPD.

Gate: Q3 native training/render chạy được; lineage đúng; RAM/VRAM/disk/ETA đã đo. Nếu fail, resolve trước full run, không tự hạ iterations/resolution.

### Step 10 — Full input/capacity validation và dry-run

```bash
# TO IMPLEMENT
python tools/validate_prepared_content.py \
  --config "$FULL_CFG" --stage dataset \
  --report "$FULL_ROOT/checks/dataset.json" --resume

python tools/content_preparation/prepare_content.py \
  --config "$FULL_CFG" --dry-run
```

Check: 300 frames, four qualities, progressive only, 6,048 adaptation renders; capacity forecast từ native pilot.

### Step 11 — Full preprocessing

```bash
python tools/content_preparation/prepare_content.py \
  --config "$FULL_CFG" --stage preprocess --resume

# TO IMPLEMENT
python tools/validate_prepared_content.py \
  --config "$FULL_CFG" --stage preprocess \
  --report "$FULL_ROOT/checks/preprocess.json" --resume
```

Expected: source datasets cho 300 frame/four scales và full camera counts.

### Step 12 — Full training và lineage gate

```bash
python tools/content_preparation/prepare_content.py \
  --config "$FULL_CFG" --stage train --resume

# TO IMPLEMENT
python tools/validate_prepared_content.py \
  --config "$FULL_CFG" --stage checkpoints \
  --report "$FULL_ROOT/checks/checkpoints.json" --resume
```

Expected: **1,200 checkpoints**. Chỉ tiếp tục khi validated lineage pass.

### Step 13 — Export → Draco encode → package → decode

```bash
for CP_STAGE in export encode package decode; do
  python tools/content_preparation/prepare_content.py \
    --config "$FULL_CFG" --stage "$CP_STAGE" --resume || break
done

# TO IMPLEMENT
python tools/validate_prepared_content.py \
  --config "$FULL_CFG" --stage decode \
  --report "$FULL_ROOT/checks/decode.json" --resume
```

Expected: 1,200 progressive packets và reconstructed states.

Initial 0.5 s network grid: 20 segments/layer × four layers = **80 CPSEG files**, hai representation epochs; mọi state exact.

### Step 14 — Full profile và JSON manifest

```bash
python tools/content_preparation/prepare_content.py \
  --config "$FULL_CFG" --stage profile --resume

python tools/content_preparation/prepare_content.py \
  --config "$FULL_CFG" --stage manifest --resume
```

Expected: 6,048 raw metric rows; **4,536 adjacent quality gains**; grouped aggregates và valid manifests.

### Step 15 — Zlib control, không train lại

`longdress_zlib.yaml` dùng:

```text
dataset.kind: checkpoints
checkpoint_manifest: output/content_preparation/longdress_full/longdress/training.json
training.backend: existing
encoding: gaussian_attribute_zlib / f32 / zlib level 6
```

Lặp các explicit stages với `$CONTROL_CFG`; profile chỉ dùng two times/three views/one scale.

Generic validator thêm **TO IMPLEMENT** option:

```bash
python tools/validate_prepared_content.py \
  --config "$CONTROL_CFG" --stage final \
  --compare-decoded-root "$FULL_ROOT/longdress" \
  --report "$CONTROL_ROOT/checks/codec_comparison.json" --resume
```

Check: mọi decoded numeric state bằng Draco; actual rates/encode/decode timings report riêng.

### Step 16 — Matching-GT report

**TO IMPLEMENT `--mode progressive`:**

```bash
python tools/evaluate_decoded_gt.py \
  --object-root "$FULL_ROOT/longdress" \
  --mode progressive \
  --source-template "$FULL_ROOT/longdress/input/{frame}/res1" \
  --poses third_party/dynamic-lapis-gs/transforms_test.json \
  --frames 1051 1066 1081 1096 1111 1126 1141 \
           1156 1171 1186 1201 1216 1231 1246 \
           1261 1276 1291 1306 1321 1336 1350 \
  --views 5 10 \
  --output "$FULL_ROOT/gt" --resume
```

Expected: 168 comparisons, labels tách rõ GT và adaptation reference.

### Step 17 — Full viewer và visual acceptance

```bash
# TO IMPLEMENT
python tools/export_decoded_viewer_assets.py \
  --prepared-root "$FULL_ROOT" --mode progressive --frames all --resume

python -m http.server 8766 \
  --bind 127.0.0.1 --directory "$FULL_ROOT"
```

Mở:

```text
http://127.0.0.1:8766/?object=longdress&quality=Q3&display=gaussians
```

Check Q0–Q3, first/middle/last frame và 5.3/6.0 s; run extended validator; lưu manual acceptance. Đóng browser trước representative GPU checks khác.

### Step 18 — MPD và refresh ablation

```bash
# TO IMPLEMENT
python tools/export_content_mpd.py \
  --prepared-root "$FULL_ROOT" \
  --output "$FULL_ROOT/manifest.mpd" --resume

python tools/run_refresh_ablation.py \
  --prepared-root "$FULL_ROOT" \
  --config configs/content_preparation/refresh_ablation.yaml \
  --output "$ABLATION_ROOT" --resume
```

Expected: XSD-valid experimental MPD, exact sidecars; 14 configurations có journal/report độc lập, null-effect control và Pareto figures.

### Step 19 — Final validation và resume audit

```bash
# TO IMPLEMENT
python tools/validate_prepared_content.py \
  --config "$FULL_CFG" --stage final \
  --report "$FULL_ROOT/checks/final.json" --resume
```

Rerun explicit eight stages với `--resume`; expected no completed training/encode/render tasks recomputed. Resume ablation và viewer exporter cũng skip unchanged work.

Final report phải ghi actual task counts, memory/resource measurements, codec boundary, browser evidence và remaining limitations.

## 17. Final server asset layout và blockers

Server-ready assets:

```text
manifest.mpd
manifest.json
content_index.json
longdress/manifest.json
longdress/package_index.json
longdress/media/progressive/...
longdress/profiles/{profile,aggregates,gains}.json
codec/provenance metadata
optional externally prepared proxy reference
```

Viewer assets, decoded NPZ, checkpoints và raw/prepared sources là preparation/diagnostic artifacts, account storage riêng.

Các blockers/gates đã xác định:

| Gate | Cách xử lý trong plan |
|---|---|
| Exact enhanced Draco LTS không public | Không block baseline; label reproduction boundary rõ |
| Draco adapter chưa có | Implement/test trước smoke |
| GPU default có thể import Scaffold extension | Explicit verified vendor path; preflight trước training |
| MPD chưa có | Project writer từ existing manifests |
| Viewer PNG chưa đáp ứng free camera | Extend existing shell |
| Fresh manifest hiện cần proxy | Optional external/null reference; không proxy execution |
| Full resource usage chưa đo | Native source-default pilot trước 300-frame run |
| Refresh optimum với non-temporal codec | Report null/packaging effect; không claim temporal benefit |
| Q4 operating point chưa chọn | Deferred; arbitrary N architecture đã có |
| Manual visual acceptance chưa thực hiện | Bắt buộc sau implementation, lưu evidence |

Không còn quyết định implementation cần coding agent tự đoán; các gates phụ thuộc kết quả kiểm tra thực tế.

## 18. Stage summary

| Stage | Existing file | New file/change | Command | Input | Output | Validation |
|---|---|---|---|---|---|---|
| Environment/dataset | `paths.py`, `upstream.py`, downloader | `validate_prepared_content.py` | `V --stage environment/dataset` | Pins/inventory/runtime | Checks/resource report | Hashes, tiny GPU, camera/frame map |
| Preprocess | `pipeline.py`, `upstream.py`, backend functions | Validator | `P --stage preprocess` | Raw/poses/scales | Input datasets/index | Dimensions/canonical/counts |
| Train | `backend.py`, upstream `train.py` | Không trainer mới | `P --stage train` | Prepared/foundation/previous | PLY/logs/training index | Saved checkpoint/argv |
| Checkpoints | Lineage/state helpers | Generic validator | `V --stage checkpoints` | PLY/manifest | Lineage report | Stable correspondence |
| Export | `assets.py`, stage export | Validator | `P --stage export` | Checkpoints | State NPZ/export index | Numeric equality |
| Progressive/encode | `codec_adapter.py`, stage encode | Draco adapter/native bridge | `P --stage encode` | Target/decoded parent | Packets/cache/index | Exact closed-loop decode |
| Package | `packaging.py` | Ablation wrapper | `P --stage package` | Actual packets/config | CPSEG/index | Bytes/offsets/boundaries |
| Decode | Stage decode/extractor | Draco dispatch | `P --stage decode` | CPSEG/parent | Decoded NPZ/index | Segment-only/exact |
| Profile | Renderer/metrics/profile modules | Conditional grouping | `P --stage profile` | Decoded states/grid | Images/metrics/profile/gains | Grid/reference/query |
| JSON manifest | `manifest.py`, stage manifest | Remove proxy dependency | `P --stage manifest` | Package/profiles/provenance | Object/server JSON | References/hashes/access |
| Viewer | Existing HTML/PLY/catalog/browser tools | Exporter/viewport/vendor helper | Viewer export + http.server | Decoded NPZ | PLY/catalog/UI/evidence | Free camera/switching |
| GT | `evaluate_decoded_gt.py` | Progressive mode flag | GT CLI | Decoded/GT/cameras | Fidelity report | Matching cameras |
| MPD | JSON manifests | `mpd.py`, export CLI | MPD export | Manifests/package | MPD/content index | XSD/project semantics |
| Refresh ablation | Package/access validators | Ablation module/CLI | Ablation CLI | Immutable packets | Measurements/Pareto | Null control/correctness |
| Final | Existing validation/journals | Generic final validator | `V --stage final` | All deliverables | Readiness evidence | Completeness/resume/provenance |

## 19. Recommended first implementation batch

Nhóm tối thiểu trước end-to-end smoke:

1. **Draco adapter/native bridge và dispatch**, giữ zlib compatibility; bit-exact/refinement tests.
2. **Proxy-off orchestration và optional manifest reference**, để explicit pipeline kết thúc mà không gọi proxy.
3. **Generic validator và smoke config**, reuse existing Journal/validators.
4. **MPD writer/CLI**, experimental signaling và XSD validation.
5. **Decoded PLY exporter + free-camera viewport trong viewer cũ**, pinned local dependencies và browser checks.

Sau khi batch này pass smoke/visual import, bổ sung grouped aggregation, progressive GT option, refresh ablation và full/native-pilot configs. Freeze implementation/configs rồi mới đi theo full runbook.

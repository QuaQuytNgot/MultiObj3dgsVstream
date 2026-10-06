# Content Preparation Readiness

## Current first-batch result — 2026-10-06

**PASS in `MultiObj3dgsVstream`**, limited to the approved first implementation
batch. All six components are operational: Draco native bridge/adapter, codec
integration, proxy-off orchestration, generic validator, MPD writer and the
decoded Gaussian extension to the existing viewer.

- **139 preparation self-tests passed, no failures/skips**, including real
  pretrained LPIPS and the legacy Chromium viewer check. Syntax/import checks
  and config dry-run passed.
- Actual smoke used **Longdress Q0/Q1, frames 1051–1052**, imported from existing
  checkpoints. No new training, native pilot or full sequence was started.
- Export → Draco → CPSEG → decode → original CUDA renderer → metrics → profile
  → MPD → decoded PLY → WebGL passed. There are 4 exported/encoded/decoded states,
  4 segments, 12 view-conditioned metric samples and 6 transition gain samples.
- `gaussian_attribute_draco_byteplanes` **1.0.0** uses pinned public Draco 1.5.7.
  Numeric f32 state is bit-exact. It is project code, not unpublished LTS.
  Progressive shared-state replacements and arbitrary quality counts are
  supported. Production smoke selected only progressive mode; legacy codec and
  representation behavior remain covered by tests.
- Actual media is **39,486,478 bytes**, including 973 bytes of CPSEG container
  overhead. No temporal prediction/history is introduced. The decoded Q1 is
  this smoke's highest-quality adaptation reference; GT quality is not measured
  by this smoke.
- MPD writer **1.1.0** passed pinned offline XSD and file/timing/dependency checks.
  Q1's exact required cumulative peak is **4,895,983,920 bits/s**, carried by a
  mandatory extended-rate property/JSON because the standard bandwidth field is
  uint32. This experimental CPSEG profile is not ordinary DASH video playback.
- Real WebGL browser audit passed **4/4 quality/frame assets** with verified
  served PLY/UI/vendor hashes, rotation/pan/zoom, camera preservation, rapid
  switches and one resident mesh. There were no JS/HTTP errors. The screenshot
  was inspected; personal visual acceptance remains the user's browser review.
- GPU guard recorded `peak_active=1`, `active=0`, 12 views. Peak **PyTorch CUDA
  allocation** was **58,848,768 bytes**, at 128×128. This excludes driver/context
  memory and does not establish full/native training capacity on 6 GB.
  LPIPS ran on CPU, batch 1. Proxy preparation was disabled.
- Resume executed **0 tasks**: pipeline skipped 49, MPD 1, viewer export 17 and
  validator 1. Original upstream source hashes stayed unchanged.

Run root: `output/content_preparation/smoke/`. Final audit is `checks/all.json`;
browser evidence is `viewer_validation.json`. Open
[decoded Gaussian smoke viewer](http://127.0.0.1:8766/?object=longdress&quality=Q1&display=gaussians)
while its local server is running. Exact commands and bounds are in
[first-batch runbook](CONTENT_PREPARATION_BATCH1.md), with portable evidence in
[batch1 smoke summary](validation/content_preparation_batch1_smoke_summary.json).

For a clean RTX 5060 Ti checkout, follow the CUDA 13/PyTorch and pinned extension
setup in [DEPENDENCIES.md](DEPENDENCIES.md#clean-machine-setup-rtx-5060-ti), then
the new-machine run order in
[CONTENT_PREPARATION_STEP_BY_STEP.md](CONTENT_PREPARATION_STEP_BY_STEP.md#14-new-machine-execution-order-visual-gate-native-pilot-full-longdress).
The runtime counts below are historical GTX 1660 measurements, not a forecast
for the RTX 5060 Ti.

**Stopping condition reached.** Grouped aggregation, progressive GT diagnostics,
refresh ablation and native/full-run configs/execution remain deferred. The
material below preserves historical migration/source-workspace observations.

> **Migration note (2026-10-05):** project-owned preparation code and this document
> were migrated from the sibling `dynamic-lapis-gs` workspace. Measured results,
> PASS counts and hardware observations below describe that historical workspace;
> they do not establish a completed run in `MultiObj3dgsVstream`. Dataset, models,
> vendor binaries and generated `output/` artifacts are not bundled. Commands now
> use this project root and the pinned backend in `third_party/dynamic-lapis-gs`.
> See [migration/status](CONTENT_PREPARATION_MIGRATION.md) before reproducing a run.

## Trạng thái tại thời điểm migration — 2026-10-05

Code project-owned nằm trong `tools/`, `scripts/`, `configs/`; Dynamic-LapisGS và
CUDA-extension source nằm trong pinned submodule `third_party/dynamic-lapis-gs`.
Migration validation chỉ kiểm tra portability, CLI/configs và các tests nhỏ. Chưa
chạy lại acquisition, Longdress training, full preparation, CUDA render/profile
hoặc viewer browser audit trong repo đích. Không build extensions trong turn này.

Trong repo đích ngày 2026-10-05: **184 pytest tests PASS, 1 skipped, 121 subtests
PASS**; preparation self-test chạy 105 tests với một optional browser skip vì
chưa cài Playwright. CPU mini pipeline dùng pretrained VGG LPIPS thật. Compilation,
18 CLI help checks, sáu config dry-runs và hai codec self-tests PASS; xem
[migration note](CONTENT_PREPARATION_MIGRATION.md) để phân biệt với evidence GPU cũ.

Checkout mới chưa có dữ liệu/model/output; config import checkpoint cần inputs
được cung cấp riêng. Source/backend layout mới thay fingerprints: dùng output mới
cho run mới thay vì tiếp tục journal/guard của workspace cũ. Xem
[hướng dẫn từng bước](CONTENT_PREPARATION_STEP_BY_STEP.md) và
[migration note](CONTENT_PREPARATION_MIGRATION.md).

## Evidence lịch sử của workspace nguồn

Lần đo gần nhất trong workspace `dynamic-lapis-gs`: **2026-10-05, Longdress bốn
trained qualities Q0–Q3 trên 1051–1055**. Các kết quả dưới đây giữ nguyên evidence;
không phải tuyên bố đã chạy lại trong repo đích. Không chạy full-sequence training
Longdress/Soldier/Loot.

### Kết quả Longdress bốn level đã chạy tại nguồn

Raw có đủ **300 frames / 10 s**; trial train/encoded chỉ **5 frames / 0.166667 s**, timestamp span 0.133333 s. Reuse Q0/res8 và Q1/res4; original `train.py` đã train Q2/res2 và Q3/res1 với foundation/temporal lineage được verify. Budget first/dynamic là 6,000/1,500 cho Q0/Q1 và 10,000/1,500 cho Q2/Q3; không tuyên bố full convergence 30,000 iterations. Trainer, loss, renderer và CUDA-extension source không thay đổi.

| Quality | Verified training images | Gaussians ở frame đầu |
|---|---:|---:|
| Q0 / res8 | 128×128 | 41,700 |
| Q1 / res4 | 256×256 | 78,421 |
| Q2 / res2 | 512×512 | 144,612 |
| Q3 / res1 | 1024×1024 | 231,823 |

Hai variant đã hoàn thành đủ chín stages, cả independent và progressive Base/E1/E2/E3. Mọi quality render **1024×1024** bằng original renderer từ decoded packaged payload; profile giữ 3 views × 2 scales × 3 keyframes × 4 qualities × 2 modes và dùng decoded Q3 làm reference.

| Kết quả mỗi variant | f32 lossless | f16 |
|---|---:|---:|
| Audit | PASS, numeric bit-exact | PASS, exact float16 rounding |
| Exported / decoded states | 20 / 40 | 20 / 40 |
| Profile rows | 144 | 144 |
| Segments, cả hai modes | 32 | 32 |
| Stored media bytes, cả hai modes | 856,942,065 | 449,653,951 |
| Frozen proxy | 256 Gaussians × 5 frames | 256 Gaussians × 5 frames |
| Full resume executed / skipped | 0 / 460 | 0 / 460 |

Byte totals gồm containers/headers và tất cả qualities/layers của cả hai modes; không phải bytes chỉ để xem Q3. Independent/progressive numeric states khớp tại mỗi quality/frame. Extension resume chạy 0 tasks, skip 32/32; guarded entry point đóng băng runtime/source/extension hashes trong sidecar trước khi resume.

**90 self-tests PASS**, syntax/import checks PASS. Native catalog có 288 PNG 1024×1024; browser kiểm tra đủ **288/288 conditions**, đúng PNG URL, training labels, per-view metrics, fit không upscale, zoom 1:1 và mobile không tràn trang, không có lỗi JS/HTTP. Mở [viewer tại Q3](http://127.0.0.1:8765/?quality=Q3) khi server phục vụ `output/content_prepare_longdress_4level/`. Đây là sampled offline PNG viewer; không phải continuous video hoặc runtime 3DGS renderer.

Matching-GT diagnostic riêng hoàn thành **16 comparisons**: f32 decoded independent states, frames 1051/1055 và test cameras 5/10, bốn comparisons mỗi quality. Max camera world-view error là 8.22×10⁻⁷; mọi metric finite, resume chạy 0 tasks và skip 16/16.

| Quality | Mean MSE vs matching GT | Mean PSNR (dB) | Mean LPIPS |
|---|---:|---:|---:|
| Q0 | 0.00260485 | 25.84 | 0.068016 |
| Q1 | 0.00122150 | 29.13 | 0.053845 |
| Q2 | 0.00051612 | 32.91 | 0.041993 |
| Q3 | 0.00042250 | 33.85 | 0.038619 |

MSE đo full-image RGB trên background đen, PSNR là mean của từng sample, LPIPS dùng pretrained VGG batch 1. Adaptation profile vẫn reference highest **decoded Q3**, giữ raw view/scale/time rows; bảng GT không thay phương pháp đó. GT sampled và mixed training budgets không chứng minh toàn sequence đã hội tụ.

Hai config native bốn level đặt LPIPS **CUDA batch 1** rõ ràng sau preflight; mặc định framework và config raw 30-frame vẫn CPU LPIPS. Mỗi variant ghi **306 GPU operations = 162 renders + 144 LPIPS batches**, guard `peak_active=1`, cuối cùng `active=0`. Renderer peak PyTorch allocation là 255,027,712 bytes; VGG LPIPS preflight 1024 px là 1,754,546,176 bytes. Một sample LPIPS CPU/CUDA chênh tuyệt đối 3.72529×10⁻⁹. Peak device memory quan sát trong 1,559 samples mỗi 2 s là **2,005 MiB**; đây là sampled peak, không bảo đảm instantaneous peak hoặc full training luôn vừa 6 GB.

Bằng chứng gọn có thể giữ trong Git: [longdress_4level_trial_summary.json](validation/longdress_4level_trial_summary.json). Artifacts thực tế: `output/content_prepare_longdress_4level/{audit_f32.json,audit_f16.json,gt_diagnostic/report.json,viewer_validation.json}`, các manifests/profiles/proxies dưới từng variant và `output/longdress_4level_training/manifest.json`. Commands reproduce/resume và các bước kiểm tra nằm trong [hướng dẫn từng bước](CONTENT_PREPARATION_STEP_BY_STEP.md).

## Kết quả lịch sử và baseline nhỏ

Trial hai level trước đó: 76 self-tests PASS, 144 browser combinations PASS, mỗi variant resume skip 256/256 tasks. Xem [LONGDRESS_ENCODING_TRIAL.md](LONGDRESS_ENCODING_TRIAL.md) và `output/content_prepare_longdress_trial/validation_summary.json`; giữ output này riêng để đối chiếu. Các con số smoke bên dưới là baseline 2026-10-04 trong `output/content_prepare_smoke/`, không phải kích thước trial bốn level.

## Kết quả kiểm tra lịch sử (workspace nguồn)

- **48 self-tests PASS**, 0 failed, 0 skipped; 7.192 giây trong môi trường Hoang. Có mini end-to-end CPU với LPIPS VGG pretrained thật.
- Syntax/compile và import toàn bộ module PASS. `train.py --help` nguyên bản PASS; lệnh help không chạy training.
- CUDA smoke PASS trên **NVIDIA GTX 1660 6 GB**: synthetic fixture, 2 frames, 2 qualities, 3 views/frame, 64×64, cả independent và progressive. Synthetic checkpoint generation không phải training experiment.
- 99 tác vụ hoàn thành; `--resume` chạy **0** tác vụ và skip đủ **99** tác vụ.
- 24 profile rows, 12 transition gains, 6 requestable segments, **29,202 media bytes**, proxy cố định 6 Gaussians × 2 frames. Byte media bao gồm toàn bộ codec/container headers; auxiliary assets tính riêng.
- 30 GPU view operations; guard ghi nhận `peak_active=1`, cuối cùng `active=0`. Peak PyTorch CUDA allocation của smoke là **8,649,216 bytes (~8.25 MiB)**; không bao gồm CUDA driver/context/ngoài allocator và không đại diện peak full training.
- Highest decoded self-reference: MSE=0, PSNR=`"inf"`, LPIPS=0. Q0 có MSE/LPIPS dương; ảnh có foreground/alpha/depth khác 0. Independent/progressive reconstruct cùng numeric state tại mỗi quality/frame.
- Kiểm tra decode độc lập đã xóa input/checkpoint/export/standalone encoded/cache của mini test rồi dùng CLI `--stage decode --overwrite`; chỉ packaged segments vẫn đủ để reconstruct.
- **Mini upstream training PASS**: prepared synthetic images, 512 initial points, 2 frames/2 qualities, chỉ 3 iterations ban đầu và 2 iterations dynamic; chạy `train.py` thật, verify foundation/temporal lineage, rồi toàn pipeline hoàn thành 59 tác vụ, 8 profile rows. Output riêng ở `output/content_prepare_smoke/training_mini/`; không dùng dataset thật.
- **Raw preprocessing smoke PASS**: 512 colored raw points tổng hợp, 1 frame, 2 train/2 test views, base 64×64; nguyên `dataset_prepare.render_2d_image` và resize. 2 tác vụ hoàn thành; canonical metadata, nonempty resized images và seeded 100,000-point CPU initialization hợp lệ. Chỉ preprocessing; không train raw fixture này. Output: `output/content_prepare_smoke/raw_mini/`.

Bằng chứng machine-readable: validation_summary.json (`output/content_prepare_smoke/validation_summary.json`, historical artifact), GPU/object validation (`output/content_prepare_smoke/analytic_fixture/validation.json`, historical artifact), source integrity (`output/content_prepare_smoke/validation_upstream.json`, historical artifact), object manifest (`output/content_prepare_smoke/analytic_fixture/manifest.json`, historical artifact), server index (`output/content_prepare_smoke/manifest.json`, historical artifact).

## Trả lời A–M từ experiment lịch sử

| Câu hỏi | Trạng thái và giới hạn |
| --- | --- |
| A. Pipeline end-to-end? | **Có**: Longdress 5-frame/4-quality real training extension → encode/package/decode/render/profile/proxy/manifest, cả f32/f16 và hai modes. Baseline fixture/prepared/raw preprocessing cũng đã kiểm tra riêng. Không chạy full dataset/full convergence experiment. |
| B. Stage nào smoke-tested? | Tất cả 9 stage. Main smoke dùng synthetic checkpoint generation; mini thứ hai dùng original training thật với vài iterations, foundation và dynamic update; mini thứ ba chạy original raw Open3D preprocessing. Encode/package/decode/profile/proxy/manifest và renderer CUDA dùng code production. |
| C. Upstream renderer/training? | **Giữ nguyên source có sẵn**. Renderer gọi trực tiếp `gaussian_renderer.render`; alpha/depth dùng thêm 2 pass override-color tuần tự. Training wrapper gọi `train.py`, giữ loss/optimization, dùng đúng foundation/previous-frame flags. Existing modifications của workspace trước turn này được giữ nguyên. Python/CUDA/header source hashes và extension binary hashes được lưu. |
| D. Codec chính xác? | `gaussian_attribute_zlib` **1.0.0**, state `gaussian-state-v1`, packet `CPGAUS01`: float32 lossless hoặc f16/q8…q16 configurable, bit packing, zlib. Smoke: **f16 + zlib level 6**. Longdress bốn level đã audit cả **f32 và f16 + zlib level 6**. Runtime smoke: Python 3.11.16, PyTorch 2.3.1+cu118, NumPy 2.4.6, zlib 1.3.2. Optional zstd adapter path chưa smoke-tested. |
| E. Phần giống LTS? | Content preparation ordering, layered dynamic assets và GoF/requestable segmentation ở mức kiến trúc. Dynamic-LapisGS trainer/renderer là upstream hiện tại. Public paper mô tả enhanced Draco lossless; không có exact implementation trong local/public repo đã inspect. [Chi tiết và primary sources](CONTENT_PREPARATION_CODEC.md). |
| F. Phần reproduction/approximation? | Attribute+zlib codec, sparse shared-state replacements, CPSEG1 containers, proxy và profile/index là baseline của repo này; **không bitstream-compatible với LTS enhanced Draco**, không tái tạo tiling/DASH/ABR của LTS. Lossy f16/q* không được mô tả là codec LTS. |
| G. Progressive? | **Có**: Base/E1/E2…; all-attribute shared/static/dynamic replacements, new/deleted IDs, decoded-parent checksum. Unit test gồm đổi SH schema/order. Smoke dùng Base/E1; trial Longdress đã kiểm tra Base/E1/E2/E3 và numeric parity với independent. Imported PLY phải có argv/hash gốc và verified lineage; không suy luận matching từ vị trí gần nhau. |
| H. Independent? | **Có**: Q0/Q1/Q2… tự chứa, không cần lower quality khi switch. Smoke và graph/access tests PASS. |
| I. View-conditioned profile? | **Có**: raw object/mode/quality/view/scale/time bins; index + nearest/inverse-distance/multilinear query; azimuth periodic; giữ raw samples cùng aggregates/gains. Default distortion MSE; PSNR/LPIPS reporting; không cộng arbitrary weights. Native Longdress: 3 views × 2 scales × 3 times × 4 qualities × 2 modes = 144 rows mỗi precision variant. |
| J. Proxy? | **Có**: frozen stable-ID Gaussian subset, descriptor/load/render, silhouette/alpha/expected-depth validation đối với highest decoded. Không tối ưu proxy hoặc làm online multi-object visibility scheduler. |
| K. Resume/reproducibility? | **Có**: per-frame/per-view task journal, config/input/implementation/runtime signatures, deterministic state/packet/container files, hashes/versions/provenance, overwrite protection, process locks. Native Longdress resume skip 460/460 mỗi variant, extension 32/32 và GT 16/16; đều 0 executed. Tests crash tại view mới giữ view cũ; tamper/config/input changes fail explicit; checkpoint import không tự tạo bằng chứng lineage. |
| L. Nguy cơ OOM 6 GB? | **Có ở larger training**, nhất là densification, first higher-quality foundation merge và render/training resolution cao. Bốn-level/5-frame trial res1 đã vừa GTX 1660; sampled device peak 2,005 MiB không bảo đảm full sequence fit. CPU training images, sequential renderer/LPIPS, batch 1, cleanup/process isolation giảm peak. Native trial dùng CUDA LPIPS đã preflight; default/raw config vẫn CPU. OOM report stage/log, gợi ý explicit setting/run mới; không tự đổi methodology. |
| M. Command full Longdress? | Command bên dưới chạy sequence **1051–1080, 4 qualities** theo config; chưa được thực thi. Cần raw PLY đủ frame ở đường dẫn cấu hình hoặc chỉnh input mode/path trước khi chạy. |

## Lệnh dùng sau này

```bash
cd MultiObj3dgsVstream
conda activate Hoang

# Kiểm tra config và plan, không tạo output hoặc training.
python tools/content_preparation/prepare_content.py \
  --config configs/content_prepare_longdress.yaml --dry-run

# Chỉ chạy khi muốn bắt đầu preparation thật.
python tools/content_preparation/prepare_content.py \
  --config configs/content_prepare_longdress.yaml --stage all --resume
```

Raw path hiện tại: `output/datasets/8i/longdress/Ply/longdress_vox10_{frame:04d}.ply`, trực tiếp từ downloader mới `tools/download_8i_object.py`. Config training này vẫn chọn 1051–1080; thay `frames.end: 1350` nếu muốn chuẩn bị đủ 300 frame ở phase sau. `prepare_content.py` không tự download dataset. Trial bốn level thực tế dùng output riêng `output/content_prepare_longdress_4level/` và `training.backend: existing` sau guarded training extension; historical trial hai level giữ tại `output/content_prepare_longdress_trial/`.

Để dùng checkpoint chuẩn bị sẵn, đổi `dataset.kind` thành `checkpoints`, thêm `checkpoint_manifest`, và chọn `training.backend: existing`. Để dùng prepared NeRF images, chọn `dataset.kind: prepared`, `source_template` chứa `{frame}`, `{quality}` hoặc `{scale}`, `training.backend: upstream`; pipeline tạo private copies và không sửa input nguồn. Soldier/Loot/object khác dùng object ID, frame map và templates riêng; implementation không hard-code Longdress.

```bash
python tools/content_preparation/self_test.py
python tools/content_preparation/prepare_content.py \
  --config configs/content_prepare_smoke.yaml --resume
python tools/content_preparation/prepare_content.py \
  --config configs/content_prepare_longdress.yaml --stage encode --resume
```

Stage riêng cần artifacts prerequisite đã tồn tại; không implicit training. Changed/tampered completed task cần `--overwrite` rõ ràng hoặc output mới. `--resume` sau crash chỉ làm lại tác vụ chưa hoàn thành. Fresh smoke rerun cần `--overwrite`; không chạy self-test/config-CUDA task song song với smoke đang giữ exclusive GPU lock.

## Coverage A–O lịch sử

| Yêu cầu test | Bằng chứng |
| --- | --- |
| A config parsing | YAML/defaults, frame/time errors, codec/memory contracts |
| B manifest roundtrip | Validated save/load, actual relative paths/checksums |
| C codec roundtrip | f32/f16/q8…q16, all-attribute refinements, corruption |
| D decoded render | CPU integration + production original CUDA smoke |
| E metric sanity | Real pretrained VGG, zero self MSE/LPIPS, infinite PSNR |
| F indexing | Bin lookup, periodic/interpolated view/scale/time queries |
| G aggregation | Uniform/weighted/median/p95/worst; raw rows retained |
| H progressive graph | Required same-frame immediate parent, cycle/reordering rejection |
| I independent graph | Self-contained qualities with zero layer dependencies |
| J refresh/access | Common/layer refresh, boundary unions/GoF reset/access schedules |
| K resume | Crash recovery, 99 skips, mutation/tamper/overwrite checks |
| L provenance | State/file hashes, original argv/hash requirements, runtime/source hashes |
| M proxy | Frozen IDs, load/render/checksum, silhouette/depth/alpha validation |
| N GPU sequential | Real guard peak=1/active=0, process lock overlap rejection, LPIPS batch 1 |
| O unchanged upstream | Before/after Python/CUDA/header hashes; original train CLI import; existing local patches preserved |

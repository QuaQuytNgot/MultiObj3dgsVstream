# Content preparation: hướng dẫn từng bước

> **Migration note (2026-10-05):** project-owned preparation code and this document
> were migrated from the sibling `dynamic-lapis-gs` workspace. Measured results,
> PASS counts and hardware observations below describe that historical workspace;
> they do not establish a completed run in `MultiObj3dgsVstream`. Dataset, models,
> vendor binaries and generated `output/` artifacts are not bundled. Commands now
> use this project root and the pinned backend in `third_party/dynamic-lapis-gs`.
> See [migration/status](CONTENT_PREPARATION_MIGRATION.md) before reproducing a run.

Làm việc tại project root `MultiObj3dgsVstream/`. Hướng dẫn này dành cho người hoặc agent chạy pipeline offline theo từng stage sau migration. Đọc config và kiểm tra output sau mỗi bước; không tự mở rộng số frame/quality, đổi precision hay giảm training budget khi một bước lỗi.

## 0. Bắt đầu từ checkout mới

```bash
git clone --recursive https://github.com/QuaQuytNgot/MultiObj3dgsVstream.git
cd MultiObj3dgsVstream
conda activate Hoang
python -m pip install -r requirements.txt
python -m pip install -r requirements-content-preparation.txt
python tools/content_preparation/prepare_content.py \
  --config configs/content_prepare_smoke.yaml --dry-run
python tools/content_preparation/self_test.py
```

`requirements-content-preparation.txt` giữ dependencies offline riêng với transport.
PyTorch/CUDA và hai extensions cần khớp môi trường đang dùng; xem
[dependency note](DEPENDENCIES.md). Migration này không build CUDA, download dataset,
train hoặc chạy lại render/profile của Longdress. `self_test.py` là kiểm tra nhỏ;
config smoke production vẫn cần original CUDA renderer khi chạy cả pipeline.

Với checkout mới, bắt đầu từ smoke/dry-run, rồi acquisition/preprocessing cho một
output mới. Các mục completed/PASS bên dưới là kết quả lịch sử của workspace nguồn.
Chỉ các JSON summary gọn nằm trong Git; `output/` cũ, checkpoints, poses và binary
vendor chưa được chuyển. Config `existing` cần manifest và checkpoint nguồn thật
có thể đọc được trước khi chạy. Di chuyển workspace/backend làm đổi fingerprints:
không coi journal hoặc guarded resume của workspace cũ là resume hợp lệ trong cây
mới. Dùng output mới cho run migrated, hoặc giữ môi trường nguồn để audit lịch sử.

## 1. Phân biệt assets lịch sử và checkout hiện tại

| Asset / công việc | Trạng thái đã xác thực | Ý nghĩa |
|---|---|---|
| Raw Longdress 1051–1350 | Đủ 300 frame; inventory hoàn thành | 30 FPS, 10 s dữ liệu point cloud; chưa đồng nghĩa 10 s 3DGS đã train/encode |
| Checkpoint Q0/res8 và Q1/res4, 1051–1055 | Đã có; training provenance và hashes đã kiểm tra | Hai trained representations; Q1 cao nhất trong cặp này |
| f32 và f16 của hai quality trên | Encode/package/decode/profile/proxy/manifest đã chạy | Hai codec precision variants của cùng checkpoint ladder |
| Bốn trained quality res8/res4/res2/res1 | Hoàn thành trên 1051–1055; hai variants đã audit | Reuse Q0/Q1, train thêm Q2/Q3; mọi quality render native 1024×1024 |
| Năm trained quality | Chưa có experiment xác thực | Cần quyết định thêm một training operating point có phương pháp; không nhân đôi checkpoint hoặc đổi precision để đếm thành quality mới |

Nguồn evidence lịch sử: [summary bốn level](validation/longdress_4level_trial_summary.json), `output/content_prepare_longdress_4level/audit_f32.json`, `audit_f16.json`, `gt_diagnostic/report.json`, `viewer_validation.json` và `output/longdress_4level_training/manifest.json`. [Trial Longdress hai level](LONGDRESS_ENCODING_TRIAL.md) là báo cáo lịch sử riêng. Raw inventory nằm tại `output/datasets/8i/longdress/inventory.json`. Báo cáo của từng run là bằng chứng cuối cùng; tên folder hoặc số entry trong YAML không chứng minh quality đã được train.

## 2. Hiểu đúng preview mờ và “highest quality”

Preview ban đầu dùng original renderer ở **256×256**. Helper xuất PNG tối đa 256×256; viewer ban đầu đặt vùng ảnh rộng 400 px. Phóng PNG lên không bổ sung chi tiết. Camera distance và projected scale còn quyết định nhân vật chiếm bao nhiêu pixel trong ảnh.

Checkpoint trial ban đầu được train 6,000 iterations/frame đầu: Q0 từ **res8**, Q1 từ **res4**, tương ứng 128/256 px khi nguồn preprocessing là 1024 px. Q1 không phải model res1/native 1024 px. Chuyển từ Q0 sang Q1 cải thiện representation; chuyển f16 sang f32 chỉ thay precision của codec trên cùng checkpoint.

Trial f32, gồm cả bốn level mới, đã kiểm tra numeric state **checkpoint → export → decode bit-exact**. Vì vậy không thể quy mọi blur cho nén. Bốn level mới có checkpoint res1 thật; Q3 cho mean sampled matching-GT PSNR **33.85 dB**, Q1 **29.13 dB** trên cùng cameras/frames, với training budgets khác nhau được ghi bên dưới. Các con số sampled không chứng minh mọi view đều sắc nét hoặc đã hội tụ.

Kiểm tra theo thứ tự: chọn highest **trained** quality → f32 decoded → cùng camera/frame/scale → xem native render ở kích thước pixel thật → mới so với f16. Render native lớn hơn cần gọi lại original renderer, không resize ảnh 256 px. Nếu muốn thêm chi tiết model, phải có checkpoint higher quality đã được train từ input phù hợp.

## 3. Mở đúng môi trường và kiểm tra máy

```bash
cd MultiObj3dgsVstream
conda activate Hoang
python --version
nvidia-smi
python tools/content_preparation/self_test.py
```

Môi trường Hoang của experiment nguồn có PyTorch/CUDA extensions của backend.
Config migrated mặc định `runtime.extension_path: null`, dùng extensions đã cài
trong environment. Nếu dùng binary vendor đã xác minh, đặt `runtime.extension_path`
rõ ràng; checkout mới cần extensions đúng pinned source. Không lấy binary của
renderer khác chỉ vì package có cùng tên. Original Python modules đến từ
`third_party/dynamic-lapis-gs`; project helpers dùng `tools/content_preparation/paths.py`. Chỉ một process preparation dùng GPU; không mở nhiều object/quality/view jobs đồng thời. LPIPS mặc định CPU, batch 1. Hai config native bốn quality đặt `metrics.device: cuda` sau preflight trên GTX 1660: VGG 1024 px batch 1 vừa VRAM; Gaussian renderer và LPIPS GPU chạy lần lượt, criterion được trả về CPU sau từng batch. Không đổi device giữa một run mà bỏ qua fingerprint checks.

Encode, package và decode baseline dùng NumPy/CPU và disk, không chịu giới hạn VRAM theo cách training/render chịu. Higher quality vẫn tăng RAM, dung lượng checkpoint/payload và thời gian CPU. Training res1, foundation merge/densification và render độ phân giải lớn có thể vượt **6 GB** dù encode cùng state thành công.

## 4. Chọn một run và đọc config trước khi chạy

Run chuẩn cho checkout mới là smoke/dry-run rồi raw preparation với output mới.
Các config import bên dưới reproduce trial lịch sử **sau khi đã cung cấp inputs**;
trạng thái complete/audit trong bảng chỉ mô tả workspace nguồn:

| Config có thật | Input / training backend | Output |
|---|---|---|
| `configs/content_prepare_longdress_trial_f32.yaml` | Historical `checkpoints` / `existing`; 1051–1055, Q0/Q1 | `output/content_prepare_longdress_trial/f32_lossless/` |
| `configs/content_prepare_longdress_trial_f16.yaml` | Historical cùng checkpoint, encoding f16 | `output/content_prepare_longdress_trial/f16/` |
| `configs/content_prepare_longdress_4level_f32.yaml` | `checkpoints` / `existing`; 1051–1055, Q0/Q1/Q2/Q3, cần manifest verified | `output/content_prepare_longdress_4level/f32_lossless/`; complete, audit passed |
| `configs/content_prepare_longdress_4level_f16.yaml` | Cùng manifest bốn quality, encoding f16 | `output/content_prepare_longdress_4level/f16/`; complete, audit passed |
| `configs/content_prepare_longdress.yaml` | `raw` / `upstream`; 1051–1080, bốn quality 8/4/2/1 | `output/content_prepare_longdress/`; full 30-frame training chưa chạy |

Trong các lệnh stage dưới đây, dùng một biến duy nhất. Ví dụ import bốn quality
chỉ chạy sau khi `output/longdress_4level_training/manifest.json` và checkpoint paths
trong manifest tồn tại. Nếu chưa có inputs, dùng smoke config hoặc raw config và
thực hiện acquisition/preprocessing trước:

```bash
CP_CONFIG=configs/content_prepare_longdress_4level_f32.yaml
python tools/content_preparation/prepare_content.py --config "$CP_CONFIG" --dry-run
```

Kiểm tra plan: object `longdress`, 5 frames, qualities Q0/Q1/Q2/Q3, backend `existing`, renderer 1024×1024, output `output/content_prepare_longdress_4level/f32_lossless`. Nếu plan khác dự định, sửa/lựa chọn config trước khi chạy. Không dùng config 30-frame `upstream` khi chỉ muốn encode checkpoint trial.

`--stage STAGE` chạy **chỉ stage đó**, không tự chạy prerequisites. `--stage all` chạy đủ chín stage. Không sửa config/code giữa một run rồi kỳ vọng mọi task cũ tiếp tục skip.

## 5. Chuẩn bị raw hoặc kiểm tra checkpoint nguồn

Với trial `existing`, raw đã tải không bị train lại. Nguồn thực sự của bốn-level state là `output/longdress_4level_training/manifest.json`, trong đó có frame → quality → checkpoint path, `commands[].argv` và SHA-256 gốc. `output/progressive_gap_real/manifest.json` là nguồn Q0/Q1 lịch sử được wrapper reuse. Progressive import đòi hỏi foundation/temporal lineage đã xác thực; tự tính một hash mới cho PLY không tạo được bằng chứng training lineage.

Với run `raw`, cần PLY đủ frame tại template cấu hình và poses train/test hợp lệ. `prepare_content.py` không tự tải dataset. Kiểm tra nhanh inventory đã có:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('output/datasets/8i/longdress/inventory.json')
d = json.loads(p.read_text())
print(d['object'], len(d['frames']), d['complete_for_selection'], d['capture']['fps'])
assert d['object'] == 'longdress'
assert d['complete_for_selection'] is True
assert len(d['frames']) == 300
PY
```

Reproduce/resume downloader, nếu thực sự cần acquisition/verification lại:

```bash
python tools/download_8i_object.py --object longdress --all-frames \
  --output output/datasets/8i --cache output/progressive_gap_real/raw \
  --allow-http-fallback --download-workers 4
```

Command trên giữ setting transport của trial: HTTP fallback được cho phép rõ ràng khi official HTTPS certificate verification lỗi; inventory ghi transport. Download workers chỉ là network/CPU. Lệnh resume xác minh lại raw files, có thể mất thời gian đọc khoảng 5.69 GB; không cần lặp acquisition cho mỗi encoding variant.

## 6. Chạy preprocess rồi kiểm tra đầu vào

```bash
python tools/content_preparation/prepare_content.py \
  --config "$CP_CONFIG" --stage preprocess --resume
```

Kiểm tra `<output>/longdress/preprocess.json`:

- `checkpoints`: frame IDs và source manifest hash được ghi; không generate lại training images.
- `prepared`: source images được copy vào private output; có `transforms_train.json`, `transforms_test.json`, initialization PLY.
- `raw`: dùng upstream preprocessing, tạo `longdress/input/<frame>/res<scale>/`, camera JSON và canonical metadata. Log ở `longdress/logs/preprocess_<frame>.log`.

Không tiếp tục nếu thiếu frames, camera transforms sai hoặc canonical/up-axis không nhất quán. Longdress trial dùng `renderer.up_axis: z`; không tự đổi thành Y-up để chữa preview.

## 7. Chạy train hoặc import; xác minh số quality thực

```bash
python tools/content_preparation/prepare_content.py \
  --config "$CP_CONFIG" --stage train --resume
```

Với backend `existing`, stage này chỉ import/verify checkpoint; `training.json` sẽ có `backend: existing`, frame map, command provenance và lineage. Không gọi nó là một training experiment mới.

Với backend `upstream`, wrapper gọi nguyên `train.py`. Thứ tự là một quality → frame đầu và các frame tiếp theo, rồi quality kế. Higher quality frame đầu dùng foundation checkpoint của quality trước; các frame sau dùng checkpoint quality đó ở frame trước. Không thay loss/training strategy hoặc tự bỏ foundation để giảm memory. Training logs nằm tại `longdress/logs/train_<quality>_<frame>.log`; checkpoint mới tại `longdress/models/<quality>/<frame>/point_cloud/iteration_<steps>/point_cloud.ply`.

Sau bước này phải có **mỗi configured quality ở mọi selected frame**, hashes tương ứng và `lineage_verified` khi progressive. Nếu thiếu Q2/Q3, không tiếp tục export bằng cách trỏ chúng vào Q1.

Trong source upstream hiện tại, `training_report` ở frame dùng foundation render `scene.gaussians` cũ, còn optimization/saving dùng model merged `gaussians`. Vì vậy PSNR in từ nhánh report này có thể không đánh giá checkpoint merged thực sự. Không dùng con số đó để kết luận Q2/Q3 hỏng hoặc đã hội tụ. Kiểm tra saved/decoded state với original renderer; nếu cần GT PSNR, đánh giá riêng trên matching cameras/GT. Giữ source trainer nguyên bản; không sửa training để chữa một nhánh logging trong lượt preparation này.

## 8. Export canonical Gaussian states

```bash
python tools/content_preparation/prepare_content.py \
  --config "$CP_CONFIG" --stage export --resume
```

Output: `longdress/export.json` và `longdress/qualities/<quality>/<frame>.npz`. State gồm xyz, raw wxyz rotation, log-scale, logit-opacity, SH và IDs. Export giữ checkpoint SHA-256; progressive sử dụng stable IDs đã được verify. Đếm số state bằng `number of frames × number of qualities`, không nhân thêm delivery modes ở bước export.

## 9. Encode, package, decode theo thứ tự

```bash
python tools/content_preparation/prepare_content.py \
  --config "$CP_CONFIG" --stage encode --resume
python tools/content_preparation/prepare_content.py \
  --config "$CP_CONFIG" --stage package --resume
python tools/content_preparation/prepare_content.py \
  --config "$CP_CONFIG" --stage decode --resume
```

| Stage | Output chính | Kiểm tra |
|---|---|---|
| encode | `longdress/encoding.json`, `encoded/<mode>/<layer>/<frame>.cpgs` | Codec name/version/config, actual bytes/SHA, input/decoded/parent state hashes |
| package | `longdress/package.json`, `package_index.json`, `media/<mode>/<layer>/gof_*/segment_*.cpseg` | Actual requestable containers, member offsets/hashes, frame/time range, access points, dependencies |
| decode | `longdress/decoding.json`, `decoded/<mode>/<layer>/<frame>.npz` | Reconstruction từ **segment bytes**, decoded hash khớp encoder, parent closure đúng |

Codec hiện tại: `gaussian_attribute_zlib` 1.0.0, CPGAUS01 payload và CPSEG1 segment container. Đây không phải exact LTS enhanced-Draco bitstream. Với f32, mục tiêu là numeric checkpoint state lossless; f16/q* là lossy và cần profile/codec comparison tương ứng.

Independent Qk tự chứa state cùng frame. Progressive Qk cần Base + E1 + … + Ek; refinement có thể thay shared/static/dynamic attributes, thêm/xóa IDs. Không giả định E chỉ chứa Gaussians mới. GoF/reset, configured layer refresh và network segment là ba khái niệm riêng; baseline không có temporal prediction, nên mỗi frame là codec access point với same-frame layer closure.

## 10. Profile decoded state, tạo proxy, xuất manifest

```bash
python tools/content_preparation/prepare_content.py \
  --config "$CP_CONFIG" --stage profile --resume
python tools/content_preparation/prepare_content.py \
  --config "$CP_CONFIG" --stage proxy --resume
python tools/content_preparation/prepare_content.py \
  --config "$CP_CONFIG" --stage manifest --resume
```

Reference là highest-quality **decoded** state của run. `profiles/profile.json` giữ từng view/scale/time/quality/mode; `gains.json` chứa measured transition gains; `aggregates.json` là tiện ích báo cáo. MSE là distortion mặc định; PSNR là transform của MSE, LPIPS là reporting/perceptual metric riêng.

Highest quality so với chính reference của nó phải cho MSE/LPIPS≈0; đây là sanity check, không phải bằng chứng chất lượng so với GT hoặc convergence. Matching-GT diagnostic của run bốn level đã hoàn thành riêng với 16 comparisons; xem mục 12.1.

Proxy dùng frozen Gaussian subset từ highest decoded representation, cố định IDs theo thời gian và quality-independent đối với ABR. Output `proxy/index.json`, frame states và `proxy/validation.json`; không gọi proxy là final display quality.

Kiểm tra `longdress/manifest.json`, `longdress/validation.json`, server `<output>/manifest.json`, `<output>/validation_upstream.json`, `.preparation/journal.json`. `passed`, state hashes, profile row count, frozen proxy IDs và `upstream_unchanged` phải phù hợp run. Đừng chỉ dựa vào exit code hoặc thấy có file manifest.

## 11. Audit trial, tạo báo cáo và mở local viewer

Với trial f32/Q0–Q3 đã có:

```bash
python tools/validate_8i_trial.py \
  --root output/content_prepare_longdress_4level/f32_lossless/longdress \
  --source-manifest output/longdress_4level_training/manifest.json --precision f32 \
  --report output/content_prepare_longdress_4level/audit_f32.json
python tools/content_preparation/run_report.py \
  output/content_prepare_longdress_4level/f32_lossless \
  --fps 30 --variant-id f32_lossless
python tools/summarize_8i_trial.py \
  --trial-root output/content_prepare_longdress_4level \
  --inventory output/datasets/8i/longdress/inventory.json --preview-max-size native
python -m http.server 8765 --bind 127.0.0.1 \
  --directory output/content_prepare_longdress_4level
```

Mở [viewer tại Q3](http://127.0.0.1:8765/?quality=Q3); bật PNG tỉ lệ 1:1 hoặc mở PNG đầy đủ. Summarizer mặc định yêu cầu hai variant folders `f32_lossless` và `f16` đã hoàn thành. Muốn catalog một variant dùng `--variants f32_lossless`. Web đọc metadata và PNG caches, không render 3DGS trực tiếp, không chạy training/GPU hoặc continuous video. Native previews của lượt bốn level có 1024×1024 pixels; không dùng thumbnail upscaling làm bằng chứng model sắc hơn.

Phân biệt **raw sequence 10 s**, **encoded 5-frame clip 0.166666667 s**, timestamp span **0.133333333 s**, và **encode-task wall seconds**. Report layer bytes khác cumulative quality bytes; cold access tính cả headers và complete segments phải request. Encoder task timing gồm CPU load/encode/verification decode/cache write; latest stage wall time sau resume có thể chỉ là hash checks/skip.

## 12. Chuẩn bị bốn hoặc năm trained quality

Ladder bốn mức theo upstream scale convention:

| Nominal quality | Training/preprocessing scale | Pixel resolution nếu nguồn 1024 px | Progressive layer |
|---|---:|---:|---|
| Q0 | 8 | 128×128 | Base |
| Q1 | 4 | 256×256 | E1 |
| Q2 | 2 | 512×512 | E2 |
| Q3 | 1 | 1024×1024 | E3 |

`resolution_scale: 1` chỉ có ý nghĩa khi checkpoint thật sự được train từ input res1. Đổi nhãn một checkpoint res4 thành Q3 không cải thiện state. Nếu chỉ có Q0/Q1, phải train tiếp Q2/Q3 bằng đúng foundation lineage, hoặc import một manifest bốn quality đã xác thực. Config bốn quality cần có đủ qualities, input sources, refresh map Base/E1/E2/E3 hoặc aliases Q0/Q1/Q2/Q3, view samples và output riêng. Phải chạy dry-run trước.

Config cho lượt bốn quality đã được chọn: `configs/content_prepare_longdress_4level_f32.yaml` và `_f16.yaml`; mỗi config cần **manifest mới** `output/longdress_4level_training/manifest.json`. Hai config này dùng backend `existing`, vì wrapper riêng tạo ladder trước; stage `train` trong preparation sau đó chỉ import/verify cả bốn quality.

Trước khi dùng wrapper mới, kiểm tra nguồn res1 và poses, không giả sử transforms JSON đã nằm trong các folder res1 legacy:

```bash
python - <<'PY'
import json
from pathlib import Path
from PIL import Image
poses = Path('output/progressive_gap_real/poses')
counts = {split: len(json.loads((poses / f'transforms_{split}.json').read_text())['frames'])
          for split in ('train', 'test')}
for frame in range(1051, 1056):
    source = Path(f'output/progressive_gap_real/source/8i/longdress/longdress_res1/{frame}')
    for split, count in counts.items():
        for index in range(count):
            assert (source / split / f'r_{index}.png').is_file(), (frame, split, index)
        with Image.open(source / split / 'r_0.png') as image:
            assert image.size == (1024, 1024), (frame, split, image.size)
    print(frame, 'native res1 images và poses đủ', counts)
PY
```

Dùng entry point `tools/extend_quality_ladder_guard.py` để đóng băng source, runtime và CUDA-extension hashes trong sidecar cạnh work root trước khi gọi wrapper `tools/extend_quality_ladder.py`. Guard từ chối resume nếu môi trường thay đổi; giữ nguyên fingerprints của wrapper. Cả hai hỗ trợ CLI `prepare|train|all`, dry-run và resume. Nó reuse Q0/Q1 verified checkpoint, tạo private prepared sources res2/res1 và gọi original `train.py` để train Q2/Q3: **10,000 iterations frame đầu, 1,500 iterations từng frame sau**, 1051–1055, CPU images, `-r 1` để giữ pixel resolution input. Q0/Q1 cũ giữ budget **6,000/1,500**; đây là mixed-provenance quality ladder, không phải controlled experiment dùng cùng số iterations cho mọi quality. Không tuyên bố đạt convergence của budget 30,000 iterations.

Work lịch sử đã được guard xác thực trong workspace nguồn; sidecar ở đó là
`output/.longdress_4level_training.quality-ladder-environment.json`. Sidecar không
được chuyển và không được tự adopt sau thay đổi paths/source/runtime. Các command
resume dưới đây chỉ dành cho một work tree đã được tạo/xác thực trong checkout
hiện tại. Với run migrated mới, dùng guarded entry point ngay từ đầu và work/output
riêng; bỏ `--resume` ở lần đầu khi chưa có journal.

Thứ tự lệnh cụ thể cho bounded extension mới:

```bash
python tools/extend_quality_ladder_guard.py \
  --config configs/content_prepare_longdress_4level_f32.yaml \
  --source-manifest output/progressive_gap_real/manifest.json \
  --work-root output/longdress_4level_training \
  --fullres-template 'output/progressive_gap_real/source/8i/longdress/longdress_res1/{frame}' \
  --poses-root output/progressive_gap_real/poses --stage all --dry-run

python tools/extend_quality_ladder_guard.py \
  --config configs/content_prepare_longdress_4level_f32.yaml \
  --source-manifest output/progressive_gap_real/manifest.json \
  --work-root output/longdress_4level_training \
  --fullres-template 'output/progressive_gap_real/source/8i/longdress/longdress_res1/{frame}' \
  --poses-root output/progressive_gap_real/poses --stage prepare --resume

python tools/extend_quality_ladder_guard.py \
  --config configs/content_prepare_longdress_4level_f32.yaml \
  --source-manifest output/progressive_gap_real/manifest.json \
  --work-root output/longdress_4level_training \
  --fullres-template 'output/progressive_gap_real/source/8i/longdress/longdress_res1/{frame}' \
  --poses-root output/progressive_gap_real/poses --stage train --resume
```

`prepare` không train; nó tạo private source dữ liệu phù hợp với quality mới. `train` gọi GPU tuần tự và cuối cùng xác minh lineage trước khi commit atomic `output/longdress_4level_training/manifest.json`. Work journal nằm tại `.preparation/journal.json` dưới work root; logs/checkpoints nằm trong work tree. Nếu job chưa xong hoặc manifest chưa có, chưa bắt đầu preparation hai variant dựa trên manifest đó.

Precheck manifest sau extension:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('output/longdress_4level_training/manifest.json')
d = json.loads(p.read_text())
qualities = ['Q0', 'Q1', 'Q2', 'Q3']
assert [r['frame'] for r in d['frames']] == list(range(1051, 1056))
assert all(Path(r[q]).is_file() for r in d['frames'] for q in qualities)
commands = {(c['frame'], c['level']): c for c in d['commands']}
assert len(commands) == 20
assert all(commands[r['frame'], q]['argv'] and len(commands[r['frame'], q]['sha256']) == 64
           for r in d['frames'] for q in qualities)
print('20 checkpoint mappings và original command/hash entries đủ')
PY
```

Sau khi wrapper hoàn thành và verified manifest tồn tại, chạy preparation bằng đầy đủ các lệnh dưới đây. Mỗi lệnh phải thành công và output phải đạt kiểm tra tương ứng trong các bước 6–10 trước khi chạy lệnh tiếp theo:

```bash
CP_CONFIG=configs/content_prepare_longdress_4level_f32.yaml
python tools/content_preparation/prepare_content.py --config "$CP_CONFIG" --dry-run
python tools/content_preparation/prepare_content.py --config "$CP_CONFIG" --stage preprocess --resume
python tools/content_preparation/prepare_content.py --config "$CP_CONFIG" --stage train --resume
python tools/content_preparation/prepare_content.py --config "$CP_CONFIG" --stage export --resume
python tools/content_preparation/prepare_content.py --config "$CP_CONFIG" --stage encode --resume
python tools/content_preparation/prepare_content.py --config "$CP_CONFIG" --stage package --resume
python tools/content_preparation/prepare_content.py --config "$CP_CONFIG" --stage decode --resume
python tools/content_preparation/prepare_content.py --config "$CP_CONFIG" --stage profile --resume
python tools/content_preparation/prepare_content.py --config "$CP_CONFIG" --stage proxy --resume
python tools/content_preparation/prepare_content.py --config "$CP_CONFIG" --stage manifest --resume
```

Plan phải có **5 frame, Q0/Q1/Q2/Q3, renderer 1024×1024 trong config**, input manifest mới, output `output/content_prepare_longdress_4level/f32_lossless/`. Kiểm tra 20 checkpoint entries trước khi import. Final audit lịch sử của **mỗi variant** đã xác nhận 20 exported states, 40 decoded states cho hai modes, 144 profile rows, 32 segments và frozen proxy 256 Gaussians × 5 frames. Full preparation resume đã chạy 0 tasks và skip 460/460 ở mỗi variant; extension resume chạy 0 tasks, skip 32/32.

Tổng stored media của mọi quality/layer và cả hai modes là **856,942,065 bytes f32**, **449,653,951 bytes f16**, gồm headers. **90 self-tests PASS**. Renderer/LPIPS guard ghi 306 operations mỗi variant (162 renders + 144 LPIPS batches), `peak_active=1`; peak device memory quan sát khi sample mỗi 2 s là **2,005 MiB**. Đây là measured bounded trial, không bảo đảm full sequence fit 6 GB. [Summary bốn level](validation/longdress_4level_trial_summary.json) giữ evidence gọn trong Git.

Audit numeric state, source hashes, packaged payload, decoded-parent refinements và rendered cache của f32 trước khi làm f16:

```bash
python tools/validate_8i_trial.py \
  --root output/content_prepare_longdress_4level/f32_lossless/longdress \
  --source-manifest output/longdress_4level_training/manifest.json --precision f32 \
  --report output/content_prepare_longdress_4level/audit_f32.json
```

Sau f32 đã audit, dùng `CP_CONFIG=configs/content_prepare_longdress_4level_f16.yaml` với cùng chín lệnh stage trên cho variant f16. Hai variants không train Q2/Q3 hai lần. Audit f16 bằng đúng manifest mới:

```bash
python tools/validate_8i_trial.py \
  --root output/content_prepare_longdress_4level/f16/longdress \
  --source-manifest output/longdress_4level_training/manifest.json --precision f16 \
  --report output/content_prepare_longdress_4level/audit_f16.json
```

Sau mỗi variant hoàn thành, tạo báo cáo CPU:

```bash
python tools/content_preparation/run_report.py \
  output/content_prepare_longdress_4level/f32_lossless --fps 30 --variant-id f32_lossless
python tools/content_preparation/run_report.py \
  output/content_prepare_longdress_4level/f16 --fps 30 --variant-id f16
```

Khi cả hai variant đã có manifest/profile/proxy, tạo catalog trong **root mới** và mở viewer:

```bash
python tools/summarize_8i_trial.py \
  --trial-root output/content_prepare_longdress_4level \
  --inventory output/datasets/8i/longdress/inventory.json --preview-max-size native
python -m http.server 8765 --bind 127.0.0.1 \
  --directory output/content_prepare_longdress_4level
```

Dừng HTTP server cũ đang giữ port 8765 trước khi khởi động server root mới; giữ `output/content_prepare_longdress_trial/` làm historical two-level output. Kiểm tra ảnh source/native PNG dimensions trong catalog: render 1024 px không có nghĩa thumbnail 256 px tự trở thành native; native preview phải được export/serve từ đúng cache 1024 px.

Config 30-frame `content_prepare_longdress.yaml` cũng đã cập nhật bốn quality; nó dùng raw/upstream và training budget riêng, chưa chạy. Nó không phải đường tắt thay wrapper reuse năm frame. Rendering settings là một phần config khác với training resolution: run mới bốn quality profile **mọi Q ở 1024×1024** để so sánh trên cùng điều kiện; không cho Q0 render nhỏ rồi dùng interpolation để so với Q3.

Để có **năm** quality thực sự, cần chọn thêm một training operating point theo phương pháp đã định nghĩa, train/import checkpoint riêng và đo profile. Một lựa chọn resolution tự nhiên là **16/8/4/2/1** trên nguồn 1024 px, tương ứng 64/128/256/512/1024 px; nó cần một lowest-quality model mới và một progressive lineage phù hợp với ladder mới. Không thể chỉ chèn model res16 rồi đổi tên prefix của các checkpoint res8/res4 cũ. Lựa chọn này chưa chạy; run hiện tại ưu tiên bốn native levels đã định nghĩa. Thêm Q4 dùng cùng Q3 checkpoint, upscale nguồn lên 2048 px hoặc đếm f32/f16 thành hai training levels không chứng minh có thêm model detail hay trained representation độc lập.

Không bắt đầu 300-frame × 4/5-level run chỉ vì đã có raw 300 frames. Chạy bounded subset đã được chọn, kiểm tra correctness/memory/provenance, rồi mới mở rộng phạm vi bằng config/output riêng. Tăng training resolution có thể tăng peak GPU memory; encoder CPU không bảo đảm training res1 vừa 6 GB.

### 12.1. Đánh giá decoded state với matching GT

Sau khi variant f32 hoàn thành, dừng các GPU jobs khác rồi chạy diagnostic riêng trên hai frame và hai test cameras. Tool dựng camera theo đúng `transforms_test.json`, render decoded independent state ở 1024×1024 và so với ảnh GT cùng camera/frame; không resize GT hoặc thay reference của adaptation profile.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tools/evaluate_decoded_gt.py \
  --object-root output/content_prepare_longdress_4level/f32_lossless/longdress \
  --source-template 'output/progressive_gap_real/source/8i/longdress/longdress_res1/{frame}' \
  --poses output/progressive_gap_real/poses/transforms_test.json \
  --frames 1051 1055 --views 5 10 \
  --output output/content_prepare_longdress_4level/gt_diagnostic --resume
```

Diagnostic đã hoàn thành **16 comparisons**, mọi MSE/PSNR/LPIPS finite, max camera world-view error **8.22×10⁻⁷**. Resume chạy 0 tasks, skip 16/16 và giữ pretrained LPIPS method metadata. Mean của bốn comparisons mỗi quality trong `gt_diagnostic/report.json`:

| Quality | Mean MSE | Mean PSNR (dB) | Mean LPIPS |
|---|---:|---:|---:|
| Q0 | 0.00260485 | 25.84 | 0.068016 |
| Q1 | 0.00122150 | 29.13 | 0.053845 |
| Q2 | 0.00051612 | 32.91 | 0.041993 |
| Q3 | 0.00042250 | 33.85 | 0.038619 |

MSE đo full-image RGB trên background đen; PSNR là mean của từng comparison, không phải transform của mean MSE. LPIPS dùng pretrained VGG trên CUDA, batch 1. Đây là diagnostic được lấy mẫu, không phải evaluation toàn sequence hoặc bằng chứng đã hội tụ; budgets Q0/Q1 khác Q2/Q3. RGB PNG và sample metrics nằm cạnh report.

### 12.2. Kiểm tra browser tự động, tùy chọn

Tạo catalog native bằng command phía trên trước. Playwright chỉ phục vụ verification web, không phải dependency của preparation. Nếu môi trường Python đang dùng chưa có package/browser, cài riêng:

```bash
python -m pip install playwright
python -m playwright install chromium
```

Mở HTTP server ở terminal riêng; port 8765 phải đang phục vụ **root bốn level mới**:

```bash
python -m http.server 8765 --bind 127.0.0.1 \
  --directory output/content_prepare_longdress_4level
```

Trong terminal khác, cùng môi trường Python, chạy Chromium CPU-only:

```bash
CUDA_VISIBLE_DEVICES='' python tools/validate_trial_viewer.py \
  --url 'http://127.0.0.1:8765/?quality=Q3' \
  --output output/content_prepare_longdress_4level/viewer_validation.json
```

Validator trong experiment nguồn đã kiểm tra đủ **288/288 conditions** với `status: passed`, không có lỗi JS/HTTP: 2 variants × 2 modes × 4 qualities × 3 frames × 3 views × 2 scales. Kiểm tra gồm đúng URL PNG, native dimensions, training labels, metrics của view, fit không upscale, zoom 1:1 có scroll và mobile không tràn trang. JSON ghi catalog SHA-256 và từng condition; screenshot desktop/mobile nằm cạnh report. Command trên reproduce verification của catalog đang serve, không chạy renderer/GPU.

## 13. Resume, lỗi và những việc không được làm tự động

Chạy lại **cùng command/config/output** với `--resume` để skip task complete có config/input/output hashes khớp. Crash ở một view/task mới sẽ retry task đó; các task hoàn thành trước đó được giữ. Journal ghi status, fingerprint, timestamps, command/version và output hashes.

Nếu báo artifact/config/input/runtime changed, kiểm tra nguyên nhân. Chọn output mới cho experiment mới. `--overwrite` chỉ dùng khi chủ đích recompute artifacts pipeline-owned đã xác định; nó không cấp phép xóa input nguồn và không cần dùng cho resume hợp lệ. Đổi code trong `tools/content_preparation/` thay implementation fingerprint; scripts report/viewer bên ngoài package không làm điều đó.

Nếu thiếu prerequisites: chạy stage còn thiếu theo thứ tự 6–10. Nếu imported lineage sai: cung cấp argv/hash gốc và đúng checkpoint chain; không sửa manifest để “pass”. Nếu OOM: giữ log/stage/config, rồi thay memory setting hoặc render resolution **rõ ràng trong một run riêng**; không silently giảm iterations, số quality, batch hoặc tắt LPIPS để báo thành công.

Hướng dẫn không triển khai LoD/ABR selection, viewport/bandwidth prediction, runtime scheduler, HTTP client, DRL hoặc full DASH. Artifact server-side có thể dùng cho các phase đó sau khi content preparation đã được validate.

# StreamSTGS multi-object client benchmark

Ngày chạy: **2026-10-23**. Máy thử: **GTX 1660 6 GiB**, CUDA/EGL headless. Trial dùng StreamSTGS Soldier GOP 0–4, nguồn frames 536–835, 30 FPS, 60 frames/GOP, 148.996 Gaussians/GOP và CRF 20/28/36/40.

**Cập nhật 2026-10-24:** runtime native mới đã được profile và tối ưu bằng progressive temporal-window readiness, bounded asynchronous upload và xử lý FAST JIT pauses. Xem [startup/GOP optimization](STREAMSTGS_STARTUP_GOP_OPTIMIZATION.md) để lấy kết quả trước/sau trên cùng workload, source snapshots và regression evidence. Các bảng dưới giữ lại phép đo prototype cũ; phần kết luận current+next chưa realtime mô tả prototype đó. Không so trực tiếp FPS giữa hai báo cáo vì contents, camera, resolution và môi trường Torch khác nhau. Lỗi atlas layout của trial cũ và reference thay thế được giải thích trong [rendering audit](NATIVE_VIEWER_FIX.md).

## Kết quả chính

Ở warm state, tối đa **2 Soldier instances nhân bản** đạt trung bình trên 30 scene frames/s: 32,96 FPS ở 512p, 32,56 FPS ở 720p và 32,15 FPS ở 1080p theo đường tái dựng tuần tự rồi render chung. N=3 chỉ đạt 22,24 FPS tại 512p; N=4 đạt 16,78 FPS.

Đã xác nhận N=1 và N=2 phát đủ **300/300 deadline ở 30 Hz** sau khi preload cả 5 GOP/object và warm tất cả tổ hợp GOP mà timeline lệch pha tạo ra. N=2 có p95 30,51 ms, startup load+warm 20,13 s và peak device memory 5.328 MiB/6.144 MiB. Kết quả đó giữ 10 GOP models trên GPU suốt video. Các bounded policies current-only và current+next chưa đạt deadline: current+next N=1 trễ 35,7%, N=2 trễ 95,3%; preload-all N=4 OOM. Do đó **2 objects phát liên tục đã được chứng minh khi preload toàn bộ representation**, nhưng chưa đạt với buffer chỉ current+next trên GTX 1660.

“Objects” ở đây là các bản sao cùng representation Soldier. Mỗi bản có decoder/model/feature buffer riêng, object ID, world transform và animation phase riêng; compressed payload và Gaussian geometry giống nhau. Đây là phép đo replicated workload, chưa phải benchmark nhiều nội dung độc lập. `cut_roasted_beef` không được tính là human object vì đó là dynamic scene với convention/hình học khác.

## Scene composition và kiểm tra correctness

Mỗi timestamp tái dựng Gaussian state cho từng instance, đổi tọa độ object-local sang world-space, nối attributes rồi gọi **một FAST raster call** cho một framebuffer. Không ghép ảnh 2D. Camera position được đổi ngược về object-local trước khi đi vào các neural head phụ thuộc view; rotation dùng convention scalar-first WXYZ của StreamSTGS.

Trial đã so render identity N=1 với đường StreamSTGS gốc ở ba thời điểm/góc; sai số ảnh tối đa 1/255 do FAST dùng render target uint8. Batched state của cùng model có sai số tối đa 0 trong các attributes được so sánh. Với N=4, ba yaw camera tạo ảnh có lần lượt 39.480, 35.174 và 43.332 pixel khác nền. Phép thử che khuất có 8.548 pixel giao nhau; object trước chiếm màu 96,0% vùng giao nhau trong diagnostic opaque. Phép thử opaque chỉ cô lập depth sorting; các scene benchmark bình thường dùng opacity được StreamSTGS tái dựng.

![Bốn Soldier trong một scene joint render](../output/codec_trials/streamstgs_multi_object_client/artifacts/layout_four_objects_front.png)

Ảnh orbit: [góc trái](../output/codec_trials/streamstgs_multi_object_client/artifacts/layout_four_objects_orbit_left.png), [góc phải](../output/codec_trials/streamstgs_multi_object_client/artifacts/layout_four_objects_orbit_right.png), [hai object cách xa](../output/codec_trials/streamstgs_multi_object_client/artifacts/layout_two_objects_separated.png), [depth/occlusion diagnostic](../output/codec_trials/streamstgs_multi_object_client/artifacts/occlusion_two_objects_front_blue_rear_red.png). Kết quả máy đọc được ở [scene_composition_validation.json](../output/codec_trials/streamstgs_multi_object_client/metrics/scene_composition_validation.json).

## Phương pháp đo

FAST chạy CUDA/OpenGL 4.6 qua EGL, không cần desktop hoặc X11. Cold GOP load gọi decoder thật: software HEVC decode 64 feature atlases, JXL/canonical reconstruction, feature tile unpack/normalize/upload và nạp MLP. Model/GOP ban đầu của từng object được nạp tuần tự; cold startup matrix vì thế là phép đo tuần tự, còn current+next playback dùng loader nền.

Warm matrix giữ model/features trên GPU, warm-up renderer, đồng bộ CUDA trước/sau mỗi sample; chạy 3×60 scene timestamps ở 512p, 2×45 ở 720p và 2×30 ở 1080p. Training, metric evaluation, ảnh lưu và encode ảnh không nằm trong vòng đo. FPS tính theo **scene frames hoàn chỉnh / giây**, không nhân N. Scene latency p50/p95 là tái dựng cộng nối state và một joint draw. Joint draw timer bao gồm `torch.cat` cần để gom Gaussian arrays; đây là số conservative cho đường scene thực tế. Warm host-to-device model transfer là 0 byte/frame vì model resident trên CUDA; cold upload được tính trong stage load nhưng số byte copy riêng không được đo.

Threaded reconstruction dùng `ThreadPoolExecutor` (tối đa 4 workers), mỗi object gọi state path độc lập rồi nối và render một ảnh chung; có ba probe có đồng bộ CUDA. Đường MLP batching chỉ bật sau khi kiểm tra weights của các bản Soldier giống nhau. GPU memory gồm PyTorch allocated/reserved và `nvidia-smi` lấy mẫu mỗi 250 ms; RAM là process peak RSS. Device-wide sample có thể chứa memory của driver/display và bỏ sót spike ngắn.

## Warm decode/reconstruction/render scaling

Latency là p50/p95 của serial state reconstruction cộng joint composition/render. N=8 không hoàn tất do CUDA OOM trong lúc giữ các model riêng biệt trên GPU; số Gaussian kỳ vọng nếu đủ bộ nhớ là 1.191.968.

| Resolution | N | Gaussians/scene | Scene latency p50/p95 (ms) | FPS serial / batched | FPS threaded + joint | Joint draw FPS | VRAM allocated / device peak (MiB) | CPU RSS peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512×512 | 1 | 148.996 | 15,64 / 15,94 | 63,76 / 59,83 | 62,51 | 173,5 | 746 / 1.778 | 2.798 |
| 512×512 | 2 | 297.992 | 30,31 / 30,56 | 32,96 / 30,64 | 32,55 | 94,1 | 1.456 / 2.764 | 2.798 |
| 512×512 | 3 | 446.988 | 44,93 / 45,20 | 22,24 / 20,60 | 21,96 | 64,8 | 2.181 / 3.758 | 2.869 |
| 512×512 | 4 | 595.984 | 59,55 / 59,77 | 16,78 / 15,50 | 16,21 | 49,4 | 2.903 / 4.644 | 3.808 |
| 512×512 | 6 | 893.976 | 89,10 / 89,41 | 11,22 / 10,42 | OOM ở threaded probe | 33,3 | 3.670 / 5.745 | 4.658 |
| 512×512 | 8 | 1.191.968 | — | CUDA/cuBLAS initialization OOM | — | — | — | — |
| 1280×720 | 1 | 148.996 | 15,89 / 16,22 | 62,72 / 59,36 | — | 166,2 | 806 / 1.971 | 3.032 |
| 1280×720 | 2 | 297.992 | 30,66 / 31,10 | 32,56 / 30,33 | — | 91,2 | 1.522 / 2.957 | 3.045 |
| 1280×720 | 3 | 446.988 | 45,16 / 45,37 | 22,12 / 20,52 | — | 63,6 | 2.143 / 3.767 | 3.377 |
| 1280×720 | 4 | 595.984 | 59,94 / 60,35 | 16,67 / 15,41 | — | 48,6 | 3.020 / 4.840 | 3.837 |
| 1280×720 | 6 | 893.976 | 89,20 / 89,54 | 11,20 / 10,39 | OOM ở threaded probe | 32,9 | 3.674 / 5.714 | 4.659 |
| 1280×720 | 8 | 1.191.968 | — | OOM khi nạp models | — | — | — | — |
| 1920×1080 | 1 | 148.996 | 16,28 / 16,75 | 61,16 / 58,06 | — | 155,6 | 874 / 2.063 | 3.033 |
| 1920×1080 | 2 | 297.992 | 31,04 / 31,45 | 32,15 / 29,96 | — | 87,8 | 1.557 / 2.939 | 2.935 |
| 1920×1080 | 3 | 446.988 | 45,96 / 46,30 | 21,74 / 20,17 | — | 60,6 | 2.147 / 3.777 | 2.869 |
| 1920×1080 | 4 | 595.984 | 60,43 / 60,92 | 16,53 / 15,31 | — | 47,3 | 3.054 / 4.783 | 3.794 |
| 1920×1080 | 6 | 893.976 | 90,12 / 90,55 | 11,09 / 10,30 | OOM ở threaded probe | 32,0 | 3.663 / 5.728 | 4.683 |

GPU utilization mean trong các warm cases nằm khoảng 30–76%; p95 thường 97–99%. CSV có từng case, GPU utilization và RSS theo sample: [matrix_summary.csv](../output/codec_trials/streamstgs_multi_object_client/metrics/matrix_summary.csv). Chart: [scene FPS scaling](../output/codec_trials/streamstgs_multi_object_client/artifacts/joint_scene_fps_scaling.png) và [GPU memory scaling](../output/codec_trials/streamstgs_multi_object_client/artifacts/gpu_memory_scaling.png).

State reconstruction chiếm khoảng 9,9 ms/object tại N=1, 19,7 ms ở N=2, 39,4 ms ở N=4 và 58,8 ms ở N=6; raster tăng thêm khoảng 5,8 ms/object ở 512p. Hai phần cùng tăng gần tuyến tính, nhưng neural reconstruction là chi phí lớn hơn khi N cao. N=6 serial joint vẫn render được ở khoảng 11 FPS; VRAM device-wide đạt 5.714/6.144 MiB tại 720p. Threaded reconstruction không tăng throughput: joint FPS của N=1/2/3/4 lần lượt khoảng 62,5/32,5/22,0/16,2 so với serial 63,8/33,0/22,2/16,8. Threaded N=6 cần thêm 38 MiB và OOM.

### Joint raster và batching

Ở 512p, N draw độc lập tạo N ảnh rời, chỉ dùng làm baseline chi phí raster (không phải phép composition tương đương). Một draw chung, kể cả thời gian nối Gaussian state, nhanh hơn baseline khoảng **6,8% tại N=2, 10,1% tại N=3 và 12,0% tại N=4**. Đây là lợi ích hữu ích nhưng nhỏ hơn nhiều so với phần thời gian reconstruction; joint render một mình không làm N=3/4 đạt 30 FPS.

Batch deformation cho copies có cùng weights không cải thiện: tại 512p, batched path thấp hơn serial khoảng 6,2% ở N=1, 7,0% ở N=2, 7,4% ở N=3 và 7,6% ở N=4. Chunking giúp giới hạn activation memory ở N=6 nhưng không biến batching thành tăng tốc trên GPU này. Shared-weight batching cũng không áp dụng trực tiếp cho model weights khác nhau.

## Cold GOP, CRF và amortized throughput

Một object GOP 0 mất khoảng 2,05–2,21 giây để nạp lạnh ở CRF20, trong đó đo mẫu N=2 cho mỗi object có trung bình 0,731 s software HEVC, 0,107 s canonical/JXL reconstruction/upload, 1,060 s feature tile unpack/normalize/upload và 0,002 s MLP weights. Còn lại là model setup và các thao tác phụ trợ. Nạp tuần tự N object trong matrix cuối mất: N=1 **2,03 s**, N=2 **4,02 s**, N=3 **6,41 s**, N=4 **8,53 s**, N=6 **12,37 s**.

Amortized 60-frame FPS dưới đây lấy `60 / (N-GOP cold startup + 60 / warm serial scene FPS)`. Đây là startup serial chia trên một GOP, không phải tốc độ sau prefetch.

| Resolution | N | Cold initial GOPs (s) | Warm FPS | Amortized 60-frame FPS |
|---|---:|---:|---:|---:|
| 512×512 | 1 | 2,03 | 63,76 | 20,2 |
| 512×512 | 2 | 4,02 | 32,96 | 10,3 |
| 512×512 | 3 | 6,41 | 22,24 | 6,6 |
| 512×512 | 4 | 8,53 | 16,78 | 5,0 |
| 512×512 | 6 | 12,37 | 11,22 | 3,4 |
| 1280×720 | 1 | 2,10 | 62,72 | 19,6 |
| 1280×720 | 2 | 4,05 | 32,56 | 10,2 |
| 1920×1080 | 1 | 2,05 | 61,16 | 19,8 |
| 1920×1080 | 2 | 4,05 | 32,15 | 10,1 |

CRF được đo ở N=2, GOP0, 512p. GOP0 package giảm từ 15.039.062 B (60,156 Mbps) tại CRF20 xuống 7.684.135 B (30,737 Mbps), 3.912.584 B (15,650 Mbps) và 3.515.502 B (14,062 Mbps) ở CRF28/36/40. Với N=2, warm serial FPS lần lượt **32,96 / 32,90 / 32,90 / 32,92**; batched là **30,64 / 30,61 / 30,59 / 30,64**. Runtime warm hầu như không phụ thuộc CRF. Cold startup giảm từ 4,28 s ở CRF20 xuống 3,48 s ở CRF40; HEVC stage trung bình giảm 0,731→0,455 s, còn feature unpack vẫn khoảng 0,97–1,06 s. Mẫu này cho thấy CRF ảnh hưởng đáng kể hơn đến cold HEVC load và bytes hơn đến warm neural/render runtime.

| CRF | GOP0 bytes gồm manifest | GOP0 bitrate trung bình 2 s | N=2 serial warm FPS | N=2 batched warm FPS | N=2 cold startup |
|---:|---:|---:|---:|---:|---:|
| 20 | 15.039.062 | 60,156 Mbps | 32,96 | 30,64 | 4,28 s |
| 28 | 7.684.135 | 30,737 Mbps | 32,90 | 30,61 | 3,75 s |
| 36 | 3.912.584 | 15,650 Mbps | 32,90 | 30,59 | 3,59 s |
| 40 | 3.515.502 | 14,062 Mbps | 32,92 | 30,64 | 3,48 s |

Bitrate lấy từ archive audit hiện có, không tính lại trên frame output benchmark: [archive_audit.json](../output/codec_trials/streamstgs_soldier_536_835/metrics/archive_audit.json). Chất lượng CRF và đánh giá rate–distortion trước đó nằm trong [codec benchmark](STREAMSTGS_CODEC_BENCHMARK.md) và [decode/render evaluation](STREAMSTGS_DECODE_RENDER_EVALUATION.md); report này không tạo quality reference mới.

## GOP buffering, deadline và continuity

Mỗi playback policy phát theo nhịp 30 Hz trong 300 scene timestamps; N object có timeline lệch pha 0/7/14/21 frames. Báo cáo giữ riêng thời gian startup, số deadline miss và wall FPS. Prototype không drop frame: frame trễ vẫn được tạo và ghi log, nên “timeline liên tục” dưới đây xác nhận source frame/GOP index không bị nhảy trong pipeline, nhưng không có nghĩa người xem thực nhận từng frame đúng lúc.

| Policy | N | Startup load (s) | Fresh scene FPS | Deadline miss | Completion latency p50/p95 (ms) | Resident GOP slots | Device peak (MiB) | CPU RSS peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Preload all, chỉ GOP0 warm | 1 | 9,77 | 30,00 | 8,7% | 14,6 / 232,6 | 5 | 3.174 | 2.862 |
| Preload all, chỉ GOP0 warm | 2 | 18,41 | 30,00 | 33,3% | 30,5 / 528,9 | 10 | 5.328 | 3.510 |
| Preload all | 4 | OOM khi nạp GOP | — | — | — | tối đa 13/20 model đã nạp | — | — |
| Current only | 1 | 2,01 | 22,64 | 80,0% | 1.826 / 4.148 | 1 | 2.006 | 3.623 |
| Current only | 2 | 4,10 | 11,68 | 82,7% | 6.685 / 13.735 | 2 | 2.458 | 3.496 |
| Current only | 4 | 8,10 | 5,79 | 100% | 17.530 / 37.272 | 4 | 3.624 | 4.282 |
| Current + next nền | 1 | 2,01 | 30,00 | 35,7% | 17,7 / 640,2 | 2 | 1.910 | 3.202 |
| Current + next nền | 2 | 4,21 | 16,87 | 95,3% | 2.897 / 6.231 | 4 | 3.060 | 4.253 |
| Current + next nền | 4 | 8,04 | 8,46 | 100% | 10.705 / 22.564 | 8 | 4.674 | 4.602 |

Thử riêng preload-all với warm-up cho từng **tổ hợp GOP theo timeline lệch pha**. GOP models đã được load trước; một offscreen joint draw cho mỗi trạng thái GOP được thực hiện trước khi bắt đầu đồng hồ 30 Hz.

| Preload và warm | N | Tổ hợp GOP warm | Load + warm startup (s) | Fresh FPS | Deadline miss | Latency p50/p95 (ms) | Model slots | Device peak (MiB) | CPU RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Toàn bộ GOP + toàn bộ tổ hợp timeline | 1 | 5 | 10,91 | 30,00 | 0% | 14,45 / 16,17 | 5 | 3.174 | 2.773 |
| Toàn bộ GOP + toàn bộ tổ hợp timeline | 2 | 10 | 20,13 | 30,00 | 0% | 27,35 / 30,51 | 10 | 5.328 | 3.491 |

Current-only cần 1,66–1,94 s cho từng GOP boundary ở N=1; với N=2 mỗi boundary mất 1,63–2,27 s. Current+next dùng hai worker nhưng GOP kế tiếp không kịp sẵn: N=1 vẫn đợi 0,26–0,44 s tại bốn boundary; N=2 có boundary wait tới 1,83 s; N=4 tới 4,28 s. Chạy nền đồng thời decode/upload và playback cũng tranh GPU/CPU, làm full scene throughput giảm còn 16,87 FPS ở N=2 và 8,46 FPS ở N=4. Chỉ tăng worker count không giải quyết được giới hạn tài nguyên này.

Preload-all giữ 5 GOP/object. Warm GOP0-only cần 9,77 s cho N=1 và 18,41 s cho N=2; N=2 dùng device-wide 5.328 MiB/6.144 MiB. N=4 chạm OOM khi đã nạp khoảng 13/20 GOP models, Torch báo 4,76 GiB allocated, 5,01 GiB reserved và chỉ còn khoảng 21 MiB trống. Warm năm GOP đồng bộ nhưng bỏ qua các tổ hợp lệch pha vẫn trễ 25% deadline ở N=2 (p95 390 ms). Sau khi warm mọi tổ hợp timeline, N=1/2 không còn miss deadline; riêng N=2 cần 20,13 s trước khi phát và giữ đủ 10 model GOP. JSON cuối: [N=1](../output/codec_trials/streamstgs_multi_object_client/metrics/playback_preload_all_layout_gops_warmed_n1_512x512_crf20.json), [N=2](../output/codec_trials/streamstgs_multi_object_client/metrics/playback_preload_all_layout_gops_warmed_n2_512x512_crf20.json). Đây là một startup/VRAM tradeoff rõ ràng: GOP prefetch giới hạn 2 GOP/object hiện chưa warm/đưa đủ trạng thái kế tiếp vào đúng hạn.

Các playback log kiểm tra `source_frame` tăng liên tiếp modulo 300 và ghi mỗi GOP transition; tất cả các run hoàn chỉnh có timeline index liên tục, với 4/9/19 transitions tương ứng N=1/2/4. Chưa có reference hình Soldier khớp chính xác để kết luận chuyển động/appearance không có seam thị giác ở biên GOP. Per-frame timing và GOP records: [buffering_summary.csv](../output/codec_trials/streamstgs_multi_object_client/metrics/buffering_summary.csv) và các JSON `playback_*.json` trong [metrics](../output/codec_trials/streamstgs_multi_object_client/metrics/).

## Bottleneck và kiến trúc client

Ở 512p, N=2 nằm sát ngân sách 33,3 ms/frame (p95 30,56 ms); N=3 đã vượt 45 ms. Neural reconstruction tăng gần 9,8 ms/object, joint raster tăng khoảng 5–10 ms/object tùy resolution/cardinality, còn cold GOP initialization khoảng 2 s/object. N=6 warm chạy được nhưng VRAM gần đầy; threaded state calls cần thêm activation buffers và OOM. Khi current+next tranh tài nguyên, cold decode/upload làm trễ playback ngay cả khi số GOP resident được giới hạn.

Client multi-object thực tế nên có per-object decoder và model state; GOP ring buffer giữ current/next với explicit VRAM budget; scheduler phân tách CPU HEVC/JXL read/decode và CUDA upload/deformation; state manager áp world transforms rồi gom Gaussian arrays; một global depth/alpha renderer; timeline đồng bộ theo object timestamps; camera controller và telemetry deadline/memory. Cần ưu tiên object còn hiển thị/đang chuyển động và bỏ hoặc giảm update của object ít quan trọng khi không đủ ngân sách.

Independent decoders, asynchronous prefetch, state batching và joint raster là các kỹ thuật hệ thống/engineering đã phổ biến; bản thân một player mới chưa đủ novelty. Trong trial này joint render chỉ nhanh hơn 6,8–12,0%, còn MLP batching chậm hơn. Có thể nghiên cứu thành contribution nếu đề xuất scheduler multi-object có quality/deadline/visibility objective, memory-bounded GOP policy và adaptive per-object update rate, rồi chứng minh lợi ích định lượng trên nhiều **representation độc lập**, scene/layout, GPU và tải mạng với baselines current-only, prefetch FIFO, per-object rendering. Kết quả replicated Soldier hiện tại chỉ là microbenchmark, không chứng minh novelty.

## Viewer, artifacts và lệnh tái lập

Prototype đã render joint scene thành ảnh thật qua EGL headless; không cần graphical desktop, X11 hay browser. Chưa viết interactive native viewer trong trial này; camera orbit/zoom được kiểm chứng bằng camera yaw và khoảng cách khác nhau trong composition validation. Đã xuất thêm năm frame cùng camera tại source-frame pairs **536/543, 548/555, 560/567, 572/579, 584/591**; hashes khác nhau, mỗi ảnh dùng một joint draw. Xem [manifest và timestamp/hash](../output/codec_trials/streamstgs_multi_object_client/artifacts/timeline_preview/manifest.json), [frame 0](../output/codec_trials/streamstgs_multi_object_client/artifacts/timeline_preview/scene_frame_000.png), [frame 12](../output/codec_trials/streamstgs_multi_object_client/artifacts/timeline_preview/scene_frame_012.png), [frame 24](../output/codec_trials/streamstgs_multi_object_client/artifacts/timeline_preview/scene_frame_024.png), [frame 36](../output/codec_trials/streamstgs_multi_object_client/artifacts/timeline_preview/scene_frame_036.png), [frame 48](../output/codec_trials/streamstgs_multi_object_client/artifacts/timeline_preview/scene_frame_048.png). Đây là static-frame evidence, không phải UI camera tương tác; playback JSON ghi đủ 300 frame/object.

Tất cả scripts, configs, logs, JSON/CSV và hình ảnh nằm trong [trial directory](../output/codec_trials/streamstgs_multi_object_client/). Lệnh từ repo root:

```bash
# Ma trận warm N=1,2,3,4,6,8 ở 512p; đại diện 720p và 1080p
bash output/codec_trials/streamstgs_multi_object_client/scripts/benchmark_matrix.sh

# Bổ sung N=3 tại 720p/1080p và N=6 tại 1080p
bash output/codec_trials/streamstgs_multi_object_client/scripts/benchmark_remaining_resolutions.sh
bash output/codec_trials/streamstgs_multi_object_client/scripts/benchmark_n6_1080.sh

# Retry high-N, CRF28/36/40 và các trường hợp mở rộng
bash output/codec_trials/streamstgs_multi_object_client/scripts/benchmark_extended.sh

# Threaded per-object reconstruction rồi joint render
bash output/codec_trials/streamstgs_multi_object_client/scripts/benchmark_concurrent_joint.sh

# Preload toàn bộ GOP, warm mỗi GOP trước khi phát
bash output/codec_trials/streamstgs_multi_object_client/scripts/benchmark_preload_warmed.sh

# 300 timestamps @30Hz: preload-all/current-only/current+next background
bash output/codec_trials/streamstgs_multi_object_client/scripts/benchmark_buffering.sh

# Composition images, orbit, separation, depth/occlusion checks
bash output/codec_trials/streamstgs_multi_object_client/scripts/run_trial.sh \
  output/codec_trials/streamstgs_multi_object_client/scripts/validate_scene_composition.py

# Xuất nhiều timestamp joint-render cùng camera để xem motion
bash output/codec_trials/streamstgs_multi_object_client/scripts/run_trial.sh \
  output/codec_trials/streamstgs_multi_object_client/scripts/export_timeline_frames.py
```

`benchmark_n6_1080.sh` và `retry_n8.sh` chạy các case memory-bound tương ứng. Từng benchmark dùng `scripts/run_trial.sh` để nạp environment StreamSTGS/FAST đã có. Toàn bộ thay đổi của task này nằm dưới trial directory và report; production codec, DASH, Planner và frontend không bị chỉnh sửa.

## Giới hạn còn lại

- Chỉ có một Soldier representation phù hợp cho phép nhân bản; chưa đo N content độc lập, mixed CRF per object hoặc các convention hình học khác nhau.
- N=8 không đạt bước nạp đồng thời model/feature state trên 6 GiB; N=6 threaded reconstruction cũng OOM. N=6 đã chạy ở 1080p nhưng chỉ đạt 11,09 FPS.
- Playback prototype không drop frame và không mô phỏng network transfer; throughput đo là remote GPU offscreen, không phải physical display FPS.
- Basic preload-all chỉ warm GOP0; phép đo bổ sung warm mọi tổ hợp GOP lệch pha cho N=1/2 đã tách first-use renderer cost khỏi steady playback. Bounded current+next chưa có bước prewarm đầy đủ trước GOP boundary.
- Không có phép đo riêng byte-by-byte PCIe copy; các stage decode/unpack/upload đã được tính chung theo stage.
- Timestamp continuity được kiểm tra theo source/GOP index; visual seam quality tại GOP boundary chưa được so với ground truth độc lập.

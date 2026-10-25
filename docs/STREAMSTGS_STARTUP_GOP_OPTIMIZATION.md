# StreamSTGS native startup và GOP playback

Ngày đo: **2026-10-24**. GPU: **GTX 1660 6 GiB**; PyTorch **2.3.1+cu118**, NVIDIA **595.99.02**, FAST CUDA/OpenGL qua native GLFW/GLX. Các phép đo chính dùng **1024×768, CRF20, 30 Hz, đủ 300 scene frames** và current+next GOP. Không preload toàn bộ video, không giảm quality, resolution hoặc số object.

Các bảng cuối dùng source và harness đóng băng tại `output/startup_gop_optimization/final_runtime/`, đo độc lập trong `paper_final/`: [source SHA ledger](../output/startup_gop_optimization/final_runtime/source_sha256.json), [harness SHA ledger](../output/startup_gop_optimization/final_runtime/harness_sha256.json), [baseline SHA ledger](../output/startup_gop_optimization/final/baseline_sha256.json). Không chọn lượt phát triển nhanh nhất. Các artifacts intermediate và run bị oracle mới reject vẫn được giữ để kiểm tra quá trình phát hiện lỗi.

## Kết quả và cách diễn giải

Startup giảm từ 5,45/7,03/12,08 giây xuống **2,13/2,20/3,05 giây** cho N=1/2/4, giảm61,0%/68,7%/74,7%. Sau frame đầu, N=1 đạt **30,05 FPS**; N=2 đạt **29,41 FPS** với jitter; N=4 đạt **17,47 FPS**. Cả ba lượt CRF20 cố định không đợi GOP mới tại boundary. N=4 vẫn không đạt30FPS vì reconstruction và joint draw của một scene đã mất47,06/53,84ms ở P50/P95. Loại bỏ các pause khởi tạo GOP dài không giải quyết được giới hạn steady-state này.

| N | Launch → first render, trước/sau (s) | Launch → first native swap, trước/sau (s) | Fresh scene FPS sau frame đầu, trước/sau | Scene compute P50/P95, trước → sau (ms) | Frames |
|---:|---:|---:|---:|---:|---:|
| 1 | 5,453 / 2,124 | 5,455 / 2,125 | 23,53 / 30,05 | 15,88/18,55 → 16,03/19,26 | 300/300 |
| 2 | 7,024 / 2,200 | 7,026 / 2,202 | 15,48 / 29,41 | 23,40/28,37 → 25,22/29,14 | 300/300 |
| 4 | 12,079 / 3,052 | 12,080 / 3,054 | 6,12 / 17,47 | 44,73/52,23 → 47,06/53,84 | 300/300 |

FPS sau frame đầu là `(frames−1)/(last scene completion−first scene completion)`, gồm toàn bộ waiting và GOP transitions sau startup. Scene compute latency đo từ lúc bắt đầu tái dựng scene đến khi joint draw hoàn tất; không gồm chờ presentation clock, preparation hoặc UI swap. Đây là FPS của scene gồm đủ N objects, không nhân FPS với N. Native swap completion xác nhận `glfw.swap_buffers` đã trả về; chưa đo physical scanout trên màn hình.

| N | Boundary transitions có frame đo | Boundary interval vượt 33,33 ms, mean/max trước (ms) | Mean/max sau (ms) | GOP readiness wait sau | Scene compute >33,33 ms, trước/sau |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 417,63 / 583,56 | 0,68 / 1,37 | 0 | 2 / 1 |
| 2 | 4 (hai objects đồng pha) | 1590,68 / 2073,76 | 4,61 / 6,90 | 0 | 2 / 1 |
| 4 | 19 (objects lệch pha) | 1782,64 / 4182,70 | 26,35 / 37,99 | 0 | 300 / 300 |

Boundary interval là khoảng giữa hai fresh scene completions quanh lúc ít nhất một object đổi GOP; phần vượt một frame period là chỉ số có cùng định nghĩa ở bản trước và sau. Nó bao gồm compute, UI/scheduling và preparation nên không đồng nhất với GOP readiness wait. Bản sau ghi riêng cặp sự kiện `readiness_stall_start/end` và phân loại startup, temporal window, GOP boundary. N=4 có interval dài ở tất cả scene frames vì compute chậm hơn33,33ms dù GOP đã sẵn sàng.

Lượt cuối N=1 không có temporal-feature wait sau startup. N=2 có hai wait ban đầu: frame3=7,72ms và frame4=11,47ms. N=4 có frame1=40,50ms. Những waits này được giữ trong FPS/timing, không bỏ mẫu. Một lượt intermediate `paper_after/n4` có15,52ms GOP wait; vì vậy zero GOP wait trong lượt final không phải bảo đảm deadline tuyệt đối ở mọi lần chạy.

Deadline trên completion gaps (299 khoảng, first frame loại khỏi denominator) được tính lại với cùng harness từ raw logs của cả hai versions:

| N | Gaps >33,33ms trước/sau | Gaps >34,33ms trước/sau | Gap P95 trước/sau (ms) | Gap max trước/sau (ms) |
|---:|---:|---:|---:|---:|
| 1 | 232 / 143 | 222 / 49 | 43,81 / 35,12 | 616,89 / 43,40 |
| 2 | 150 / 164 | 150 / 106 | 61,55 / 38,57 | 2107,09 / 77,97 |
| 4 | 299 / 299 | 299 / 299 | 623,09 / 67,03 | 4216,03 / 118,79 |

Strict count N=2 tăng dù long stalls và overall FPS cải thiện: baseline có chuỗi completion gaps ngắn/dài xen kẽ, còn bản sau nhiều gaps chỉ vượt33,33ms rất ít. Không dùng mean FPS hoặc tolerance1ms để tuyên bố mọi deadline đạt. Các số `deadline_misses` do runtime báo không dùng để so trực tiếp hai versions: thời điểm lấy scheduled deadline đã được sửa để loại startup stall khỏi deadline của frame đầu. N=2 còn jitter; cả N=1 cũng chưa đạt tất cả299 gaps đúng33,33ms tuyệt đối.

Raw số liệu: [before reanalysis](../output/startup_gop_optimization/before_reanalyzed.json), [paper_final](../output/startup_gop_optimization/paper_final/summary.json); mỗi thư mục case có `invocation.json`, `timing.json`, `playback.jsonl`, `samples.json`, `summary.json` và log của player. Bảng máy đọc: [comparison.csv](../output/startup_gop_optimization/figures_final/comparison.csv), [comparison.json](../output/startup_gop_optimization/figures_final/comparison.json). Charts được tạo lại bằng [report_native.py](../benchmarks/streaming/report_native.py); có [PDF startup/FPS](../output/startup_gop_optimization/figures_final/startup_fps.pdf) và [PDF frame gaps](../output/startup_gop_optimization/figures_final/frame_gaps.pdf) cho export.

[Validation index](../output/startup_gop_optimization/validation_index.json) tổng hợp các checks cuối và xác nhận source Python trong workspace khớp snapshot đã đo.

![Launch-to-first-swap và fresh scene FPS trước/sau](../output/startup_gop_optimization/figures_final/startup_fps.png)

![Fresh completion gaps qua đủ300 frames](../output/startup_gop_optimization/figures_final/frame_gaps.png)

## Workload và phương pháp đo

Baseline là source copy được giữ trước sửa đổi tại `output/startup_gop_optimization/baseline/src_final/`. Cả hai dùng cùng renderer FAST, camera tự calibration, joint scene, transform, compressed payload, local MPD và màn hình native. Mỗi run là Python process mới với application cache rỗng; CUDA/GL context mới. Filesystem/driver cache của hệ điều hành không bị flush, vì vậy “cold” ở đây là runtime/process/GOP cold, không phải cold disk hoặc cold machine. Không có network transfer trong bảng chính.

Scenes riêng của phép đo nằm trong `output/startup_gop_optimization/scenes/content-n{1,2,4}/manifest.mpd`:

- N=1: Soldier, offset0.
- N=2: Longdress và Soldier, offset0, transforms tách hai objects.
- N=4: Longdress/Soldier/Longdress/Soldier, offsets 0/7/14/21 frames, mỗi instance loop độc lập. Các instances có GPU model, temporal storage và GOP state riêng; chỉ immutable CPU preparation của đúng cùng payload được dùng chung. Đây là hai contents khác nhau với bốn instances, chưa phải bốn contents khác nhau.

Source data là 300 frames, GOP60 @30FPS, CRF20. N=4 vượt wrap-around của các timeline lệch pha trước khi scene hết 10 giây. So sánh trước/sau dùng đúng cùng MPD. Assets/segments của scenes benchmark là hardlinks tới publication hiện có; không thay đổi MPD public đang dùng cho 2 hoặc 3 contents.

Harness [profile_native.py](../benchmarks/streaming/profile_native.py) khởi chạy child process và dùng `perf_counter`/monotonic timestamps từ launch. [profile_entry.py](../benchmarks/streaming/profile_entry.py) bọc các method để ghi stage và native swap đầu có nội dung. CPU% và RSS là tổng process tree gồm FFmpeg; CPU100% tương đương một logical core. GPU utilization và device memory được lấy bằng `nvidia-smi` mỗi250ms. Vì lấy mẫu, peak có thể bỏ sót spike ngắn và chứa driver/desktop/GL memory. Torch counters được giữ riêng. Không cộng stage times của các workers chạy overlap để suy ra startup wall time.

Các reports lịch sử ở [decode/render evaluation](STREAMSTGS_DECODE_RENDER_EVALUATION.md) và [multi-object benchmark](STREAMSTGS_MULTI_OBJECT_CLIENT_BENCHMARK.md) cung cấp cold/warm và preload tradeoffs, nhưng khác workload, environment hoặc resolution; không dùng những dòng cũ làm baseline số cho bảng này. Baseline ở đây là native implementation sau rendering fixes, nên cả hai versions đã dùng đúng feature atlas layout và framebuffer state.

## Root cause và thay đổi runtime

1. **GOP host preparation là một barrier toàn bộ64 temporal atlases.** Reconstruction upstream tại một timestamp chỉ đọc cửa sổ ba feature vectors (`get_time_features(..., training=False, noise=0)`). Runtime cũ đợi64 atlases rồi mới đưa model lên GPU. Adapter mới publish canonical attributes/weights sớm, tiếp tục decode/unpack temporal atlases trên persistent CPU pool và chỉ cho render khi đúng window cần dùng đã copy xong. Không thay interpolation, temporal index hoặc MLP.
2. **Owner thread upload trọn GOP và synchronize CUDA toàn device.** Soldier GOP0 chứa khoảng466MB canonical/features. Bản mới dùng một CUDA copy stream, hai pinned buffers8MiB và copy chunks vào destination cuối cùng. Event sau feature hoàn chỉnh bảo vệ read-after-copy; slot chỉ reuse sau event completion. Render/release chỉ đợi render stream cần thiết. Bản mới không có full GPU staging copy thứ hai.
3. **FAST TorchScript specialize khi cardinality thay đổi.** Microprofile độc lập chỉ ra `build_cov6` có pause140–332ms ở call sau khi N Gaussian thay đổi; generic graph cùng phép toán ổn định khoảng3,9–4,1ms. Wrapper FAST dùng scoped `torch.jit.optimized_execution(False)` cho draw, không sửa vendor graph hoặc covariance formula. Flag `--fast-jit-specialization` giữ lại đường cũ để tái lập ablation.
4. **Idle loop rerun neural reconstruction/rasterization của cùng timestamp.** Khi camera không đổi và chưa đến scene frame mới, native renderer khôi phục scene sạch từ FBO cache rồi vẽ UI. Orbit/pan/fly/resize vẫn làm refresh scene. UI tĩnh giới hạn60Hz; input/GOP polling tiếp tục ở idle1ms. Cache lưu scene trước UI, tránh chồng UI qua nhiều vòng.
5. **Hai instances cùng payload decode lại immutable CPU data.** Singleflight theo payload/shared-asset SHA dùng chung đúng kết quả CPU; handles/GPU state luôn độc lập. Host budget refcount tính một lần cho allocation vật lý, không nhân đôi reservation của host dùng chung.
6. **Feature unpack tạo nhiều scratch arrays và memory passes.** Tối ưu tile channel-first dùng scratch float64 rồi giữ kết quả contiguous float32. Vẫn divide/multiply/add cùng thứ tự rounding và giữ layout atlas đã sửa trong rendering audit. SHA và tensor equality xác nhận không đổi giá trị.
7. **Clock cộng dồn float làm sai source index và lỡ quality boundary.** Trace600 frames phát hiện `position += 1/fps` có thể tạo3,999999999999996 thay vì4s. GOP lookup còn ở GOP trước; frame kế tiếp đã vượt boundary nên pending quality không activate, và runtime bỏ prefetch đúng target để nạp lại CRF cũ. Clock mới giữ seek anchor và integer frame ticks; media mapping chỉ normalize roundoff trong4ULP của operands, giữ các phase thực sự lệch frame grid. Natural scene loop giữ generation và reuse GOP0 đã prefetch; explicit seek vẫn invalidates jobs. Đây là sửa correctness được xác nhận bằng source-index oracle độc lập, không đổi temporal interpolation hoặc pacing để làm đẹp FPS.

Fetch và CPU preparation bắt đầu trước window/renderer initialization để overlap với OpenGL/CUDA import. CPU pool mặc định2 workers, FFmpeg2 decoder threads dựa trên microprofile và playback đo thực tế. Không thêm decoder process thường trực vì FFmpeg startup không phải phần chi phối đã đo; thêm process sẽ có chi phí IPC/lifetime cần chứng minh riêng.

## Startup và cold GOP breakdown

Trong baseline N=1, native window mất263ms, renderer/CUDA setup1379ms; GOP CPU preparation trung bình2727ms, upload/model bind59ms. GOP0 đầu tiên CPU3189,7ms, trong đó feature stage3036,9ms, rồi GPU realization71,0ms. Các stage cũ nằm trên đường chờ đầu tiên nên first swap5,455s.

| Stage, trung bình mỗi logical GOP | Trước N=1 (ms) | Sau N=1 (ms) | Diễn giải |
|---|---:|---:|---|
| Full CPU preparation | 2726,9 | 1894,39 | Bản sau vẫn hoàn thành và kiểm tra đủ bitstream |
| Metadata read | Không tách timer | 1,10 | Reconstruction JSON/CSV/config |
| Canonical+weights host ready | Không tách timer | 108,86 | Publish host sớm, không cần đợi64 features |
| Weights CPU load | Không tách timer | 3,50 | Là một phần canonical-ready interval |
| Feature read + unpack stage | 2594,6 | 1785,52 | Decoder và unpack overlap |
| Feature producer read wait | Không tách timer | 1080,42 | Consumer chờ FFmpeg; không phải isolated HEVC runtime |
| Feature unpack CPU | Không tách timer | 704,52 | Không gồm read waiting |
| Codec model import | Trong full realize | 0,66 | First GOP2,71ms; mean có cached imports |
| GPU model initialization | 2,39 | 2,37 | Không phải bottleneck chính |
| Weight binding/upload | 0,44 | 0,39 | Không đổi weights |
| Host pack CPU, cộng qua chunks | Không tách timer | 91,54 | Pin/copy trước async H2D |
| GPU destination allocation CPU | Trong allocate+upload | 5,55 | Cấp phát trực tiếp đích cuối |
| H2D CUDA event time, cộng qua chunks | Không tách timer | 48,94 | Khoảng hoạt động copy stream giữa timing events |
| GPU upload owner CPU work, cộng qua chunks | 55,2 (allocate+copy blocking) | 140,13 | Thêm pin/copy nhỏ nhưng chia trên nhiều pumps |
| First three GPU features ready, từ begin upload | Đợi full upload | 243,79 | Mean gồm queue/poll scheduling; GOP0 riêng74,27ms |
| Full upload pipeline elapsed | 58,6 (full realize) | 1731,85 | Bản sau overlap CPU producer và playback; **không** là PCIe copy time |

Window và CUDA/renderer vẫn tốn khoảng0,28–0,29s và1,34–1,41s ở bản sau. Đây là phần còn lại đáng kể trong first swap~2,1–2,2s của N=1/2; tối ưu GOP overlap không loại bỏ context/import cost. Stage timing tách host pack, destination allocation, CUDA-event H2D copy và upload pipeline elapsed để tránh đánh đồng idle/producer wait với transfer latency.

Timeline N=1 theo giây từ process launch, cho thấy stage overlap trực tiếp:

| Mốc | Baseline (s) | Optimized (s) |
|---|---:|---:|
| Entry imports | 0,024–0,110 | 0,024–0,133 |
| Native window init | 0,226–0,488 | 0,268–0,562 |
| FAST/CUDA renderer init | 0,488–1,867 | 0,564–1,900 |
| Host readiness được owner harvest | 5,202 (complete GOP) | 1,901 (canonical/weights, feature producer tiếp tục) |
| GPU decoder first required window ready | 5,273 | 1,976 |
| First neural reconstruction + joint draw | 5,274–5,453 | 1,977–2,124 |
| First content swap complete | 5,455 | 2,125 |

GOP0 CPU producer của bản sau hoàn thành đủ64 atlases ở3,925s; full GPU transfer xong3,958s, sau first content swap1,833s. First-frame compute còn147,08ms, gồm first neural/CUDA operations và draw setup; scoped generic graph loại repeated shape-specialization stalls ở GOP sau, không làm lần đầu mọi CUDA operation miễn phí. CPU/GPU stage means ở bảng là các GOP đã hoàn tất trong telemetry, không cộng với timeline trên.

Tổng CPU preparation sau tối ưu không bằng thời gian chặn UI: scene được present trước khi toàn bộ64 atlases và toàn bộ transfer hoàn tất. Các future windows chỉ được đọc sau completion event. `PreparedGopHost.nbytes` và GPU admission vẫn reserve kích thước đầy đủ từ đầu, vì partial readiness không làm allocation tương lai biến mất.

## CPU microprofile và ablations

Soldier GOP0,64 atlases; hai lần/variant, CPU-only. Mọi atlas của16 combinations đã khớp SHA256. Sau đó đo lần riêng không hashing để tránh hash CPU làm thay đổi producer overlap.

| FFmpeg threads | Unpack | Full feature wall (s) |
|---:|---|---:|
| 1 | Original | 2,602 |
| 1 | Vectorized | 2,378 |
| 2 | Original | 1,954 |
| 2 | Vectorized, giữ gray16be | 1,874 |

Giữ gray16be,2 threads và vectorized unpack giảm28,0% whole stage. Little-endian/filter1 tốt nhất1,845s nhưng lợi ích phụ nhỏ và không giữ thay đổi endian. Microprofile layout sau đó cho retained C-contiguous channel-first unpack5,077ms/atlas so với8,972ms của vector đầu (giảm43,4%); F-contiguous view3,385ms nhanh hơn nhưng không giữ vì rủi ro thêm pack/strided transfer ở GPU pipeline. [CPU recommendation](../output/startup_gop_optimization/cpu_trials/RECOMMENDATION.md), [layout timing](../output/startup_gop_optimization/cpu_trials/unpack_layout/report.json).

| Lượt thử | N=1 first swap/FPS | N=2 first swap/FPS | N=4 first swap/FPS | Kết luận |
|---|---:|---:|---:|---|
| Baseline | 5,455s / 23,53 | 7,026s / 15,48 | 12,080s / 6,12 | CPU barrier, idle rerender, shape compilation |
| Baseline, chỉ tăng workers2 + FFmpeg2 | — | 5,575s / 22,05 | — | Có ích nhưng vẫn chờ full GOP và shape JIT |
| v1: cache + full-GOP async upload + CPU changes | 4,219s / 26,29 | 4,405s / 25,31 | 4,727s / 16,41 | Startup vẫn đợi complete GOP |
| v2: progressive temporal windows | 2,216s / 27,41 | 2,184s / 26,83 | 3,115s / 16,82 | GOP ready nhưng còn FAST JIT spikes |
| v3: generic FAST graph + shared-host accounting/upload budget | 2,128s / 30,02 | 2,200s / 29,31 | 3,043s / 17,73 | Loại shape compile và prep starvation |

Các lượt là ablation theo các bản phát triển liên tiếp, không phải randomized factorial experiment; không gán toàn bộ delta mỗi dòng cho duy nhất một thay đổi. Artifacts `after_v1/`, `after_v2/`, `after_v3/` giữ nguyên. Lượt [workers_only/n2](../output/startup_gop_optimization/workers_only/n2/summary.json) dùng baseline snapshot với hai flags duy nhất workers2/FFmpeg2: worst completion gap488,75ms và FPS22,05 vẫn thấp hơn final29,41. Việc tăng workers riêng không đạt mục tiêu.

Hai thử nghiệm không giữ:

- **Upload chỉ trong idle slack:** N=2 đạt29,51FPS nhưng phát sinh thêm temporal waits và25,49ms GOP-boundary readiness wait. Khi render sát33ms, không đủ slack để copy kịp. Retain bounded copy budget mỗi pump; ngân sách nhân số active instances (cap4), tăng trong lúc scene chưa ready.
- **Clock catch-up:** giữ đủ frames nhưng29,53FPS, không cải thiện rõ và deadline behavior phức tạp hơn. Không giữ thay đổi clock pacing. Tiếp tục báo jitter thay vì đổi định nghĩa playback để nhận30FPS.

Microprofile [jit_covariance.json](../output/startup_gop_optimization/jit_covariance.json) cô lập đúng graph/cardinality, tránh suy đoán GL driver là nguyên nhân của các spike còn lại. Bản v2 có draw max~454/471/493ms N=1/2/4 dù GOP readiness đã gần như đủ; cache/prefetch riêng không xử lý được nguyên nhân này.

## Bounded memory và contention

Default giữ current+next tối đa hai GOP models mỗi active instance, bị giới hạn tiếp bằng byte admission. N=1/2/4 tương ứng tối đa2/4/8 slots, thấp hơn preload-all5/10/20 models. GPU model/feature storage của hai instances dùng cùng CPU payload vẫn độc lập. GPU budget default4096MiB, transient256MiB; host2048MiB; raw512MiB. Hai pinned8MiB slots được charge16MiB vào raw budget và dùng lại suốt runtime.

| N | Process-tree peak RSS trước/sau (MiB) | Device peak trước/sau (MiB) | CPU mean trước/sau (%) | GPU mean trước/sau (%) |
|---:|---:|---:|---:|---:|
| 1 | 1419,6 / 1795,2 | 1024 / 1470 | 177,1 / 347,2 | 41,8 / 34,5 |
| 2 | 1442,5 / 1711,8 | 1342 / 1650 | 175,0 / 480,6 | 46,6 / 52,7 |
| 4 | 1789,8 / 1883,5 | 2514 / 3416 | 189,4 / 394,7 | 59,9 / 64,6 |

Memory tăng so baseline vì decoder và next GOP thực sự overlap, GPU prefetch hoàn tất sớm hơn; baseline thấp hơn một phần do next states đến muộn và chặn playback. Đây là tradeoff được đo, không phải tối ưu giảm mọi chỉ số. Peak vẫn dưới6GiB và không tăng theo tổng số GOP của video. RSS bao gồm allocator, FFmpeg và Python/GL/Torch ngoài codec budgets; byte-budget counters không phải toàn-process RSS limit. N=4 dùng chung CPU preparation làm giảm duplicate work, không chia sẻ mutable GPU models.

CPU utilization tăng vì tận dụng nhiều cores để chuẩn bị trước deadline. N=1 GPU mean thấp hơn vì không còn rerender frame cũ lúc idle. N=4 GPU mean cao hơn khi loại long preparation waits. Các sample utilization toàn lượt gồm startup; không dùng mean thấp để kết luận compute còn đủ khả năng30FPS. P50/P95 scene compute cho biết N=4 vượt ngân sách ngay cả lúc resident.

## Correctness và regression

- Host regression: **112 passed** trong [regression_all_final.log](../output/startup_gop_optimization/regression_all_final.log), gồm progressive-window gating, shared host budget/refcounts, cancel/seek producer lifetime, failure propagation, independent offsets, quality activation ở2/4/6/8s, source continuity300/600 frames, arbitrary seek phase,4ULP roundoff bound và GPU event availability.
- GPU full-GOP oracle so progressive: **18 packages** = Longdress/RedAndBlack/Soldier × CRF20/40 × GOP0/4 (12 packages), bổ sung GOP2 × CRF20/28 cho cả ba contents (6 packages). Mỗi package kiểm first/middle/last frame, canonical và đủ64 temporal CPU hashes, năm Gaussian attributes, image EGL, independent GPU allocations, unavailable-window rejection, producer/transfer drain. Tất cả tensors và images có **max error0**: [GOP0/4](../output/startup_gop_optimization/pipeline_gpu.json), [GOP2](../output/startup_gop_optimization/pipeline_gpu_gop2.json).
- Native parity/cache: **24 cases passed**, N=1/2/4, GOP0/4, ba timestamps/angles và resize800×600/1024×768/1280×720. Cố ý làm hỏng backbuffer/scissor/viewport trước restore; cache restore/UI redraw/full clear đều **max error0**. Native so EGL: max2/255, mean absolute error tối đa0,0027 trên thang8-bit, fraction vượt1/255 tối đa0,00195%; state SHA khớp. [report và PNGs](../output/startup_gop_optimization/native_parity/report.json).
- Native controls: **passed lại trên final runtime** với public MPD ba contents, actual GLFW/CUDA window. Pause/resume; paused orbit/pan/scroll; Q quality+seek; home/loop/fly/move; resize; đủ12 fresh native swaps: [final controls](../output/startup_gop_optimization/native_controls_final/controls.json).
- Composition/depth/translucency: **passed**, synthetic joint FAST depth swap và per-object contribution checks: [composition](../output/startup_gop_optimization/composition.json).
- Final N=1/2/4 playback: **300/300 scene frames** mỗi case, đúng300/600/1200 object-frame mappings bằng oracle rational độc lập, không sai; N=4 kiểm cả local phased wrap. Legacy snapshot thiếu source telemetry được ghi unknown. Oracle mới có6 regression fixtures, reject cả source duplicate/skip nhất quán với timestamp sai và deferred quality hợp lệ nếu chỉ xét standalone decoder.
- Standalone quality switching sweep: **192/192 passed**, ba contents × bốn GOP boundaries ×16 previous/target CRF pairs, so standalone target reference: [switching192.json](../output/startup_gop_optimization/switching192.json). Đây là kiểm pairwise decoder/target correctness, chưa đủ để chứng minh live runtime áp mọi request đúng boundary.
- Full-publication sweep: **60/60 passed**, ba contents ×5GOP ×4CRF, first/middle/last reconstruction và EGL render: [publication60_verified.json](../output/startup_gop_optimization/publication60_verified.json). Underflow oracle được sửa theo bằng chứng nêu dưới đây; không đổi codec output.
- Final live quality/loop trace: **600/600 scene frames,1200/1200 rational source mappings,8/8 quality intents activate đúng next GOP, không còn request pending**. Global loop tại frame300 reuse prefetched GOP0 và không phải đợi restart: [quality_loop_final](../output/startup_gop_optimization/quality_loop_final/quality_loop/summary.json).

Trace quality/loop final vẫn còn performance hạn chế: **28,61FPS**, scene compute P50/P95 **25,54/30,31ms**, max completion gap **176,23ms**, peak RSS **1942,3MiB**, device **2010MiB**. Các quality requests gần boundary hủy/replan phần prefetch trước đó; bốn boundary readiness waits tại scene frames60/120/180/240 là **27,87/60,87/9,89/134,41ms**. Không tuyên bố seamless representation switching hoặc30FPS tuyệt đối cho trace này. Requests tới10–30frames trước boundary cho target~0,33–1,0s lead time, ngắn hơn preparation của một GOP; pipeline bounded current+next không bảo đảm kịp mọi request sát deadline.

Trace đầu tiên [quality_loop](../output/startup_gop_optimization/quality_loop/quality_loop/summary.json) đạt26,33FPS nhưng không áp đúng mọi request ở2/4/6/8s; oracle cũ suy expected source từ observed `scene_time` đã chấp nhận duplicate/skip khi timestamp cũng sai. Oracle rational mới suy source từ `frame_id/fps`, per-instance phase/loop và raw index, đồng thời kiểm intent độc lập. Nó reject old quality trace2880 lỗi và old N=4 intermediate2458 lỗi: [oracle_old_runs_rejected.json](../output/startup_gop_optimization/oracle_old_runs_rejected.json). Sau clock/media correction, final trace mới được công nhận correctness. Artifacts cũ được giữ; không ghi đè thành kết quả pass.

Full-publication sweep đầu tiên [publication60.json](../output/startup_gop_optimization/publication60.json) có58 pass/2 fail vì assertion test yêu cầu mọi scale strictly positive ở RedAndBlack GOP2 CRF20/28. Điều tra xác nhận đây là **float32 exponential underflow có sẵn ở baseline**: cả sáu frame probes (CRF20/28 × first/middle/last) có cùng scale SHA/zero counts ở [baseline](../output/startup_gop_optimization/scales_baseline.json) và [optimized](../output/startup_gop_optimization/scales_optimized.json), không có negative scale. Preactivation của zero scales ≤−103,97965, `exp` float64 tối đa6,9536×10⁻⁴⁶, nhỏ hơn ngưỡng làm tròn về0 là2⁻¹⁵⁰ (nửa subnormal float32 nhỏ nhất2⁻¹⁴⁹). Test được sửa để reject negative/unexplained zero và kiểm chính preactivation underflow trước khi chấp nhận zero; không clamp scale, không sửa dữ liệu hoặc production decoder. Sweep gốc được giữ nguyên và GOP2 progressive oracle bổ sung nêu trên đã khớp chính xác.

Reference ở đây kiểm mathematical/image equivalence với complete decode và EGL path đã sửa rendering. Không có ground-truth RGB khớp các archives để suy ra PSNR/SSIM/LPIPS; không dùng rendered image làm quality ground truth. Scoped generic JIT phải giữ parity; không chỉ dựa vào warm FPS để chấp nhận thay đổi.

## Tái lập

Từ repository root, chỉ chạy một GPU job cùng lúc để không làm sai contention measurements. Native environment và Python dùng thống nhất:

```bash
env DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  MULTIOBJ3DGS_VENDOR_ROOT="$PWD/third_party/streamstgs" \
  /home/fil/miniconda3/envs/Hoang/bin/python \
  benchmarks/streaming/profile_native.py \
  --source-root output/startup_gop_optimization/baseline/src_final \
  --case n1=output/startup_gop_optimization/scenes/content-n1/manifest.mpd \
  --case n2=output/startup_gop_optimization/scenes/content-n2/manifest.mpd \
  --case n4=output/startup_gop_optimization/scenes/content-n4/manifest.mpd \
  --output output/startup_gop_optimization/reproduce_before \
  --python /home/fil/miniconda3/envs/Hoang/bin/python --frames 300 --workers 1
```

Đo bản sau bằng `--source-root output/startup_gop_optimization/final_runtime/src_final`, `--workers 2` và output mới. Default của bản sau là FFmpeg2, upload2ms/pump, prefetch1 GOP/instance, scoped generic JIT. `--case-args 'n2=["--upload-budget-ms","0"]'` tắt progressive/async path; `--case-args 'n2=["--fast-jit-specialization"]'` bật lại FAST shape specialization để ablate. Không truyền hai flags mới khi chạy baseline snapshot.

Tái lập600-frame quality/loop validation bằng cùng harness/source, `--frames 600`, một case tên `quality_loop` dùng N=2MPD ở trên và:

```bash
--case-args 'quality_loop=["--loop","--control-trace","output/startup_gop_optimization/final_runtime/quality_trace.json"]'
```

Harness tách frame timestamp/source continuity khỏi reported clock flags, kiểm manual intents tại next local GOP. Oracle này dành cho fixed-step/loop run không có seek; seek/paused interaction được kiểm riêng bằng native controls và unit tests.

Chạy player thông thường với public MPD nhất quán:

```bash
env PYTHONPATH=src_final DISPLAY=:1 \
  XAUTHORITY=/run/user/1000/gdm/Xauthority \
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  /home/fil/miniconda3/envs/Hoang/bin/python -m multiobj3dgs.client \
  --mpd output/src_final_validation/http_full/content-n2/manifest.mpd \
  --mode native
```

Đổi `content-n2` thành `content-n3` để mở ba contents Longdress/RedAndBlack/Soldier. MPDs này không phải N=1/2/4 shifted scenes của benchmark trên.

## Files thay đổi và hướng tiếp

Production: `codecs/streamstgs/adapter.py` (CPU progressive preparation, exact unpack, event-gated chunked transfer); `client/runtime/playback.py` (per-instance jobs, singleflight CPU leases, upload admission/scheduling và readiness telemetry); `client/runtime/clock.py` (anchor/integer ticks, natural-loop prefetch reuse); `client/scene/timeline.py` (ULP-bounded frame-grid roundoff); `client/rendering/fast.py` (clean FBO cache, scoped JIT policy và stream-local synchronization); `client/__main__.py` (startup overlap, defaults, cached idle/UI pacing và first content swap telemetry). Tất cả nằm dưới `src_final/multiobj3dgs/`. Không thay codec bitstream/algorithm, vendor StreamSTGS, DASH Adaptation Planner hoặc ABR solver.

Reproduction/reporting tại `benchmarks/streaming/{profile_native,profile_entry,trace_fast,report_native}.py`; regression fixtures ở `tests_final/unit/{test_runtime_pipeline,test_timeline_precision,test_profile_native,test_scale_validation}.py` và `benchmarks/streaming/test_profile_native.py`. GPU progressive oracle ở `tests_final/gpu/validate_pipeline.py`; native parity test bổ sung checks cache/resize/UI; publication validator ghi bằng chứng scale underflow thực tế.

N=2 còn jitter từ UI/owner scheduling, CPU decode cạnh tranh và first neural/render setup; cần trace riêng queueing/owner work quanh late frame, không tăng worker một cách mặc định. Manual quality sát deadline cần target scheduling/preparation linh hoạt hơn hoặc lead time sớm hơn; không giải quyết bằng thay CRF tự động hoặc trì hoãn request vô hạn. N=4 cần tối ưu steady reconstruction và raster (ví dụ giảm kernel launch overhead hoặc fuse operations mà giữ nguyên toán học) hoặc GPU mạnh hơn để đạt30FPS; chỉ prefetch sớm hơn không đủ. Các bảng chính là một run/process mới mỗi case, chưa phải confidence intervals qua nhiều repetitions. Đo thêm repetitions trên source frozen, warm thermal state, bốn contents thật và network throughput trước khi khái quát kết quả cho paper. Startup context/import~1,7s là mục tiêu kế tiếp; persistent process/context chỉ có ích cho reopen và phải đo tách khỏi cold process startup.

"""Tables and scientific plots from completed native experiment artifacts."""
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil

import numpy as np

from experiments.adaptation_baselines.methods import METHODS
from experiments.visibility_weighted_sse.common import digest, read_json
from .runtime_native import commit

LABELS = dict(NAVA='NAVA',VISIBILITY_PSNR='Visibility × PSNR',VISIBILITY_MSE='Visibility × MSE',
              SOLO_SSE='Solo SSE',VISIBILITY_WEIGHTED_SSE='Weighted SSE')


def available(root):
    rows = []
    for marker in sorted((root/'matrix').glob('*/*/*/*/completion.json')):
        folder = marker.parent
        family,n,duration,method = folder.relative_to(root/'matrix').parts
        metrics, performance = read_json(folder/'metrics.json'), read_json(folder/'performance.json')
        quality, bandwidth, playback, adaptation = (metrics[k] for k in ('quality','bandwidth','playback','adaptation'))
        summary = performance['summary']
        rows.append(dict(scenario=family,N=int(n[1:]),duration_s=int(duration[:-1]),method=method,
            status=metrics['status'],pooled_psnr_db=quality['pooled_psnr_db'],
            pooled_psnr_infinite=quality['pooled_psnr_infinite'],mean_psnr_db=quality['average_frame_psnr_db'],
            mean_psnr_infinite=quality['average_frame_psnr_infinite'],measured_frames=quality['measured_frame_count'],
            frame_coverage=quality['measurement_coverage'],requested_bytes=bandwidth['total_requested_bytes'],
            delivered_bytes=bandwidth['total_delivered_bytes'],utilization=bandwidth['bandwidth_utilization'],
            actual_consumption_bps=bandwidth['actual_consumption_bps'],startup_s=playback['startup_delay_s'],
            rebuffering_s=playback['total_rebuffering_s'],stall_count=playback['stall_count'],
            completion_s=playback['completion_wall_time_s'],switches=adaptation['decision_switching']['switch_count'],
            q0_fallbacks=adaptation['q0_fallback_count'],budget_violations=adaptation['budget_violation_count'],
            solver_ms=summary['native_solver_ms']['mean_ms'],
            decision_compute_ms=summary['decision_compute_excluding_preparation_ms']['mean_ms'],
            visibility_compute_ms=summary['visibility_analysis_ms']['mean_ms'],
            visibility_original_ms=summary['visibility_original_analysis_ms']['mean_ms'],
            visibility_lookup_ms=summary['visibility_cache_lookup_ms']['mean_ms'],
            folder=str(folder)))
    return rows


def _plots(root, rows):
    if not rows:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    families = ['camera_rotation','long_line','random_spatial']
    outputs = []
    for field,title in [('pooled_psnr_db','Pooled joint viewport PSNR (dB)'),
        ('delivered_bytes','Delivered bytes'),('rebuffering_s','Rebuffering (s)'),
        ('switches','Quality switches'),('decision_compute_ms','Decision CPU excluding visibility preparation (ms)'),
        ('visibility_original_ms','Original native visibility analysis per GOP (ms)'),
        ('visibility_lookup_ms','Visibility cache lookup per GOP (ms)')]:
        target = figures/f'{field}_versus_N'
        signature = digest([{k:r[k] for k in ('scenario','N','duration_s','method',field)} for r in rows])
        if target.with_suffix('.json').exists() and read_json(target.with_suffix('.json'))['data_sha256'] == signature:
            outputs.append(target.with_suffix('.png'))
            continue
        fig,axes = plt.subplots(3,3,figsize=(15,10),squeeze=False)
        for i,family in enumerate(families):
            for j,duration in enumerate((10,20,30)):
                ax = axes[i,j]
                for method in METHODS:
                    series = sorted([r for r in rows if r['scenario']==family and r['duration_s']==duration
                                     and r['method']==method],key=lambda r:r['N'])
                    if series:
                        ax.plot([r['N'] for r in series],
                                [np.nan if r[field] is None else r[field] for r in series],marker='o',
                                label=LABELS[method],alpha=.85)
                ax.set_title(f'{family}, {duration}s'); ax.set_xlabel('Independent instances N')
                ax.grid(alpha=.25)
                if j==0:
                    ax.set_ylabel(title)
        handles,labels = next(((ax.get_legend_handles_labels()) for ax in axes.flat
                               if ax.get_legend_handles_labels()[0]),([],[]))
        fig.legend(handles,labels,loc='lower center',ncol=5)
        fig.suptitle(title+' — completed runs only; ∞ remains flagged in raw data')
        fig.tight_layout(rect=(0,.045,1,.96))
        fig.savefig(target.with_suffix('.png'),dpi=150);fig.savefig(target.with_suffix('.pdf'))
        plt.close(fig);commit(target.with_suffix('.json'),dict(data_sha256=signature))
        outputs.append(target.with_suffix('.png'))
    # Preserve each actual duration, N and policy in its own common-axis time plot.
    groups = sorted({(r['scenario'],r['N'],r['duration_s']) for r in rows})
    for family,n,duration in groups:
        group = [r for r in rows if (r['scenario'],r['N'],r['duration_s'])==(family,n,duration)]
        target = figures/f'time_{family}_N{n:02d}_{duration:02d}s'
        signature = digest([(r['method'],r['pooled_psnr_db'],r['measured_frames']) for r in group])
        if target.with_suffix('.json').exists() and read_json(target.with_suffix('.json'))['data_sha256']==signature:
            continue
        fig,axes=plt.subplots(2,1,figsize=(10,7))
        for row in group:
            folder=Path(row['folder'])
            with (folder/'frames.csv').open(newline='') as stream:
                frames=list(csv.DictReader(stream))
            x=[float(f['global_time_s']) for f in frames]
            y=[float(f['psnr_db']) if f.get('psnr_db') else np.nan for f in frames]
            axes[0].plot(x,y,label=LABELS[row['method']],alpha=.85)
            with (folder/'buffer.csv').open(newline='') as stream:
                buffer=list(csv.DictReader(stream))
            axes[1].plot([float(f['wall_time_s']) for f in buffer],
                         [float(f['buffer_s']) for f in buffer],label=LABELS[row['method']],alpha=.85)
        axes[0].set(xlabel='Playback media time (s)',ylabel='Joint viewport PSNR (dB)',
                    title='Finite frame PSNR; exact-reference frames (∞) remain in raw records')
        axes[1].set(xlabel='Network/playback wall time (s)',ylabel='Buffer (s)')
        for ax in axes:
            ax.grid(alpha=.25)
        axes[1].legend(ncol=3)
        fig.suptitle(f'{family}, N={n}, {duration}s');fig.tight_layout()
        fig.savefig(target.with_suffix('.png'),dpi=150);fig.savefig(target.with_suffix('.pdf'))
        plt.close(fig);commit(target.with_suffix('.json'),dict(data_sha256=signature))
        outputs.append(target.with_suffix('.png'))
    return outputs


def make_report(root, project, state):
    root,project=Path(root),Path(project)
    rows=available(root)
    root.mkdir(parents=True,exist_ok=True)
    if rows:
        fields=list(rows[0])
        with (root/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    commit(root/'summary.json',dict(completed_runs=len(rows),runs=rows))
    plots=_plots(root,rows)
    report=project/'reports/STREAMING_EXPERIMENT_RESULTS.md'
    old=project/'reports/STREAMING_EXPERIMENT_PREFLIGHT_10S_SOURCE.md'
    if not old.exists() and report.exists() and report.read_text().startswith('# Task 4 — Multi-object streaming experiment status'):
        shutil.copyfile(report,old)
    stats=state.get('workload',{})
    elapsed=state.get('elapsed_seconds',0)
    manifest=read_json(root/'manifest.json') if (root/'manifest.json').exists() else {}
    cfg=manifest.get('config',{})
    feasibility=[]
    representations=manifest.get('representations',{})
    sources=cfg.get('sources',[])
    if representations and sources:
        for n in (3,5,7,8,10,12,15,20):
            costs=[sum(representations[f'{sources[i%len(sources)]}/{g}/40']['physical_bytes']
                       for i in range(n)) for g in range(3)]
            feasibility.append(dict(N=n,q0_bytes_per_source_loop=sum(costs),
                average_q0_mbps=sum(costs)*8/6/1e6,peak_gop_q0_mbps=max(costs)*8/2/1e6))
        commit(root/'q0_feasibility.json',feasibility)
    text=[ '# Streaming experiments — six-second source', '',
        f"Updated {datetime.now(timezone.utc).isoformat()}. Status: **{state['status']}**. "
        f"Validated matrix results: **{len(rows)}/{state['total_runs']}**. "
        f"N=2 six-second smoke policy runs: {state.get('smoke_completed_runs',0)}/5. "
        f"Failed execution attempts: {state.get('failed_runs',0)}.", '',
        'This report supersedes the earlier ten-second/four-source preflight. Its original '
        'measurements remain in `STREAMING_EXPERIMENT_PREFLIGHT_10S_SOURCE.md`.', '',
        '## Frozen setup and runtime integration', '',
        'Sources: Longdress and Soldier, first 180 frames (GOP0–2), 30 FPS, six seconds. '
        'Playback durations: 10/20/30 s. N=3/5/7/8/10/12/15; camera rotation, exact line placement '
        'and seeded random spatial placement. All five original utilities and the native solver '
        'are unchanged. Lumos5G HV starts at original offset 18 s, retaining all measured rates '
        'and a 102-second service horizon. Each policy starts from the same causal 80 Mbps '
        'bootstrap estimate, safety 0.9, 20 ms per-resource request latency and serial FIFO.', '',
        'Every global GOP invokes the runtime Visibility Analyzer on the complete upcoming '
        'ground-truth 30-FPS trajectory and CRF40 native states. Exact repeated physical inputs '
        'reuse experimental caches; original analysis, current computation, native pipeline and '
        'cache lookup latencies are separately exported. No permanent offline visibility '
        'content metadata is required. Camera trace and content repeat every six seconds; '
        'global timestamps, new GOP/request identities, estimator and buffer remain continuous.', '',
        'Instances alternate the two sources, have distinct spatial origins and unique IDs. '
        'Positive scales and rotations preserve the native decoder’s exact recorded local eye '
        'while placing the origins on a line, an ellipse, or seeded separated 3D positions. '
        'The eye is fixed and its orientation changes; no zoom or translating unsupported orbit '
        'is used. These scene transforms are frozen identically across methods and durations.', '',
        'Inter-instance payload deduplication and downloaded-segment residency are disabled. '
        'Transfer aliases namespace each physical source SHA by instance. Every loop requests '
        'the entire selected source-GOP representation again. Rendering/metadata cache hits '
        'never reduce network bytes. Raw requests include instance and original physical SHA.', '',
        '## Base representation bandwidth feasibility', '',
        'Actual manifest-plus-component sizes are counted independently for every instance. '
        'These Q0 payload rates exclude request latency and enhancements; N=20 is an estimate only.', '',
        '| N | Average Q0 Mbps | Peak GOP Q0 Mbps |', '| ---: | ---: | ---: |']
    text += [f"| {r['N']} | {r['average_q0_mbps']:.3f} | {r['peak_gop_q0_mbps']:.3f} |" for r in feasibility]
    text += ['', '## Actual quality protocol', '',
        'Only selected allocations and their joint CRF20 references are rendered. Primary '
        'viewport PSNR evaluates all 640×480 RGB pixels, black background, clipped [0,1], '
        'against a same-camera/frame/transform joint CRF20 reference. This fixed full-viewport '
        'protocol also measures empty views. Reference-alpha foreground metrics are retained '
        'as supplementary scalar cache records where the reference mask is nonempty; they '
        'are not pooled with the primary metric. Offline PSNR/MSE/SSE utilities never supply '
        'frame-quality values. Exact-reference frames have +∞ PSNR, preserved by flags; '
        'pooled PSNR sums RGB SSE/count, and is the main comparison statistic. Finite-only '
        'time plots leave ∞ frames as gaps rather than drawing invented values.', '',
        '## Coverage and measurements', '',
        '| Scenario | N | 10 s | 20 s | 30 s |', '| --- | ---: | ---: | ---: | ---: |']
    for family in cfg.get('scenarios',[]):
        for n in cfg.get('object_counts',[]):
            counts=[sum(r['scenario']==family and r['N']==n and r['duration_s']==d for r in rows) for d in (10,20,30)]
            text.append(f"| {family} | {n} | {counts[0]}/5 | {counts[1]}/5 | {counts[2]}/5 |")
    text+=['','All completed runs, including unfavorable stalls, fallbacks and network failures, '
        f"are listed in `{root.relative_to(project)}/summary.csv` and `summary.json`. "
        'Per-run files contain full request, decision, buffer, frame and latency time series. '
        'A completion marker verifies all JSON/CSV checksums. Partial quality runs have no '
        'completion marker and retain their network checkpoint and available frame metrics.', '',
        '| Scenario | N | Duration | Method | Pooled PSNR | Delivered MB | Rebuffer s | Q0 fallbacks |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    # Show every N=3 policy/duration and the largest available N as a compact report;
    # detailed CSV above contains every other completed result.
    largest=max([r['N'] for r in rows],default=3)
    for r in rows:
        if r['N'] not in {3,largest}:
            continue
        psnr='∞' if r['pooled_psnr_infinite'] else ('unavailable' if r['pooled_psnr_db'] is None else f"{r['pooled_psnr_db']:.3f}")
        text.append(f"| {r['scenario']} | {r['N']} | {r['duration_s']} | {LABELS[r['method']]} | {psnr} | "
                    f"{r['delivered_bytes']/1e6:.3f} | {r['rebuffering_s']:.3f} | {r['q0_fallbacks']} |")
    partial=[]
    for metrics_path in sorted((root/'matrix').glob('*/*/*/*/metrics.json')):
        if not (metrics_path.parent/'completion.json').exists():
            metric=read_json(metrics_path)
            partial.append(dict(run=str(metrics_path.parent.relative_to(root)),status=metric['status'],
                                measured_frames=metric['quality']['measured_frame_count']))
    commit(root/'partial_runs.json',partial)
    if partial:
        text+=['', 'Partial runs retained without a completion marker:', '']
        text += [f"- `{r['run']}`: {r['status']}, {r['measured_frames']} actual measured frames." for r in partial]
    text+=['','## Progress, resource use and ETA','',
        f"Current configuration: `{state.get('current')}`. Stop reason: `{state.get('stop_reason','none')}`. "
        f"Elapsed invocation wall time: {elapsed:.1f} s. "
        f"Native GPU pipeline wall time (decode/reconstruction/rasterization): {stats.get('native_wall_seconds',0):.1f} s. "
        f"Completed native jobs: {stats.get('native_jobs_completed',0)}, failed native jobs: "
        f"{stats.get('native_jobs_failed',0)}. Image jobs: {stats.get('image_jobs',0)}, visibility jobs: "
        f"{stats.get('visibility_jobs',0)}. Exact metric cache hits: {stats.get('metric_cache_hits',0)}. "
        f"Output disk usage: {state.get('disk_bytes',0)/1024**3:.3f} GiB.", '',
        f"Operational limits for this invocation: `{state.get('limits')}`. Limits are checked "
        'before each native job and frame lookup. A current native job may finish before '
        'the next check; its validated result is retained. The estimate is updated from '
        'the complete five-policy N=2 smoke workload and measured per-instance native costs. '
        f"Remaining native ETA: `{state.get('estimated_remaining_native_seconds')}` seconds "
        '(approximate; selection diversity and N scaling can change it). Pure GPU kernel time '
        'is not claimed; native wall time includes CPU decoding and synchronous device work.', '',
        'The configured 20 ms latency is charged to every manifest/component request. '
        'At high N, serialized request latency can dominate throughput and cause stalls '
        'despite high measured 5G capacity. Those outcomes remain in the results. Decision '
        'CPU plots exclude visibility preparation; visibility computation and lookup curves '
        'are separate, avoiding comparisons between native work and cached work as if equal.', '',
        '## Tables, plots and reproduction','',
        f"Figures: `{root.relative_to(project)}/figures/` (PNG and PDF). {len(plots)} plot groups "
        'were available for this update. The figures compare pooled PSNR, delivered bytes, '
        'rebuffering, switches, decision/visibility latency against N for each duration, '
        'plus per-frame quality and buffer time series for each available scene/N/duration.', '',
        '```bash', 'bash scripts/run_streaming_experiments.sh',
        '# The same command resumes validated results and exact frame caches.',
        '# Resource limits can be changed explicitly for a later invocation:',
        'bash scripts/run_streaming_experiments.sh --max-wall-seconds 3600 --max-native-seconds 3000 --max-storage-gib 32',
        'tmux attach -t streaming_experiments', '```', '',
        'Configuration: `experiments/streaming_evaluation/configs/runtime_six_second.json`. '
        f"Progress and persistent logs: `{root.relative_to(project)}/progress.json` and `logs/execution.log`. "
        'Runtime modules reuse the existing renderer, Visibility Analyzer validation/aggregation, '
        'five-method AdaptationPlanner, solver, causal network link, PlaybackBuffer and quality '
        'metric collectors. New code supplies runtime binding, exact experimental caches, '
        'resource limits, checkpointed execution and plots.']
    report.write_text('\n'.join(text)+'\n')

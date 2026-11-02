"""Report streaming coverage separately from measured native-quality coverage."""
import csv
from pathlib import Path

from experiments.visibility_weighted_sse.common import read_json
from .runtime_matrix import PROJECT
from .runtime_results import make_report


def update_report(root, state=None):
    root=Path(root).resolve()
    if state is None:
        state=read_json(root/'progress.json')
    make_report(root,PROJECT,state)
    markers=sorted((root/'matrix').glob('*/*/*/*/phase_a/completion.json'))
    quality=list((root/'matrix').glob('*/*/*/*/completion.json'))
    manifest_path=root/'manifest.json'
    cfg=read_json(manifest_path).get('config',{}) if manifest_path.exists() else {}
    visibility_description=('checksum-verified native visibility reused before the transport clock' if
        cfg.get('visibility_cache_from') else 'the runtime Visibility Analyzer')
    rows=[]
    for marker in markers:
        phase=marker.parent;folder=phase.parent
        family,n,duration,method=folder.relative_to(root/'matrix').parts
        m=read_json(phase/'metrics.json');p=read_json(phase/'performance.json')
        rows.append(dict(scenario=family,N=int(n[1:]),duration_s=int(duration[:-1]),method=method,
            status=m['status'],played_duration_s=m['playback']['played_duration_s'],
            requested_bytes=m['bandwidth']['total_requested_bytes'],
            delivered_bytes=m['bandwidth']['total_delivered_bytes'],
            startup_s=m['playback']['startup_delay_s'],rebuffering_s=m['playback']['total_rebuffering_s'],
            stall_count=m['playback']['stall_count'],completion_s=m['playback']['completion_wall_time_s'],
            q0_fallbacks=m['adaptation']['q0_fallback_count'],
            budget_violations=m['adaptation']['budget_violation_count'],
            decision_ms=p['summary']['decision_compute_excluding_preparation_ms']['mean_ms'],
            played_frames=read_json(marker)['frame_count']))
    if rows:
        with (root/'streaming_summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plan=root/'phase_b_jobs.json'
    jobs=read_json(plan) if plan.exists() else None
    text=['## Two-phase execution', '',
        f"Phase A streaming checkpoints: **{len(markers)}/{state['total_runs']}**; "
        f"Phase B native-quality-complete runs: **{len(quality)}/{state['total_runs']}**. "
        f"Current phase: **{state.get('phase','A')}**, status **{state['status']}**.", '',
        'The former interleaved runner was interrupted with SIGINT; its finally block flushed pending '
        'lossless images and released the GPU. The migration preserves every completed run and image. '
        'Existing validated simulations are imported from their checksum-verified checkpoints; '
        'their original decisions and computation timings are retained. No completed simulation is repeated.', '',
        f"**Phase A:** execute the configured {state['total_runs']}-run matrix with {visibility_description}, "
        'the unchanged five planners, causal network estimation, independent per-instance/per-loop '
        'requests and continuous buffer/playback state. Native joint quality is deferred. '
        '`matrix/<scenario>/Nxx/<duration>/<method>/phase_a/` stores simulation, metrics, original '
        'performance, requests/buffer/decisions CSVs and `played_frames.json`. The latter retains '
        'every actually played 30-FPS frame, global GOP/request/loop identities, source frame time '
        'and independent selected-quality vector. Its camera sample resolves in the checksum-pinned '
        'parent `trace.json`, including exact camera matrices, per-instance transforms, source identity '
        'and source conditioning. `configuration.json` pins the scene; simulation retains download '
        'completion, startup, stalls and media/wall-time playback state. Missing PSNR remains unavailable.', '',
        '**Phase B:** collect all Phase A frames into `phase_b_jobs.json`, including frames belonging '
        'to previously completed quality runs. Its scalar keys are the existing exact candidate-render '
        'identity plus reference/protocol identity. Each render key includes the original validated '
        'source/artifact namespace, full camera, source frame/GOP, ordered instances/transforms, '
        'source conditioning and full quality vector. Global timestamps, method and loop are rebound '
        'only after this exact physical match; the job manifest preserves every run/frame binding. '
        'References are actual joint all-CRF20 images. Jobs run source-GOP first, then scene/frame/vector; '
        'eight resident models cover both sources and all four qualities within a GOP. The existing '
        'bounded CPU lossless writer overlaps compression with one CUDA caller. No unsupported GPU '
        'multi-view batching or independent-object image substitution is introduced.', '',
        'Cached joint RGB/alpha remain original-precision compressed NPZ arrays, with immutable payload '
        'SHA markers and adjacent exact SSE/PSNR/sample-count/render-identity sidecars. Existing images '
        'and scalar records are validated and reused. Network requests never consult image cache hits. '
        'After global jobs finish, cache-only merges restore chronological full-frame quality results; '
        'previous completed result files and completion hashes are left byte-for-byte intact. '
        'Rendering times belong to the global GPU phase; per-run merge events explicitly record cache '
        'lookup time rather than claiming a new render.', '',
        '`phase_a_progress.json` and `phase_b_progress.json` retain separate elapsed time, counts and '
        'ETA; `progress.json` identifies the active phase. ETA is an observed wall-time extrapolation '
        'and is unknown before work completes. Native/image-write/cache counters and storage usage '
        'remain in workload/progress. A single exclusive writer lock covers both phases. '
        'SIGINT and resource limits flush image writes, preserve atomic checkpoints and exit safely. '
        'The next invocation imports all finished streaming runs and reuses all valid images. '
        'Phase B is refused until every configured streaming checkpoint exists.', '',
        'Commands: `bash scripts/run_streaming_experiments.sh` runs A then B sequentially. '
        'Use `--phase A` for streaming only or `--phase B` for the globally deduplicated quality pass '
        'after A. Existing `--max-wall-seconds`, `--max-native-seconds` and `--max-storage-gib` '
        'execution caps remain available. The scientific configuration and validated renderer are unchanged.', '']
    if jobs:
        text += [f"Global plan: {jobs['frame_occurrences']} played frame occurrences, "
                 f"{jobs['unique_quality_jobs']} unique frame-quality jobs, "
                 f"{jobs['unique_images']} unique joint/reference images.", '']
    validation=root/'two_phase_validation.json'
    if validation.exists():
        checked=read_json(validation)
        text += [f"Two-phase validation: **{checked['passed']} passed, "
            f"{checked['subtests_passed']} subtests passed**. Regression coverage includes all five methods "
            'with no Phase A joint rendering, checksum-verified checkpoint imports, full 30-FPS contexts, '
            'exact loop/duration/method deduplication, changed-pose/vector/transform rejection, deterministic '
            'job-plan resume, source-GOP grouping and SSE-based merge without changing playback or transfers. '
            'Command: `'+checked['command']+'`.', '']
    stats=state.get('workload',{}).get('execution_revisions',{}).get('two_phase_v1',{})
    seconds=stats.get('image_native_wall_seconds',0)
    if seconds:
        text += [f"Observed Phase B native pipeline: {stats['image_jobs']} new joint images / "
            f"{seconds:.3f} s = {stats['image_jobs']/seconds:.3f} images/s "
            '(includes reconstruction and GPU rendering; excludes cached images and separate compression time).', '']
    report=root/'REPORT.md'
    original=report.read_text()
    first,rest=original.split('\n',1)
    report.write_text(first+'\n\n'+'\n'.join(text)+rest)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    update_report(parser.parse_args().root)

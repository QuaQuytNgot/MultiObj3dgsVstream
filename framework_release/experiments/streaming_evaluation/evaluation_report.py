"""Incremental report/CSV for terminal timeouts, real streaming and native PSNR."""
from collections import Counter
import csv
from pathlib import Path
import statistics
import time

from experiments.visibility_weighted_sse.common import read_json
from .runtime_native import commit

LABELS=dict(NAVA='NAVA',VISIBILITY_PSNR='Visibility × PSNR',VISIBILITY_MSE='Visibility × MSE',
            VISIBILITY_WEIGHTED_SSE='Visibility × SSE',EXHAUSTIVE_SEARCH='Exhaustive Search')
NA={'TIMEOUT','MEMORY_LIMIT','INFEASIBLE','ERROR'}


def collect_rows(root):
    plan=root/'run_plan.json'
    if not plan.exists():return []
    rows=[]
    for case in read_json(plan)['cases']:
        folder=root/'matrix'/case['scenario']/f"N{case['N']:02d}"/f"{case['duration_s']:02d}s"/case['method']
        terminal=read_json(folder/'terminal.json') if (folder/'terminal.json').exists() else {}
        phase=folder/'phase_a';status=terminal.get('status','PENDING')
        row=dict(**case,status=status,quality_status='NA' if status in NA else 'PENDING',
            startup_s=None,rebuffering_s=None,delivered_bytes=None,body_mbps=None,goodput_mbps=None,
            q0_fallbacks=None,utility_sum=None,solver_median_ms=None,solver_peak_rss_mib=None,
            dp_cells_max=None,selected_crf_counts=None,pooled_psnr_db=None,pooled_psnr_infinite=None,
            measured_frames=None,displayed_frames=None,measurement_coverage=None)
        if status!='PENDING' and (phase/'utility_solver_summary.json').exists():
            records=read_json(phase/'utility_solver_summary.json')['records']
            timings=[r['timings'] for r in records]
            values=[t['solver_wall_ms'] for t in timings if t.get('solver_wall_ms') is not None]
            rss=[t['process_peak_rss_kib'] for t in timings if t.get('process_peak_rss_kib') is not None]
            row['solver_median_ms']=statistics.median(values) if values else None
            row['solver_peak_rss_mib']=max(rss)/1024 if rss else None
            row['dp_cells_max']=max((r['diagnostics'].get('native_dp_cells_max',0) for r in records),default=0)
            if status not in NA:
                utilities=[r['utility'] for r in records if r['utility'] is not None]
                row['utility_sum']=sum(utilities) if utilities else None
                row['selected_crf_counts']=dict(Counter(q for r in records for q in (r.get('selected_quality') or {}).values()))
        if status not in NA and (phase/'metrics.json').exists():
            m=read_json(phase/'metrics.json');b=m['bandwidth'];p=m['playback'];a=m['adaptation']
            row.update(startup_s=p['startup_delay_s'],rebuffering_s=p['total_rebuffering_s'],
                delivered_bytes=b['total_delivered_bytes'],body_mbps=(b.get('body_throughput_bps') or 0)/1e6,
                goodput_mbps=(b.get('application_goodput_bps') or 0)/1e6,q0_fallbacks=a['q0_fallback_count'],
                displayed_frames=read_json(phase/'completion.json')['frame_count'])
        if (folder/'completion.json').exists() and status not in NA:
            q=read_json(folder/'metrics.json')['quality']
            row.update(quality_status='COMPLETE',pooled_psnr_db=q['pooled_psnr_db'],
                pooled_psnr_infinite=q['pooled_psnr_infinite'],measured_frames=q['measured_frame_count'],
                measurement_coverage=q['measurement_coverage'])
        elif (folder/'phase_b_status.json').exists():
            row['quality_status']=read_json(folder/'phase_b_status.json')['status']
        rows.append(row)
    return rows


def update_report(root,state,report_path):
    root=Path(root);rows=collect_rows(root)
    if rows:
        temporary=root/'summary.csv.tmp'
        with temporary.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader()
            for row in rows:
                record=dict(row)
                for key in ('selected_crf_counts',):
                    if record[key] is not None:
                        import json
                        record[key]=json.dumps(record[key],sort_keys=True)
                writer.writerow(record)
        temporary.replace(root/'summary.csv')
    counts=Counter(r['status'] for r in rows);quality=Counter(r['quality_status'] for r in rows)
    commit(root/'summary.json',dict(schema='streaming-http2-dp015-summary.v1',rows=rows,
        phase_a_status_counts=dict(counts),phase_b_status_counts=dict(quality)))
    lines=['# Real HTTP/2 streaming evaluation — DP Δ = 0.15 Mbps','',
        f"Updated {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}. Run **{state.get('run_kind')}**; "
        f"status **{state['status']}**, phase **{state.get('phase','A')}**.", '',
        f"Cases terminal: **{len(rows)-counts['PENDING']}/{len(rows)}**; Phase A statuses `{dict(counts)}`; "
        f"rendered-quality statuses `{dict(quality)}`. Active: `{state.get('current')}`.", '',
        f'Output: `{root}`. Checkpoints, raw requests/decisions/buffer/solver diagnostics, display records, '
        'per-frame PSNR CSV/JSON, lossless image/metric caches and logs remain versioned here.', '',
        'The full matrix is N=3/5/7/10/12/15/20 × camera_rotation/long_line/random_spatial × '
        '10/20/30 s × exactly NAVA, Visibility × PSNR, Visibility × SSE, Visibility × MSE, Exhaustive Search. '
        'Smoke mode is explicitly a two-second N=3/N=20 subset and is not a completed matrix.', '',
        'Protocol retains Longdress/Soldier, six-second source, 30 FPS, 640×480, original seeded placements '
        'and recorded camera trajectories, Lumos5G HV offset 18 s (102 s finite service), 80 Mbps causal '
        'bootstrap, safety 0.9, 20 ms per-resource latency, 17 resources/object/GOP and independent '
        'instance/loop transfers. HTTPX negotiates TLS/ALPN h2 against Hypercorn with one persistent '
        'connection per case and 16 multiplexed streams. One aggregate application-body shaper serves '
        'the trace; it is not packet/radio emulation. The measured Python host CPU ceiling still applies.', '',
        'The four utility functions are unchanged. DP uses conservative Δ=150000 bps and exact-byte '
        'feasibility. Exhaustive Search uses the identical Visibility × PSNR utility rows and exact '
        'physical unions, enumerates combinations without DP, and returns a selection only after '
        'complete enumeration. Per-GOP limits are 5 s, 64 MiB extra process RSS, 4 GiB absolute RSS; '
        'no incumbent, DP or Q0 substitution is returned on a limit. TIMEOUT/MEMORY_LIMIT/INFEASIBLE '
        'cases have NA QoE/PSNR, while measured attempt latency/memory remains available. The common '
        '80 Mbps bootstrap may itself make Exhaustive Search infeasible at larger N; this is retained '
        'explicitly rather than relaxing its bandwidth budget. Existing DP Q0 overrun behavior remains '
        'explicitly identified as a non-feasible delivery fallback.', '',
        'Phase A prepares candidate-independent oracle-trace visibility before the media clock, recording '
        'preparation cost separately. Online utility/optimization, requests and playback waits consume '
        'real wall time. TLS setup/warmup is separate. It saves each scheduled displayed frame’s wall/media '
        'timestamp, source frame/GOP, selected CRFs, exact camera intrinsics/extrinsics and all transforms '
        'in phase_a/played_frames.json. Stall holds remain in the continuous buffer intervals. These are '
        'the existing modeled playback display events, not a live interactive renderer.', '',
        'Phase B reads those persisted poses without regenerating trajectories. It renders only selected '
        'joint scenes and corresponding all-CRF20 joint references, uses validated full-viewport clipped '
        'RGB SSE/sample-count PSNR and checksum-verified exact-identity caches. All-CRF20 matches retain '
        '+∞ flags. Pooled PSNR sums SSE/counts. No Phase B image or score feeds adaptation.', '',
        'One writer lock, atomic terminal/phase/image checkpoints, bounded native/GPU/disk/wall resources, '
        'two attempts per failed case/job and report heartbeats support continuation. Repeating the same '
        'command resumes verified completed cases and image caches. Historical results and production '
        'codec/renderer remain untouched. No full-matrix result is claimed before completion.', '',
        '| Scene | N | Duration s | Method | Status | Startup s | Rebuffer s | Body Mbps | Solver ms | PSNR dB | Frames |',
        '|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):return 'NA' if value is None else f'{value:.3f}'
    for row in rows:
        psnr='+∞' if row['pooled_psnr_infinite'] else fmt(row['pooled_psnr_db'])
        lines.append(f"| {row['scenario']} | {row['N']} | {row['duration_s']} | {LABELS[row['method']]} | "
            f"{row['status']}/{row['quality_status']} | {fmt(row['startup_s'])} | {fmt(row['rebuffering_s'])} | "
            f"{fmt(row['body_mbps'])} | {fmt(row['solver_median_ms'])} | {psnr} | "
            f"{row['measured_frames'] if row['measured_frames'] is not None else 'NA'} |")
    if state.get('stop_reason'):lines+=['',f"Stop reason: `{state['stop_reason']}`."]
    lines+=['','Measured process RSS belongs to the experiment process (including preceding native allocations); '
        'it is not isolated solver-only RAM. Solver tables, utility/CRF choices, startup/rebuffering and '
        'throughput are retained in summary.csv/JSON and immutable per-case Phase A artifacts. '
        'Failures have adjacent error JSON; unavailable values remain blank in CSV and null in JSON.','']
    text='\n'.join(lines)
    for path in (root/'REPORT.md',Path(report_path)):
        path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.md.tmp');temp.write_text(text);temp.replace(path)

#!/usr/bin/env python3
"""Export paired native profiles to CSV and standalone paper figures.

Consumes raw artifacts without touching the source/runtime or original reports.
Uses the same offline summary implementation as reanalyze_native.py.
"""
import argparse
import csv
import json
from pathlib import Path

from reanalyze_native import reanalyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profiles = {version: [reanalyze(path.parent) for path in sorted(root.glob('*/summary.json'))]
                for version, root in [('Before', args.before), ('After', args.after)]}
    (args.output/'comparison.json').write_text(json.dumps(profiles, indent=2)+'\n')
    rows = []
    for version, cases in profiles.items():
        for case in cases:
            rows.append(dict(version=version, case=case['case'], frames=case['frames'],
                first_swap_s=case['launch_to_first_swap_s'],
                fresh_fps=case['post_first_frame_fps'],
                compute_p50_ms=case['frame_latency_s']['p50']*1000,
                compute_p95_ms=case['frame_latency_s']['p95']*1000,
                gap_p95_ms=case['frame_interval_s']['p95']*1000,
                gap_max_ms=case['frame_interval_s']['max']*1000,
                completion_misses_strict=case['frame_completion_miss_count_strict'],
                completion_misses_over_1ms=case['frame_completion_miss_count_over_1ms'],
                boundary_excess_mean_ms=case['boundary_excess_s']['mean']*1000,
                boundary_excess_max_ms=case['boundary_excess_s']['max']*1000,
                peak_tree_rss_mib=case['peak_tree_rss_bytes']/1024**2,
                device_peak_mib=case['device_peak_mib'],
                cpu_mean_percent=case['cpu_percent']['mean'],
                gpu_mean_percent=case['gpu_utilization_percent']['mean']))
    with (args.output/'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = {'Before': '#b35a36', 'After': '#247e99'}
    names = [case['case'] for case in profiles['Before']]
    if set(names) != {case['case'] for case in profiles['After']}:
        raise ValueError('Before and after must have identical case names')
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout='constrained')
    for version, shift in [('Before', -.2), ('After', .2)]:
        cases = {case['case']: case for case in profiles[version]}
        for axis, metric in zip(axes, ['launch_to_first_swap_s', 'post_first_frame_fps']):
            bars = axis.bar([i+shift for i in range(len(names))],
                            [cases[name][metric] for name in names], width=.38,
                            label=version, color=colors[version])
            axis.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    for axis in axes:
        axis.set_xticks(range(len(names)), [name.upper() for name in names])
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_ylim(0, axis.get_ylim()[1]*1.12)
    axes[0].set_ylabel('Launch to first native swap (s)')
    axes[1].set_ylabel('Fresh scene completions / s')
    axes[1].axhline(30, ls='--', lw=1, color='0.4', label='30 Hz target')
    axes[0].legend(frameon=False)
    for suffix in ['png', 'pdf']:
        figure.savefig(args.output/f'startup_fps.{suffix}', dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(len(names), 1, figsize=(9, 6), sharex=True,
                                layout='constrained', squeeze=False)
    for axis, name in zip(axes[:, 0], names):
        after = next(case for case in profiles['After'] if case['case'] == name)
        for boundary in after['boundary_intervals']:
            axis.axvline(boundary['frame_id'], color='0.85', lw=.7, zorder=0)
        for version, root in [('Before', args.before), ('After', args.after)]:
            events = [json.loads(line) for line in (root/name/'playback.jsonl').read_text().splitlines()]
            frames = [event for event in events if event.get('stage') == 'scene_frame' and event.get('fresh', True)]
            gaps = [(current['monotonic_s']-previous['monotonic_s'])*1000
                    for previous, current in zip(frames, frames[1:])]
            axis.plot([frame['frame_id'] for frame in frames[1:]], gaps,
                      color=colors[version], lw=.8, label=version)
        axis.axhline(1000/30, color='0.3', ls='--', lw=.7)
        axis.set_yscale('log'); axis.set_ylim(10, 6000)
        axis.set_ylabel(f'{name.upper()} gap (ms)')
        axis.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=2)
    axes[-1, 0].set_xlabel('Source scene presentation frame (0–299)')
    for suffix in ['png', 'pdf']:
        figure.savefig(args.output/f'frame_gaps.{suffix}', dpi=180)
    plt.close(figure)
    print(args.output/'comparison.csv')


if __name__ == '__main__':
    main()

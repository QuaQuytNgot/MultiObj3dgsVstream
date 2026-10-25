#!/usr/bin/env python3
"""Recompute common profile metrics from saved raw events without rerunning GPU.

The original artifact directories remain unchanged. Example:
python benchmarks/streaming/reanalyze_native.py --input output/RUN \
    --output output/RUN_reanalyzed.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from profile_native import summarize


def reanalyze(case):
    def load(name):
        return json.loads((case/name).read_text())
    previous = load('summary.json')
    timing = load('timing.json')
    # Recover the very same parent-process launch clock from its saved delta.
    # Do not replace launch with the later instrumentation-module entry time.
    launch = None
    for clock, delta in (('first_swap_monotonic_s', 'launch_to_first_swap_s'),
                         ('first_render_monotonic_s', 'launch_to_first_rendered_s')):
        if timing.get(clock) is not None and previous.get(delta) is not None:
            launch = timing[clock] - previous[delta]
            origin = f'{clock} minus original {delta}'
            break
    if launch is None:
        raise ValueError(f'{case}: original process launch timing cannot be recovered')
    events = [json.loads(line) for line in (case/'playback.jsonl').read_text().splitlines()]
    mpd = previous.get('mpd', '')
    index_path = Path(mpd).parent/'index.json' if mpd and urlsplit(mpd).scheme not in ('http', 'https') else None
    index = json.loads(index_path.read_text()) if index_path is not None and index_path.is_file() else None
    result = dict(previous)
    invocation=load('invocation.json')
    result.update(summarize(load('playback.json'), events, timing, load('samples.json'), launch, index,
                            scene_loop='--loop' in invocation.get('argv', [])))
    result['offline_reanalysis'] = dict(input_directory=str(case.resolve()), launch_clock_origin=origin,
        original_artifacts_modified=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Matrix root containing case/summary.json')
    parser.add_argument('--output', required=True, type=Path, help='New JSON file; existing files are rejected')
    args = parser.parse_args(argv)
    cases = sorted(path.parent for path in args.input.glob('*/summary.json'))
    if not cases:
        parser.error('--input contains no case/summary.json artifacts')
    rows = [reanalyze(case) for case in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as destination:
        destination.write(json.dumps(rows, indent=2)+'\n')
    for row in rows:
        print(json.dumps({key: row[key] for key in ('case', 'frames',
            'frame_completion_miss_count_strict', 'frame_completion_miss_count_over_1ms',
            'source_mapping_valid')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

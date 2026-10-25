#!/usr/bin/env python3
"""Isolated preparation, native solver timing, validation, and report CLI."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCES = ['brute_force.cpp', 'nava.cpp', 'lts_inspired.cpp', 'proposed_dp.cpp']
FLAGS = ['-std=c++17', '-O3', '-Wall', '-Wextra', '-Wpedantic']


def execute(command, log, **kwargs):
    result = subprocess.run([str(c) for c in command], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    (ROOT / 'logs' / log).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed; see logs/{log}:\n{result.stdout[-3000:]}")
    print(result.stdout.strip(), flush=True)
    return result.stdout


def compile_binaries():
    (ROOT / 'build').mkdir(exist_ok=True)
    source_paths = [ROOT / 'cpp' / s for s in SOURCES]
    for filename, extra in [('benchmark', ROOT / 'cpp' / 'benchmark.cpp'),
                            ('solver_tests', ROOT / 'tests' / 'solver_tests.cpp')]:
        execute(['g++', *FLAGS, '-I', ROOT / 'cpp', *source_paths, extra,
                 '-o', ROOT / 'build' / filename], f'build_{filename}.log')


def prepare(cfg):
    from workloads import prepare as prepare_workloads
    data = (ROOT / cfg['data_root']).resolve()
    cost = (ROOT / cfg['cost_root']).resolve() if cfg.get('cost_root') else None
    prepared = prepare_workloads(data, cost, ROOT / 'inputs',
                                 sizes=cfg['object_counts'], seeds=cfg['seeds'], eta=cfg.get('eta', 0.9))
    print(f"Prepared 80 measured profiles and {prepared['case_count']} common decision instances.", flush=True)


def test():
    execute([ROOT / 'build' / 'solver_tests'], 'solver_tests.log')
    execute([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'], 'python_tests.log')


def run(cfg):
    allowed = sorted(os.sched_getaffinity(0))
    cpu = allowed[0] if cfg.get('cpu') == 'first_allowed' else int(cfg.get('cpu', allowed[0]))
    if cpu not in allowed:
        raise ValueError(f'CPU {cpu} is outside affinity mask {allowed}')
    command = ['taskset', '-c', str(cpu), ROOT / 'build' / 'benchmark',
               ROOT / 'inputs' / 'runner_input.txt', ROOT / 'results' / 'raw.jsonl',
               cfg['warmups'], cfg['repetitions'], cfg['timeout_seconds'],
               cfg['max_combinations'], cfg['max_labels'], cfg['max_dp_cells'],
               ','.join(map(str, cfg['delta_bps'])), cfg['order_seed']]
    sources = [*sorted((ROOT / 'cpp').glob('*')), ROOT / 'metadata.py', ROOT / 'workloads.py', Path(__file__)]
    manifest = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'platform': platform.platform(), 'python': sys.version, 'cpu_affinity': cpu,
                'available_affinity': allowed, 'compiler_flags': FLAGS,
                'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
                'cpu_info': subprocess.check_output(['lscpu'], text=True), 'config': cfg,
                'command': list(map(str, command)),
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'input_sha256': hashlib.sha256((ROOT / 'inputs' / 'runner_input.txt').read_bytes()).hexdigest(),
                'solver_peak_memory': 'not measured; process RSS is not attributed to individual solver calls',
                'timer': 'C++ std::chrono::steady_clock, individual solver calls, nanosecond durations',
                'scope': 'validation, quantization/graph setup, allocation, search, and backtracking inside solver; input parsing, workload construction, post-validation, serialization outside'}
    (ROOT / 'results' / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    execute(command, 'benchmark.log')
    validate_results(cfg)


def load_cases():
    data = json.loads((ROOT / 'inputs' / 'prepared_cases.json').read_text())
    if isinstance(data, dict):
        data = data['cases']
    return {c['id']: c for c in data}


def validate_results(cfg):
    cases = load_cases()
    rows = [json.loads(line) for line in (ROOT / 'results' / 'raw.jsonl').read_text().splitlines()]
    oracle, violations, checks, matches = {}, 0, 0, 0
    for row in rows:
        if row['status'] == 'optimal' and row['algorithm'] in ('Brute-force', 'LTS-inspired', 'NAVA'):
            key = row['case_id']
            if key in oracle and abs(oracle[key] - row['utility']) > 1e-7:
                raise AssertionError(f'exact solver objective mismatch {key}')
            oracle[key] = row['utility']
    for row in rows:
        case = cases[row['case_id']]
        row.update(n=case['n'], matrix_sha256=case['matrix_sha256'])
        row['elapsed_ms'] = row['elapsed_ns'] / 1e6
        selection = row['selection']
        row['feasibility_violation'] = False
        if selection:
            assert len(selection) == case['n']
            assert all(isinstance(k, int) and 0 <= k < 4 for k in selection)
            actual = sum(case['costs_bps'][i][k] for i, k in enumerate(selection))
            utility = sum(case['utilities'][i][k] for i, k in enumerate(selection))
            row['feasibility_violation'] = actual > case['budget_bps']
            violations += int(row['feasibility_violation'])
            assert actual == row['cost_bps'] and abs(utility - row['utility']) < 1e-7
            checks += 1
        row['bandwidth_utilization_pct'] = 100 * row['cost_bps'] / case['budget_bps'] if selection else None
        row['oracle_utility'] = oracle.get(row['case_id'])
        row['optimality_gap_pct'] = None
        if row['status'] == 'optimal' and row['oracle_utility'] is not None:
            gap = row['oracle_utility'] - row['utility']
            if gap < -1e-7:
                raise AssertionError('solver exceeds exact oracle')
            row['optimality_gap_pct'] = max(0, gap) / abs(row['oracle_utility']) * 100 if row['oracle_utility'] else 0
            matches += 1
    assert violations == 0
    fields = list(rows[0])
    with (ROOT / 'results' / 'raw.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    # JSON remains lossless, including selections, null gaps, and all warmup/timeout samples.
    (ROOT / 'results' / 'validated_results.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    report = {'cases': len(cases), 'raw_samples_including_warmups': len(rows),
              'selection_checks': checks, 'feasibility_violations': violations,
              'cases_with_exact_oracle': len(oracle), 'samples_with_exact_gap': matches,
              'same_input_interface': 'all C++ solver jobs receive same const Instance reference',
              'status': 'PASS'}
    (ROOT / 'results' / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['all', 'prepare', 'build', 'test', 'run', 'report'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'default.json')
    parser.add_argument('--data-root')
    parser.add_argument('--cost-root')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    for field in ['data_root', 'cost_root']:
        if getattr(args, field): cfg[field] = str(Path(getattr(args, field)).resolve())
    for d in ['logs', 'inputs', 'results', 'figures', 'reports']: (ROOT / d).mkdir(exist_ok=True)
    if args.command in ['all', 'prepare']: prepare(cfg)
    if args.command in ['all', 'build']: compile_binaries()
    if args.command in ['all', 'test']: test()
    if args.command in ['all', 'run']: run(cfg)
    if args.command in ['all', 'report']:
        from reporting import generate_report
        generate_report(ROOT)


if __name__ == '__main__':
    main()

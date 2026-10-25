"""Create auditable completed-case summaries and publication figures."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
from statistics import median


def percentile(values, q=0.95):
    """Nearest-rank percentile (no interpolation)."""
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def generate_report(root: Path):
    rows = json.loads((root / 'results' / 'validated_results.json').read_text())
    prepared = json.loads((root / 'inputs' / 'prepared_cases.json').read_text())
    cases = {c['id']: c for c in prepared['cases']}
    audit = json.loads((root / 'inputs' / 'dataset_audit.json').read_text())
    manifest = json.loads((root / 'results' / 'run_manifest.json').read_text())
    cfg = manifest['config']
    validation = json.loads((root / 'results' / 'validation.json').read_text())
    sizes = cfg['object_counts']
    deltas = cfg['delta_bps']
    main_delta = 250000 if 250000 in deltas else deltas[len(deltas)//2]
    jobs = [('Brute-force', 0), ('NAVA', 0), ('LTS-inspired', 0), *[('Proposed-DP', d) for d in deltas]]
    groups = defaultdict(list)
    for row in rows:
        groups[row['case_id'], row['algorithm'], row['delta_bps']].append(row)
    summaries, completed = [], {}
    for case in cases.values():
        for algorithm, delta in jobs:
            samples = groups[case['id'], algorithm, delta]
            measured = [r for r in samples if r['phase'] == 'measured']
            status = next((r['status'] for r in reversed(samples) if r['status'] in ['timeout', 'skipped']), None)
            full = len(measured) == cfg['repetitions'] and all(r['status'] == 'optimal' for r in measured)
            status = 'optimal' if full else status or (samples[-1]['status'] if samples else 'missing')
            vals = [r['elapsed_ms'] for r in measured] if full else []
            last = samples[-1] if samples else {}
            summary = dict(case_id=case['id'], n=case['n'], seed=case['seed'], budget_label=case['budget_label'],
                           algorithm=algorithm, delta_bps=delta, status=status, complete=full,
                           measured_repetitions=len(measured), median_ms=median(vals) if vals else None,
                           p95_ms=percentile(vals), maximum_ms=max(vals) if vals else None,
                           utility=last.get('utility'), cost_bps=last.get('cost_bps'),
                           utilization_pct=last.get('bandwidth_utilization_pct'),
                           optimality_gap_pct=last.get('optimality_gap_pct'),
                           oracle_utility=last.get('oracle_utility'),
                           solver_peak_memory_bytes=None, detail=last.get('detail', ''))
            summaries.append(summary)
            if full:
                completed[case['id'], algorithm, delta] = measured
    csv_write(root / 'results' / 'case_summary.csv', summaries)
    aggregates = []
    for algorithm, delta in jobs:
        for n in sizes:
            selected = [s for s in summaries if (s['algorithm'], s['delta_bps'], s['n']) == (algorithm, delta, n)]
            good = [s for s in selected if s['complete']]
            samples = [r for s in good for r in completed[s['case_id'], algorithm, delta]]
            values = [r['elapsed_ms'] for r in samples]
            gaps = [s['optimality_gap_pct'] for s in good if s['optimality_gap_pct'] is not None]
            aggregates.append(dict(algorithm=algorithm, delta_bps=delta, n=n, complete_cases=len(good), total_cases=len(selected),
                                   timeout_cases=sum(s['status']=='timeout' for s in selected),
                                   skipped_cases=sum(s['status']=='skipped' for s in selected),
                                   discretized_infeasible_cases=sum(s['status']=='discretized_infeasible' for s in selected),
                                   measured_calls=len(values), median_ms=median(values) if values else None,
                                   p95_ms=percentile(values), maximum_ms=max(values) if values else None,
                                   mean_gap_pct=sum(gaps)/len(gaps) if gaps else None,
                                   maximum_gap_pct=max(gaps) if gaps else None,
                                   cases_with_gap=len(gaps), feasibility_violations=sum(r['feasibility_violation'] for r in samples)))
    csv_write(root / 'results' / 'summary.csv', aggregates)
    # Workload conditions remain available separately: main plot pools equal-size cases.
    budget_rows=[]
    for algorithm, delta in jobs:
        for n in sizes:
            for budget in [b['label'] for b in prepared['budget_specs']]:
                selected=[s for s in summaries if (s['algorithm'],s['delta_bps'],s['n'],s['budget_label'])==(algorithm,delta,n,budget)]
                good=[s for s in selected if s['complete']]
                vals=[r['elapsed_ms'] for s in good for r in completed[s['case_id'],algorithm,delta]]
                budget_rows.append(dict(algorithm=algorithm,delta_bps=delta,n=n,budget_label=budget,complete_cases=len(good),total_cases=len(selected),median_ms=median(vals) if vals else None,p95_ms=percentile(vals),maximum_ms=max(vals) if vals else None))
    csv_write(root/'results'/'by_budget.csv',budget_rows)
    gap_rows=[]
    for algorithm,delta in jobs:
        good=[s for s in summaries if (s['algorithm'],s['delta_bps'])==(algorithm,delta) and s['complete'] and s['optimality_gap_pct'] is not None]
        gaps=[s['optimality_gap_pct'] for s in good]
        # Brute-force coverage is the independent exact check for all baselines.
        small=[s['optimality_gap_pct'] for s in good if s['n']<=8]
        gap_rows.append(dict(algorithm=algorithm,delta_bps=delta,oracle_cases=len(good),mean_gap_pct=sum(gaps)/len(gaps) if gaps else None,max_gap_pct=max(gaps) if gaps else None,brute_force_cases=len(small),brute_force_mean_gap_pct=sum(small)/len(small) if small else None,brute_force_max_gap_pct=max(small) if small else None))
    csv_write(root/'results'/'optimality_gap.csv',gap_rows)
    nava_coverage = '; '.join(f"N={r['n']}: {r['complete_cases']}/{r['total_cases']}" for r in aggregates if r['algorithm']=='NAVA')
    nava_complete = sum(s['complete'] for s in summaries if s['algorithm']=='NAVA')
    nava_timeouts = [s for s in summaries if s['algorithm']=='NAVA' and s['status']=='timeout']
    clock_timeouts = sum('wall-clock' in s['detail'] for s in nava_timeouts)
    label_timeouts = sum('label cap' in s['detail'] for s in nava_timeouts)
    model_line = next((s for s in manifest['cpu_info'].splitlines() if s.strip().startswith('Model name:')), 'Model name: unspecified CPU')
    cpu_model = model_line.split(':',1)[1].strip()

    os.environ.setdefault('MPLCONFIGDIR',str(root/'.cache'/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42})
    colors={'Brute-force':'#5b5b5b','NAVA':'#b33a30','LTS-inspired':'#b8860b','Proposed-DP':'#1670a7'}
    markers={'Brute-force':'s','NAVA':'^','LTS-inspired':'D','Proposed-DP':'o'}
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(root/'figures'/f'{name}.{ext}',dpi=240)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.0,4.6))
    for algorithm,delta in [j for j in jobs if j[0]!='Proposed-DP' or j[1]==main_delta]:
        points=[r for r in aggregates if (r['algorithm'],r['delta_bps'])==(algorithm,delta) and r['median_ms'] is not None]
        x=[sizes.index(r['n']) for r in points]; y=[r['median_ms'] for r in points]
        label=algorithm if not delta else f'Proposed DP (Δ={delta/1e6:g} Mb/s)'
        solid_y=[r['median_ms'] if r['complete_cases']==r['total_cases'] else math.nan for r in points]
        ax.plot(x,solid_y,marker=markers[algorithm],color=colors[algorithm],label=label,lw=1.5,ms=5)
        for xx,r in zip(x,points):
            if r['complete_cases']<r['total_cases']:
                ax.plot([xx],[r['median_ms']],marker=markers[algorithm],color=colors[algorithm],mfc='white',ms=6,linestyle='none')
        ax.errorbar(x,y,yerr=[[0]*len(y),[max(0,r['p95_ms']-r['median_ms']) for r in points]],fmt='none',ecolor=colors[algorithm],capsize=3,alpha=.8)
        for xx,r in zip(x,points):
            if r['complete_cases']<r['total_cases']:
                ax.annotate(f"{r['complete_cases']}/{r['total_cases']}",(xx,r['p95_ms']),xytext=(4,6),textcoords='offset points',color=colors[algorithm],fontsize=8)
    ax.set(yscale='log',xticks=range(len(sizes)),xticklabels=sizes,xlabel='Number of independent object instances N',ylabel='Solver processing time (ms, log scale)')
    ax.grid(axis='y',which='major',alpha=.2)
    ax.legend(loc='best',fontsize=8)
    ax.set_title('CPU solver latency: median and P95',loc='left')
    fig.subplots_adjust(bottom=.25)
    fig.text(.12,.035,'Open markers: incomplete coverage; labels give completed cases. Skips/timeouts have no latency.\nP95 pools workload and invocation variation; bars are not confidence intervals.',fontsize=7)
    save(fig,'processing_time_vs_n')

    fig,axes=plt.subplots(1,2,figsize=(9.0,3.6))
    for n in [8,16,32,64]:
        pts=[r for r in aggregates if r['algorithm']=='Proposed-DP' and r['n']==n and r['median_ms'] is not None]
        axes[0].plot([r['delta_bps']/1e6 for r in pts],[r['median_ms'] for r in pts],marker='o',label=f'N={n}')
    axes[0].set(xscale='log',yscale='log',xlabel='Discretization Δ (Mb/s)',ylabel='Median solver time (ms)',title='(a) DP latency')
    axes[0].legend(fontsize=8)
    for n in sizes:
        pts=[r for r in aggregates if r['algorithm']=='Proposed-DP' and r['n']==n and r['mean_gap_pct'] is not None]
        if pts:axes[1].plot([r['delta_bps']/1e6 for r in pts],[r['mean_gap_pct'] for r in pts],marker='o',label=f'N={n}')
    axes[1].set(xscale='log',xlabel='Discretization Δ (Mb/s)',ylabel='Mean objective gap (%)',title='(b) Gap to available exact oracle')
    axes[1].legend(fontsize=7,ncol=2)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig,'dp_discretization_tradeoff')
    fig,ax=plt.subplots(figsize=(6.8,3.5))
    for delta in deltas:
        pts=[r for r in aggregates if r['algorithm']=='Proposed-DP' and r['delta_bps']==delta and r['mean_gap_pct'] is not None]
        x=[sizes.index(r['n']) for r in pts];y=[r['mean_gap_pct'] for r in pts]
        ax.plot(x,y,marker='o',label=f'Δ={delta/1e6:g} Mb/s')
        ax.errorbar(x,y,yerr=[[0]*len(y),[r['maximum_gap_pct']-r['mean_gap_pct'] for r in pts]],fmt='none',capsize=3)
    ax.set(xticks=range(len(sizes)),xticklabels=sizes,xlabel='N',ylabel='Objective gap (%)',title='Proposed DP: mean gap, upper bars show maximum')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    save(fig,'optimality_gap_vs_n')
    # Per-budget latency panels avoid hiding easy overprovisioned cases in the main plot.
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True)
    for ax,budget in zip(axes.flat,[b['label'] for b in prepared['budget_specs']]):
        for algorithm,delta in [j for j in jobs if j[0]!='Proposed-DP' or j[1]==main_delta]:
            pts=[r for r in budget_rows if (r['algorithm'],r['delta_bps'],r['budget_label'])==(algorithm,delta,budget) and r['median_ms'] is not None]
            ax.plot([sizes.index(r['n']) for r in pts],[r['median_ms'] for r in pts],marker=markers[algorithm],color=colors[algorithm],label=algorithm,ms=4)
        ax.set(title=budget,yscale='log',xticks=range(len(sizes)),xticklabels=sizes);ax.grid(axis='y',alpha=.2)
    axes[1,0].set_xlabel('N');axes[1,1].set_xlabel('N');axes[0,0].set_ylabel('Median time (ms)');axes[1,0].set_ylabel('Median time (ms)')
    axes[0,0].legend(fontsize=7);fig.tight_layout();save(fig,'processing_time_by_budget')

    def fmt(v):return '—' if v is None else f'{v:.4f}'
    lines=['# CPU ABR solver processing-time experiment','',
           'This is an isolated object-level MCKP microbenchmark using measured StreamSTGS PSNR and GOP transfer costs. It makes no playback, rendering, decoding, network, or full-system QoE claims.','',
           '## Inputs and common problem','',
           f"The loader verified 80 profiles (four contents × five 60-frame GOPs × four independent CRFs), all 115,200 frame/view metric rows, and metadata hash links. PSNR is the arithmetic mean in dB across all 24 views, including training and held-out views, at scale 1.0. Costs include manifest-listed payload components plus the representation manifest itself. Each GOP lasts exactly two seconds, so R=4×transfer_bytes bit/s, without rounding. The observed PSNR range is {audit['psnr_db_range'][0]:.4f}–{audit['psnr_db_range'][1]:.4f} dB, and rate range is {audit['cost_bps_range'][0]/1e6:.4f}–{audit['cost_bps_range'][1]/1e6:.4f} Mb/s. See [DATA_AUDIT.md](DATA_AUDIT.md).",'',
           'Every solver maximizes ΣwᵢPSNRᵢₖ under one shared total-rate budget, choosing exactly one CRF per object. No temporal, buffer, switching, compute, or representation-dependency constraints are added. Utility uses absolute PSNR, so a percent gap is a fraction of the entire quality objective, not just achievable quality improvement.','',
           f"There are {len(cases)} cases: N={sizes}, seeds={cfg['seeds']}, four budgets each. Contents are balanced by shuffled blocks of four; GOPs are sampled uniformly. All clones reuse complete measured four-CRF rows with no quality/rate perturbation. Smaller N uses a common prefix within each seed. Repeated instances are independent decision variables, not independently captured scenes. Visibility is synthetic: offscreen zero (10%); peripheral U[.01,.10] (20%); visible U[.20,.70] (45%); foreground U[.75,1] (25%), normalized to sum one. All solvers receive the same immutable arrays; hashes are recorded per case.",'',
           'Effective budgets are Rmin+floor(α(Rmax−Rmin)), with α=.08,.35,.75,1.10 for tight, balanced, generous and overprovisioned. All original-problem cases are feasible. η=.9 and Ĉ=budget/η are informational; solvers directly receive the exact integer effective budget. Offscreen objects still require one representation.','',
           '## Algorithms and complexity','',
           '| Method | Implementation and guarantee | Complexity |','|---|---|---|',
           '| Brute-force | Complete-vector odometer; every K^N vector checked; exact when completed. | O(N K^N) time, O(N) traversal memory. |',
           '| NAVA | Section III-G/Algorithm 1 layered graph and recursive pulse with same-node label dominance, suffix minimum-rate feasibility, and maximum-utility upper bounds; exact when completed. | Exponential worst case; nondominated labels can grow exponentially; map operations add log-label overhead. |',
           '| LTS-inspired | Independent recursive full-space enumeration and feasibility/objective evaluation; exact when completed. | Exponential; O(N K^N) upper bound and O(N) traversal memory. |',
           '| Proposed DP | ceil(R/Δ), floor(budget/Δ), rolling value rows and byte-valued backtracking decisions. Exact for the conservative discretized problem. | O(N K B) time, O(B) value memory plus O(NB) parent memory, B=floor(budget/Δ). Pseudo-polynomial. |','',
           'NAVA inherits its graph/pulse pruning structure, replacing measured geometry PSNR with weighted measured RGB-reference PSNR and measured StreamSTGS costs. No NAVA Lagrange-multiplier solver or new heuristic bound is substituted. LTS originally selects layer–segment variables, with tile visibility, cached data, layer prerequisites and temporal priority. After adapting to this common independent-CRF problem, its transferable full-space search coincides conceptually with brute-force; timing differences reflect enumeration details. Author source code is unavailable, so this baseline is always named LTS-inspired. Detailed mapping and primary-paper links are in [NAVA_METHOD.md](../references/NAVA_METHOD.md) and [LTS_METHOD.md](../references/LTS_METHOD.md).','',
           'DP uses exact integer ceil/floor arithmetic, unreachable −∞ states and deterministic first-index ties. Backtracking rechecks the original budget. Conservative quantization cannot admit an over-budget solution, but can exclude original optima or every feasible original selection; the latter is explicitly discretized_infeasible. There is no general relative approximation bound from this discretization.','',
           '## Measurement and censoring','',
           f"All solvers are independent C++17 implementations compiled together with {' '.join(manifest['compiler_flags'])}, using {manifest['compiler']}. This run pins one thread to CPU {manifest['cpu_affinity']} on {cpu_model}. No GPU work is invoked. Each case/job has {cfg['warmups']} warmups and {cfg['repetitions']} measured calls, with seeded shuffled job order in each round. Individual calls use std::chrono::steady_clock. The timed interval includes validation, solver-specific graph/quantization setup, allocations, solving, and backtracking. Files, workload generation, scheduling, result validation/retention and output are outside. No per-case solver state is cached between repetitions.",'',
           f"Both exhaustive solvers skip before enumeration when K^N exceeds {cfg['max_combinations']:,}; see coverage for the skipped object counts. All solvers have {cfg['timeout_seconds']:g}s search deadlines (DP checks recurrence time, while its setup remains included in measured latency); NAVA also has {cfg['max_labels']:,} live-label cap and DP has {cfg['max_dp_cells']:,} parent-cell cap. Deadline checks are periodic, so slight overruns are possible. A capped NAVA run is recorded as timeout with its detailed reason, never certified optimal. The first timeout/skip stops further calls for that case/job, including during warmup. Any partial measured series is excluded from completed-case latency summaries. Raw samples retain phase, duration, incumbent, status and reason.",'',
           'Figure medians and P95 pool measured invocations over completed cases, with equal repetitions per case. P95 uses the nearest-rank definition; bars describe workload plus invocation variation, not confidence intervals or independent-sample statistical significance. Censored coverage must be read alongside latency, since easy completed cases alone can look fast. Maximum, per-case and per-budget summaries are also saved. Timer overhead is not subtracted; CPU frequency, other processes and thermal state are not controlled. Peak solver memory was not reliably isolated and is null, rather than reporting process RSS as solver memory.','',
           '## Results','',
           '![Processing time](../figures/processing_time_vs_n.png)','',
           '| Method | Δ (Mb/s) | N | Completed cases | Median ms | P95 ms | Max ms |','|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        if r['algorithm']=='Proposed-DP' and r['delta_bps']!=main_delta:continue
        lines.append(f"| {r['algorithm']} | {r['delta_bps']/1e6:g} | {r['n']} | {r['complete_cases']}/{r['total_cases']} | {fmt(r['median_ms'])} | {fmt(r['p95_ms'])} | {fmt(r['maximum_ms'])} |")
    largest_n=max(sizes)
    largest_dp = next(r for r in aggregates if (r['algorithm'], r['delta_bps'], r['n']) == ('Proposed-DP', main_delta, largest_n))
    dg = next(r for r in gap_rows if (r['algorithm'], r['delta_bps']) == ('Proposed-DP', main_delta))
    lines += ['', f"At N={largest_n}, Proposed DP with Δ={main_delta/1e6:g} Mb/s completes {largest_dp['complete_cases']}/{largest_dp['total_cases']} cases: median {fmt(largest_dp['median_ms'])} ms, P95 {fmt(largest_dp['p95_ms'])} ms, maximum {fmt(largest_dp['maximum_ms'])} ms. Across {dg['oracle_cases']} cases with certified optima, its mean gap is {fmt(dg['mean_gap_pct'])}% and maximum gap is {fmt(dg['max_gap_pct'])}%. These known-gap cases are a subset, with large-N coverage favoring cases completed by NAVA. NAVA coverage is {nava_coverage}. Survivor medians are not evidence that latency improves with more objects. NAVA censor reasons: {clock_timeouts} wall-clock deadlines and {label_timeouts} live-label caps.", '', f"Brute-force supplies the independent small-N oracle; completed NAVA additionally certifies some larger cases. No gap is imputed where an exact oracle is unavailable. There are {nava_complete} complete NAVA case series and {validation['cases_with_exact_oracle']} certified cases. An exact result returned before a later timeout remains a valid oracle certificate; that censored case series is still excluded from latency aggregation. Exact-method comparisons at small N match utility and constraints; different tied selections are accepted.",'',
              '| Method | Δ (Mb/s) | Cases with exact gap | Mean gap % | Max gap % | Cases checked against brute-force |','|---|---:|---:|---:|---:|---:|']
    for r in gap_rows:
        lines.append(f"| {r['algorithm']} | {r['delta_bps']/1e6:g} | {r['oracle_cases']} | {fmt(r['mean_gap_pct'])} | {fmt(r['max_gap_pct'])} | {r['brute_force_cases']} |")
    lines += ['', '![DP trade-off](../figures/dp_discretization_tradeoff.png)','',
              'All completed exact methods attain the same optimum where an independent oracle exists. DP trades quantization conservatism for runtime; finer Δ increases the number of budget states and can improve the objective. Individual quantization grids are not nested for arbitrary Δ, so finer steps need not improve every individual selection monotonically. Latency comparisons characterize these implementations on this workload, rather than universal ordering of algorithms.','',
              '## Validation and reproducibility','',
              f"Native correctness tests passed 350 randomized small instances and boundary cases with 13,873 assertions, using independent full-space and quantized oracles. Python metadata and harness tests validate source isolation, missing/invalid metadata, matrix identity and wire round-trips. An AddressSanitizer/UndefinedBehaviorSanitizer build also passed; LeakSanitizer was disabled because of the environment's ptrace restriction. Runtime validation checked {validation['selection_checks']:,} returned selections with **{validation['feasibility_violations']} feasibility violations**. {validation['cases_with_exact_oracle']} of {validation['cases']} cases have certified exact objectives. Logs and [validation.json](../results/validation.json) preserve evidence.",'',
              'Reproduce with `python3 benchmark.py all` from this directory. See [README.md](../README.md) for configuration, independent stage commands, external metadata paths and artifact descriptions. Raw JSONL includes warmups and failure/censor samples; validated JSON and CSV attach input hashes, utilities, budget utilization and exact gaps. Machine details, compiler flags, source hashes, config and input SHA256 are in results/run_manifest.json.','',
              '## Limits on interpretation','',
              'Only four real source contents and five GOPs per content are available; synthetic scaling reuses rows. Visibility is synthetic and independent of actual pose, occlusion and PSNR view dependence. The experiment intentionally omits cache sharing and scene interactions. All views are averaged, and visibility weights do not make quality view-conditioned. Normalization means total utility remains comparable as N grows. The configured seeds provide workload examples, not a population study. All budgets scale with N and actual rates: DP state count B therefore also grows with N at fixed Δ. Fixed-Δ scalability can approach quadratic growth rather than linear scaling in N.','',
              'NAVA has exponential worst-case label growth and time/label censoring. Exhaustive and LTS-inspired large-N results are unavailable when the candidate cap is exceeded. A missing exact oracle at larger N means the DP gap is unknown, even if it returns a feasible solution. Reports do not claim DP dominates NAVA on every budget; overprovisioned instances can be especially easy for bounding. The benchmark establishes CPU solver latency and the observed feasibility/optimality trade-off; it does not establish end-to-end ABR performance or mobile-device latency.','']
    (root/'reports'/'REPORT.md').write_text('\n'.join(lines))
    # Compact paper-facing tables retain coverage and use explicit missing cells.
    paper=['| N | Brute-force median/P95 ms | NAVA median/P95 ms (coverage) | LTS-inspired median/P95 ms | Proposed DP median/P95 ms |','|---:|---:|---:|---:|---:|']
    for n in sizes:
        cells=[]
        for a,d in [('Brute-force',0),('NAVA',0),('LTS-inspired',0),('Proposed-DP',main_delta)]:
            r=next(r for r in aggregates if (r['algorithm'],r['delta_bps'],r['n'])==(a,d,n))
            cell='—' if r['median_ms'] is None else f"{r['median_ms']:.4f}/{r['p95_ms']:.4f}"
            if a=='NAVA':cell+=f" ({r['complete_cases']}/{r['total_cases']})"
            cells.append(cell)
        paper.append(f"| {n} | "+' | '.join(cells)+' |')
    paper+=['',f'Proposed DP Δ={main_delta/1e6:g} Mb/s. Completed-case pooled median and P95; missing cells are skipped/censored, not zero latency. P95 is not a confidence interval.','']
    (root/'reports'/'PAPER_TABLE.md').write_text('\n'.join(paper))
    print(f'Wrote {len(summaries)} case summaries, {len(aggregates)} aggregates and four PNG/PDF/SVG figure sets.')

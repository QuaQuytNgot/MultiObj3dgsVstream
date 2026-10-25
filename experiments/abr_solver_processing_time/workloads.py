"""Prepare deterministic measured-cost/PSNR MCKP workloads outside timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from metadata import CRFS, DEFAULT_DATA_ROOT, GOPS, OBJECTS, MetadataError, load_dataset, write_audit_report

DEFAULT_SIZES = (2, 4, 8, 16, 32, 64)
DEFAULT_SEEDS = (2027, 2028, 2029)
DEFAULT_BUDGETS = (("tight", 0.08), ("balanced", 0.35),
                   ("generous", 0.75), ("overprovisioned", 1.10))


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def generate_workloads(profiles: list[dict], sizes=DEFAULT_SIZES, seeds=DEFAULT_SEEDS,
                       budget_specs=DEFAULT_BUDGETS, eta: float = 0.9) -> list[dict]:
    """Reuse real object/GOP rows; synthesize only independent visibility weights.

    Each seed builds a shuffled, balanced sequence of contents. GOPs are sampled
    uniformly. Sizes are prefixes of that sequence, with weights renormalized
    within each size. Bitrates and qualities are never perturbed or imputed.
    """
    sizes = tuple(sizes)
    seeds = tuple(seeds)
    if not sizes or any(not isinstance(n, int) or n <= 0 for n in sizes):
        raise ValueError("Object counts must be positive integers")
    if len(set(sizes)) != len(sizes) or len(set(seeds)) != len(seeds):
        raise ValueError("Object counts and seeds must not repeat")
    if not math.isfinite(eta) or not 0 < eta <= 1:
        raise ValueError("eta must lie in (0, 1]")
    names = [name for name, _ in budget_specs]
    if len(set(names)) != len(names) or not names:
        raise ValueError("Budget labels must be unique")
    if any(not math.isfinite(alpha) or alpha < 0 for _, alpha in budget_specs):
        raise ValueError("Budget fractions must be finite and nonnegative")
    profile_map = {(p["object"], p["gop"], p["crf"]): p for p in profiles}
    if len(profile_map) != len(profiles) or set(profile_map) != {
            (o, g, c) for o in OBJECTS for g in GOPS for c in CRFS}:
        raise MetadataError("Workloads require all 80 unique measured profiles; no imputation")
    cases = []
    for seed in seeds:
        rng = random.Random(seed)
        instances = []
        while len(instances) < max(sizes):
            block = list(OBJECTS)
            rng.shuffle(block)
            for obj in block:
                if len(instances) == max(sizes):
                    break
                gop = rng.choice(GOPS)
                mixture = rng.random()
                if mixture < 0.10:
                    category, raw_weight = "offscreen", 0.0
                elif mixture < 0.30:
                    category, raw_weight = "peripheral", rng.uniform(0.01, 0.10)
                elif mixture < 0.75:
                    category, raw_weight = "visible", rng.uniform(0.20, 0.70)
                else:
                    category, raw_weight = "foreground", rng.uniform(0.75, 1.0)
                instances.append({"instance_id": len(instances), "object": obj, "gop": gop,
                                  "visibility_category": category, "raw_visibility_weight": raw_weight})
        # Avoid the low-probability all-zero prefix at N=2 while preserving the
        # other mixture draws and the prefix reuse rule at every larger size.
        if not any(i["raw_visibility_weight"] > 0 for i in instances[:min(sizes)]):
            instances[0]["raw_visibility_weight"] = 1.0
            instances[0]["visibility_category"] = "foreground_fallback"
        for n in sizes:
            objects = [dict(i) for i in instances[:n]]
            weight_sum = math.fsum(i["raw_visibility_weight"] for i in objects)
            weights = [i["raw_visibility_weight"] / weight_sum for i in objects]
            costs = [[profile_map[i["object"], i["gop"], c]["cost_bps"] for c in CRFS]
                     for i in objects]
            psnr = [[profile_map[i["object"], i["gop"], c]["psnr_db"] for c in CRFS]
                    for i in objects]
            utilities = [[w * q for q in row] for w, row in zip(weights, psnr)]
            for i, weight in zip(objects, weights):
                i["visibility_weight"] = weight
                i["profile_sha256_by_crf"] = {
                    str(c): profile_map[i["object"], i["gop"], c]["quality_profile_sha256"]
                    for c in CRFS}
            min_total = sum(min(row) for row in costs)
            max_total = sum(max(row) for row in costs)
            matrix_hash = _hash_json({"costs_bps": costs, "utilities": utilities})
            for label, alpha in budget_specs:
                budget = min_total + math.floor(alpha * (max_total - min_total))
                case = {
                    "id": f"n{n:02d}_s{seed}_{label}", "n": n, "k": len(CRFS),
                    "seed": seed, "budget_label": label, "budget_fraction_of_upgrade_span": alpha,
                    "budget_bps": budget, "eta": eta,
                    "estimated_capacity_bps": budget / eta,
                    "budget_semantics": "Effective integer constraint eta*C_hat; C_hat is informational only, solver receives budget_bps directly",
                    "minimum_feasible_bps": min_total, "maximum_selection_bps": max_total,
                    "costs_bps": costs, "utilities": utilities, "psnr_db": psnr,
                    "weights": weights, "crfs": list(CRFS), "instances": objects,
                    "matrix_sha256": matrix_hash,
                    "decision_sha256": _hash_json({"matrix_sha256": matrix_hash, "budget_bps": budget}),
                    "real_quality_and_cost": True,
                    "synthetic_scaling": "Independent object-instance clones reuse measured sequence/GOP rows; no contention, inter-object correlation, caching or actual scene geometry is modeled",
                }
                if budget < min_total:
                    raise AssertionError("Benchmark budget must be feasible in the original problem")
                cases.append(case)
    return cases


def write_runner_input(cases: list[dict], path: str | Path) -> None:
    """Whitespace wire format: count; id N budget; N (4 costs + 4 utilities)."""
    lines = [str(len(cases))]
    for case in cases:
        if any(character.isspace() for character in case["id"]):
            raise ValueError("Case identifiers cannot contain whitespace")
        lines.append(f"{case['id']} {case['n']} {case['budget_bps']}")
        for costs, utilities in zip(case["costs_bps"], case["utilities"]):
            lines.append(" ".join([*(str(c) for c in costs), *(format(u, ".17g") for u in utilities)]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def prepare(data_root=DEFAULT_DATA_ROOT, cost_root=None, output_dir=None,
            sizes=DEFAULT_SIZES, seeds=DEFAULT_SEEDS, eta=0.9) -> dict:
    experiment = Path(__file__).resolve().parent
    output_dir = Path(output_dir) if output_dir is not None else experiment / "inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles, audit = load_dataset(data_root, cost_root)
    cases = generate_workloads(profiles, sizes=sizes, seeds=seeds, eta=eta)
    workload_metadata = {
        "schema": "abr-solver-prepared-cases.v1", "case_count": len(cases),
        "sizes": list(sizes), "seeds": list(seeds), "eta": eta,
        "budget_specs": [{"label": label, "fraction_of_upgrade_span": alpha}
                         for label, alpha in DEFAULT_BUDGETS],
        "visibility_distribution": "Seeded Python random.Random; P(offscreen)=0.10 => 0; P(peripheral)=0.20 => U[.01,.10]; P(visible)=0.45 => U[.20,.70]; P(foreground)=0.25 => U[.75,1]. Normalize weights to sum 1 within each workload. All-zero smallest prefix gets one weight of 1.",
        "reuse_rule": "Each seed shuffles each block of four contents once, samples GOP uniformly in 0..4, and builds max(N) independent clones. Smaller N uses the same prefix. Every clone offers all four real CRF profiles from its one chosen sequence/GOP. Reused quality/rate rows are identical, weights differ; only instance multiplicity and visibility are synthetic.",
        "source_checksum_inventory_sha256": audit["checksum_inventory_sha256"],
        "cases_sha256": _hash_json(cases), "cases": cases,
    }
    for filename, value in (("dataset_audit.json", audit), ("measured_profiles.json", profiles),
                            ("prepared_cases.json", workload_metadata)):
        (output_dir / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    write_runner_input(cases, output_dir / "runner_input.txt")
    write_audit_report(audit, experiment / "reports" / "DATA_AUDIT.md")
    return workload_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cost-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "inputs")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--eta", type=float, default=0.9)
    args = parser.parse_args()
    try:
        prepared = prepare(args.data_root, args.cost_root, args.output_dir,
                           sizes=args.sizes, seeds=args.seeds, eta=args.eta)
    except (MetadataError, ValueError, KeyError) as error:
        parser.exit(2, f"Input preparation failed: {error}\n")
    print(f"Prepared {prepared['case_count']} cases from 80 measured profiles in {args.output_dir}")
    print(f"Cases SHA256: {prepared['cases_sha256']}")


if __name__ == "__main__":
    main()

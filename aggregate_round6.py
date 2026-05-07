#!/usr/bin/env python3
"""Aggregate round-6 ablation results and check whether all B2 deltas turn negative.

Reads:
    outputs/clean_ablation_v2_1/<benchmark>/<run_id>/<ablation>/seed<seed>/metrics.json

Outputs:
    * Per-ablation table (mean ± std test_mcc, Δ vs full)
    * Sign consistency across seeds
    * Pass/fail verdict for the round-6 acceptance criteria

Usage:
    /home/fyh0106/miniconda3/envs/sucf_run/bin/python aggregate_round6.py <run_id> [--benchmark B2|B1|both]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev


def load_metrics(run_id: str, benchmark: str):
    base = Path('outputs/clean_ablation_v2_1') / benchmark / run_id
    if not base.exists():
        return {}
    results = {}
    for ablation_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        seed_results = {}
        for seed_dir in sorted(p for p in ablation_dir.iterdir() if p.is_dir() and p.name.startswith('seed')):
            metrics_path = seed_dir / 'metrics.json'
            if metrics_path.exists():
                with open(metrics_path) as f:
                    seed_results[seed_dir.name] = json.load(f)
        if seed_results:
            results[ablation_dir.name] = seed_results
    return results


def report(run_id: str, benchmark: str):
    print(f'=== {benchmark} | {run_id} ===')
    results = load_metrics(run_id, benchmark)
    if not results:
        print('  (no results found)')
        return False

    full_mccs = []
    if 'full' in results:
        full_mccs = [r['test_mcc'] for r in results['full'].values()]
    if not full_mccs:
        print('  (full ablation has no metrics, cannot compute deltas)')
        return False

    full_mean = mean(full_mccs)
    full_std = stdev(full_mccs) if len(full_mccs) > 1 else 0.0
    print(f"  full              n={len(full_mccs)} mean={full_mean:.4f} std={full_std:.4f}")

    expected_negative = ['wo_structure', 'wo_plddt_gate_keep_structure',
                         'wo_scgc_keep_structure', 'wo_bimamba_keep_fusion']
    all_negative = True

    for abl in expected_negative:
        if abl not in results:
            continue
        mccs = [r['test_mcc'] for r in results[abl].values()]
        if not mccs:
            continue
        m = mean(mccs)
        s = stdev(mccs) if len(mccs) > 1 else 0.0
        delta = m - full_mean
        # Per-seed signed deltas (paired by seed when possible)
        seed_signs = []
        for seed_name, full_seed_metric in results.get('full', {}).items():
            ablation_seed_metric = results[abl].get(seed_name)
            if ablation_seed_metric is None:
                continue
            d = ablation_seed_metric['test_mcc'] - full_seed_metric['test_mcc']
            seed_signs.append('-' if d < 0 else '+')
        sign_str = ''.join(seed_signs)
        verdict = 'OK' if delta < 0 else 'BAD'
        if delta >= 0:
            all_negative = False
        print(f"  {abl:<35} n={len(mccs)} mean={m:.4f} std={s:.4f} Δ={delta:+.4f} signs={sign_str} {verdict}")

    print(f"  ==> {'ALL DELTAS NEGATIVE (acceptance met)' if all_negative else 'SOME DELTAS POSITIVE — FAIL'}")
    return all_negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    parser.add_argument('--benchmark', choices=['B1', 'B2', 'both'], default='B2')
    args = parser.parse_args()

    if args.benchmark == 'both':
        ok_b2 = report(args.run_id, 'B2')
        print()
        ok_b1 = report(args.run_id, 'B1')
        sys.exit(0 if ok_b2 and ok_b1 else 1)
    else:
        ok = report(args.run_id, args.benchmark)
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Run SUCF full training across multiple seeds and repetitions on GPU 0.
Creates a per-run temporary config and directories under outputs/seed_runs/
Collects final_test_results.json (if produced) and aggregates metrics.

Usage: python3 scripts/run_seed_experiments.py

Note: This script runs sequentially and may take a very long time for full training.
"""

import os
import sys
import shutil
import subprocess
import yaml
import json
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO_ROOT / 'configs' / 'training_config.yaml'
TRAIN_SCRIPT = REPO_ROOT / 'train_sucf.py'
OUTPUTS_ROOT = REPO_ROOT / 'outputs' / 'seed_runs'
GPU_ID = '0'

SEEDS = [32, 37, 42, 47, 52]
REPEATS = 1


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_config(cfg, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f)


def run_one(run_dir, config_path):
    """Run one training process under conda env 'multi' on specified GPU."""
    run_log = run_dir / 'run.log'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Use conda-run to ensure the 'multi' environment is used
    cmd = ['conda', 'run', '-n', 'multi', sys.executable, str(TRAIN_SCRIPT), '--config', str(config_path)]

    with open(run_log, 'wb') as logf:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
        return_code = proc.wait()
    return return_code


def collect_metrics(run_dir):
    # final_test_results.json is written into run-specific log_dir by train_sucf
    log_dir = run_dir / 'logs'
    result_file = log_dir / 'final_test_results.json'
    if result_file.exists():
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    # fallback: try to read run.log and look for JSON blob or metrics lines
    run_log = run_dir / 'run.log'
    if run_log.exists():
        # last-resort: return None
        return None
    return None


def main():
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(BASE_CONFIG)

    all_results = {}

    for seed in SEEDS:
        seed_results = []
        for r in range(1, REPEATS + 1):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = OUTPUTS_ROOT / f'seed_{seed}' / f'run_{r}_{ts}'
            (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
            (run_dir / 'logs').mkdir(parents=True, exist_ok=True)

            # prepare config copy
            cfg = dict(base_cfg)
            # set random seed at top-level (train_sucf reads config.get('random_seed'))
            cfg['random_seed'] = int(seed)
            # set run-specific paths
            paths = cfg.get('paths', {})
            paths['checkpoint_dir'] = str(run_dir / 'checkpoints')
            paths['log_dir'] = str(run_dir / 'logs')
            cfg['paths'] = paths

            temp_cfg_path = run_dir / 'config.yaml'
            write_config(cfg, temp_cfg_path)

            print(f"Starting seed={seed} repeat={r} -> {run_dir}")
            sys.stdout.flush()
            rc = run_one(run_dir, temp_cfg_path)
            print(f"Run finished (rc={rc}): {run_dir}")

            metrics = collect_metrics(run_dir)
            if metrics is None:
                print(f"Warning: no metrics found for run {run_dir}")
            else:
                print(f"Collected metrics for run {run_dir}: {metrics}")

            # save per-run metadata
            with open(run_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
                json.dump({'seed': seed, 'repeat': r, 'rc': rc, 'metrics': metrics}, f, indent=2)

            seed_results.append({'run_dir': str(run_dir), 'rc': rc, 'metrics': metrics})

        all_results[seed] = seed_results

    # aggregate simple stats
    summary = {'by_seed': {}, 'seeds': SEEDS, 'repeats': REPEATS}
    for seed, runs in all_results.items():
        # collect numeric metric keys
        metric_keys = set()
        for run in runs:
            if run['metrics']:
                for k in run['metrics'].keys():
                    metric_keys.add(k)
        seed_summary = {}
        for k in sorted(metric_keys):
            vals = [run['metrics'].get(k) for run in runs if run['metrics'] and isinstance(run['metrics'].get(k), (int, float))]
            if vals:
                import statistics
                seed_summary[k] = {'mean': statistics.mean(vals), 'std': statistics.pstdev(vals) if len(vals)>1 else 0.0, 'n': len(vals)}
        summary['by_seed'][str(seed)] = seed_summary

    # write summary files
    with open(OUTPUTS_ROOT / 'all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    with open(OUTPUTS_ROOT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # also write a human-readable markdown
    md_lines = [f"# Seed runs summary\n", f"Date: {datetime.now().isoformat()}\n", f"Seeds: {SEEDS}\n", f"Repeats per seed: {REPEATS}\n\n"]
    for seed, ssum in summary['by_seed'].items():
        md_lines.append(f"## Seed {seed}\n")
        if not ssum:
            md_lines.append("No numeric metrics collected for this seed.\n\n")
            continue
        for k, v in ssum.items():
            md_lines.append(f"- {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, n={v['n']}\n")
        md_lines.append("\n")

    with open(OUTPUTS_ROOT / 'summary.md', 'w', encoding='utf-8') as f:
        f.writelines(md_lines)

    print("All runs finished. Summaries written to:")
    print(str(OUTPUTS_ROOT))


if __name__ == '__main__':
    main()

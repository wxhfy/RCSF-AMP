#!/usr/bin/env python3
"""Run PepNet (fast or standard mode) on benchmark1 across multiple seeds."""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
PEPNET_SCRIPT_DIR = REPO_ROOT / 'baseline' / 'PepNet' / 'script'
PEPNET_DATASET_DIR = REPO_ROOT / 'baseline' / 'PepNet' / 'datasets' / 'AMP'
DEFAULT_DATA_ROOT = REPO_ROOT / 'data' / 'benchmark1'
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'baseline' / 'PepNet' / 'outputs_benchmark1'
DEFAULT_SEEDS = [32, 37, 42, 47, 52]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PepNet benchmark1 runner")
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT,
                        help='Path to benchmark1 FASTA directory')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help='Directory to store per-seed results')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help='Random seeds to evaluate')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs per seed')
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--n-transformer', type=int, default=1, help='Transformer layers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--conda-env', type=str, default='acep',
                        help='Conda environment to run PepNet scripts inside')
    parser.add_argument('--mode', choices=['fast', 'standard'], default='fast',
                        help='PepNet mode to execute (fast skips ProtT5 features)')
    return parser.parse_args()


def run_prepare_script(data_root: Path) -> None:
    cmd = [sys.executable,
           str((REPO_ROOT / 'scripts' / 'prepare_pepnet_benchmark1.py').resolve()),
           '--data-root', str(data_root),
           '--pepnet-root', str(PEPNET_DATASET_DIR)]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def run_training(seed: int, args: argparse.Namespace, run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'conda', 'run', '-n', args.conda_env, 'python', 'train.py',
        '--type', 'AMP',
        '--train_fasta', 'train.txt',
        '--test_fasta', 'test.txt',
        '--hidden', str(args.hidden),
        '--batch_size', str(args.batch_size),
        '--epoch', str(args.epochs),
        '--seed', str(seed),
        '--lr', str(args.lr),
        '--n_transformer', str(args.n_transformer),
        '--mode', args.mode,
        '--output_dir', str(run_dir),
    ]

    subprocess.run(cmd, check=True, cwd=str(PEPNET_SCRIPT_DIR))


def collect_metrics(seed: int, run_dir: Path, dest_file: Path) -> Dict:
    metrics_path = run_dir / 'log' / 'metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found for seed {seed}: {metrics_path}")
    with metrics_path.open('r', encoding='utf-8') as handle:
        metrics = json.load(handle)
    with dest_file.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def summarize(per_seed: Dict[str, Dict]) -> Dict:
    metric_keys = ['accuracy', 'auc', 'sensitivity', 'specificity', 'matthews_corrcoef']
    summary = {}
    for key in metric_keys:
        values: List[float] = []
        for seed_metrics in per_seed.values():
            values.append(seed_metrics['test'][key])
        summary[key] = {
            'mean': statistics.mean(values),
            'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def main() -> None:
    args = parse_args()
    run_prepare_script(args.data_root)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    per_seed_metrics: Dict[str, Dict] = {}

    for seed in args.seeds:
        seed_dir = output_root / f'seed_{seed}'
        print(f"[PepNet] Running seed {seed} -> {seed_dir}")
        run_training(seed, args, seed_dir)
        metrics = collect_metrics(seed, seed_dir, seed_dir / f'metrics_seed_{seed}.json')
        per_seed_metrics[str(seed)] = metrics

    summary = summarize(per_seed_metrics)
    summary_payload = {
        'seeds': args.seeds,
        'per_seed': per_seed_metrics,
        'summary': summary,
        'params': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'hidden': args.hidden,
            'lr': args.lr,
            'n_transformer': args.n_transformer,
            'mode': args.mode,
        },
    }

    with (output_root / 'summary_metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"[PepNet] Summary saved to {output_root / 'summary_metrics.json'}")


if __name__ == '__main__':
    main()

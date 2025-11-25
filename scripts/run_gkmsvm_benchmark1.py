#!/usr/bin/env python3
"""Run gkmSVM on benchmark1 splits across multiple seeds.

The script wraps the compiled CLI tools in baseline/gkmSVM/bin to:
1. build the gapped k-mer kernel on the training AMP/DECOY FASTA files,
2. train the SVM model,
3. score the benchmark1 test splits, and
4. compute requested metrics (ACC, AUC, SEN, SPEC, MCC + confusion counts).

Usage example:
    python scripts/run_gkmsvm_benchmark1.py \
        --data-root data/benchmark1 \
        --output-root baseline/gkmSVM/outputs_benchmark1

The seeds list mimics other baselines even though gkmSVM itself is deterministic.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "benchmark1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "baseline" / "gkmSVM" / "outputs_benchmark1"
GKM_ROOT = REPO_ROOT / "baseline" / "gkmSVM"
BIN_DIR = GKM_ROOT / "bin"
DEFAULT_ALPHABET = GKM_ROOT / "peptide_alphabet.txt"
DEFAULT_SEEDS = [32, 37, 42, 47, 52]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gkmSVM benchmark1 evaluation across multiple seeds.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Directory containing AMP_* and DECOY_* FASTA files (default: data/benchmark1)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Directory to store per-seed artifacts and metrics")
    parser.add_argument("--alphabet", type=Path, default=DEFAULT_ALPHABET,
                        help="Alphabet file listing allowed amino acids (default: peptide_alphabet.txt)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Seeds to label each run (default: 32 37 42 47 52)")
    parser.add_argument("-l", type=int, default=10, dest="lmer_length",
                        help="Word length parameter L for gkm kernel (default: 10)")
    parser.add_argument("-k", type=int, default=6, dest="informative_cols",
                        help="Number of informative columns K (default: 6)")
    parser.add_argument("-d", type=int, default=3, dest="max_mismatches",
                        help="Maximum mismatches to consider (default: 3)")
    parser.add_argument("-T", "--threads", type=int, default=16,
                        help="Maximum threads for kernel computation (default: 16)")
    return parser.parse_args()


def ensure_files_exist(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def run_command(cmd: Sequence[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as logf:
        log_line = "$ " + " ".join(cmd)
        logf.write(log_line + "\n")
        process = subprocess.run(cmd, stdout=logf, stderr=logf)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(cmd)}")


def load_scores(score_file: Path) -> List[float]:
    scores: List[float] = []
    with open(score_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                scores.append(float(parts[1]))
            except ValueError:
                continue
    return scores


def compute_metrics(scores: List[float], labels: List[int], seed: int) -> Dict[str, float]:
    if len(scores) != len(labels):
        raise ValueError("Scores and labels must be the same length.")

    preds = [1 if s >= 0 else 0 for s in scores]
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5

    if len(set(preds)) == 1 or len(set(labels)) == 1:
        mcc = 0.0
    else:
        mcc = matthews_corrcoef(labels, preds)

    return {
        "seed": seed,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "matthews_corrcoef": mcc,
        "auc": auc,
    }


def summarize_metrics(per_seed: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    metric_keys = ["accuracy", "auc", "sensitivity", "specificity", "matthews_corrcoef"]
    for key in metric_keys:
        values = [metrics[key] for metrics in per_seed.values() if key in metrics]
        if not values:
            continue
        mean_val = statistics.mean(values)
        std_val = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary[key] = {"mean": mean_val, "std": std_val}
    return summary


def write_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def main() -> None:
    args = parse_args()

    ensure_files_exist([
        BIN_DIR / "gkmsvm_kernel",
        BIN_DIR / "gkmsvm_train",
        BIN_DIR / "gkmsvm_classify",
        args.alphabet,
    ])

    amp_train = args.data_root / "AMP_train.fasta"
    decoy_train = args.data_root / "DECOY_train.fasta"
    amp_test = args.data_root / "AMP_test.fasta"
    decoy_test = args.data_root / "DECOY_test.fasta"
    ensure_files_exist([amp_train, decoy_train, amp_test, decoy_test])

    per_seed_metrics: Dict[str, Dict[str, float]] = {}
    args.output_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        seed_dir = args.output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        log_file = seed_dir / "run.log"

        kernel_file = seed_dir / f"kernel_seed_{seed}.txt"
        model_prefix = seed_dir / f"gkmsvm_seed_{seed}"
        sv_seq_file = Path(f"{model_prefix}_svseq.fa")
        sv_alpha_file = Path(f"{model_prefix}_svalpha.out")
        amp_scores_file = seed_dir / "scores_amp.txt"
        decoy_scores_file = seed_dir / "scores_decoy.txt"
        metrics_file = seed_dir / f"metrics_seed_{seed}.json"

        print(f"[seed {seed}] Computing kernel...")
        kernel_cmd = [
            str(BIN_DIR / "gkmsvm_kernel"),
            "-l", str(args.lmer_length),
            "-k", str(args.informative_cols),
            "-d", str(args.max_mismatches),
            "-A", str(args.alphabet),
            "-R",
            "-T", str(args.threads),
            str(amp_train),
            str(decoy_train),
            str(kernel_file),
        ]
        run_command(kernel_cmd, log_file)

        print(f"[seed {seed}] Training SVM...")
        train_cmd = [
            str(BIN_DIR / "gkmsvm_train"),
            str(kernel_file),
            str(amp_train),
            str(decoy_train),
            str(model_prefix),
        ]
        run_command(train_cmd, log_file)

        print(f"[seed {seed}] Scoring AMP test set...")
        classify_amp_cmd = [
            str(BIN_DIR / "gkmsvm_classify"),
            "-l", str(args.lmer_length),
            "-k", str(args.informative_cols),
            "-d", str(args.max_mismatches),
            "-A", str(args.alphabet),
            "-R",
            str(amp_test),
            str(sv_seq_file),
            str(sv_alpha_file),
            str(amp_scores_file),
        ]
        run_command(classify_amp_cmd, log_file)

        print(f"[seed {seed}] Scoring DECOY test set...")
        classify_decoy_cmd = [
            str(BIN_DIR / "gkmsvm_classify"),
            "-l", str(args.lmer_length),
            "-k", str(args.informative_cols),
            "-d", str(args.max_mismatches),
            "-A", str(args.alphabet),
            "-R",
            str(decoy_test),
            str(sv_seq_file),
            str(sv_alpha_file),
            str(decoy_scores_file),
        ]
        run_command(classify_decoy_cmd, log_file)

        amp_scores = load_scores(amp_scores_file)
        decoy_scores = load_scores(decoy_scores_file)
        labels = [1] * len(amp_scores) + [0] * len(decoy_scores)
        scores = amp_scores + decoy_scores

        metrics = compute_metrics(scores, labels, seed)
        write_json(metrics, metrics_file)
        per_seed_metrics[str(seed)] = metrics

        print(f"[seed {seed}] Metrics saved -> {metrics_file}")

    summary = summarize_metrics(per_seed_metrics)
    summary_payload = {
        "seeds": args.seeds,
        "per_seed": per_seed_metrics,
        "summary": summary,
        "params": {
            "L": args.lmer_length,
            "K": args.informative_cols,
            "max_mismatches": args.max_mismatches,
            "threads": args.threads,
            "alphabet": str(args.alphabet),
        },
    }
    summary_file = args.output_root / "summary_metrics.json"
    write_json(summary_payload, summary_file)
    print(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()

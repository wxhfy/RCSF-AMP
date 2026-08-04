#!/usr/bin/env python3
"""Aggregate same-split 2026 AMP baseline reproductions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUTPUT_ROOT = Path("outputs/baselines_2026")
MODELS = ("ampidentifier", "cars_amp", "amp_capsnet")
BENCHMARKS = ("benchmark1", "benchmark2")
SEEDS = (32, 37, 42, 47, 52)
METRICS = ("ACC", "AUC", "AUPR", "SEN", "SPEC", "F1", "MCC")


def main():
    rows = []
    for model in MODELS:
        for benchmark in BENCHMARKS:
            for seed in SEEDS:
                path = OUTPUT_ROOT / model / benchmark / f"seed{seed}" / "metrics.json"
                payload = json.loads(path.read_text())
                rows.append(
                    {
                        "path": str(path),
                        "model": model,
                        "dataset": benchmark,
                        "seed": seed,
                        "speed": payload["test_inference_ms_per_1000"],
                        **payload["test"],
                    }
                )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUTPUT_ROOT / "per_seed_results.csv", index=False)
    aggregate = per_seed.groupby(["model", "dataset"])[[*METRICS, "speed"]].agg(["mean", "std"])
    aggregate.to_csv(OUTPUT_ROOT / "aggregate_results.csv")


if __name__ == "__main__":
    main()

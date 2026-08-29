# Results Archive

This directory is reserved for compact, reviewable result summaries.

Recommended files:

- `benchmark_metrics.csv`: aggregate Benchmark 1 and Benchmark 2 metrics;
- `component_ablation.csv`: matched component controls;
- `per_seed_metrics.csv`: seed-level records for the reported configurations;
- `protocol_manifest.json`: partitions, seeds, threshold, and checkpoint-selection rules.

The dated `archive_20260804/` directory contains retained historical summaries and is separated from the current compact result files.

Use the common classification metric set `ACC`, `AUC`, `AUPR`, `F1`, and `MCC`. Large prediction files, model checkpoints, generated structures, and training logs remain outside the public repository.

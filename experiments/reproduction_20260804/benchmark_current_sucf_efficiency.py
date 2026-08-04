#!/usr/bin/env python3
"""Benchmark the current full SUCF predictor with cached graph features."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader


PROJECT_ROOT = Path("/home/20T-1/fyh0106/SUCF")
sys.path.insert(0, str(PROJECT_ROOT))

from models.sucf_model import create_sucf_model  # noqa: E402
from utils.config_utils import load_config  # noqa: E402
from utils.datasets import AMPGraphDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timed", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def create_test_loader(config: dict, args: argparse.Namespace) -> PyGDataLoader:
    data_root = Path(config["paths"]["data_root"])
    dataset = AMPGraphDataset(
        str(data_root),
        split_file=str(data_root / "test.txt"),
        extra_embeddings=config.get("data", {}).get("extra_embeddings", {}),
    )
    return PyGDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def load_model(config: dict, checkpoint_path: Path | None, device: torch.device):
    model = create_sucf_model(config["model"])
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.inference_mode()
def benchmark(model, loader, args: argparse.Namespace, device: torch.device) -> list[dict]:
    rows = []
    for repeat in range(args.repeats):
        iterator = itertools.cycle(loader)
        for _ in range(args.warmup):
            model(next(iterator).to(device, non_blocking=True))
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        graph_count = 0
        start = time.perf_counter()
        for _ in range(args.timed):
            batch = next(iterator).to(device, non_blocking=True)
            graph_count += int(batch.num_graphs)
            model(batch)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "repeat": repeat + 1,
                "graphs": graph_count,
                "elapsed_seconds": elapsed,
                "inference_ms_per_1000": elapsed * 1_000_000.0 / graph_count,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    loader = create_test_loader(config, args)
    model = load_model(config, args.checkpoint, device)
    rows = benchmark(model, loader, args, device)
    latencies = [row["inference_ms_per_1000"] for row in rows]
    memories = [row["peak_gpu_memory_mb"] for row in rows]
    architecture = config["model"].get("architecture", {})
    summary = {
        "model_id": "SUCF-AMP_full",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "batch_size": args.batch_size,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.timed,
        "repeats": args.repeats,
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "post_scgc_structure_source": architecture.get("post_scgc_structure_source"),
        "inference_ms_per_1000_mean": statistics.mean(latencies),
        "inference_ms_per_1000_std": statistics.stdev(latencies),
        "peak_gpu_memory_mb_mean": statistics.mean(memories),
        "repeats_detail": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

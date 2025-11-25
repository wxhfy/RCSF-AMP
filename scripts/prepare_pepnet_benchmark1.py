#!/usr/bin/env python3
"""Prepare benchmark1 FASTA files for PepNet training/evaluation.

This script converts the existing AMP/DECOY FASTA splits under data/benchmark1
into PepNet's expected format (>name|label) and creates the required
properties.pkl descriptor used for handcrafted residue features.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
PROPERTY_FLAGS: List[Tuple[str, Iterable[str]]] = [
    ("hydrophobic", "AVILMFYW"),
    ("polar", "STNQCYDEHKR"),
    ("positively_charged", "KRH"),
    ("negatively_charged", "DE"),
    ("aromatic", "FWYH"),
    ("aliphatic", "AVILM"),
    ("tiny", "AGS"),
    ("small", "ACGNPSTV"),
    ("sulfur", "CM"),
    ("hydroxyl", "STY"),
    ("amide", "NQ"),
    ("helix_preferring", "AEHILKMQR"),
    ("sheet_preferring", "VIYFWTC"),
    ("turn_preferring", "GNDPS"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PepNet benchmark1 dataset")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/benchmark1"),
        help="Directory containing AMP_*.fasta and DECOY_*.fasta",
    )
    parser.add_argument(
        "--pepnet-root",
        type=Path,
        default=Path("baseline/PepNet/datasets/AMP"),
        help="Target directory where PepNet expects AMP train/test files",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="AMP",
        choices=["AMP"],
        help="PepNet ligand type (currently only AMP is supported)",
    )
    return parser.parse_args()


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    name = None
    seq_lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    entries.append((name, "".join(seq_lines).upper()))
                name = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        if name is not None:
            entries.append((name, "".join(seq_lines).upper()))
    return entries


def write_pepnet_fasta(entries: List[Tuple[str, str, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for name, seq, label in entries:
            handle.write(f">{name}|{label}\n")
            handle.write(f"{seq}\n")


def build_property_matrix() -> Dict[str, List[float]]:
    prop_map: Dict[str, List[float]] = {}
    for aa in AA_ORDER + "X":
        vector = [1.0 if aa in group else 0.0 for _, group in PROPERTY_FLAGS]
        prop_map[aa] = vector
    return prop_map


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    pepnet_root = args.pepnet_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root '{data_root}' does not exist")

    splits = {
        "train": (
            read_fasta(data_root / "AMP_train.fasta"),
            read_fasta(data_root / "DECOY_train.fasta"),
        ),
        "test": (
            read_fasta(data_root / "AMP_test.fasta"),
            read_fasta(data_root / "DECOY_test.fasta"),
        ),
        "eval": (
            read_fasta(data_root / "AMP_eval.fasta"),
            read_fasta(data_root / "DECOY_eval.fasta"),
        ),
    }

    for split_name, (amps, decoys) in splits.items():
        combined: List[Tuple[str, str, int]] = []
        for name, seq in amps:
            combined.append((f"AMP_{split_name.upper()}_{name}", seq, 1))
        for name, seq in decoys:
            combined.append((f"DECOY_{split_name.upper()}_{name}", seq, 0))
        out_file = pepnet_root / f"{split_name}.txt"
        write_pepnet_fasta(combined, out_file)

    # Ensure companion directories exist for PepNet bookkeeping
    (pepnet_root / "feature").mkdir(parents=True, exist_ok=True)
    (pepnet_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Create/amend the shared properties.pkl expected by PepNet preprocess
    # PepNet scripts expect ../datasets/properties.pkl relative to baseline/PepNet/script
    properties_path = (pepnet_root.parent / "properties.pkl").resolve()
    properties_path.parent.mkdir(parents=True, exist_ok=True)
    with properties_path.open("wb") as handle:
        pickle.dump(build_property_matrix(), handle)

    summary = {
        "data_root": str(data_root),
        "pepnet_root": str(pepnet_root),
        "splits": {
            split: {
                "positives": len(splits[split][0]),
                "negatives": len(splits[split][1]),
            }
            for split in splits
        },
        "properties_file": str(properties_path),
    }

    summary_path = pepnet_root / "prepared_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        import json

        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()

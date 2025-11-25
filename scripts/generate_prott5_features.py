#!/usr/bin/env python3
"""Generate ProtT5-XL-U50 embeddings for PepNet FASTA splits."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import h5py
import torch
from transformers import T5EncoderModel, T5Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PepNet ProtT5 feature generator")
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file with name|label format")
    parser.add_argument("--output", type=Path, required=True, help="Destination HDF5 file")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Directory containing ProtT5-XL-U50 weights and tokenizer files")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of sequences per forward pass")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum number of tokens (residues) to keep per sequence")
    return parser.parse_args()


def read_fasta(path: Path) -> Tuple[List[str], List[str]]:
    names: List[str] = []
    seqs: List[str] = []
    name: str | None = None
    seq_lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    names.append(name)
                    seqs.append("".join(seq_lines).upper())
                name = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if name is not None:
            names.append(name)
            seqs.append("".join(seq_lines).upper())
    return names, seqs


def chunk_indices(total: int, chunk_size: int) -> Iterable[range]:
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield range(start, end)


def format_sequence(seq: str) -> str:
    cleaned = seq.replace("U", "X").replace("Z", "X").replace("B", "X")
    return " ".join(cleaned)


def run_embedding(names: Sequence[str], seqs: Sequence[str], args: argparse.Namespace) -> None:
    tokenizer = T5Tokenizer.from_pretrained(str(args.model_dir), do_lower_case=False)
    model = T5EncoderModel.from_pretrained(str(args.model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as h5f:
        for batch_ids in chunk_indices(len(names), args.batch_size):
            batch_names = [names[i] for i in batch_ids]
            batch_seqs = [format_sequence(seqs[i]) for i in batch_ids]
            encoded = tokenizer(batch_seqs,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                add_special_tokens=False,
                                max_length=args.max_length)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
                embeddings = outputs.last_hidden_state
            attention = encoded["attention_mask"].cpu()
            for idx, name in enumerate(batch_names):
                length = int(attention[idx].sum().item())
                seq_embed = embeddings[idx, :length, :].detach().cpu().numpy()
                h5f.create_dataset(name, data=seq_embed, compression="gzip")
            print(f"Embedded {batch_ids.stop} / {len(names)} sequences", flush=True)


def main() -> None:
    args = parse_args()
    names, seqs = read_fasta(args.fasta)
    if not names:
        raise ValueError(f"No sequences found in {args.fasta}")
    run_embedding(names, seqs, args)


if __name__ == "__main__":
    main()

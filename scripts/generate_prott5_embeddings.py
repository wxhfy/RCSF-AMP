#!/usr/bin/env python3
"""Generate ProtT5 embeddings for FASTA files using a local or remote checkpoint.

This is a self-contained variant of `prott5_embedder.py` that can run fully
offline by pointing `--model-path` to a directory containing the downloaded
ProtT5 weights (config.json, pytorch_model.bin, tokenizer config etc.).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import torch
from transformers import T5EncoderModel, T5Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ProtT5 embeddings for FASTA sequences")
    parser.add_argument('--input', '-i', required=True, type=Path,
                        help='Path to FASTA file (headers must be unique).')
    parser.add_argument('--output', '-o', required=True, type=Path,
                        help='Destination HDF5 file to store embeddings.')
    parser.add_argument('--model-path', type=Path, default=None,
                        help='Directory containing the ProtT5 checkpoint (config.json, tokenizer files, weights).')
    parser.add_argument('--per-protein', action='store_true',
                        help='If set, save mean pooled per-protein vectors instead of per-residue embeddings.')
    parser.add_argument('--max-residues', type=int, default=4000,
                        help='Upper bound for summed residue count per embedding batch.')
    parser.add_argument('--max-seq-len', type=int, default=1000,
                        help='Sequences longer than this are processed individually to avoid OOM errors.')
    parser.add_argument('--max-batch', type=int, default=100,
                        help='Maximum number of sequences per batch.')
    return parser.parse_args()


def read_fasta(fasta_path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current_id: str | None = None
    seq_chunks: List[str] = []
    with fasta_path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(seq_chunks)
                current_id = line[1:].replace('/', '_').replace('.', '_')
                seq_chunks = []
            else:
                seq_chunks.append(line.upper().replace('-', ''))
        if current_id is not None:
            sequences[current_id] = ''.join(seq_chunks)
    if not sequences:
        raise ValueError(f"No sequences found in FASTA {fasta_path}")
    return sequences


def load_model(model_path: Path | None) -> Tuple[T5EncoderModel, T5Tokenizer]:
    identifier = model_path if model_path is not None else 'Rostlab/prot_t5_xl_half_uniref50-enc'
    print(f"Loading ProtT5 checkpoint from: {identifier}")
    tokenizer = T5Tokenizer.from_pretrained(str(identifier), do_lower_case=False)
    model = T5EncoderModel.from_pretrained(str(identifier))

    if not torch.cuda.is_available():
        print('Casting model to float32 for CPU execution...')
        model.to(torch.float32)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    return model, tokenizer


def batch_iterator(items: List[Tuple[str, str]], max_batch: int, max_residues: int,
                   max_seq_len: int) -> Iterable[List[Tuple[str, str]]]:
    batch: List[Tuple[str, str]] = []
    residue_count = 0
    for idx, (identifier, seq) in enumerate(items, start=1):
        seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        seq_len = len(seq)
        batch.append((identifier, seq))
        residue_count += seq_len
        close_batch = (
            len(batch) >= max_batch
            or residue_count >= max_residues
            or seq_len > max_seq_len
            or idx == len(items)
        )
        if close_batch:
            yield batch
            batch = []
            residue_count = 0


def embed_sequences(model: T5EncoderModel, tokenizer: T5Tokenizer,
                    sequences: Dict[str, str], per_protein: bool,
                    max_residues: int, max_seq_len: int, max_batch: int,
                    output_path: Path) -> None:
    device = next(model.parameters()).device
    ordered = sorted(sequences.items(), key=lambda kv: len(kv[1]), reverse=True)
    t0 = time.time()
    written = 0

    with h5py.File(str(output_path), 'w') as handle:
        with torch.no_grad():
            for batch in batch_iterator(ordered, max_batch, max_residues, max_seq_len):
                ids = [identifier for identifier, _ in batch]
                tokenized = tokenizer([' '.join(list(seq)) for _, seq in batch],
                                      add_special_tokens=True, padding='longest',
                                      return_tensors='pt')
                input_ids = tokenized['input_ids'].to(device)
                attention_mask = tokenized['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state.cpu()
                for idx, identifier in enumerate(ids):
                    seq_len = len(batch[idx][1])
                    embedding = hidden_states[idx, :seq_len]
                    if per_protein:
                        embedding = embedding.mean(dim=0, keepdim=True)
                    handle.create_dataset(identifier, data=embedding.numpy())
                    written += 1

    elapsed = time.time() - t0
    print(f"Saved {written} embeddings to {output_path} in {elapsed:.2f}s")


def main() -> None:
    args = parse_args()
    sequences = read_fasta(args.input)
    model, tokenizer = load_model(args.model_path)
    embed_sequences(model, tokenizer, sequences, args.per_protein,
                    args.max_residues, args.max_seq_len, args.max_batch,
                    args.output)


if __name__ == '__main__':
    main()

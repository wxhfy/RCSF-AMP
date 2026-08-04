#!/usr/bin/env python3
"""Same-split reproductions of public 2026 sequence-only AMP baselines."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from modlamp.descriptors import GlobalDescriptor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model", choices=("ampidentifier", "cars_amp", "amp_capsnet"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(root: Path, split: str):
    sequences, labels, ids = [], [], []
    for filename in (root / f"{split}.txt").read_text().splitlines():
        filename = filename.strip()
        if not filename:
            continue
        graph = torch.load(root / "graphs" / filename, map_location="cpu")
        ids.append(str(graph.seq_id))
        sequences.append(str(graph.original_seq))
        labels.append(int(graph.y.item()))
    return ids, sequences, np.asarray(labels, dtype=np.int64)


def metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "ACC": float(accuracy_score(labels, predictions)),
        "AUC": float(roc_auc_score(labels, probabilities)),
        "AUPR": float(average_precision_score(labels, probabilities)),
        "SEN": float(tp / (tp + fn)),
        "SPEC": float(tn / (tn + fp)),
        "F1": float(f1_score(labels, predictions)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
    }


def physicochemical_features(sequences):
    descriptor = GlobalDescriptor(sequences)
    descriptor.calculate_all(amide=True)
    values = np.nan_to_num(np.asarray(descriptor.descriptor, dtype=np.float64))
    return values, list(descriptor.featurenames)


def dpc_features(sequences):
    amino_acid_index = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
    features = np.zeros((len(sequences), 400), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        valid_sequence = [amino_acid for amino_acid in sequence if amino_acid in amino_acid_index]
        for left, right in zip(valid_sequence, valid_sequence[1:]):
            pair_index = amino_acid_index[left] * 20 + amino_acid_index[right]
            features[row, pair_index] += 1.0
    return features


def run_ampidentifier(args, splits):
    train_x, names = physicochemical_features(splits["train"][1])
    val_x, _ = physicochemical_features(splits["val"][1])
    test_x, _ = physicochemical_features(splits["test"][1])
    scaler = StandardScaler().fit(train_x)
    train_x, val_x, test_x = map(scaler.transform, (train_x, val_x, test_x))
    train_y, val_y, test_y = (splits[key][2] for key in ("train", "val", "test"))
    models = {
        "rf": RandomForestClassifier(n_estimators=100, random_state=args.seed, class_weight="balanced", n_jobs=-1),
        "svm": SVC(probability=True, random_state=args.seed, class_weight="balanced"),
        "gb": GradientBoostingClassifier(n_estimators=100, random_state=args.seed),
    }
    for model in models.values():
        model.fit(train_x, train_y)
    val_probabilities = np.mean([model.predict_proba(val_x)[:, 1] for model in models.values()], axis=0)
    test_started = time.perf_counter()
    test_probabilities = np.mean([model.predict_proba(test_x)[:, 1] for model in models.values()], axis=0)
    elapsed = time.perf_counter() - test_started
    for name, model in models.items():
        joblib.dump(model, args.output_dir / f"{name}.pkl")
    joblib.dump(scaler, args.output_dir / "scaler.pkl")
    return val_probabilities, test_probabilities, elapsed, {"feature_names": names, "models": list(models)}


def encode(sequence, max_length=50):
    lookup = {aa: index + 1 for index, aa in enumerate(AMINO_ACIDS)}
    encoded = [lookup.get(aa, 0) for aa in sequence[:max_length]]
    return np.asarray(encoded + [0] * (max_length - len(encoded)), dtype=np.int64)


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.x = torch.as_tensor(np.stack([encode(sequence) for sequence in sequences]), dtype=torch.long)
        self.y = torch.as_tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, index): return self.x[index], self.y[index]


class CARSAMP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(21, 20, padding_idx=0)
        self.attention = nn.MultiheadAttention(20, 4, dropout=0.2, batch_first=True)
        self.gru = nn.GRU(20, 25, num_layers=3, dropout=0.2, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(150, 15), nn.LeakyReLU(0.2), nn.Dropout(0.25), nn.Linear(15, 1))
    def forward(self, tokens):
        embedded = self.embedding(tokens)
        attended, _ = self.attention(embedded, embedded, embedded, need_weights=False)
        _, hidden = self.gru(attended)
        return self.classifier(hidden.transpose(0, 1).reshape(tokens.size(0), -1)).squeeze(1)


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.x = torch.as_tensor(features, dtype=torch.float32)
        self.y = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class CapsuleLayer(nn.Module):
    def __init__(self, input_dim=8, num_capsules=10, capsule_dim=20, routings=3):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routings = routings
        self.projection = nn.Linear(input_dim, num_capsules * capsule_dim, bias=False)

    @staticmethod
    def squash(inputs):
        squared_norm = inputs.square().sum(dim=-1, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm) / torch.sqrt(squared_norm + 1e-7)
        return scale * inputs

    def forward(self, inputs):
        projected = self.projection(inputs)
        projected = projected.view(inputs.size(0), inputs.size(1), self.num_capsules, self.capsule_dim)
        projected = projected.permute(0, 2, 1, 3)
        routing_logits = projected.new_zeros(projected.shape[:-1])
        for routing_index in range(self.routings):
            coupling = routing_logits.softmax(dim=1)
            capsules = self.squash((coupling.unsqueeze(-1) * projected).sum(dim=2))
            if routing_index < self.routings - 1:
                routing_logits = routing_logits + (projected * capsules.unsqueeze(2)).sum(dim=-1)
        return capsules


class AMPCapsNet(nn.Module):
    def __init__(self, input_length=400):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 64, kernel_size=6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        with torch.no_grad():
            flattened_dim = self.convolutions(torch.zeros(1, 1, input_length)).numel()
        self.primary_capsules = nn.Sequential(nn.Flatten(), nn.Linear(flattened_dim, 128), nn.ReLU())
        self.capsules = CapsuleLayer(input_dim=8, num_capsules=10, capsule_dim=20, routings=3)
        self.classifier = nn.Linear(200, 1)

    def forward(self, features):
        features = self.convolutions(features.unsqueeze(1))
        primary_capsules = self.primary_capsules(features).view(features.size(0), -1, 8)
        return self.classifier(self.capsules(primary_capsules).flatten(1)).squeeze(1)


@torch.inference_mode()
def predict_torch(model, loader, device):
    model.eval(); probabilities=[]
    for tokens, _ in loader:
        probabilities.extend(torch.sigmoid(model(tokens.to(device))).cpu().numpy().tolist())
    return np.asarray(probabilities)


def train_torch_model(args, splits, datasets, model):
    device = torch.device(args.device)
    loaders = {
        key: DataLoader(dataset, batch_size=args.batch_size, shuffle=(key == "train"))
        for key, dataset in datasets.items()
    }
    model = model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters()) if args.model == "amp_capsnet" else torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_mcc, best_state, stale = -2.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        value = metrics(splits["val"][2], predict_torch(model, loaders["val"], device))["MCC"]
        if value > best_mcc:
            best_mcc, stale = value, 0
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.to(device)
    val_probabilities = predict_torch(model, loaders["val"], device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    test_probabilities = predict_torch(model, loaders["test"], device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    torch.save({"model_state_dict": best_state}, args.output_dir / "best.pt")
    return val_probabilities, test_probabilities, elapsed, {
        "epochs_run": epoch + 1,
        "best_validation_mcc": best_mcc,
        "params": sum(parameter.numel() for parameter in model.parameters()),
    }


def run_cars(args, splits):
    datasets = {key: SequenceDataset(value[1], value[2]) for key, value in splits.items()}
    return train_torch_model(args, splits, datasets, CARSAMP())


def run_amp_capsnet(args, splits):
    train_features = dpc_features(splits["train"][1])
    validation_features = dpc_features(splits["val"][1])
    test_features = dpc_features(splits["test"][1])
    scaler = StandardScaler().fit(train_features)
    train_features, validation_features, test_features = map(
        lambda values: scaler.transform(values).astype(np.float32),
        (train_features, validation_features, test_features),
    )
    joblib.dump(scaler, args.output_dir / "scaler.pkl")
    datasets = {
        "train": FeatureDataset(train_features, splits["train"][2]),
        "val": FeatureDataset(validation_features, splits["val"][2]),
        "test": FeatureDataset(test_features, splits["test"][2]),
    }
    result = train_torch_model(args, splits, datasets, AMPCapsNet())
    result[3]["features"] = "DPC (400), best configuration reported by AMP-CapsNet"
    result[3]["official_optimizer"] = "RMSprop"
    return result


def main():
    args = parse_args(); set_seed(args.seed); args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {key: load_split(args.data_root, key) for key in ("train", "val", "test")}
    started = time.perf_counter()
    if args.model == "ampidentifier":
        result = run_ampidentifier(args, splits)
    elif args.model == "cars_amp":
        result = run_cars(args, splits)
    else:
        result = run_amp_capsnet(args, splits)
    val_probabilities, test_probabilities, test_elapsed, details = result
    payload = {
        "model": args.model, "seed": args.seed, "threshold": 0.5,
        "split_sizes": {key: len(value[2]) for key, value in splits.items()},
        "validation": metrics(splits["val"][2], val_probabilities),
        "test": metrics(splits["test"][2], test_probabilities),
        "test_inference_ms_per_1000": test_elapsed * 1_000_000 / len(splits["test"][2]),
        "total_elapsed_seconds": time.perf_counter() - started,
        "details": details,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame({"id": splits["test"][0], "label": splits["test"][2], "probability": test_probabilities}).to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()

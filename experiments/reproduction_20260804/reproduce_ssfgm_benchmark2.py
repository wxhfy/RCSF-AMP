#!/usr/bin/env python3

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from Bio.PDB import PDBParser
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


MAX_LENGTH = 160
MAX_VERTICES = 5109
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/home/20T-1/fyh0106/SUCF/baseline/SSFGM-Model"),
    )
    parser.add_argument(
        "--protbert-dir",
        type=Path,
        default=Path(
            "/home/20T-1/fyh0106/compare2/merged_amp_decoy/embeddings/protbert_embedding"
        ),
    )
    parser.add_argument(
        "--esm-dir",
        type=Path,
        default=None,
        help="Override the ESM embedding directory.",
    )
    parser.add_argument(
        "--graph-data-root",
        type=Path,
        default=None,
        help="Use test.txt and graph .pt files directly from a merged dataset root.",
    )
    parser.add_argument(
        "--masif-root",
        type=Path,
        default=Path("/home/20T-1/fyh0106/compare2/masif_site"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ssfgm_reproduction_benchmark2"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override the checkpoint under baseline-root.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalized_records(fasta_path):
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequence_id = record.id.split("|")[0]
        records.append((sequence_id, str(record.seq)))
    return records


def load_and_pad_embedding(path, max_length):
    feature = np.squeeze(np.load(path))
    if feature.ndim != 2:
        raise ValueError(f"Expected a 2D embedding at {path}, got {feature.shape}")
    feature = feature[:max_length]
    if feature.shape[0] < max_length:
        padding = np.zeros(
            (max_length - feature.shape[0], feature.shape[1]), dtype=feature.dtype
        )
        feature = np.vstack([feature, padding])
    return feature


def one_hot(sequence, max_length):
    identity = np.eye(len(AMINO_ACIDS), dtype=np.float32)
    lookup = {amino_acid: identity[index] for index, amino_acid in enumerate(AMINO_ACIDS)}
    encoded = np.asarray(
        [lookup.get(amino_acid, np.zeros(len(AMINO_ACIDS), dtype=np.float32)) for amino_acid in sequence],
        dtype=np.float32,
    )[:max_length]
    if encoded.shape[0] < max_length:
        encoded = np.pad(encoded, ((0, max_length - encoded.shape[0]), (0, 0)))
    return encoded


def combined_node_features(sequence_id, sequence, protbert_dir, esm_dir):
    protbert = load_and_pad_embedding(protbert_dir / f"{sequence_id}.npy", MAX_LENGTH)
    esm = load_and_pad_embedding(esm_dir / f"{sequence_id}.npy", MAX_LENGTH)
    residue_identity = one_hot(sequence, MAX_LENGTH)
    combined = np.concatenate([protbert, esm, residue_identity], axis=1)
    if combined.shape != (MAX_LENGTH, 3604):
        raise ValueError(f"Unexpected combined feature shape for {sequence_id}: {combined.shape}")
    return combined.astype(np.float32, copy=False)


def residue_positions(pdb_path, max_residues):
    if pdb_path.suffix == ".pt":
        graph = torch.load(pdb_path, map_location="cpu")
        return graph.coords.detach().cpu().numpy()[:max_residues]
    structure = PDBParser(QUIET=True).get_structure("PDB", pdb_path)
    positions = []
    for chain in structure[0]:
        for residue in chain:
            if residue.id[0] == " " and "CA" in residue:
                positions.append(residue["CA"].coord)
                if len(positions) >= max_residues:
                    return positions
    return positions


def graph_edges(positions, cutoff=10.0):
    edges = []
    distances = []
    for source in range(len(positions)):
        for target in range(source + 1, len(positions)):
            distance = float(np.linalg.norm(positions[source] - positions[target]))
            if distance < cutoff:
                edges.extend([[source, target], [target, source]])
                distances.extend([distance, distance])
    if not edges:
        raise ValueError("No graph edges were generated")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(distances, dtype=torch.float32)
    return edge_index, edge_weight


class SSFGMDataset(Dataset):
    def __init__(self, entries, protbert_dir, esm_dir, masif_root):
        self.entries = entries
        self.protbert_dir = protbert_dir
        self.esm_dir = esm_dir
        self.masif_root = masif_root

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        sequence_id, sequence, label, pdb_path = self.entries[index]
        node_features = combined_node_features(
            sequence_id, sequence, self.protbert_dir, self.esm_dir
        )
        positions = residue_positions(pdb_path, MAX_LENGTH)
        edge_index, edge_weight = graph_edges(positions)
        graph = Data(
            x=torch.from_numpy(node_features),
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=torch.tensor([label], dtype=torch.long),
        )

        surface_dir = self.masif_root / sequence_id
        surface = {}
        filenames = {
            "input_feat": "p1_input_feat.npy",
            "rho_coords": "p1_rho_wrt_center.npy",
            "theta_coords": "p1_theta_wrt_center.npy",
            "mask": "p1_mask.npy",
        }
        for key, filename in filenames.items():
            values = np.nan_to_num(np.load(surface_dir / filename))
            values = values[:MAX_VERTICES]
            if values.shape[0] < MAX_VERTICES:
                padding = np.zeros(
                    (MAX_VERTICES - values.shape[0],) + values.shape[1:],
                    dtype=values.dtype,
                )
                values = np.concatenate([values, padding], axis=0)
            surface[key] = torch.tensor(values, dtype=torch.float32)
        return sequence_id, graph, surface


def collate_batch(items):
    sequence_ids = [item[0] for item in items]
    graph_batch = Batch.from_data_list([item[1] for item in items])
    surface_batch = {
        key: torch.stack([item[2][key] for item in items]) for key in items[0][2]
    }
    return sequence_ids, graph_batch, surface_batch


class ImprovedGCN(nn.Module):
    def __init__(self, num_features, num_classes, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, 1024)
        self.conv2 = GCNConv(1024, 512)
        self.conv3 = GCNConv(512, 256)
        self.conv4 = GCNConv(256, 128)
        self.conv5 = GCNConv(128, 64)
        self.conv6 = GCNConv(64, 32)
        self.attn1 = GATConv(32, 16 // heads, heads=heads, concat=True)
        self.fc = nn.Linear(16, num_classes)
        self.dropout = dropout

    def forward(self, data, return_features=False):
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv4(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv5(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv6(x, edge_index, edge_weight=edge_weight))
        x = F.elu(self.attn1(x, edge_index, edge_attr=edge_weight))
        x = global_mean_pool(x, batch)
        if return_features:
            return x
        return F.log_softmax(self.fc(x), dim=1)


class MaSIFSitePyTorch(nn.Module):
    def __init__(self, n_thetas, n_rhos, n_feat, n_rotations):
        super().__init__()
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.n_feat = n_feat
        self.n_rotations = n_rotations
        self.mu_rho = nn.Parameter(torch.empty(n_rotations, 1))
        self.sigma_rho = nn.Parameter(torch.empty(n_rotations, 1))
        self.mu_theta = nn.Parameter(torch.empty(n_rotations, 1))
        self.sigma_theta = nn.Parameter(torch.empty(n_rotations, 1))
        nn.init.uniform_(self.mu_rho, 0, 1)
        nn.init.constant_(self.sigma_rho, 0.5)
        nn.init.uniform_(self.mu_theta, 0, 2 * np.pi)
        nn.init.constant_(self.sigma_theta, 0.5)
        self.avgpool1d = nn.AvgPool1d(kernel_size=6, stride=5)
        self.fc1 = nn.Linear(40840, 2)

    def forward(self, input_feat, rho_coords, theta_coords, mask, return_features=False):
        batch_size, _, _, n_feat = input_feat.size()
        input_feat = input_feat.mean(dim=2)
        output_features = []
        for rotation in range(self.n_rotations):
            rotated_theta = theta_coords + rotation * 2 * np.pi / self.n_rotations
            rotated_theta %= 2 * np.pi
            rho_gauss = torch.exp(
                -torch.square(rho_coords - self.mu_rho[rotation])
                / (2 * torch.square(self.sigma_rho[rotation]) + 1e-5)
            )
            theta_gauss = torch.exp(
                -torch.square(rotated_theta - self.mu_theta[rotation])
                / (2 * torch.square(self.sigma_theta[rotation]) + 1e-5)
            )
            activations = rho_gauss * theta_gauss * mask
            activations /= torch.sum(activations, dim=1, keepdim=True) + 1e-5
            activations = activations.unsqueeze(3).expand(-1, -1, -1, n_feat)
            descriptor = torch.sum(activations * input_feat.unsqueeze(2), dim=2)
            output_features.append(descriptor)
        output = torch.cat(output_features, dim=2)
        output = self.avgpool1d(output.permute(0, 2, 1))
        output = output.permute(0, 2, 1).reshape(batch_size, -1)
        if return_features:
            return output
        return self.fc1(output)


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_gcn = ImprovedGCN(num_features=3604, num_classes=2)
        self.model_masif = MaSIFSitePyTorch(
            n_thetas=16, n_rhos=5, n_feat=5, n_rotations=8
        )
        self.reduce_masif = nn.Linear(40840, 16)
        self.fusion_layer = nn.Linear(32, 2)

    def forward(self, graph, surface):
        graph_features = self.model_gcn(graph, return_features=True)
        surface_features = self.model_masif(
            surface["input_feat"],
            surface["rho_coords"],
            surface["theta_coords"],
            surface["mask"],
            return_features=True,
        )
        surface_features = F.relu(self.reduce_masif(surface_features))
        combined = F.relu(torch.cat([graph_features, surface_features], dim=1))
        return F.log_softmax(self.fusion_layer(combined), dim=1)


def build_entries(data_root):
    specifications = [
        (data_root / "fasta/AMP_test.fasta", 1, data_root / "pdb/amp"),
        (data_root / "fasta/DECOY_test.fasta", 0, data_root / "pdb/decoy"),
    ]
    entries = []
    for fasta_path, label, pdb_dir in specifications:
        for sequence_id, sequence in normalized_records(fasta_path):
            entries.append((sequence_id, sequence, label, pdb_dir / f"{sequence_id}.pdb"))
    return entries


def build_entries_from_graph_split(graph_data_root):
    entries = []
    split_path = graph_data_root / "test.txt"
    for filename in split_path.read_text(encoding="utf-8").splitlines():
        filename = filename.strip()
        if not filename:
            continue
        graph_path = graph_data_root / "graphs" / filename
        graph = torch.load(graph_path, map_location="cpu")
        entries.append(
            (
                str(graph.seq_id),
                str(graph.original_seq),
                int(graph.y.item()),
                graph_path,
            )
        )
    return entries


def validate_inputs(entries, protbert_dir, esm_dir, masif_root, checkpoint):
    missing = []
    for sequence_id, _, _, pdb_path in entries:
        paths = [
            protbert_dir / f"{sequence_id}.npy",
            esm_dir / f"{sequence_id}.npy",
            pdb_path,
            masif_root / sequence_id / "p1_input_feat.npy",
            masif_root / sequence_id / "p1_rho_wrt_center.npy",
            masif_root / sequence_id / "p1_theta_wrt_center.npy",
            masif_root / sequence_id / "p1_mask.npy",
        ]
        missing.extend(str(path) for path in paths if not path.is_file())
    if not checkpoint.is_file():
        missing.append(str(checkpoint))
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing[:50]))


def calculate_metrics(labels, probabilities, predictions):
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    return {
        "n_samples": int(len(labels)),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "sensitivity": float(true_positive / (true_positive + false_negative)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "confusion_matrix": matrix.tolist(),
    }


def main():
    args = parse_args()
    set_deterministic(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.baseline_root / "b2_test_data"
    esm_dir = args.esm_dir or (data_root / "esm_npy")
    checkpoint = args.checkpoint or (args.baseline_root / "SSFGM-Model.pth")
    if args.graph_data_root is None:
        entries = build_entries(data_root)
    else:
        entries = build_entries_from_graph_split(args.graph_data_root)
    validate_inputs(entries, args.protbert_dir, esm_dir, args.masif_root, checkpoint)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = SSFGMDataset(entries, args.protbert_dir, esm_dir, args.masif_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    model = FusionModel().to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    sequence_ids = []
    labels = []
    probabilities = []
    predictions = []
    started = time.time()
    with torch.no_grad():
        for batch_index, (batch_ids, graph, surface) in enumerate(loader, start=1):
            graph = graph.to(device)
            surface = {key: value.to(device) for key, value in surface.items()}
            output = model(graph, surface)
            probability = torch.softmax(output, dim=1)[:, 1]
            prediction = torch.argmax(output, dim=1)
            sequence_ids.extend(batch_ids)
            labels.extend(graph.y.cpu().numpy().tolist())
            probabilities.extend(probability.cpu().numpy().tolist())
            predictions.extend(prediction.cpu().numpy().tolist())
            print(
                f"batch={batch_index}/{len(loader)} samples={len(sequence_ids)}/{len(dataset)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    metrics = calculate_metrics(labels_array, probabilities_array, predictions_array)
    metrics.update(
        {
            "model": "SSFGM-Model",
            "checkpoint": str(checkpoint),
            "protbert_dir": str(args.protbert_dir),
            "esm_dir": str(esm_dir),
            "masif_root": str(args.masif_root),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "device": str(device),
            "elapsed_seconds": time.time() - started,
        }
    )
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        sequence_ids=np.asarray(sequence_ids),
        labels=labels_array,
        probabilities=probabilities_array,
        predictions=predictions_array,
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

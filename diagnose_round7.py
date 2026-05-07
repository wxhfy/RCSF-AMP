"""Round 7 per-bin specialization diagnosis.

Goal: verify that the redesigned modules behave as the per-bin specialization
narrative predicts:
  * pLDDT Gate: hurts low-bin most when ablated   (wo_plddt_gate)
  * SCGC mid-peaked: hurts mid-bin most            (wo_scgc)
  * Bi-Mamba reliability: hurts high-bin most      (wo_bimamba)
  * Quality router (Fusion): mixes them adaptively per graph

Loads ckpts from `outputs/clean_ablation_v2_1/B2/{RUN_ID}/{ablation}/seed{seed}/`
and computes per-graph (mean pLDDT bin) MCC for the test set.

Run:
    /home/fyh0106/miniconda3/envs/sucf/bin/python diagnose_round7.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.sucf_clean_ablation import create_clean_ablation_model
from utils.config_utils import load_config
from utils.datasets import AMPGraphDataset
from utils.metrics import calculate_metrics


RUN_ID = 'round6_20260506_201243'
ROOT = Path('outputs/clean_ablation_v2_1/B2') / RUN_ID
DATA_ROOT = '/home/20T-1/fyh0106/compare2/merged_amp_decoy/'
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
ABLATIONS = [
    'full',
    'wo_plddt_gate_keep_structure',
    'wo_scgc_keep_structure',
    'wo_bimamba_keep_fusion',
]
SEEDS = [37, 42, 123]


def patched_b2_config(config):
    config['paths']['data_root'] = DATA_ROOT
    extra = config.get('data', {}).get('extra_embeddings', {}) or {}
    for key in ['esm_amp', 'protbert']:
        if key in extra and 'path' in extra[key]:
            extra[key]['path'] = extra[key]['path'].replace('/compare/', '/compare2/')
    return config


def gather_predictions(model, loader, device):
    model.eval()
    probs, labels, plddts, gates, reliabs, scgc_strs, router_ws, batch_idx_all = [], [], [], [], [], [], [], []
    n_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            p = torch.sigmoid(out['activity_pred'].squeeze(-1)).cpu().numpy()
            t = batch.y.cpu().numpy()
            probs.append(p)
            labels.append(t)
            plddts.append(batch.plddt.cpu().numpy())
            batch_idx_all.append(batch.batch.cpu().numpy() + n_graphs)
            n_graphs += int(batch.batch.max().item() + 1)
            if out.get('gate_per_residue') is not None:
                gates.append(out['gate_per_residue'].view(-1).cpu().numpy())
            else:
                gates.append(np.full(batch.plddt.shape[0], np.nan))
            if out.get('reliability_per_residue') is not None:
                reliabs.append(out['reliability_per_residue'].view(-1).cpu().numpy())
            else:
                reliabs.append(np.full(batch.plddt.shape[0], np.nan))
            if out.get('scgc_strength_per_residue') is not None:
                scgc_strs.append(out['scgc_strength_per_residue'].view(-1).cpu().numpy())
            else:
                scgc_strs.append(np.full(batch.plddt.shape[0], np.nan))
            if out.get('router_weights_per_graph') is not None:
                router_ws.append(out['router_weights_per_graph'].cpu().numpy())
    return {
        'probs': np.concatenate(probs),
        'labels': np.concatenate(labels),
        'plddt': np.concatenate(plddts),
        'gate': np.concatenate(gates),
        'reliability': np.concatenate(reliabs),
        'scgc_strength': np.concatenate(scgc_strs),
        'router_w': np.concatenate(router_ws) if router_ws else None,
        'batch_idx': np.concatenate(batch_idx_all),
    }


def per_graph_pLDDT_means(test_ds):
    return np.array([float(d.plddt.mean().item()) for d in test_ds])


def main():
    config = patched_b2_config(load_config('configs/training_config.yaml'))
    test_ds = AMPGraphDataset(DATA_ROOT, split_file=str(Path(DATA_ROOT) / 'test.txt'))
    test_loader = PyGDataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    print(f'Test set: {len(test_ds)} graphs')

    graph_means = per_graph_pLDDT_means(test_ds)
    bin_edges = [(0, 65), (65, 75), (75, 100)]
    print(f'Per-bin graph counts:')
    for lo, hi in bin_edges:
        mask = (graph_means >= lo) & (graph_means < hi)
        print(f'  pLDDT[{lo}, {hi}): n={int(mask.sum())}')

    # Aggregate per (ablation, seed, bin) MCC
    per_bin_results = {}  # (ablation, bin) -> [mcc per seed]
    for abl in ABLATIONS:
        per_bin_results[abl] = {f'{lo}-{hi}': [] for lo, hi in bin_edges}

    print('\n=== Per-bin MCC (test set, all seeds) ===')
    for abl in ABLATIONS:
        for seed in SEEDS:
            ckpt_path = ROOT / abl / f'seed{seed}' / 'best_model.pth'
            if not ckpt_path.exists():
                print(f'[skip] {abl}/seed{seed}: no ckpt')
                continue
            print(f'\n--- {abl} / seed{seed} ---')
            model = create_clean_ablation_model(config, abl).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if missing:
                print(f'  [info] {len(missing)} missing keys (e.g. {missing[:2]})')

            diag = gather_predictions(model, test_loader, DEVICE)
            test_metrics = calculate_metrics(diag['labels'], diag['probs'])
            print(f"  test_mcc={test_metrics['mcc']:.4f} auc={test_metrics['auc']:.4f}")

            # Per-bin MCC
            for lo, hi in bin_edges:
                mask = (graph_means >= lo) & (graph_means < hi)
                if mask.sum() < 10:
                    continue
                sub = calculate_metrics(diag['labels'][mask], diag['probs'][mask])
                per_bin_results[abl][f'{lo}-{hi}'].append(sub['mcc'])
                print(f'    bin[{lo},{hi}): n={int(mask.sum())} mcc={sub["mcc"]:.4f}')

            # Round 7 specifics for full only
            if abl == 'full':
                if not np.isnan(diag['reliability']).all():
                    print('  Reliability by pLDDT bin (per residue):')
                    for lo, hi in [(0, 50), (50, 70), (70, 90), (90, 101)]:
                        mask = (diag['plddt'] >= lo) & (diag['plddt'] < hi)
                        if mask.any():
                            print(f'    plddt[{lo},{hi}): n={int(mask.sum())} reliability={diag["reliability"][mask].mean():.4f}')
                if not np.isnan(diag['scgc_strength']).all():
                    print('  SCGC strength by pLDDT bin:')
                    for lo, hi in [(0, 50), (50, 70), (70, 90), (90, 101)]:
                        mask = (diag['plddt'] >= lo) & (diag['plddt'] < hi)
                        if mask.any():
                            print(f'    plddt[{lo},{hi}): n={int(mask.sum())} scgc_strength={diag["scgc_strength"][mask].mean():.4f}')
                if diag['router_w'] is not None:
                    rw = diag['router_w']
                    print('  Router weights by graph mean pLDDT bin:')
                    for lo, hi in bin_edges:
                        mask = (graph_means >= lo) & (graph_means < hi)
                        if mask.sum() < 10:
                            continue
                        ws = rw[mask].mean(axis=0)
                        print(f'    bin[{lo},{hi}): w_seq={ws[0]:.3f} w_mid={ws[1]:.3f} w_high={ws[2]:.3f}')

            del model
            torch.cuda.empty_cache()

    # Per-bin Δ (ablation - full) summary
    print('\n=== Per-bin Δ MCC vs full (mean across 3 seeds) ===')
    print(f'{"Ablation":35s} | bin[0,65)  bin[65,75)  bin[75,100)')
    full_per_bin = {b: np.mean(per_bin_results['full'][b]) if per_bin_results['full'][b] else np.nan
                    for b in [f'{lo}-{hi}' for lo, hi in bin_edges]}
    print(f'{"full (reference)":35s} | ' + '  '.join([f'{full_per_bin[b]:.4f}'
        for b in [f'{lo}-{hi}' for lo, hi in bin_edges]]))
    for abl in ABLATIONS:
        if abl == 'full':
            continue
        deltas = []
        for lo, hi in bin_edges:
            b = f'{lo}-{hi}'
            if not per_bin_results[abl][b] or not per_bin_results['full'][b]:
                deltas.append(np.nan)
            else:
                d = np.mean(per_bin_results[abl][b]) - np.mean(per_bin_results['full'][b])
                deltas.append(d)
        print(f'{abl:35s} | ' + '  '.join([f'{d:+.4f}' if not np.isnan(d) else 'nan' for d in deltas]))

    # Per-seed paired Δ for sign consistency
    print('\n=== Per-seed paired Δ (each seed) ===')
    for abl in ABLATIONS:
        if abl == 'full':
            continue
        print(f'\n  {abl}:')
        for lo, hi in bin_edges:
            b = f'{lo}-{hi}'
            full_seeds = per_bin_results['full'][b]
            abl_seeds = per_bin_results[abl][b]
            if len(full_seeds) != len(abl_seeds):
                continue
            line = f'    bin[{lo},{hi}): '
            for i in range(len(full_seeds)):
                d = abl_seeds[i] - full_seeds[i]
                line += f'seed{SEEDS[i]}={d:+.4f}  '
            print(line)


if __name__ == '__main__':
    main()

"""Round-6 seed-123 outlier diagnosis.

Loads the trained checkpoints for full / wo_scgc / wo_bimamba on seed 123 and
inspects:
  1. Learned PLDDTGating threshold / slope (Fix 1 stickiness)
  2. Per-bin gate scalar mean (4 pLDDT bins) — does the gate stay monotone?
  3. Reliability MLP outputs by pLDDT bin — does the round-6 reliability fire
     differently for the seed-123 outlier?
  4. Per-pLDDT-bin test MCC — where does full lose ground vs wo_scgc on seed 123?
  5. Train/val/test pLDDT distribution comparison.

Run from repo root:
    /home/fyh0106/miniconda3/envs/sucf_run/bin/python diagnose_seed123.py
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


RUN_ID = 'round6_20260506_123248'
ROOT = Path('outputs/clean_ablation_v2_1/B2') / RUN_ID
DATA_ROOT = '/home/20T-1/fyh0106/compare2/merged_amp_decoy/'
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
SEED = 123


def patched_b2_config(config):
    config['paths']['data_root'] = DATA_ROOT
    extra = config.get('data', {}).get('extra_embeddings', {}) or {}
    for key in ['esm_amp', 'protbert']:
        if key in extra:
            extra[key]['path'] = extra[key]['path'].replace('/compare/', '/compare2/')
    return config


def load_ckpt(ablation: str):
    ckpt_path = ROOT / ablation / f'seed{SEED}' / 'best_model.pth'
    return torch.load(ckpt_path, map_location='cpu')


def gather_per_residue(model, loader, device):
    """Run model on a loader, collect per-residue gate, plddt, reliability, and per-graph (label, prob)."""
    model.eval()
    plddt_list, gate_list, reliability_list = [], [], []
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            p = torch.sigmoid(out['activity_pred'].squeeze(-1)).cpu().numpy()
            t = batch.y.cpu().numpy()
            probs.append(p)
            labels.append(t)
            plddt_list.append(batch.plddt.cpu().numpy())
            if out.get('gate_per_residue') is not None:
                gate_list.append(out['gate_per_residue'].view(-1).cpu().numpy())
            else:
                gate_list.append(np.full(batch.plddt.shape[0], np.nan))
            if out.get('reliability_per_residue') is not None:
                reliability_list.append(out['reliability_per_residue'].view(-1).cpu().numpy())
            else:
                reliability_list.append(np.full(batch.plddt.shape[0], np.nan))
    return {
        'plddt': np.concatenate(plddt_list),
        'gate': np.concatenate(gate_list),
        'reliability': np.concatenate(reliability_list),
        'probs': np.concatenate(probs),
        'labels': np.concatenate(labels),
    }


def bin_stats(x, plddt, bins=((0, 50), (50, 70), (70, 90), (90, 101))):
    out = []
    for lo, hi in bins:
        mask = (plddt >= lo) & (plddt < hi)
        if mask.any():
            vals = x[mask]
            out.append((f'{lo}-{hi}', int(mask.sum()), float(np.nanmean(vals))))
        else:
            out.append((f'{lo}-{hi}', 0, float('nan')))
    return out


def main():
    config = patched_b2_config(load_config('configs/training_config.yaml'))
    test_ds = AMPGraphDataset(DATA_ROOT, split_file=str(Path(DATA_ROOT) / 'test.txt'))
    val_ds = AMPGraphDataset(DATA_ROOT, split_file=str(Path(DATA_ROOT) / 'val.txt'))
    test_loader = PyGDataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    val_loader = PyGDataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    print(f'=== Seed {SEED} outlier diagnosis (round-6) ===')
    print(f'Test set: {len(test_ds)} graphs')
    print(f'Val set:  {len(val_ds)} graphs')

    for ablation in ['full', 'wo_scgc_keep_structure', 'wo_bimamba_keep_fusion',
                     'wo_plddt_gate_keep_structure']:
        print(f'\n--- {ablation} ---')
        model = create_clean_ablation_model(config, ablation).to(DEVICE)
        ckpt = load_ckpt(ablation)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f'  [info] {len(missing)} missing keys (e.g. {missing[:2]})')
        if unexpected:
            print(f'  [info] {len(unexpected)} unexpected keys (e.g. {unexpected[:2]})')

        # Gate parameter inspection
        if hasattr(model, 'plddt_gating') and model.plddt_gating is not None:
            g = model.plddt_gating
            delta_t = (10.0 * torch.tanh(g.delta_t_raw)).item()
            slope = (torch.nn.functional.softplus(g.raw_slope) + 5.0).item()
            threshold = g.init_threshold + delta_t
            print(f'  Learned gate: threshold={threshold:.2f} (delta={delta_t:+.3f}) slope={slope:.2f} gate_min={g.gate_min}')

        # Per-residue diagnostics on test set
        diag = gather_per_residue(model, test_loader, DEVICE)
        test_metrics = calculate_metrics(diag['labels'], diag['probs'])
        print(f"  test_mcc={test_metrics['mcc']:.4f} auc={test_metrics['auc']:.4f}")

        # Per-bin gate
        if not np.isnan(diag['gate']).all():
            print('  Gate by pLDDT bin:')
            for lo_hi, n, m in bin_stats(diag['gate'], diag['plddt']):
                print(f'    {lo_hi:>8s}: n={n:>5d}  mean_gate={m:.4f}')
        else:
            print('  Gate disabled for this ablation.')

        # Per-bin reliability
        if not np.isnan(diag['reliability']).all():
            print('  Reliability by pLDDT bin:')
            for lo_hi, n, m in bin_stats(diag['reliability'], diag['plddt']):
                print(f'    {lo_hi:>8s}: n={n:>5d}  mean_relia={m:.4f}')

        # Per-bin MCC: compare full vs ablations on the same test set
        # We need per-graph mean pLDDT then bin graphs.
        # Group residues by graph using graph batch boundaries (rebuild here from dataset).
        print('  Per-graph MCC by mean pLDDT bin:')
        # Recompute per-graph mean pLDDT
        graph_means = []
        for d in test_ds:
            graph_means.append(float(d.plddt.mean().item()))
        graph_means = np.array(graph_means)
        for lo, hi in [(0, 65), (65, 75), (75, 100)]:
            mask = (graph_means >= lo) & (graph_means < hi)
            if mask.sum() < 10:
                continue
            sub = calculate_metrics(diag['labels'][mask], diag['probs'][mask])
            print(f'    pLDDT[{lo},{hi}): n={int(mask.sum()):>4d} mcc={sub["mcc"]:.4f} auc={sub["auc"]:.4f}')

        del model
        torch.cuda.empty_cache()

    # Train/val/test pLDDT distribution
    print('\n=== Dataset pLDDT distribution ===')
    for split, ds in [('val', val_ds), ('test', test_ds)]:
        all_p = np.concatenate([d.plddt.numpy() for d in ds])
        print(f'  {split} (n={len(ds)} graphs, {len(all_p)} residues):'
              f' mean={all_p.mean():.2f} pct<70={(all_p<70).mean()*100:.1f}%'
              f' pct<50={(all_p<50).mean()*100:.1f}%')


if __name__ == '__main__':
    main()

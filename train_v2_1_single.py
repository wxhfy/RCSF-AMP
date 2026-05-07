#!/usr/bin/env python3
"""
V2.1 Single-task launcher: TWO-STAGE training matching paper narrative.
Stage 1: Feature alignment (15 epochs, alignment_contrastive only)
Stage 2: Fine-tuning (35 epochs, activity + supervised_contrastive)

This matches train_new_baselines.py and the paper's claimed two-stage protocol.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from torch_geometric.loader import DataLoader as PyGDataLoader

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from models.sucf_clean_ablation import create_clean_ablation_model
from utils.sucf_losses import create_sucf_loss_function
from utils.config_utils import load_config
from utils.datasets import AMPGraphDataset
from utils.metrics import calculate_metrics
from utils.early_stopping import EarlyStopping


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            p = torch.sigmoid(out['activity_pred'].squeeze(-1)).cpu().numpy()
            t = batch.y.cpu().numpy()
            probs.append(p)
            targets.append(t)
    return calculate_metrics(np.concatenate(targets), np.concatenate(probs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--benchmark', choices=['B1', 'B2'], required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--s1_epochs', type=int, default=15)
    parser.add_argument('--s2_epochs', type=int, default=35)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='AdamW weight decay for both stages.')
    args = parser.parse_args()

    data_roots = {
        'B1': '/home/20T-1/fyh0106/compare/merged_amp_decoy/',
        'B2': '/home/20T-1/fyh0106/compare2/merged_amp_decoy/',
    }
    data_root = data_roots[args.benchmark]

    config = load_config('configs/training_config.yaml')
    config['paths']['data_root'] = data_root
    if args.benchmark == 'B2':
        for key in ['esm_amp', 'protbert']:
            extra_emb = config.get('data', {}).get('extra_embeddings', {})
            if key in extra_emb:
                old_path = extra_emb[key].get('path', '')
                if 'compare' in old_path and 'compare2' not in old_path:
                    extra_emb[key]['path'] = old_path.replace('/compare/', '/compare2/')

    out_dir = project_root / 'outputs' / 'clean_ablation_v2_1' / args.benchmark / args.run_id / args.ablation / f'seed{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)

    model = create_clean_ablation_model(config, args.ablation).to(device)
    loss_fn = create_sucf_loss_function(config)

    # data
    train_ds = AMPGraphDataset(data_root, split_file=str(Path(data_root) / 'train.txt'))
    val_ds = AMPGraphDataset(data_root, split_file=str(Path(data_root) / 'val.txt'))
    test_ds = AMPGraphDataset(data_root, split_file=str(Path(data_root) / 'test.txt'))
    train_loader = PyGDataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = PyGDataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    test_loader = PyGDataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    # ============= Stage 1: Feature Alignment =============
    s1_lr = 2.27e-5
    opt_s1 = torch.optim.AdamW(model.parameters(), lr=s1_lr, weight_decay=args.weight_decay)
    stage1_info = {
        'active_losses': ['alignment_contrastive'],
        'loss_weights': {'activity': 0.0, 'alignment_contrastive': 1.0, 'supervised_contrastive': 0.0},
    }

    t0 = time.time()
    for epoch in range(args.s1_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt_s1.zero_grad()
            out = model(batch)
            loss_dict = loss_fn(out, batch, stage1_info)
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_s1.step()

    s1_time = time.time() - t0

    # ============= Stage 2: Fine-tuning =============
    s2_lr = 1.0e-5
    opt_s2 = torch.optim.AdamW(model.parameters(), lr=s2_lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=args.s2_epochs)
    early = EarlyStopping(patience=args.patience, mode='max', verbose=False)

    stage2_info = {
        # FIX (B2 anomaly diag): include alignment_contrastive in active_losses
        # so the configured weight 0.1 actually contributes to Stage 2 loss
        # (matches paper narrative of preserving cross-modal alignment).
        # FIX (round 6): also include gate_monotonicity to prevent the pLDDT gate
        # from collapsing to a flat ~0.5 across all pLDDT (root cause of the
        # previous B2 ablation anomaly).
        'active_losses': ['activity', 'supervised_contrastive', 'alignment_contrastive', 'gate_monotonicity', 'reliability_reg'],
        'loss_weights': {
            'activity': 1.0,
            'alignment_contrastive': 0.1,
            'supervised_contrastive': 0.5,
            'gate_monotonicity': 0.05,
            'reliability_reg': 0.02,
        },
    }

    best_val = -float('inf')
    best_state = None
    epochs_run = 0

    t1 = time.time()
    for epoch in range(args.s2_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt_s2.zero_grad()
            out = model(batch)
            loss_dict = loss_fn(out, batch, stage2_info)
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_s2.step()
        sched.step()
        epochs_run += 1

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics['mcc'] > best_val:
            best_val = val_metrics['mcc']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if early(val_metrics['mcc']):
            break

    s2_time = time.time() - t1

    # final test
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    # save best checkpoint to disk so post-hoc analysis can use it
    torch.save(best_state, out_dir / 'best_model.pth')

    elapsed = time.time() - t0
    result = {
        'ablation': args.ablation,
        'seed': args.seed,
        'benchmark': args.benchmark,
        'best_val_mcc': float(best_val),
        'test_mcc': float(test_metrics['mcc']),
        'test_auc': float(test_metrics['auc']),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1': float(test_metrics['f1']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        's1_time': s1_time,
        's2_time': s2_time,
        's2_epochs_run': epochs_run,
        'elapsed_sec': elapsed,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[DONE-V2.1] {args.benchmark}/{args.ablation}/seed{args.seed} "
          f"val={best_val:.4f} test={test_metrics['mcc']:.4f} "
          f"s1={s1_time:.0f}s s2={s2_time:.0f}s ({epochs_run}ep)")


if __name__ == '__main__':
    main()

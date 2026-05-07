"""Round 7 smoke test: verify forward + backward + per-bin behavior of the
redesigned reliability / SCGC mid-peak / quality router."""

import sys
sys.path.insert(0, '/home/fyh0106/SUCF')

import torch
from pathlib import Path
from torch_geometric.loader import DataLoader as PyGDataLoader
from models.sucf_clean_ablation import create_clean_ablation_model
from utils.config_utils import load_config
from utils.datasets import AMPGraphDataset


DATA_ROOT = '/home/20T-1/fyh0106/compare2/merged_amp_decoy/'


def main():
    config = load_config('configs/training_config.yaml')
    # Patch to B2 paths so the smoke test loads real graphs
    config['paths']['data_root'] = DATA_ROOT
    extra = config.get('data', {}).get('extra_embeddings', {}) or {}
    for key in ['esm_amp', 'protbert']:
        if key in extra and 'path' in extra[key]:
            extra[key]['path'] = extra[key]['path'].replace('/compare/', '/compare2/')

    test_ds = AMPGraphDataset(DATA_ROOT, split_file=str(Path(DATA_ROOT) / 'test.txt'))
    loader = PyGDataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader)).to('cuda:2')

    model = create_clean_ablation_model(config, 'full').to('cuda:2')
    model.train()

    print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')

    # Forward
    out = model(batch)
    print(f'\n=== Forward shapes ===')
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: {tuple(v.shape)}  device={v.device}  range=[{v.min().item():.4f}, {v.max().item():.4f}]')
        else:
            print(f'  {k}: {type(v).__name__}')

    # Per-graph router weights
    print(f'\n=== Per-graph quality + router weights ===')
    plddt = batch.plddt.detach().cpu()
    batch_idx = batch.batch.detach().cpu()
    w = out['router_weights_per_graph'].detach().cpu()
    for g in range(min(4, w.size(0))):
        m_pl = plddt[batch_idx == g].mean().item()
        print(f'  graph {g}: mean_plddt={m_pl:.1f}  router_w=[seq={w[g, 0]:.3f}, mid={w[g, 1]:.3f}, high={w[g, 2]:.3f}]')

    # Per-residue reliability + plddt monotonicity
    print(f'\n=== Reliability monotonicity check (should ↗ in pLDDT) ===')
    reliab = out['reliability_per_residue'].detach().cpu().squeeze(-1)
    pl = batch.plddt.detach().cpu()
    bins = [(0, 50), (50, 70), (70, 90), (90, 101)]
    for lo, hi in bins:
        mask = (pl >= lo) & (pl < hi)
        if mask.any():
            print(f'  pLDDT [{lo:>3d},{hi:>3d}): n={mask.sum().item():>4d}  reliability={reliab[mask].mean().item():.4f}')

    # SCGC strength per pLDDT bin (should peak in mid bin ~70)
    print(f'\n=== SCGC strength per pLDDT bin (should peak ~70) ===')
    scgc = out['scgc_strength_per_residue'].detach().cpu().squeeze(-1)
    for lo, hi in bins:
        mask = (pl >= lo) & (pl < hi)
        if mask.any():
            print(f'  pLDDT [{lo:>3d},{hi:>3d}): scgc_strength={scgc[mask].mean().item():.4f}')

    # Backward: ensure no NaN gradient
    loss = out['activity_pred'].sum()
    loss.backward()
    print(f'\n=== Backward check ===')
    nan_grads = []
    none_grads = []
    for n, p in model.named_parameters():
        if p.grad is None:
            none_grads.append(n)
        elif torch.isnan(p.grad).any():
            nan_grads.append(n)
    print(f'  NaN grads: {len(nan_grads)}  None grads: {len(none_grads)}')
    if nan_grads:
        print(f'  NaN grad params: {nan_grads[:5]}')

    # New round 7 params should have nonzero grad
    print(f'\n=== Round 7 param grads ===')
    for name in ['reliability_q_scale', 'reliability_q_bias', 'scgc_mu_mid',
                 'scgc_log_sigma_mid', 'scgc_beta']:
        p = dict(model.named_parameters())[name]
        g = p.grad.item() if p.grad is not None else float('nan')
        print(f'  grad[{name}] = {g:+.6e}')


if __name__ == '__main__':
    main()

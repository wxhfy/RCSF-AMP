#!/usr/bin/env python3
"""V2.2 round-6 redesign dispatcher (12 jobs minimum).

Validates that the architecture-level fixes (sharp pLDDT gate, RGAT-internal
sender confidence gate, log-bias SCGC, neighbour-conf SGFN, reliability-aware
Bi-Mamba, gate monotonicity regulariser) eliminate the B2 ablation anomaly.

Ablations: full / wo_plddt_gate_keep_structure / wo_scgc_keep_structure / wo_bimamba_keep_fusion
Seeds: 37 42 123
Benchmark: B2 (primary). Add B1 by passing --include-b1 (24 jobs total).
GPU slots: greedy 8-way fan-out across cuda:0..3.

Usage:
  /home/fyh0106/miniconda3/envs/sucf_run/bin/python dispatch_v2_2.py [--include-b1]
"""
import argparse
import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime
from queue import Queue
from threading import Thread, Lock

DEFAULT_ABLATIONS = [
    'full',
    'wo_plddt_gate_keep_structure',
    'wo_scgc_keep_structure',
    'wo_bimamba_keep_fusion',
]
DEFAULT_SEEDS = [37, 42, 123]
DEFAULT_SLOTS = ['cuda:0', 'cuda:0', 'cuda:1', 'cuda:1',
                 'cuda:2', 'cuda:2', 'cuda:3', 'cuda:3']

PYTHON = '/home/fyh0106/miniconda3/envs/sucf/bin/python'
SCRIPT = 'train_v2_1_single.py'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-b1', action='store_true',
                        help='Also run B1 (24 total jobs instead of 12).')
    parser.add_argument('--ablations', nargs='+', default=DEFAULT_ABLATIONS)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--slots', nargs='+', default=DEFAULT_SLOTS)
    parser.add_argument('--s2_epochs', type=int, default=35)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    args = parser.parse_args()

    benchmarks = ['B2'] + (['B1'] if args.include_b1 else [])

    run_id = 'round6_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs/ablation_round6') / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(b, abl, s) for b in benchmarks for abl in args.ablations for s in args.seeds]

    print('=== SUCF round-6 redesign dispatcher ===', flush=True)
    print('Run ID:', run_id, flush=True)
    print('Benchmarks:', benchmarks, flush=True)
    print('Ablations:', args.ablations, flush=True)
    print('Seeds:', args.seeds, flush=True)
    print('Total jobs:', len(jobs), flush=True)
    print('GPU slots:', args.slots, flush=True)
    print('Logs:', log_dir, flush=True)

    q = Queue()
    for j in jobs:
        q.put(j)

    lock = Lock()
    state = {'done': 0, 'failed': 0, 'started': time.time(), 'retries': {}}

    def worker(slot_idx, device):
        while True:
            try:
                bench, abl, seed = q.get(timeout=1)
            except Exception:
                return
            log_file = log_dir / f'{bench}_{abl}_seed{seed}.log'
            cmd = [
                PYTHON, SCRIPT,
                '--ablation', abl,
                '--seed', str(seed),
                '--benchmark', bench,
                '--device', device,
                '--run_id', run_id,
                '--s1_epochs', '15',
                '--s2_epochs', str(args.s2_epochs),
                '--patience', str(args.patience),
                '--weight_decay', str(args.weight_decay),
            ]
            mode = 'a' if log_file.exists() and log_file.stat().st_size > 0 else 'w'
            with open(log_file, mode) as f:
                f.write(f'# slot {slot_idx} on {device}\n')
                f.write('# ' + ' '.join(cmd) + '\n')
                f.flush()
                try:
                    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                          check=False, timeout=5400)
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    rc = -1
                    f.write('\n# TIMEOUT 5400s\n')
            # Detect OOM and re-queue once with a small back-off.
            if rc != 0:
                try:
                    tail = log_file.read_text(errors='ignore').splitlines()[-50:]
                except Exception:
                    tail = []
                is_oom = any('OutOfMemoryError' in line or 'CUDA out of memory' in line
                             for line in tail)
                with lock:
                    retries = state['retries'].get((bench, abl, seed), 0)
                if is_oom and retries < 2:
                    with lock:
                        state['retries'][(bench, abl, seed)] = retries + 1
                    time.sleep(30)
                    q.put((bench, abl, seed))
                    with lock:
                        elapsed = (time.time() - state['started']) / 60
                        print(time.strftime('[%H:%M:%S]'),
                              f"{bench}/{abl}/seed{seed}",
                              f"OOM retry #{retries + 1}",
                              f"elapsed={elapsed:.1f}m", flush=True)
                    q.task_done()
                    continue
            with lock:
                if rc == 0:
                    state['done'] += 1
                else:
                    state['failed'] += 1
                elapsed = (time.time() - state['started']) / 60
                print(time.strftime('[%H:%M:%S]'),
                      f"{bench}/{abl}/seed{seed}",
                      f"rc={rc} done={state['done']} failed={state['failed']}",
                      f"elapsed={elapsed:.1f}m", flush=True)
            q.task_done()

    threads = []
    for i, dev in enumerate(args.slots):
        t = Thread(target=worker, args=(i, dev), daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    elapsed = (time.time() - state['started']) / 60
    print(f"=== COMPLETE run_id={run_id} done={state['done']} "
          f"failed={state['failed']} elapsed={elapsed:.1f}m ===", flush=True)
    print(f"Aggregate with: ls outputs/clean_ablation_v2_1/B2/{run_id}/", flush=True)


if __name__ == '__main__':
    main()

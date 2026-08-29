# RCSF-AMP

Residue-Confidence-Guided Sequence--Structure Fusion for antimicrobial peptide classification.

RCSF-AMP combines peptide sequence representations with ESMFold-derived residue graphs. Residue-level pLDDT scores regulate the contribution of predicted geometry through a calibrated structural representation before sequence--structure interaction and peptide-level prediction.

## Repository layout

| Path | Purpose |
| --- | --- |
| `models/` | Model, graph encoder, calibration, and fusion modules |
| `configs/` | Training and evaluation configurations |
| `data_processing/` | Sequence, embedding, and graph preparation utilities |
| `experiments/` | Benchmark, ablation, and intervention runners |
| `scripts/` | Data preparation and result utilities |
| `tests/` | Smoke tests and physical-removal checks |
| `utils/` | Dataset, loss, metric, and checkpoint helpers |
| `docs/` | Reproducibility and data-preparation notes |
| `results/` | Compact metric summaries and protocol manifests |

Raw datasets, generated structures, embedding caches, checkpoints, and training outputs are not part of this repository. Prepare them locally and point the supplied configurations to their paths.

## Environment

The reference environment uses Python 3.10, PyTorch 2.2.1, PyTorch Geometric 2.6.1, and Mamba-SSM 2.2.4.

```bash
conda create -n rcsf_amp python=3.10 -y
conda activate rcsf_amp
pip install -r requirements.txt
```

GPU builds of PyTorch Geometric and Mamba-SSM should match the installed CUDA and PyTorch versions.

## Data preparation

Prepare the benchmark split files, residue graphs, and sequence embeddings locally. Set the data and warm-start checkpoint roots before running the experiments:

```bash
export SUCF_DATA_ROOT=/path/to/prepared_data
export SUCF_WARM_START_ROOT=/path/to/warm_start_checkpoints
```

The variable names are retained for compatibility with the existing experiment scripts. The expected file layout and preprocessing sequence are described in `docs/reproducibility.md`.

## Training

Run the final configuration with:

```bash
python train_sucf.py \
  --config configs/training_config.yaml \
  --device cuda:0
```

Checkpoint selection uses validation-only criteria and test evaluation uses the fixed threshold `0.5`. The benchmark and component analyses are launched through the scripts in `experiments/` and summarized using the utilities in `scripts/`.

## Results

Compact aggregate tables and protocol notes are stored under `results/`. The reported main comparisons use five local seeds. See `results/README.md` for the file list and metric definitions.

## Scope

This release supports the reported computational AMP classification experiments. It does not include prospective wet-lab validation or raw data redistribution.

## License

MIT License

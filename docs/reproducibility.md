# Reproducibility Guide

## 1. Prepare local inputs

Create local directories for the prepared benchmark splits, ESM-2 and ProtBERT embeddings, and ESMFold residue graphs. Do not commit these generated files to the repository.

The split files should identify the training, validation, and test members used by the manuscript. Graph files must retain residue order and the normalized pLDDT attribute used by the calibrated structural interface.

## 2. Configure paths

Set `SUCF_DATA_ROOT` and `SUCF_WARM_START_ROOT`, then review `configs/training_config.yaml`. Keep the data partitions, model-selection rule, batch size, optimizer, and test threshold unchanged when reproducing the reported values.

## 3. Run the model

```bash
python train_sucf.py \
  --config configs/training_config.yaml \
  --device cuda:0
```

The final evaluation uses the fixed threshold `0.5`. Aggregate accuracy, AUC, AUPR, F1, and MCC from the seed-level prediction files.

## 4. Run controls

The physical-removal and structural-intervention entry points are in `experiments/`. Each control should use the same partitions, optimizer, checkpoint rule, and seed set as the corresponding main run. The tests in `tests/` check that removed components are absent from the forward path.

## 5. Record outputs

Keep large predictions, checkpoints, caches, and logs in local `outputs/` or `runs/` directories. Only compact aggregate tables, split manifests, and protocol descriptions belong in `results/`.

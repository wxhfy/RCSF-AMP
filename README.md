# RCSF-AMP

Residue-Confidence-Guided Sequence–Structure Fusion for Antimicrobial Peptide Prediction.

RCSF-AMP is an end-to-end AMP classifier that combines peptide sequence representations with ESMFold-derived residue graphs. Predicted Local Distance Difference Test (pLDDT) scores regulate the contribution of predicted geometry through a calibrated structural representation before sequence–structure interaction and peptide-level prediction.

## Model components

- Frozen ESM-2 and ProtBERT sequence representations with residue identity features
- ESMFold-derived residue graph with sequential and spatial relations
- Geometric encoder with GVP and relational graph-attention layers
- pLDDT-conditioned structural calibration
- One cross-attention block, projection, sequence-anchor blend, attention pooling, and MLP classifier

## Repository layout

```
configs/                 Reproducible experiment configurations
data_processing/        Data preparation utilities
models/                  Model and encoder implementations
scripts/                 Training and evaluation scripts
experiments/             Benchmark and intervention evaluations
requirements.txt         Python dependencies
```

## Reproducibility

The manuscript reports balanced and imbalanced AMP benchmarks using fixed data partitions, validation-AUPR checkpoint selection, a fixed test threshold, and repeated random seeds. The `c_revision_working/reproducibility_build/` archive contains sanitized configurations, split manifests, seed-level metric records, and figure-generation scripts for the reported analyses.

Set the local data and checkpoint roots before running the supplied configurations:

```bash
export SUCF_DATA_ROOT=/path/to/prepared_data
export SUCF_WARM_START_ROOT=/path/to/warm_start_checkpoints
```

The environment-variable names are retained for compatibility with the existing experiment scripts.

## Citation

Please cite the associated RCSF-AMP manuscript when using this repository.

## License

MIT License

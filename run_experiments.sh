datasets=("benchmark1" "benchmark1_af2")
#!/bin/bash

# This script runs dataset/seed configs found under `configs/`.
# It filters configs to exactly the seeds you requested to avoid accidental extra files.

set -euo pipefail

# Accept optional --dry-run flag
DRY_RUN=0
if [ "${1-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

# Which seeds we want to run (change here if needed)
SEEDS=(32 37 42 47 52)

# Candidate globs (we restrict to these prefixes)
CONFIG_GLOBS=("configs/training_config_benchmark1*.yaml" "configs/training_config_benchmark1_af2*.yaml")

# Expand globs and sort
mapfile -t all_configs < <(ls ${CONFIG_GLOBS[@]} 2>/dev/null | sort)

if [ ${#all_configs[@]} -eq 0 ]; then
    echo "No matching config files found in configs/. Exiting." >&2
    exit 1
fi

# Filter configs to only those matching the requested seeds (ensures exactly 10 files)
declare -a selected_configs=()
for cfg in "${all_configs[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [[ "$cfg" == *"_seed${seed}.yaml" ]]; then
            selected_configs+=("$cfg")
            break
        fi
    done
done

# Deduplicate (just in case) and sort
mapfile -t selected_configs < <(printf "%s
" "${selected_configs[@]}" | awk '!seen[$0]++' | sort)

echo "Found ${#all_configs[@]} candidate config files; ${#selected_configs[@]} selected after seed filtering."

if [ ${#selected_configs[@]} -eq 0 ]; then
    echo "No configs matched the requested seeds (${SEEDS[*]}). Exiting." >&2
    exit 1
fi

if [ ${#selected_configs[@]} -ne $(( ${#SEEDS[@]} * 2 )) ]; then
    echo "Warning: expected $(( ${#SEEDS[@]} * 2 )) configs (2 datasets × ${#SEEDS[@]} seeds), but selected ${#selected_configs[@]}." >&2
    echo "Selected files:" >&2
    for f in "${selected_configs[@]}"; do echo "  $f" >&2; done
    echo "Proceeding with the selected set."
fi

if [ $DRY_RUN -eq 1 ]; then
    echo "--dry-run enabled. The following configs would be executed in order:"
    for cfg in "${selected_configs[@]}"; do
        echo "  $cfg"
    done
    exit 0
fi

echo "Starting experiments for ${#selected_configs[@]} configs..."

for config in "${selected_configs[@]}"; do
    echo "=================================================="
    echo "Running config: $config"
    echo "=================================================="

    conda run -n multi python train_sucf.py --config "$config"

    rc=$?
    if [ $rc -ne 0 ]; then
        echo "Training failed for config: $config (exit code $rc)" >&2
    else
        echo "Finished config: $config"
    fi

    echo "--------------------------------------------------"
    sleep 5
done

echo "All experiments finished."
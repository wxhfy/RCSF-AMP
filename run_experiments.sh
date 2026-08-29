#!/bin/bash
# Run experiments in parallel across multiple GPUs
# Usage: ./run_experiments.sh <experiment_pattern> <num_gpus>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"
TRAIN_SCRIPT="$SCRIPT_DIR/train_sucf.py"

PATTERN="${1:-exp_phase2}"
GPUS="${2:-4}"

echo "=== SUCF Experiment Runner ==="
echo "Pattern: $PATTERN"
echo "Available GPUs: $GPUS"

CONFIGS=$(ls $CONFIG_DIR/${PATTERN}*.yaml 2>/dev/null | sort)
if [ -z "$CONFIGS" ]; then
    echo "No configs found matching: $PATTERN"
    exit 1
fi

TOTAL=$(echo "$CONFIGS" | wc -l)
echo "Found $TOTAL experiments:"
echo "$CONFIGS" | sed 's|.*/||' | sed 's/.yaml//'

GPU_IDS=($(seq 0 $((GPUS-1))))
PIDS=()
for i in $(seq 0 $((TOTAL-1))); do
    cfg=$(echo "$CONFIGS" | sed -n "$((i+1))p")
    gpu=${GPU_IDS[$((i % GPUS))]}
    cfg_name=$(basename "$cfg" .yaml)
    echo "GPU $gpu: $cfg_name"
    mkdir -p "$SCRIPT_DIR/outputs/logs"
    python "$TRAIN_SCRIPT" --config "$cfg" --device "cuda:$gpu" \
        > "$SCRIPT_DIR/outputs/logs/${cfg_name}.log" 2>&1 &
    PIDS+=($!)
done

echo "All $TOTAL jobs launched"
FAILED=0
for i in $(seq 0 $((TOTAL-1))); do
    cfg=$(echo "$CONFIGS" | sed -n "$((i+1))p")
    cfg_name=$(basename "$cfg" .yaml)
    if wait ${PIDS[$i]}; then
        echo "✓ $cfg_name done"
    else
        echo "✗ $cfg_name FAILED"
        FAILED=$((FAILED+1))
    fi
done

echo ""
echo "=== Done: $TOTAL total, $FAILED failed ==="
if [ $FAILED -gt 0 ]; then exit 1; fi

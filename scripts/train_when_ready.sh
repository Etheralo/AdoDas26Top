#!/usr/bin/env bash
# Wait for validated extraction, then check data and run a GPU smoke test.
# Add --train to continue to the full stage-1 run after the smoke test succeeds.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python_bin="${ADODAS_PYTHON:-/root/miniconda3/envs/adodas_train/bin/python}"
config="${ADODAS_CONFIG:-configs/server_stage1.yaml}"
status=/root/autodl-tmp/dataset-ai/extract.status
if [[ ! -x "$python_bin" ]]; then
    echo "Training interpreter not installed: $python_bin; see SERVER_PREPARATION.md" >&2
    exit 1
fi
while true; do
    extraction_status=$(cat "$status")
    printf '%s %s\n' "$(date -Is)" "$extraction_status"
    if [[ "$extraction_status" == *FAILED* ]]; then exit 1; fi
    if [[ "$extraction_status" == *COMPLETE_ARCHIVES_REMOVED* ]]; then break; fi
    sleep 30
done
"$python_bin" tools/preflight.py --config "$config"
"$python_bin" tools/check_data.py --config "$config"
"$python_bin" tools/self_test.py
"$python_bin" tools/runtime_checks.py
# Check the configured architecture at both formal sequence-length limits.
"$python_bin" tools/check_training_gpu.py --config "$config" \
    --output outputs/readiness/training_gpu_check.json
# Exercise configured batching/workers/frame lengths, including result export.
"$python_bin" train.py --config "$config" --name readiness_pipeline --set \
    debug.limit_participants=32 train.epochs=1 train.min_epochs=1 \
    train.stats_participants=8 paths.output_dir=outputs/readiness/pipeline_smoke
if [[ "${1:-}" == --train ]]; then
    "$python_bin" train.py --config "$config"
fi

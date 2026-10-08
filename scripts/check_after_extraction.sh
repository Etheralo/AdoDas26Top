#!/usr/bin/env bash
# Persistent data preparation; reports GPU readiness without starting training.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p outputs/readiness
status=outputs/readiness/post_extract.status
trap 'printf "%s FAILED\n" "$(date -Is)" > "$status"' ERR
printf '%s WAITING_FOR_EXTRACTION\n' "$(date -Is)" > "$status"
while true; do
    extraction_status=$(cat /root/autodl-tmp/dataset-ai/extract.status)
    if [[ "$extraction_status" == *FAILED* ]]; then
        printf '%s EXTRACTION_FAILED\n' "$(date -Is)" > "$status"
        exit 1
    fi
    if [[ "$extraction_status" == *COMPLETE_ARCHIVES_REMOVED* ]]; then break; fi
    sleep 30
done
printf '%s CHECKING_ALL_DATA\n' "$(date -Is)" > "$status"
python_bin=/root/miniconda3/bin/python
if [[ -x /root/miniconda3/envs/adodas_train/bin/python ]] &&
   grep -q INSTALLED outputs/readiness/environment.status; then
    python_bin=/root/miniconda3/envs/adodas_train/bin/python
fi
"$python_bin" tools/check_data.py --config configs/server_stage1.yaml > outputs/readiness/full_data_check.log 2>&1
"$python_bin" tools/preflight.py --config configs/server_stage1.yaml --allow-cpu > outputs/readiness/post_extract_preflight.json
if "$python_bin" -c 'import torch; raise SystemExit(not torch.cuda.is_available())'; then
    printf '%s DATA_CHECKED_GPU_AVAILABLE\n' "$(date -Is)" > "$status"
else
    printf '%s DATA_CHECKED_GPU_UNAVAILABLE\n' "$(date -Is)" > "$status"
fi

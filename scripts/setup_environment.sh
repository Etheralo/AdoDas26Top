#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/readiness
status=outputs/readiness/environment.status
trap 'printf "%s FAILED\n" "$(date -Is)" > "$status"' ERR
printf '%s INSTALLING\n' "$(date -Is)" > "$status"
if [[ -d /root/miniconda3/envs/adodas_train/conda-meta ]]; then
    /root/miniconda3/bin/conda env update -f environment.yaml
else
    /root/miniconda3/bin/conda env create -f environment.yaml
fi
/root/miniconda3/envs/adodas_train/bin/python -c 'import sys,torch,numpy,pandas,pyarrow.parquet,sklearn,yaml; print(sys.version,torch.__version__,torch.version.cuda,torch.cuda.is_available())'
/root/miniconda3/envs/adodas_train/bin/python -m pip check
printf '%s INSTALLED\n' "$(date -Is)" > "$status"

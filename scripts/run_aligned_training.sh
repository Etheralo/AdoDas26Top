#!/usr/bin/env bash
# Sequential, separate A1/A2 candidates; never reuse the collapsed Stage1 run.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
python_bin="${ADODAS_PYTHON:-/root/miniconda3/envs/adodas_train/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
sequence_root="${ADODAS_SEQUENCE_ROOT:-/root/autodl-tmp/adodas_sota/outputs/sequences}"
mkdir -p "$sequence_root"
exec 9>"$sequence_root/training.lock"
flock -n 9 || { echo 'Another sequence holds training.lock' >&2; exit 1; }
"$python_bin" - <<'PY'
from pathlib import Path
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try: args = (p / 'cmdline').read_bytes().decode().strip('\0').split('\0')
    except (OSError, UnicodeError): continue
    if args and Path(args[0]).name.startswith('python') and 'train.py' in args:
        raise SystemExit(f'Training already active: PID {p.name}')
PY
# Runs the real CUDA computation test before allocating any experiment directory.
"$python_bin" tools/preflight.py --config configs/aligned_a2.yaml
root="$sequence_root/aligned_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$root"
exec > >(tee -a "$root/sequence.log") 2>&1
phase=STARTING
trap 'rc=$?; printf "%s FAILED %s exit=%s\n" "$(date -Is)" "$phase" "$rc" > "$root/status"; exit "$rc"' ERR
trap 'printf "%s STOPPED %s\n" "$(date -Is)" "$phase" > "$root/status"; exit 143' TERM INT
status() { phase="$1"; printf '%s %s\n' "$(date -Is)" "$phase" | tee "$root/status"; }
# Default A1 includes ASR. Set ADODAS_A1_USE_TEXT=false only for a declared AV-only run.
a1_text="${ADODAS_A1_USE_TEXT:-true}"
text_root="${ADODAS_TEXT_ROOT:-/root/autodl-tmp/dataset-ai/text_emb}"
[[ "$a1_text" == true || "$a1_text" == false ]]
if [[ "$a1_text" == true ]]; then
    # Existing complete vectors do not require any pretrained model files.
    if ! "$python_bin" - "$text_root" <<'PY'
import sys
from pathlib import Path
from data.dataset import read_manifest
from data.text_io import text_path
root = Path('/root/autodl-tmp/dataset-ai')
for split in ('train', 'val', 'test_hidden'):
    for row in read_manifest(root / f'{split}.csv').to_dict('records'):
        if row['session'] == 'A01': continue
        path = text_path(sys.argv[1], split, row['anon_school'], row['anon_class'],
                         row['anon_pid'], 'chinese-roberta-wwm-ext-large', row['session'])
        if not path.is_file(): raise SystemExit(1)
PY
    then
        text_model="${ADODAS_TEXT_MODEL:-/root/autodl-tmp/models/chinese-roberta-wwm-ext-large}"
        [[ -d "$text_model" ]] || { echo 'A1 requires complete text vectors or a local text model. Automatic downloads are disabled.' >&2; exit 1; }
        status TEXT_EXTRACTION_LOCAL_MODEL
        "$python_bin" tools/extract_text_embeddings.py --transcript_root /root/autodl-tmp/dataset-ai/transcripts \
            --manifest_dir /root/autodl-tmp/dataset-ai --text_root "$text_root" --model "$text_model"
    fi
fi
# On changed architectures, tune a small batch first. This limit is a guard,
# not a guarantee against a single allocation causing OOM.
batch="${ADODAS_BATCH_SIZE:-36}"
accum="${ADODAS_GRAD_ACCUM:-1}"
# Both distinct architectures must fit before any formal training starts.
for task in a1 a2; do
    status "GPU_PROBE_$task"
    "$python_bin" tools/check_training_gpu.py --config "configs/aligned_${task}.yaml" \
        --output "$root/gpu_probe_${task}.json" --set "train.batch_size=$batch" \
        "paths.text_root=$text_root" "model.use_text=$([[ "$task" == a1 ]] && echo "$a1_text" || echo false)"
done
"$python_bin" tools/check_data.py --config configs/aligned_a1.yaml --set \
    "paths.text_root=$text_root" "model.use_text=$a1_text" > "$root/data_check.log"

get_run() {
    "$python_bin" - "$1" <<'PY'
import sys
from pathlib import Path
runs = list((Path(sys.argv[1]) / 'runs').glob('*'))
if len(runs) != 1: raise RuntimeError(f'Expected one run, got {runs}')
run = runs[0]
for name in ('report.json', 'checkpoints/best_a1.pt', 'checkpoints/best_a2.pt', 'predictions/val_a1.npz', 'predictions/val_a2.npz', 'predictions/test_hidden_a1.npz', 'predictions/test_hidden_a2.npz'):
    if not (run / name).is_file(): raise RuntimeError(f'Missing {run / name}')
print(run)
PY
}
read -r -a seeds <<< "${ADODAS_SEEDS:-42 43 44}"
(( ${#seeds[@]} >= 2 )) || { echo 'At least two independent candidates required for item experts' >&2; exit 1; }
for seed in "${seeds[@]}"; do
    status "A1_STAGE1_SEED_$seed"
    "$python_bin" train.py --config configs/aligned_a1.yaml --name "aligned_a1_s$seed" --set \
        "paths.output_dir=$root/a1_s$seed" "paths.text_root=$text_root" "model.use_text=$a1_text" \
        "train.seed=$seed" "train.batch_size=$batch" "train.grad_accum_steps=$accum"
    a1_run=$(get_run "$root/a1_s$seed")
    status "A1_STAGE2_SEED_$seed"
    "$python_bin" train.py --config configs/aligned_a1_stage2.yaml --name "aligned_a1_head_s$seed" --set \
        "paths.output_dir=$root/a1_head_s$seed" "paths.text_root=$text_root" \
        "train.init_checkpoint=$a1_run/checkpoints/best_a1.pt" "train.seed=$seed" \
        "train.batch_size=$batch" "train.grad_accum_steps=$accum"
    get_run "$root/a1_head_s$seed" >> "$root/a1_members.txt"
    status "A2_STAGE1_SEED_$seed"
    "$python_bin" train.py --config configs/aligned_a2.yaml --name "aligned_a2_s$seed" --set \
        "paths.output_dir=$root/a2_s$seed" "train.seed=$seed" \
        "train.batch_size=$batch" "train.grad_accum_steps=$accum"
    get_run "$root/a2_s$seed" >> "$root/a2_members.txt"
done
for task in a1 a2; do
    status "ENSEMBLE_$task"
    "$python_bin" - "$root" "$task" <<'PY'
import sys, yaml
from pathlib import Path
root, task = Path(sys.argv[1]), sys.argv[2]
cfg = {'name': f'aligned_{task}_ensemble', 'primary_task': task, 'output_dir': str(root / 'ensembles'),
       'members': (root / f'{task}_members.txt').read_text().splitlines(),
       'a2': {'expert_selection': task == 'a2', 'min_gain': 0.0}}
(root / f'{task}_ensemble.yaml').write_text(yaml.safe_dump(cfg))
PY
    "$python_bin" ensemble.py --config "$root/${task}_ensemble.yaml"
done
status COMPLETE
# Test labels remain separate; use tools/score_test.py only after all choices are frozen.

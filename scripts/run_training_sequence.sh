#!/usr/bin/env bash
# One GPU, sequential Stage1 -> Stage2 from best A1 -> Stage2 from best A2.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
python_bin="${ADODAS_PYTHON:-/root/miniconda3/envs/adodas_train/bin/python}"
stage1_config="${ADODAS_STAGE1_CONFIG:-configs/server_stage1_fast.yaml}"
stage2_config="${ADODAS_STAGE2_CONFIG:-configs/server_stage2_fast.yaml}"
sequence_root="${ADODAS_SEQUENCE_ROOT:-/root/autodl-tmp/adodas_sota/outputs/sequences}"
mkdir -p "$sequence_root"
exec 9>"$sequence_root/training.lock"
flock -n 9 || { echo 'Another training sequence holds the lock.' >&2; exit 1; }
[[ -x "$python_bin" ]]
"$python_bin" - "$$" <<'PY'
import sys
from pathlib import Path
own_pid = int(sys.argv[1])
for process in Path("/proc").iterdir():
    if not process.name.isdigit() or int(process.name) == own_pid:
        continue
    try:
        args = (process / "cmdline").read_bytes().decode().strip("\0").split("\0")
    except (OSError, UnicodeError):
        continue
    executable = Path(args[0]).name if args else ""
    training = executable.startswith("python") and "train.py" in args
    script = Path(args[1]).name if len(args) > 1 else ""
    queue = executable == "bash" and script in ("train_when_ready.sh", "run_training_sequence.sh")
    if training or queue:
        raise SystemExit(f"Training already active (PID {process.name}); refusing duplicate launch")
PY
sequence_id="av_fast_seed42_$(date +%Y%m%d_%H%M%S)_$$"
sequence_dir="$sequence_root/$sequence_id"
mkdir "$sequence_dir"
printf '%s\n' "$sequence_dir" > "$sequence_root/latest.txt"
printf '%s\n' "$$" > "$sequence_dir/sequence.pid"
exec > >(tee -a "$sequence_dir/sequence.log") 2>&1
phase=STARTING
write_status() {
    printf '%s %s\n' "$(date -Is)" "$1" > "$sequence_dir/status.tmp"
    mv "$sequence_dir/status.tmp" "$sequence_dir/status"
    printf '%s %s\n' "$(date -Is)" "$1"
}
trap 'rc=$?; write_status "FAILED phase=$phase exit=$rc"; exit "$rc"' ERR
trap 'write_status "STOPPED phase=$phase"; exit 130' INT
trap 'write_status "STOPPED phase=$phase"; exit 143' TERM
[[ -x "$python_bin" ]]
phase=READINESS
write_status "$phase"
ADODAS_CONFIG="$stage1_config" bash scripts/train_when_ready.sh

# Separate output roots make checkpoint discovery unambiguous.
completed_run() {
    "$python_bin" - "$1" <<'PY'
import sys
from pathlib import Path
runs = list((Path(sys.argv[1]) / "runs").glob("*"))
if len(runs) != 1:
    raise RuntimeError(f"Expected exactly one run; found {len(runs)}")
run = runs[0]
for relative in ("checkpoints/best_a1.pt", "checkpoints/best_a2.pt",
                 "checkpoints/last.pt", "inner_dev.csv", "history.json",
                 "calibration.json", "report.json"):
    artifact = run / relative
    if not artifact.is_file() or not artifact.stat().st_size:
        raise RuntimeError(f"Missing or empty artifact: {artifact}")
print(run)
PY
}

phase=STAGE1
write_status "$phase"
"$python_bin" train.py --config "$stage1_config" --name av_stage1 \
    --set "paths.output_dir=$sequence_dir/stage1"
stage1_run=$(completed_run "$sequence_dir/stage1")
printf '%s\n' "$stage1_run" > "$sequence_dir/stage1_run.txt"

for task in a1 a2; do
    phase="STAGE2_FROM_${task^^}"
    write_status "$phase"
    "$python_bin" train.py --config "$stage2_config" --name "av_stage2_from_$task" \
        --set "train.init_checkpoint=$stage1_run/checkpoints/best_$task.pt" \
        "paths.output_dir=$sequence_dir/stage2_from_$task"
    stage2_run=$(completed_run "$sequence_dir/stage2_from_$task")
    printf '%s\n' "$stage2_run" > "$sequence_dir/stage2_from_${task}_run.txt"
done
phase=COMPLETE
write_status "$phase"

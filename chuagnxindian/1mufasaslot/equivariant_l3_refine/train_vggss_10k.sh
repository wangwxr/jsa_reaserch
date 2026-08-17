#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 [GPU_ID] [EXPERIMENT_NAME]"
    exit 2
fi

gpu=${1:-0}
script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-python}
experiment_name=${2:-equivariant_l3_refine_vggss_10k}
experiment_dir="$project_root/checkpoints/$experiment_name"

if [ -f "$experiment_dir/latest.pth" ] \
    || [ -f "$experiment_dir/final.pth" ] \
    || [ -f "$experiment_dir/vggss_best.pth" ]; then
    echo "Refusing to overwrite existing experiment: $experiment_dir"
    exit 1
fi

log_dir="$experiment_dir/logs"
mkdir -p "$log_dir"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$log_dir/${experiment_name}_train_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

echo "Training log: $log_file"
echo "Experiment F: VGGSoundSS-10k, GPU: $gpu"
cd "$project_root"
"$python_bin" "$script_dir/train.py" \
    --experiment vggss_10k \
    --gpu "$gpu" \
    --experiment-name "$experiment_name"

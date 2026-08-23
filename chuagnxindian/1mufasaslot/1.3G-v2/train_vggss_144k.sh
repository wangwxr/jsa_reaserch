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
experiment_name=${2:-1.3G-v2_vggss_144k}
experiment_dir="$project_root/checkpoints/$experiment_name"
log_dir="$experiment_dir/logs"
mkdir -p "$log_dir"
timestamp=$(date +%Y%m%d_%H%M%S)
train_log="$log_dir/${experiment_name}_train_${timestamp}.log"
test_log="$log_dir/${experiment_name}_test_${timestamp}.log"
exec > >(tee -a "$train_log") 2>&1

echo "1.3G-v2 VGGSoundSS-144k: GPU=$gpu, epochs=50"
cd "$project_root"
"$python_bin" "$script_dir/train.py" --experiment vggss_144k --gpu "$gpu" \
    --epochs 50 --experiment-name "$experiment_name"
"$python_bin" "$script_dir/evaluate_full.py" --experiment vggss_144k --gpu "$gpu" \
    --experiment-name "$experiment_name" 2>&1 | tee -a "$test_log"

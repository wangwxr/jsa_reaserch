#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 [GPU_ID] [EXPERIMENT_NAME]"
    exit 2
fi
gpu=${1:-1}
script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-python}
experiment_name=${2:-1.3G-v2_flickr_144k_frame8_center5}
experiment_dir="$project_root/checkpoints/$experiment_name"
log_dir="$experiment_dir/logs"
mkdir -p "$log_dir"
timestamp=$(date +%Y%m%d_%H%M%S)
train_log="$log_dir/${experiment_name}_train_${timestamp}.log"
test_log="$log_dir/${experiment_name}_test_${timestamp}.log"
exec > >(tee -a "$train_log") 2>&1

echo "1.3G-v2 Flickr-144k: GPU=$gpu, epochs=50"
cd "$project_root"
"$python_bin" "$script_dir/train.py" --experiment flickr_144k --gpu "$gpu" \
    --epochs 50 --experiment-name "$experiment_name"
"$python_bin" "$script_dir/evaluate_full.py" --experiment flickr_144k --gpu "$gpu" \
    --experiment-name "$experiment_name" 2>&1 | tee -a "$test_log"

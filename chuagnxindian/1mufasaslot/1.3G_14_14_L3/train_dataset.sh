#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 {vggss|flickr} {10k|144k} GPU_ID EXPERIMENT_NAME"
    exit 2
fi
dataset=$1
split=$2
gpu=$3
experiment_name=$4
script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-/home/wxr/miniconda3/envs/wwww/bin/python}
experiment_key="${dataset}_${split}"
experiment_dir="$project_root/checkpoints/$experiment_name"
best_name="${dataset}_best.pth"

if [ -f "$experiment_dir/latest.pth" ] \
    || [ -f "$experiment_dir/final.pth" ] \
    || [ -f "$experiment_dir/$best_name" ]; then
    echo "Refusing to overwrite existing experiment: $experiment_dir"
    exit 1
fi
log_dir="$experiment_dir/logs"
mkdir -p "$log_dir"
timestamp=$(date +%Y%m%d_%H%M%S)
train_log="$log_dir/${experiment_name}_train_${timestamp}.log"
eval_log="$log_dir/${experiment_name}_full_metrics_${timestamp}.log"
exec > >(tee -a "$train_log") 2>&1

echo "Training log: $train_log"
echo "Experiment: 1.3G_14_14_L3; teacher: 1.1.1_14_14_L3 best; key: $experiment_key; GPU: $gpu"
cd "$project_root"
"$python_bin" "$script_dir/train.py" \
    --experiment "$experiment_key" \
    --gpu "$gpu" \
    --experiment-name "$experiment_name"

echo "Full six-metric log: $eval_log"
"$python_bin" "$script_dir/evaluate_full.py" \
    --experiment "$experiment_key" \
    --gpu "$gpu" \
    --experiment-name "$experiment_name" \
    2>&1 | tee -a "$eval_log"

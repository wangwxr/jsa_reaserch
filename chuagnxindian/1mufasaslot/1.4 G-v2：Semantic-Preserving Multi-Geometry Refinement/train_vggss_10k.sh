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
experiment_name=${2:-1.4G-v2-semantic_preserving_multigeom_vggss_10k}
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
train_log="$log_dir/${experiment_name}_train_${timestamp}.log"
test_log="$log_dir/${experiment_name}_full_six_metrics_${timestamp}.log"
exec > >(tee -a "$train_log") 2>&1

echo "Training log: $train_log"
echo "Experiment G-v2: VGGSoundSS-10k, GPU: $gpu, epochs: 10"
cd "$project_root"
"$python_bin" "$script_dir/train.py" \
    --experiment vggss_10k \
    --gpu "$gpu" \
    --epochs 10 \
    --experiment-name "$experiment_name"

echo "Full six-metric test log: $test_log"
"$python_bin" "$script_dir/evaluate_full.py" \
    --experiment vggss_10k \
    --gpu "$gpu" \
    --experiment-name "$experiment_name" \
    2>&1 | tee -a "$test_log"

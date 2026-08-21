#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 {vggss|flickr} GPU_ID {10k|144k} [...]"
    exit 2
fi
dataset=$1
gpu=$2
shift 2
splits=("$@")
script_dir=$(cd "$(dirname "$0")" && pwd)
g_dir="$script_dir/../1.3G_14_14_L3"
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-/home/wxr/miniconda3/envs/wwww/bin/python}

for split in "${splits[@]}"; do
    key="${dataset}_${split}"
    native_name="1.1.1_14_14_L3_${dataset}_${split}"
    g_name="1.3G_14_14_L3_${dataset}_${split}"
    native_dir="$project_root/checkpoints/$native_name"
    mkdir -p "$native_dir/logs"

    sanity_log="$native_dir/logs/${native_name}_zero_training_sanity.log"
    echo "[$(date --iso-8601=seconds)] Stage 1: zero-training sanity $key on GPU $gpu" | tee -a "$sanity_log"
    "$python_bin" "$script_dir/sanity.py" --experiment "$key" --gpu "$gpu" \
        2>&1 | tee -a "$sanity_log"

    echo "[$(date --iso-8601=seconds)] Stage 2: formal native-L3 training $key on GPU $gpu"
    bash "$script_dir/train_${dataset}_${split}.sh" "$gpu" "$native_name"

    eval_log="$native_dir/logs/${native_name}_ownership_evaluation.log"
    echo "[$(date --iso-8601=seconds)] Stage 3: formal + ownership evaluation $key" | tee -a "$eval_log"
    "$python_bin" "$script_dir/evaluate.py" --experiment "$key" --gpu "$gpu" \
        2>&1 | tee -a "$eval_log"

    echo "[$(date --iso-8601=seconds)] Stage 4: G training from $native_name best on GPU $gpu"
    bash "$g_dir/train_${dataset}_${split}.sh" "$gpu" "$g_name"
done

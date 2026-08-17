#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 EXPERIMENT_NAME [GPU_ID] [ALPHA] [CHECKPOINT]"
    exit 2
fi

experiment_name=$1
gpu=${2:-0}
alpha=${3:-0.6}
experiment_root=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$experiment_root/../.." && pwd)
metadata_root="$project_root/metadata_flickr"
dataset_root=${FLICKR_ROOT:-/data/wxr/datasets/FlickrSoundNet}
test_root="$dataset_root/test/Dataset"
python_bin=${JSA_PYTHON:-python}

if [ "$#" -ge 4 ]; then
    checkpoint=$4
elif [ -f "$project_root/checkpoints/$experiment_name/flickr_best.pth" ]; then
    checkpoint=flickr_best.pth
else
    checkpoint=final.pth
fi

log_root=${JSA_LOG_ROOT:-$project_root/checkpoints/$experiment_name/logs}
mkdir -p "$log_root"
timestamp=$(date +%Y%m%d_%H%M%S)
checkpoint_label=${checkpoint%.pth}
log_file="$log_root/${experiment_name}_test_flickr_${checkpoint_label}_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1
echo "Test log: $log_file"
echo "Architecture: mufasa_jsa_v1; experiment: $experiment_name; checkpoint: $checkpoint; GPU: $gpu"

cd "$project_root"
"$python_bin" "$experiment_root/test_mufasa.py" \
    --test_data_path "$test_root" \
    --test_manifest_path "$metadata_root/flickr_test_SLAVC.csv" \
    --test_gt_path "$test_root/Annotations" \
    --testset flickr \
    --batch_size 32 \
    --alpha "$alpha" \
    --infer_sharpening 0.1 \
    --aud_length 5.0 \
    --workers 8 \
    --gpu "$gpu" \
    --wandb false \
    --experiment_name "$experiment_name" \
    --checkpoint "$checkpoint" \
    --model_dir ./checkpoints

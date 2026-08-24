#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 {vggss|flickr} {no_spatial_att|no_image_rec|no_both} GPU EXPERIMENT"
    exit 2
fi

dataset=$1
configuration=$2
gpu=$3
experiment=$4
script_root=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_root/../.." && pwd)
python_bin=${JSA_PYTHON:-python}
checkpoint_root="$project_root/checkpoints/$experiment"

for checkpoint in latest.pth final.pth vggss_best.pth flickr_best.pth; do
    if [ -f "$checkpoint_root/$checkpoint" ]; then
        echo "Refusing to overwrite existing checkpoint: $checkpoint_root/$checkpoint"
        exit 1
    fi
done

mkdir -p "$checkpoint_root/logs"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$checkpoint_root/logs/${experiment}_train_${timestamp}.log"
workers=12
if [ "$dataset" = "vggss" ]; then
    workers=16
fi

cd "$project_root"
"$python_bin" "$script_root/train.py" \
    --dataset "$dataset" \
    --configuration "$configuration" \
    --experiment-name "$experiment" \
    --gpu "$gpu" \
    --workers "$workers" \
    2>&1 | tee -a "$log_file"

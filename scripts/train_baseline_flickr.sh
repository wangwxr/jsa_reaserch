#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 {10k|144k} [GPU_ID] [EXPERIMENT_NAME]"
    exit 2
fi

split=$1
gpu=${2:-0}
experiment_name=${3:-baseline_av_mil_flickr_${split}}
project_root=$(cd "$(dirname "$0")/.." && pwd)
dataset_root=${FLICKR_ROOT:-/data/wxr/datasets/FlickrSoundNet}
train_root="$dataset_root/prepared/jsa_flickr_${split}"
test_root="$dataset_root/test/Dataset"
python_bin=${JSA_PYTHON:-python}

case "$split" in
    10k) epochs=100 ;;
    144k) epochs=50 ;;
    *)
        echo "split must be 10k or 144k"
        exit 2
        ;;
esac

log_root=${JSA_LOG_ROOT:-$project_root/checkpoints/$experiment_name/logs}
mkdir -p "$log_root"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$log_root/${experiment_name}_train_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1
echo "Training log: $log_file"
echo "Experiment: $experiment_name, split: $split, GPU: $gpu"

cd "$project_root"
"$python_bin" train_slot.py \
    --model av_mil \
    --train_data_path "$train_root" \
    --train_manifest_path "$train_root/available_ids.txt" \
    --test_data_path "$test_root" \
    --test_gt_path "$test_root/Annotations" \
    --trainset "flickr_${split}" \
    --testset flickr \
    --epochs "$epochs" \
    --warmup -1 \
    --batch_size 256 \
    --init_lr 0.00005 \
    --weight_decay 0.01 \
    --alpha 0.6 \
    --tau 0.03 \
    --aud_length 5.0 \
    --workers 12 \
    --gpu "$gpu" \
    --wandb false \
    --hard_aud true \
    --hard_img true \
    --rand_aud false \
    --experiment_name "$experiment_name"

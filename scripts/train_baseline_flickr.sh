#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 {10k|144k} [GPU_ID] [EXPERIMENT_NAME]"
    exit 2
fi

split=$1
gpu=${2:-0}
experiment_name=${3:-baseline_av_mil_flickr_${split}}
dataset_root=${FLICKR_ROOT:-/data/wxr/datasets/FlickrSoundNet}
train_root="$dataset_root/prepared/jsa_flickr_${split}"
test_root="$dataset_root/test/Dataset"

case "$split" in
    10k) epochs=100 ;;
    144k) epochs=50 ;;
    *)
        echo "split must be 10k or 144k"
        exit 2
        ;;
esac

python train_slot.py \
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
    --alpha 0.4 \
    --tau 0.03 \
    --aud_length 5.0 \
    --workers 8 \
    --gpu "$gpu" \
    --wandb false \
    --hard_aud true \
    --hard_img true \
    --rand_aud false \
    --experiment_name "$experiment_name"

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

case "$split" in
    10k) epochs=100 ;;
    144k) epochs=50 ;;
    *) echo "split must be 10k or 144k"; exit 2 ;;
esac

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
log_file="$log_dir/${experiment_name}_train_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

if [ "$dataset" = "vggss" ]; then
    metadata_root="$project_root/metadata_vggss"
    dataset_root=${VGGSOUND_ROOT:-/data/wxr/datasets/VGGSound}
    test_root=${VGGSS_TEST_ROOT:-/data/wxr/datasets/ACL-SSL/VGGSound}
    video_root="$dataset_root/extracted_full/video"
    train_root=${VGG_PREP_ROOT:-/home/wxr/datasets/JSA/VGGSound_144k_npy}
    train_manifest="$metadata_root/vggss_${split}.csv"
    test_manifest="$metadata_root/vggss_test.csv"
    test_gt="$metadata_root/vggss.json"
    workers=16
    trainset="vggss_${split}"
    testset=vggss
    overlap_args=()
    if [ "$split" = "144k" ]; then overlap_args=(--allow-source-overlap); fi
    "$python_bin" "$project_root/tools/check_vggsound_split.py" \
        --train-manifest "$train_manifest" \
        --test-manifest "$test_manifest" \
        --annotations "$test_gt" \
        --video-dir "$video_root" \
        --test-data-root "$test_root" \
        --prepared-root "$train_root" \
        "${overlap_args[@]}"
else
    metadata_root="$project_root/metadata_flickr"
    dataset_root=${FLICKR_ROOT:-/data/wxr/datasets/FlickrSoundNet}
    train_root=${FLICKR_PREP_ROOT:-/home/wxr/datasets/JSA/FlickrSoundNet_144k_frame8_center5_npy}
    test_root="$dataset_root/test/Dataset"
    train_manifest="$metadata_root/flickr_${split}.csv"
    test_manifest="$metadata_root/flickr_test_SLAVC.csv"
    test_gt="$test_root/Annotations"
    workers=12
    trainset="flickr_${split}"
    testset=flickr
fi

echo "Training log: $log_file"
echo "Architecture: 1.1.1_14_14_L3; dataset: $dataset; split: $split; GPU: $gpu"
cd "$project_root"
"$python_bin" "$script_dir/train.py" \
    --train_data_path "$train_root" \
    --train_manifest_path "$train_manifest" \
    --train_metadata_path "$train_manifest" \
    --test_data_path "$test_root" \
    --test_manifest_path "$test_manifest" \
    --test_gt_path "$test_gt" \
    --trainset "$trainset" \
    --testset "$testset" \
    --epochs "$epochs" \
    --warmup -1 \
    --batch_size 256 \
    --init_lr 0.00005 \
    --weight_decay 0.01 \
    --alpha 0.6 \
    --lam1 0.1 \
    --lam2 0.1 \
    --lam3 100.0 \
    --tau 0.03 \
    --infer_sharpening 0.1 \
    --num_slots 2 \
    --iters 5 \
    --reciprocal_k 20 \
    --mask_ratio 0.1 \
    --aud_length 5.0 \
    --workers "$workers" \
    --gpu "$gpu" \
    --wandb false \
    --hard_aud true \
    --hard_img true \
    --rand_aud false \
    --eval_during_training true \
    --experiment_name "$experiment_name"

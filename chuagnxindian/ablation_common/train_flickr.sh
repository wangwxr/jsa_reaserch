#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ] || [ "$#" -gt 7 ]; then
    echo "Internal usage: $0 ENTRY ARCH PREFIX MODEL {10k|144k} [GPU] [EXPERIMENT]"
    exit 2
fi

entry_script=$1
architecture=$2
experiment_prefix=$3
model_name=$4
split=$5
gpu=${6:-0}
experiment_name=${7:-${experiment_prefix}_flickr_${split}_frame8_center5}
common_root=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$common_root/../.." && pwd)
metadata_root="$project_root/metadata_flickr"
dataset_root=${FLICKR_ROOT:-/data/wxr/datasets/FlickrSoundNet}
train_root=${FLICKR_PREP_ROOT:-$project_root/prepared/jsa_flickr_${split}_metadata}
test_root="$dataset_root/test/Dataset"
train_manifest="$metadata_root/flickr_${split}.csv"
test_manifest="$metadata_root/flickr_test_SLAVC.csv"
python_bin=${JSA_PYTHON:-python}

case "$split" in
    10k) epochs=100 ;;
    144k) epochs=50 ;;
    *)
        echo "split must be 10k or 144k"
        exit 2
        ;;
esac

if [ "$model_name" = "av_mil" ]; then
    lam1=0.0
    lam2=0.0
    lam3=0.0
    num_slots=0
    iters=0
    mask_ratio=0.0
else
    lam1=0.1
    lam2=0.1
    lam3=100.0
    num_slots=2
    iters=5
    mask_ratio=0.1
fi

experiment_dir="$project_root/checkpoints/$experiment_name"
if [ -f "$experiment_dir/latest.pth" ] \
    || [ -f "$experiment_dir/final.pth" ] \
    || [ -f "$experiment_dir/flickr_best.pth" ]; then
    echo "Refusing to overwrite existing experiment: checkpoints/$experiment_name"
    exit 1
fi

log_root=${JSA_LOG_ROOT:-$experiment_dir/logs}
mkdir -p "$log_root"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$log_root/${experiment_name}_train_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1
echo "Training log: $log_file"
echo "Architecture: $architecture; experiment: $experiment_name; split: $split; GPU: $gpu"

cd "$project_root"
"$python_bin" "$entry_script" \
    --model "$model_name" \
    --train_data_path "$train_root" \
    --train_manifest_path "$train_manifest" \
    --train_metadata_path "$train_manifest" \
    --test_data_path "$test_root" \
    --test_manifest_path "$test_manifest" \
    --test_gt_path "$test_root/Annotations" \
    --trainset "flickr_${split}" \
    --testset flickr \
    --epochs "$epochs" \
    --warmup -1 \
    --batch_size 256 \
    --init_lr 0.00005 \
    --weight_decay 0.01 \
    --alpha 0.6 \
    --lam1 "$lam1" \
    --lam2 "$lam2" \
    --lam3 "$lam3" \
    --tau 0.03 \
    --infer_sharpening 0.1 \
    --num_slots "$num_slots" \
    --iters "$iters" \
    --reciprocal_k 20 \
    --mask_ratio "$mask_ratio" \
    --aud_length 5.0 \
    --workers 12 \
    --gpu "$gpu" \
    --wandb false \
    --hard_aud true \
    --hard_img true \
    --rand_aud false \
    --eval_during_training true \
    --experiment_name "$experiment_name"

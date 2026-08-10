#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 {10k|144k} [GPU_ID] [EXPERIMENT_NAME]"
    exit 2
fi

split=$1
gpu=${2:-0}
experiment_name=${3:-jsa_flickr_${split}}
project_root=$(cd "$(dirname "$0")/.." && pwd)
metadata_root="$project_root/metadata_flickr"
dataset_root=${FLICKR_ROOT:-/data/wxr/datasets/FlickrSoundNet}
extracted_root=${FLICKR_EXTRACTED_ROOT:-$dataset_root/extracted}
train_root=${FLICKR_PREP_ROOT:-$project_root/prepared/jsa_flickr_${split}_metadata}
test_root="$dataset_root/test/Dataset"
train_manifest="$metadata_root/flickr_${split}.csv"
test_manifest="$metadata_root/flickr_test_SLAVC.csv"
python_bin=${JSA_PYTHON:-python}

case "$split" in
    10k)
        epochs=100
        infer_sharpening=0.1
        ;;
    144k)
        epochs=50
        infer_sharpening=1.0
        ;;
    *)
        echo "split must be 10k or 144k"
        exit 2
        ;;
esac

experiment_dir="$project_root/checkpoints/$experiment_name"
if [ -f "$experiment_dir/latest.pth" ] \
    || [ -f "$experiment_dir/final.pth" ] \
    || [ -f "$experiment_dir/flickr_best.pth" ]; then
    echo "Refusing to overwrite existing experiment: checkpoints/$experiment_name"
    exit 1
fi

log_root=${JSA_LOG_ROOT:-$project_root/checkpoints/$experiment_name/logs}
mkdir -p "$log_root"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$log_root/${experiment_name}_train_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1
echo "Training log: $log_file"
echo "Experiment: $experiment_name, split: $split, GPU: $gpu"

"$python_bin" "$project_root/tools/build_flickr_view.py" \
    --manifest "$train_manifest" \
    --extracted-dir "$extracted_root" \
    --output-dir "$train_root"

cd "$project_root"
"$python_bin" train_slot.py \
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
    --alpha 0.4 \
    --lam1 0.1 \
    --lam2 0.1 \
    --lam3 100.0 \
    --tau 0.03 \
    --infer_sharpening "$infer_sharpening" \
    --num_slots 2 \
    --iters 5 \
    --reciprocal_k 20 \
    --mask_ratio 0.1 \
    --aud_length 5.0 \
    --workers 8 \
    --gpu "$gpu" \
    --wandb false \
    --hard_aud true \
    --hard_img true \
    --rand_aud false \
    --eval_during_training true \
    --experiment_name "$experiment_name"

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 {10k|144k} [GPU_ID] [EXPERIMENT_NAME]"
    exit 2
fi

split=$1
gpu=${2:-0}
experiment_name=${3:-jsa_vggss_${split}_clean}
project_root=$(cd "$(dirname "$0")/.." && pwd)
metadata_root="$project_root/metadata_vggss"
dataset_root=${VGGSOUND_ROOT:-/data/wxr/datasets/VGGSound}
test_root=${VGGSS_TEST_ROOT:-/data/wxr/datasets/ACL-SSL/VGGSound}
video_root="$dataset_root/extracted_full/video"
prepared_root=${VGG_PREP_ROOT:-$project_root/prepared/jsa_vggss_${split}_clean}
train_manifest="$metadata_root/vggss_${split}.csv"
test_manifest="$metadata_root/vggss_test.csv"
annotations="$metadata_root/vggss.json"
prepare_workers=${VGG_PREP_WORKERS:-16}
python_bin=${JSA_PYTHON:-python}

case "$split" in
    10k)
        epochs=100
        infer_sharpening=0.1
        overlap_args=()
        ;;
    144k)
        epochs=50
        infer_sharpening=1.0
        overlap_args=(--allow-source-overlap)
        ;;
    *)
        echo "split must be 10k or 144k"
        exit 2
        ;;
esac

experiment_dir="$project_root/checkpoints/$experiment_name"
if [ -f "$experiment_dir/latest.pth" ] \
    || [ -f "$experiment_dir/final.pth" ] \
    || [ -f "$experiment_dir/vggss_best.pth" ]; then
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

"$python_bin" "$project_root/tools/check_vggsound_split.py" \
    --train-manifest "$train_manifest" \
    --test-manifest "$test_manifest" \
    --annotations "$annotations" \
    --video-dir "$video_root" \
    --test-data-root "$test_root" \
    "${overlap_args[@]}"

"$python_bin" "$project_root/tools/prepare_vggsound_split.py" \
    --manifest "$train_manifest" \
    --video-dir "$video_root" \
    --output-dir "$prepared_root" \
    --workers "$prepare_workers"

"$python_bin" "$project_root/tools/check_vggsound_split.py" \
    --train-manifest "$train_manifest" \
    --test-manifest "$test_manifest" \
    --annotations "$annotations" \
    --video-dir "$video_root" \
    --test-data-root "$test_root" \
    --prepared-root "$prepared_root" \
    "${overlap_args[@]}"

cd "$project_root"
"$python_bin" train_slot.py \
    --train_data_path "$prepared_root" \
    --train_manifest_path "$train_manifest" \
    --train_metadata_path "$train_manifest" \
    --test_data_path "$test_root" \
    --test_manifest_path "$test_manifest" \
    --test_gt_path "$annotations" \
    --trainset "vggss_${split}" \
    --testset vggss \
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

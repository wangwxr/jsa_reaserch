#!/usr/bin/env bash
set -euo pipefail

workers=${1:-4}
project_root=$(cd "$(dirname "$0")/.." && pwd)
dataset_root=${FLICKR_ROOT:-/data/wxr/datasets/FlickrSoundNet}
extracted_root=${FLICKR_EXTRACTED_ROOT:-$dataset_root/extracted}
audio_root=${FLICKR_FLAT_ROOT:-$project_root/prepared/jsa_flickr_144k_real_flat}
target_root=${FLICKR_FRAME3_NVME_ROOT:-/home/wxr/datasets/JSA/FlickrSoundNet_144k_frame3_first5_npy}
manifest=${FLICKR_MANIFEST:-$project_root/metadata_flickr/flickr_144k.csv}
python_bin=${JSA_PYTHON:-/home/wxr/miniconda3/envs/wwww/bin/python}
log_experiment=${JSA_LOG_EXPERIMENT:-jsa_flickr_144k_frame3_first5}

log_root=${JSA_LOG_ROOT:-$project_root/checkpoints/$log_experiment/logs}
mkdir -p "$log_root"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$log_root/flickr_frame3_first5_prepare_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

echo "Preparation log: $log_file"
echo "Frame source: $extracted_root"
echo "Audio source: $audio_root/audio"
echo "Target: $target_root"
echo "Workers: $workers"

if [ "$target_root" = "/home/wxr/datasets/JSA/FlickrSoundNet_144k_npy" ]; then
    echo "Refusing to overwrite the existing center-crop dataset."
    exit 1
fi

mkdir -p "$target_root/audio" "$target_root/frames"

echo "Copying exact 00000003 JPG frames directly to NVMe..."
"$python_bin" "$project_root/tools/copy_flickr_frame3.py" \
    --manifest "$manifest" \
    --lists-dir "$extracted_root/lists" \
    --extracted-frames-dir "$extracted_root/frames" \
    --output-dir "$target_root/frames" \
    --workers "$workers"

echo "Precomputing first-five-second power spectrograms to NVMe..."
cd "$project_root"
"$python_bin" tools/precompute_flickr_spectrograms.py \
    --manifest "$manifest" \
    --audio-dir "$audio_root/audio" \
    --output-dir "$target_root/audio" \
    --duration 5.0 \
    --crop first \
    --workers "$workers" \
    --report-every 1000

audio_count=$(find "$target_root/audio" -maxdepth 1 -type f -name '*.npy' | wc -l)
frame_count=$(find "$target_root/frames" -maxdepth 1 -type f -name '*.jpg' | wc -l)
symlink_count=$(find "$target_root" -type l | wc -l)
echo "Prepared audio NPY files: $audio_count"
echo "Prepared frame-3 JPG files: $frame_count"
echo "Symlinks in final dataset: $symlink_count"
if [ "$audio_count" -ne 144000 ] \
    || [ "$frame_count" -ne 144000 ] \
    || [ "$symlink_count" -ne 0 ]; then
    echo "Preparation is incomplete."
    exit 1
fi

touch "$target_root/PREPARATION_COMPLETE"
echo "Preparation completed successfully."

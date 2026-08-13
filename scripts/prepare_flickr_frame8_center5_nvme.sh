#!/usr/bin/env bash
set -euo pipefail

workers=${1:-8}
project_root=$(cd "$(dirname "$0")/.." && pwd)
source_root=${FLICKR_FLAT_ROOT:-$project_root/prepared/jsa_flickr_144k_real_flat}
selected_frame_view=${FLICKR_SELECTED_FRAME_VIEW:-$project_root/prepared/jsa_flickr_144k_metadata/frames}
lists_dir=${FLICKR_LISTS_DIR:-/data/wxr/datasets/FlickrSoundNet/extracted/lists}
reused_frame_dir=${FLICKR_REUSED_FRAME_DIR:-/home/wxr/datasets/JSA/FlickrSoundNet_144k_npy/frames}
target_root=${FLICKR_FRAME8_NVME_ROOT:-/home/wxr/datasets/JSA/FlickrSoundNet_144k_frame8_center5_npy}
manifest=${FLICKR_MANIFEST:-$project_root/metadata_flickr/flickr_144k.csv}
python_bin=${JSA_PYTHON:-/home/wxr/miniconda3/envs/wwww/bin/python}
log_experiment=${JSA_LOG_EXPERIMENT:-jsa_flickr_144k_frame8_center5}

log_root=${JSA_LOG_ROOT:-$project_root/checkpoints/$log_experiment/logs}
mkdir -p "$log_root"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$log_root/flickr_frame8_center5_prepare_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

echo "Preparation log: $log_file"
echo "Audio source: $source_root/audio"
echo "Selected-frame view: $selected_frame_view"
echo "Frame lists: $lists_dir"
echo "Reused frame directory: $reused_frame_dir"
echo "Target: $target_root"
echo "Workers: $workers"

if [ "$target_root" = "/home/wxr/datasets/JSA/FlickrSoundNet_144k_npy" ] \
    || [ "$target_root" = "/home/wxr/datasets/JSA/FlickrSoundNet_144k_frame3_first5_npy" ]; then
    echo "Refusing to overwrite an existing dataset."
    exit 1
fi
if [ ! -d "$reused_frame_dir" ]; then
    echo "Missing reused frame directory: $reused_frame_dir"
    exit 1
fi

mkdir -p "$target_root/audio"
if [ -L "$target_root/frames" ]; then
    if [ "$(readlink -f "$target_root/frames")" != "$(readlink -f "$reused_frame_dir")" ]; then
        echo "Existing frame symlink points to the wrong directory: $target_root/frames"
        exit 1
    fi
elif [ -e "$target_root/frames" ]; then
    echo "Refusing to replace existing non-symlink path: $target_root/frames"
    exit 1
else
    ln -s "$reused_frame_dir" "$target_root/frames"
fi

cd "$project_root"
"$python_bin" tools/precompute_flickr_frame_aligned_spectrograms.py \
    --manifest "$manifest" \
    --audio-dir "$source_root/audio" \
    --lists-dir "$lists_dir" \
    --output-dir "$target_root/audio" \
    --duration 5.0 \
    --workers "$workers" \
    --report-every 1000

audio_count=$(find "$target_root/audio" -maxdepth 1 -type f -name '*.npy' | wc -l)
frame_count=$(find -L "$target_root/frames" -maxdepth 1 -type f -name '*.jpg' | wc -l)
echo "Prepared audio NPY files: $audio_count"
echo "Reused JPG frames: $frame_count"
if [ "$audio_count" -ne 144000 ] || [ "$frame_count" -ne 144000 ]; then
    echo "Preparation is incomplete."
    exit 1
fi

date -Iseconds > "$target_root/PREPARATION_COMPLETE"
echo "Preparation completed successfully."

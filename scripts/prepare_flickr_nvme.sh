#!/usr/bin/env bash
set -euo pipefail

workers=${1:-4}
project_root=$(cd "$(dirname "$0")/.." && pwd)
source_root=${FLICKR_FLAT_ROOT:-$project_root/prepared/jsa_flickr_144k_real_flat}
target_root=${FLICKR_NVME_ROOT:-/home/wxr/datasets/JSA/FlickrSoundNet_144k_npy}
manifest=${FLICKR_MANIFEST:-$project_root/metadata_flickr/flickr_144k.csv}
python_bin=${JSA_PYTHON:-/home/wxr/miniconda3/envs/wwww/bin/python}
log_experiment=${JSA_LOG_EXPERIMENT:-jsa_flickr_144k_nvme_npy}

log_root=${JSA_LOG_ROOT:-$project_root/checkpoints/$log_experiment/logs}
mkdir -p "$log_root"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$log_root/flickr_nvme_prepare_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

echo "Preparation log: $log_file"
echo "Source: $source_root"
echo "Target: $target_root"
echo "Workers: $workers"

mkdir -p "$target_root/audio" "$target_root/frames"

source_device=$(stat -c '%d' "$source_root/audio")
target_device=$(stat -c '%d' "$target_root/audio")
if [ "$source_device" = "$target_device" ]; then
    echo "Refusing to continue: source and target are on the same filesystem device."
    exit 1
fi

echo "Copying JPG frames to NVMe..."
rsync -a --ignore-existing --info=progress2 \
    "$source_root/frames/" "$target_root/frames/"

echo "Precomputing power spectrograms to NVMe..."
cd "$project_root"
"$python_bin" tools/precompute_flickr_spectrograms.py \
    --manifest "$manifest" \
    --audio-dir "$source_root/audio" \
    --output-dir "$target_root/audio" \
    --duration 5.0 \
    --workers "$workers"

audio_count=$(find "$target_root/audio" -maxdepth 1 -type f -name '*.npy' | wc -l)
frame_count=$(find "$target_root/frames" -maxdepth 1 -type f -name '*.jpg' | wc -l)
echo "Prepared audio NPY files: $audio_count"
echo "Prepared JPG frames: $frame_count"
if [ "$audio_count" -ne 144000 ] || [ "$frame_count" -ne 144000 ]; then
    echo "Preparation is incomplete."
    exit 1
fi

date -Iseconds > "$target_root/PREPARATION_COMPLETE"
echo "Preparation completed successfully."

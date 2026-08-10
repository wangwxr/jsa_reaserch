#!/usr/bin/env bash
set -euo pipefail

dataset_root=${1:-/data/wxr/datasets/FlickrSoundNet}
split_root=${2:-/data/wxr/audio_video/EZ-VSL/metadata}
repo_root=$(cd "$(dirname "$0")/.." && pwd)

bash "$repo_root/scripts/extract_flickr_archives.sh" \
    "$dataset_root/raw_tar" \
    "$dataset_root/extracted"

python "$repo_root/tools/build_flickr_view.py" \
    --manifest "$split_root/flickr_10k.txt" \
    --extracted-dir "$dataset_root/extracted" \
    --output-dir "$dataset_root/prepared/jsa_flickr_10k"

python "$repo_root/tools/build_flickr_view.py" \
    --manifest "$split_root/flickr_144k.txt" \
    --extracted-dir "$dataset_root/extracted" \
    --output-dir "$dataset_root/prepared/jsa_flickr_144k"

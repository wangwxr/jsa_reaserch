#!/usr/bin/env bash
set -euo pipefail

project_root="/data/wxr/audio_video/JSA"
script_dir="${project_root}/chuagnxindian/1.3G诊断"
gpu_id="${1:-0}"
num_samples="${2:-10}"

cd "${project_root}"
python "${script_dir}/generate_visualizations.py" \
  --datasets vggss flickr \
  --gpu "${gpu_id}" \
  --num-samples "${num_samples}"

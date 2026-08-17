#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-python}
mkdir -p "$project_root/checkpoints/topdown_l3_refine_sanity/logs"
timestamp=$(date +%Y%m%d_%H%M%S)

cd "$project_root"
"$python_bin" "$script_dir/train.py" \
    --experiment vggss_10k --gpu 0 --sanity-only \
    > >(tee "$project_root/checkpoints/topdown_l3_refine_sanity/logs/vggss_10k_${timestamp}.log") 2>&1 &
vgg_pid=$!

"$python_bin" "$script_dir/train.py" \
    --experiment flickr_10k --gpu 1 --sanity-only \
    > >(tee "$project_root/checkpoints/topdown_l3_refine_sanity/logs/flickr_10k_${timestamp}.log") 2>&1 &
flickr_pid=$!

echo "Sanity checks started: VGGSS-10k PID=$vgg_pid GPU=0; Flickr-10k PID=$flickr_pid GPU=1"
wait "$vgg_pid"
wait "$flickr_pid"
echo "Both 10k sanity checks passed. No optimizer step was executed."

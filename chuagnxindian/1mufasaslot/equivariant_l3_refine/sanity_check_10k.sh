#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-python}
log_dir="$project_root/checkpoints/equivariant_l3_refine_sanity/logs"
mkdir -p "$log_dir"
timestamp=$(date +%Y%m%d_%H%M%S)

cd "$project_root"
"$python_bin" "$script_dir/train.py" \
    --experiment vggss_10k --gpu 0 --sanity-only \
    > >(tee "$log_dir/vggss_10k_${timestamp}.log") 2>&1 &
vgg_pid=$!

"$python_bin" "$script_dir/train.py" \
    --experiment flickr_10k --gpu 1 --sanity-only \
    > >(tee "$log_dir/flickr_10k_${timestamp}.log") 2>&1 &
flickr_pid=$!

echo "Experiment F sanity checks: VGG PID=$vgg_pid GPU=0; Flickr PID=$flickr_pid GPU=1"
vgg_status=0
flickr_status=0
wait "$vgg_pid" || vgg_status=$?
wait "$flickr_pid" || flickr_status=$?
if [ "$vgg_status" -ne 0 ] || [ "$flickr_status" -ne 0 ]; then
    echo "Sanity check failed: VGG status=$vgg_status, Flickr status=$flickr_status"
    exit 1
fi
echo "Both Experiment F 10k sanity checks passed; no training checkpoint was written."

#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-python}
log_dir="$project_root/checkpoints/1.4G-v2-semantic_preserving_multigeom_sanity/logs"
mkdir -p "$log_dir"
timestamp=$(date +%Y%m%d_%H%M%S)

cd "$project_root"
"$python_bin" "$script_dir/train.py" \
    --experiment vggss_10k --gpu 0 --sanity-only \
    > >(tee -a "$log_dir/vggss_10k_${timestamp}.log") 2>&1 &
vgg_pid=$!
"$python_bin" "$script_dir/train.py" \
    --experiment flickr_10k --gpu 1 --sanity-only \
    > >(tee -a "$log_dir/flickr_10k_${timestamp}.log") 2>&1 &
flickr_pid=$!

echo "Experiment G-v2 sanity: VGG PID=$vgg_pid GPU=0; Flickr PID=$flickr_pid GPU=1"
status=0
wait "$vgg_pid" || status=$?
wait "$flickr_pid" || status=$?
if [ "$status" -eq 0 ]; then
    echo "Both Experiment G-v2 sanity checks passed."
fi
exit "$status"

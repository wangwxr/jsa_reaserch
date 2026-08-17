#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-python}
vgg_exp=${VGG_EXPERIMENT:-1.5G-v3semantic_priority_gradient_refine_vggss_10k}
flickr_exp=${FLICKR_EXPERIMENT:-1.5G-v3semantic_priority_gradient_refine_flickr_10k_frame8_center5}
timestamp=$(date +%Y%m%d_%H%M%S)

mkdir -p "$project_root/checkpoints/$vgg_exp/logs"
mkdir -p "$project_root/checkpoints/$flickr_exp/logs"
cd "$project_root"
"$python_bin" "$script_dir/evaluate_full.py" \
    --experiment vggss_10k --gpu 0 --experiment-name "$vgg_exp" \
    > >(tee -a "$project_root/checkpoints/$vgg_exp/logs/${vgg_exp}_full_six_metrics_${timestamp}.log") 2>&1 &
vgg_pid=$!
"$python_bin" "$script_dir/evaluate_full.py" \
    --experiment flickr_10k --gpu 1 --experiment-name "$flickr_exp" \
    > >(tee -a "$project_root/checkpoints/$flickr_exp/logs/${flickr_exp}_full_six_metrics_${timestamp}.log") 2>&1 &
flickr_pid=$!

status=0
wait "$vgg_pid" || status=$?
wait "$flickr_pid" || status=$?
exit "$status"

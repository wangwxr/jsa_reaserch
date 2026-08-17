#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-python}
vgg_exp=${VGG_EXPERIMENT:-1.3G-multigeom_equivariant_l3_refine_vggss_144k}
flickr_exp=${FLICKR_EXPERIMENT:-1.3G-multigeom_equivariant_l3_refine_flickr_144k_frame8_center5}

mkdir -p "$project_root/checkpoints/$vgg_exp/logs"
mkdir -p "$project_root/checkpoints/$flickr_exp/logs"
timestamp=$(date +%Y%m%d_%H%M%S)
vgg_log="$project_root/checkpoints/$vgg_exp/logs/${vgg_exp}_full_six_metrics_${timestamp}.log"
flickr_log="$project_root/checkpoints/$flickr_exp/logs/${flickr_exp}_full_six_metrics_${timestamp}.log"

cd "$project_root"
"$python_bin" "$script_dir/evaluate_full.py" \
    --experiment vggss_144k --gpu 0 --experiment-name "$vgg_exp" \
    > >(tee -a "$vgg_log") 2>&1 &
vgg_pid=$!
"$python_bin" "$script_dir/evaluate_full.py" \
    --experiment flickr_144k --gpu 1 --experiment-name "$flickr_exp" \
    > >(tee -a "$flickr_log") 2>&1 &
flickr_pid=$!

status=0
wait "$vgg_pid" || status=$?
wait "$flickr_pid" || status=$?
exit "$status"

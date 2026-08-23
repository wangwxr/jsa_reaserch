#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-python}

run_vgg_pipeline() {
    bash "$script_dir/train_vggss_10k.sh" 0
    "$python_bin" "$script_dir/check_10k.py" --experiment vggss_10k
    bash "$script_dir/train_vggss_144k.sh" 0
}

run_flickr_pipeline() {
    bash "$script_dir/train_flickr_10k.sh" 1
    "$python_bin" "$script_dir/check_10k.py" --experiment flickr_10k
    bash "$script_dir/train_flickr_144k.sh" 1
}

run_vgg_pipeline &
vgg_pid=$!
run_flickr_pipeline &
flickr_pid=$!
echo "Independent pipelines started: VGG PID=$vgg_pid GPU=0; Flickr PID=$flickr_pid GPU=1"
vgg_status=0
flickr_status=0
wait "$vgg_pid" || vgg_status=$?
wait "$flickr_pid" || flickr_status=$?
if [ "$vgg_status" -ne 0 ] || [ "$flickr_status" -ne 0 ]; then
    echo "1.3G-v2 staged pipelines failed: VGG=$vgg_status Flickr=$flickr_status"
    exit 1
fi
cd "$project_root"
"$python_bin" "$script_dir/aggregate_results.py"

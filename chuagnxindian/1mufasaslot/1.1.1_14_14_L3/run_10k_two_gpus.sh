#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)

bash "$script_dir/run_dataset_pipeline.sh" vggss 0 10k &
vgg_pid=$!
bash "$script_dir/run_dataset_pipeline.sh" flickr 1 10k &
flickr_pid=$!
echo "10k pipelines started: VGGSS queue PID=$vgg_pid GPU=0; Flickr queue PID=$flickr_pid GPU=1"

vgg_status=0
flickr_status=0
wait "$vgg_pid" || vgg_status=$?
wait "$flickr_pid" || flickr_status=$?
if [ "$vgg_status" -ne 0 ] || [ "$flickr_status" -ne 0 ]; then
    echo "10k pipeline failed: VGGSS=$vgg_status Flickr=$flickr_status"
    exit 1
fi
"${JSA_PYTHON:-/home/wxr/miniconda3/envs/wwww/bin/python}" "$script_dir/aggregate_results.py"
echo "Both 10k two-stage experiments completed."

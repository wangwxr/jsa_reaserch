#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
log_dir="$script_dir/orchestration_logs"
mkdir -p "$log_dir"

bash "$script_dir/run_dataset_pipeline.sh" vggss 0 10k 144k &
vgg_pid=$!
bash "$script_dir/run_dataset_pipeline.sh" flickr 1 10k 144k &
flickr_pid=$!
state_file="$log_dir/active_pipelines.tsv"
{
    echo -e "dataset\tgpu\tqueue_pid"
    echo -e "vggss\t0\t$vgg_pid"
    echo -e "flickr\t1\t$flickr_pid"
} > "$state_file"
echo "All pipelines started: VGGSS queue PID=$vgg_pid GPU=0; Flickr queue PID=$flickr_pid GPU=1"
echo "PID audit: $state_file"

vgg_status=0
flickr_status=0
wait "$vgg_pid" || vgg_status=$?
wait "$flickr_pid" || flickr_status=$?
if [ "$vgg_status" -ne 0 ] || [ "$flickr_status" -ne 0 ]; then
    echo "Full pipeline failed: VGGSS=$vgg_status Flickr=$flickr_status"
    exit 1
fi
"${JSA_PYTHON:-/home/wxr/miniconda3/envs/wwww/bin/python}" "$script_dir/aggregate_results.py"
echo "All 10k and 144k native-L3 + G experiments completed."

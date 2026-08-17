#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
bash "$script_dir/train_vggss_10k.sh" 0 &
vgg_pid=$!
bash "$script_dir/train_flickr_10k.sh" 1 &
flickr_pid=$!

echo "Experiment D started: VGGSS-10k PID=$vgg_pid; Flickr-10k PID=$flickr_pid"
wait "$vgg_pid"
wait "$flickr_pid"
echo "Both Experiment D 10k runs completed."

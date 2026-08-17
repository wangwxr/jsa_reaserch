#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
bash "$script_dir/train_vggss.sh" 144k 0 &
vgg_pid=$!
bash "$script_dir/train_flickr.sh" 144k 1 &
flickr_pid=$!

echo "Stage-2 144k runs started: VGGSS PID=$vgg_pid GPU=0; Flickr PID=$flickr_pid GPU=1"
wait "$vgg_pid"
wait "$flickr_pid"
echo "Both stage-2 144k runs completed."

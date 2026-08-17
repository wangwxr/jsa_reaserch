#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
bash "$script_dir/train_vggss_10k.sh" 0 &
vgg_pid=$!
bash "$script_dir/train_flickr_10k.sh" 1 &
flickr_pid=$!

echo "Experiment G-v2: VGG PID=$vgg_pid GPU=0; Flickr PID=$flickr_pid GPU=1"
status=0
wait "$vgg_pid" || status=$?
wait "$flickr_pid" || status=$?
exit "$status"

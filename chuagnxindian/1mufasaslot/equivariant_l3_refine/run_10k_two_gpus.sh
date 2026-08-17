#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
bash "$script_dir/train_vggss_10k.sh" 0 &
vgg_pid=$!
bash "$script_dir/train_flickr_10k.sh" 1 &
flickr_pid=$!

echo "Experiment F started: VGGSoundSS-10k PID=$vgg_pid GPU=0; Flickr-10k PID=$flickr_pid GPU=1"
vgg_status=0
flickr_status=0
wait "$vgg_pid" || vgg_status=$?
wait "$flickr_pid" || flickr_status=$?
if [ "$vgg_status" -ne 0 ] || [ "$flickr_status" -ne 0 ]; then
    echo "Experiment F failed: VGG status=$vgg_status, Flickr status=$flickr_status"
    exit 1
fi
echo "Both Experiment F 10k runs completed."

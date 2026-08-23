#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
bash "$script_dir/train_vggss_144k.sh" 0 &
vgg_pid=$!
bash "$script_dir/train_flickr_144k.sh" 1 &
flickr_pid=$!
echo "1.3G-v2 144k started: VGG PID=$vgg_pid GPU=0; Flickr PID=$flickr_pid GPU=1"
vgg_status=0
flickr_status=0
wait "$vgg_pid" || vgg_status=$?
wait "$flickr_pid" || flickr_status=$?
if [ "$vgg_status" -ne 0 ] || [ "$flickr_status" -ne 0 ]; then
    echo "1.3G-v2 144k failed: VGG=$vgg_status Flickr=$flickr_status"
    exit 1
fi
echo "Both 1.3G-v2 144k runs completed."

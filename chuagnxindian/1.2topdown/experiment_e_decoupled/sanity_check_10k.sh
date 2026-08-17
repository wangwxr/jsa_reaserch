#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-python}
cd "$project_root"

"$python_bin" "$script_dir/train_e.py" \
    --experiment vggss_10k --gpu 0 --sanity-only &
vgg_pid=$!
"$python_bin" "$script_dir/train_e.py" \
    --experiment flickr_10k --gpu 1 --sanity-only &
flickr_pid=$!
wait "$vgg_pid"
wait "$flickr_pid"
echo "Both Experiment E sanity checks passed."

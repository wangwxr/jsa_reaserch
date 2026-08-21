#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
python_bin="${JSA_PYTHON:-python}"
cd "${project_root}"
mkdir -p "${script_dir}/smoke_results/logs"

"${python_bin}" "${script_dir}/train.py" --experiment vggss_144k --gpu 0 --smoke-only \
  2>&1 | tee "${script_dir}/smoke_results/logs/vggss_144k.log" &
vgg_pid=$!
"${python_bin}" "${script_dir}/train.py" --experiment flickr_144k --gpu 1 --smoke-only \
  2>&1 | tee "${script_dir}/smoke_results/logs/flickr_144k.log" &
flickr_pid=$!
wait "${vgg_pid}"
wait "${flickr_pid}"


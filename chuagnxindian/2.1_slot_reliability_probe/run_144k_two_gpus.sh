#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
output_root="${SLOT_RELIABILITY_OUTPUT_ROOT:-${script_dir}/results}"
python_bin="${PYTHON_BIN:-python}"
mkdir -p "${output_root}/logs"
cd "${project_root}"

run_one() {
  local experiment="$1"
  local gpu="$2"
  "${python_bin}" "${script_dir}/probe.py" \
    --experiment "${experiment}" \
    --gpu "${gpu}" \
    --output-root "${output_root}" \
    2>&1 | tee "${output_root}/logs/${experiment}.log"
}

echo "Experiment 2.1 is zero-training and read-only."
echo "VGGSoundSS-144k -> GPU 0; Flickr-144k -> GPU 1"
run_one vggss_144k 0 &
vgg_pid=$!
run_one flickr_144k 1 &
flickr_pid=$!

vgg_status=0
flickr_status=0
wait "${vgg_pid}" || vgg_status=$?
wait "${flickr_pid}" || flickr_status=$?
if (( vgg_status != 0 || flickr_status != 0 )); then
  echo "2.1 failed: VGG=${vgg_status}, Flickr=${flickr_status}" >&2
  exit 1
fi

"${python_bin}" "${script_dir}/aggregate_results.py" --result-root "${output_root}"

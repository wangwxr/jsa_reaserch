#!/usr/bin/env bash
set -euo pipefail

repo_root="/data/wxr/audio_video/JSA"
experiment_root="${repo_root}/chuagnxindian/6.1_spatial_att_temporal_schedule"

cd "${repo_root}"
python "${experiment_root}/test_schedule.py"

run_pair() {
  local schedule="$1"
  echo "Starting ${schedule}: VGGSS on GPU 0, Flickr on GPU 1"
  "${experiment_root}/run_one_10k.sh" "${schedule}" vggss 0 &
  local vgg_pid=$!
  "${experiment_root}/run_one_10k.sh" "${schedule}" flickr 1 &
  local flickr_pid=$!

  local failed=0
  wait "${vgg_pid}" || failed=1
  wait "${flickr_pid}" || failed=1
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one ${schedule} process failed; stopping scheduler." >&2
    exit 1
  fi
}

run_pair early_high_late_low
run_pair early_low_late_high

python "${experiment_root}/aggregate_results.py"

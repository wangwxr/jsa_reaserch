#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/results/logs"

CUDA_VISIBLE_DEVICES=0 python "$HERE/probe.py" \
  --experiment vggss_144k --gpu 0 \
  >"$HERE/results/logs/vggss_144k.log" 2>&1 &
PID_VGG=$!

CUDA_VISIBLE_DEVICES=1 python "$HERE/probe.py" \
  --experiment flickr_144k --gpu 0 \
  >"$HERE/results/logs/flickr_144k.log" 2>&1 &
PID_FLICKR=$!

wait "$PID_VGG"
wait "$PID_FLICKR"

python "$HERE/aggregate_results.py"

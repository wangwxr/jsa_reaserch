#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${1:-${HERE}/results}"
mkdir -p "${RESULTS_ROOT}/logs"

CUDA_VISIBLE_DEVICES=0 python "${HERE}/probe.py" \
  --experiment vggss_144k --gpu 0 --output-root "${RESULTS_ROOT}" \
  >"${RESULTS_ROOT}/logs/vggss_144k.log" 2>&1 &
VGG_PID=$!

CUDA_VISIBLE_DEVICES=1 python "${HERE}/probe.py" \
  --experiment flickr_144k --gpu 0 --output-root "${RESULTS_ROOT}" \
  >"${RESULTS_ROOT}/logs/flickr_144k.log" 2>&1 &
FLICKR_PID=$!

status=0
wait "${VGG_PID}" || status=$?
wait "${FLICKR_PID}" || status=$?
exit "${status}"

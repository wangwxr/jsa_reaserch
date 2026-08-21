#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${HERE}/results"
mkdir -p "${RESULTS}/logs"

python "${HERE}/probe.py" \
  --experiment vggss_144k --gpu 0 --output-root "${RESULTS}" \
  >"${RESULTS}/logs/vggss_144k.log" 2>&1 &
VGG_PID=$!

python "${HERE}/probe.py" \
  --experiment flickr_144k --gpu 1 --output-root "${RESULTS}" \
  >"${RESULTS}/logs/flickr_144k.log" 2>&1 &
FLICKR_PID=$!

status=0
wait "${VGG_PID}" || status=$?
wait "${FLICKR_PID}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  echo "4.1 probe failed; inspect ${RESULTS}/logs" >&2
  exit "${status}"
fi

python "${HERE}/aggregate_results.py" --results-root "${RESULTS}" --report "${HERE}/REPORT.md"
echo "Experiment 4.1 complete: ${HERE}/REPORT.md"

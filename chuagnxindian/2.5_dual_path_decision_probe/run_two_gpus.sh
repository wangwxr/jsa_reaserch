#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

(
  python "$HERE/probe.py" --experiment vggss_10k --gpu 0
  python "$HERE/probe.py" --experiment vggss_144k --gpu 0
) >"$HERE/vgg.log" 2>&1 &
vgg_pid=$!

(
  python "$HERE/probe.py" --experiment flickr_10k --gpu 1
  python "$HERE/probe.py" --experiment flickr_144k --gpu 1
) >"$HERE/flickr.log" 2>&1 &
flickr_pid=$!

wait "$vgg_pid"
wait "$flickr_pid"
python "$HERE/aggregate_results.py"


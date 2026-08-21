#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${JSA_PYTHON:-python}
mkdir -p "$HERE/logs" "$HERE/results"

run_probe() {
    local setting=$1
    local gpu=$2
    local result="$HERE/results/$setting/summary.json"
    if [ -f "$result" ]; then
        echo "[$setting] completed result exists; skipping"
        return
    fi
    "$PYTHON_BIN" "$HERE/probe.py" \
        --experiment "$setting" \
        --gpu "$gpu" \
        2>&1 | tee "$HERE/logs/${setting}.log"
}

run_probe vggss_144k 0 &
vgg_pid=$!
run_probe flickr_144k 1 &
flickr_pid=$!

wait "$vgg_pid"
wait "$flickr_pid"

"$PYTHON_BIN" "$HERE/aggregate_results.py" | tee "$HERE/logs/aggregate.log"


#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${JSA_PYTHON:-python}
mkdir -p "$HERE/logs"

run_setting() {
    local setting=$1
    local gpu=$2
    local experiment
    experiment=$(
        "$PYTHON_BIN" -c \
            "import sys; sys.path.insert(0, '$HERE'); import common; print(common.registry('$setting')['experiment'])"
    )
    local checkpoint_dir="$HERE/checkpoints/$experiment"
    local result_dir="$HERE/results/$setting"
    local train_log="$HERE/logs/${setting}_train.log"
    local eval_log="$HERE/logs/${setting}_evaluate.log"

    if [ -f "$result_dir/summary.json" ]; then
        echo "[$setting] completed result exists; skipping"
        return
    fi
    if [ ! -f "$checkpoint_dir/final.pth" ]; then
        if [ -f "$checkpoint_dir/latest.pth" ]; then
            echo "[$setting] incomplete prior run found at $checkpoint_dir; refusing implicit resume"
            return 1
        fi
        echo "[$setting] training on GPU $gpu"
        "$PYTHON_BIN" "$HERE/train.py" --setting "$setting" --gpu "$gpu" \
            2>&1 | tee "$train_log"
    else
        echo "[$setting] completed checkpoint exists; running evaluation only"
    fi
    "$PYTHON_BIN" "$HERE/evaluate.py" --setting "$setting" --gpu "$gpu" \
        2>&1 | tee "$eval_log"
}

for setting in vggss_10k vggss_144k flickr_10k flickr_144k; do
    if [ ! -f "$HERE/results/$setting/smoke.json" ]; then
        echo "Missing smoke result for $setting; run smoke.py before formal training"
        exit 1
    fi
    "$PYTHON_BIN" -c \
        "import json; assert json.load(open('$HERE/results/$setting/smoke.json'))['passed']"
done

(
    run_setting vggss_10k 0
    run_setting vggss_144k 0
) &
vgg_pid=$!

(
    run_setting flickr_10k 1
    run_setting flickr_144k 1
) &
flickr_pid=$!

wait "$vgg_pid"
wait "$flickr_pid"
"$PYTHON_BIN" "$HERE/aggregate_results.py" | tee "$HERE/logs/aggregate.log"


#!/usr/bin/env bash
set -euo pipefail
script_root=$(cd "$(dirname "$0")" && pwd)
exec bash "$script_root/../ablation_common/train_flickr.sh" \
    "$script_root/train_no_slot.py" b0_baseline b0_baseline av_mil "$@"

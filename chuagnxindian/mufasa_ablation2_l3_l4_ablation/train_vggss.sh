#!/usr/bin/env bash
set -euo pipefail
script_root=$(cd "$(dirname "$0")" && pwd)
exec bash "$script_root/../ablation_common/train_vggss.sh" \
    "$script_root/train.py" mufasa_ablation2_l3_l4_ablation \
    mufasa_ablation2_l3_l4_ablation jsa "$@"

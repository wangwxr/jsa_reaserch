#!/usr/bin/env bash
set -euo pipefail
script_root=$(cd "$(dirname "$0")" && pwd)
exec bash "$script_root/../ablation_common/test_flickr.sh" \
    "$script_root/test.py" mufasa_ablation2_l3_l4_ablation "$@"

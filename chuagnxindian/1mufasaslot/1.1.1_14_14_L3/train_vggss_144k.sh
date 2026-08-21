#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
bash "$script_dir/train_dataset.sh" vggss 144k "${1:-0}" "${2:-1.1.1_14_14_L3_vggss_144k}"

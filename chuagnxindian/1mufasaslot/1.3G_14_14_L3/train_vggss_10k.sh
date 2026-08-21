#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
bash "$script_dir/train_dataset.sh" vggss 10k "${1:-0}" "${2:-1.3G_14_14_L3_vggss_10k}"

#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
bash "$script_dir/train_dataset.sh" flickr 144k "${1:-1}" "${2:-1.3G_14_14_L3_flickr_144k}"

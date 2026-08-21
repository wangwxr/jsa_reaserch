#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
bash "$script_dir/train_dataset.sh" flickr 10k "${1:-1}" "${2:-1.1.1_14_14_L3_flickr_10k}"

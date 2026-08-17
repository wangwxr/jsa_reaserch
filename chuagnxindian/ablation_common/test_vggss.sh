#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 6 ]; then
    echo "Internal usage: $0 ENTRY ARCH EXPERIMENT [GPU] [ALPHA] [CHECKPOINT]"
    exit 2
fi

entry_script=$1
architecture=$2
experiment_name=$3
gpu=${4:-0}
alpha=${5:-0.6}
common_root=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$common_root/../.." && pwd)
metadata_root="$project_root/metadata_vggss"
test_root=${VGGSS_TEST_ROOT:-/data/wxr/datasets/ACL-SSL/VGGSound}
python_bin=${JSA_PYTHON:-python}
checkpoint=${6:-vggss_best.pth}

log_root=${JSA_LOG_ROOT:-$project_root/checkpoints/$experiment_name/logs}
mkdir -p "$log_root"
timestamp=$(date +%Y%m%d_%H%M%S)
checkpoint_label=${checkpoint%.pth}
log_file="$log_root/${experiment_name}_test_vggss_${checkpoint_label}_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1
echo "Test log: $log_file"
echo "Architecture: $architecture; experiment: $experiment_name; checkpoint: $checkpoint; GPU: $gpu"

cd "$project_root"
"$python_bin" "$entry_script" \
    --experiment_name "$experiment_name" \
    --testset vggss \
    --test_data_path "$test_root" \
    --test_manifest_path "$metadata_root/vggss_test.csv" \
    --test_gt_path "$metadata_root/vggss.json" \
    --checkpoint "$checkpoint" \
    --gpu "$gpu" \
    --alpha "$alpha" \
    --infer_sharpening 0.1

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 {10k|144k} [GPU_ID] [EXPERIMENT_NAME]"
    exit 2
fi

split=$1
gpu=${2:-0}
experiment_name=${3:-topdown_l3_refine_vggss_${split}}
script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${JSA_PYTHON:-python}
log_dir="$project_root/checkpoints/$experiment_name/logs"
mkdir -p "$log_dir"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$log_dir/${experiment_name}_test_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

echo "Test log: $log_file"
cd "$project_root"
"$python_bin" "$script_dir/test.py" \
    --experiment "vggss_${split}" \
    --gpu "$gpu" \
    --experiment-name "$experiment_name"

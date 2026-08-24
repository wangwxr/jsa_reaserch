#!/usr/bin/env bash
set -euo pipefail

script_root=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_root/../.." && pwd)
python_bin=${JSA_PYTHON:-python}
export MPLCONFIGDIR="$script_root/results/.matplotlib"
mkdir -p "$MPLCONFIGDIR" "$script_root/results/audit/logs"

wait_pair() {
    local pid0=$1
    local pid1=$2
    local label=$3
    local status0=0
    local status1=0
    wait "$pid0" || status0=$?
    if [ "$status0" -ne 0 ]; then
        kill "$pid1" 2>/dev/null || true
    fi
    wait "$pid1" || status1=$?
    if [ "$status0" -ne 0 ] || [ "$status1" -ne 0 ]; then
        echo "FAILED pair $label: GPU0=$status0 GPU1=$status1"
        exit 1
    fi
    echo "COMPLETED pair $label"
}

cd "$project_root"
if [ ! -f "$script_root/results/audit/audit_summary.json" ]; then
    "$python_bin" "$script_root/audit.py" --dataset vggss --gpu 0 --workers 16 \
        > >(tee -a "$script_root/results/audit/logs/vggss.log") 2>&1 &
    audit0=$!
    "$python_bin" "$script_root/audit.py" --dataset flickr --gpu 1 --workers 12 \
        > >(tee -a "$script_root/results/audit/logs/flickr.log") 2>&1 &
    audit1=$!
    wait_pair "$audit0" "$audit1" "Stage-A audits"
    "$python_bin" "$script_root/audit.py" --aggregate
else
    "$python_bin" -c "import json; p=json.load(open('$script_root/results/audit/audit_summary.json')); assert p['stage_a_passed'], p"
    echo "Using previously passed Stage A audit."
fi

"$python_bin" "$script_root/evaluate_reference.py" --dataset vggss --gpu 0 --workers 16 &
ref0=$!
"$python_bin" "$script_root/evaluate_reference.py" --dataset flickr --gpu 1 --workers 12 &
ref1=$!
wait_pair "$ref0" "$ref1" "FULL_REFERENCE evaluation"

configs=(no_spatial_att no_image_rec no_both)
for configuration in "${configs[@]}"; do
    case "$configuration" in
        no_spatial_att) prefix="6.0_no_spatial_att" ;;
        no_image_rec) prefix="6.0_no_image_rec" ;;
        no_both) prefix="6.0_no_both" ;;
    esac
    bash "$script_root/run_one_10k.sh" vggss "$configuration" 0 "${prefix}_vggss_10k" &
    pid0=$!
    bash "$script_root/run_one_10k.sh" flickr "$configuration" 1 "${prefix}_flickr_10k_frame8_center5" &
    pid1=$!
    wait_pair "$pid0" "$pid1" "$configuration"
done

"$python_bin" "$script_root/aggregate_results.py"
echo "Experiment 6.0 complete. No 144k job was started."

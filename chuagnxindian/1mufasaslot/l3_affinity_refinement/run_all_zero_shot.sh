#!/usr/bin/env bash
set -euo pipefail

script_root=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "$script_root/../../.." && pwd)
python_bin=${JSA_PYTHON:-python}
gpu_vgg=${GPU_VGG:-0}
gpu_flickr=${GPU_FLICKR:-1}

cd "$project_root"

echo "Stage 1/3: reproduce all four formal AUD_L4 baselines before refinement"
(
  "$python_bin" "$script_root/evaluate_refinement.py" --experiment vggss_10k --gpu "$gpu_vgg" --baseline-only
  "$python_bin" "$script_root/evaluate_refinement.py" --experiment vggss_144k --gpu "$gpu_vgg" --baseline-only
) &
vgg_gate_pid=$!
(
  "$python_bin" "$script_root/evaluate_refinement.py" --experiment flickr_10k --gpu "$gpu_flickr" --baseline-only
  "$python_bin" "$script_root/evaluate_refinement.py" --experiment flickr_144k --gpu "$gpu_flickr" --baseline-only
) &
flickr_gate_pid=$!
wait "$vgg_gate_pid"
wait "$flickr_gate_pid"

echo "Stage 2/3: run zero-training sweeps; each free GPU advances independently"
(
  "$python_bin" "$script_root/evaluate_refinement.py" --experiment vggss_10k --gpu "$gpu_vgg"
  "$python_bin" "$script_root/evaluate_refinement.py" --experiment vggss_144k --gpu "$gpu_vgg" --qualitative
) &
vgg_pid=$!
(
  "$python_bin" "$script_root/evaluate_refinement.py" --experiment flickr_10k --gpu "$gpu_flickr"
  "$python_bin" "$script_root/evaluate_refinement.py" --experiment flickr_144k --gpu "$gpu_flickr" --qualitative
) &
flickr_pid=$!

wait "$vgg_pid"
wait "$flickr_pid"

echo "Stage 3/3: merge all 88 rows and print summaries"
"$python_bin" "$script_root/aggregate_results.py"

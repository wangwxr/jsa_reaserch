#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
python_bin="${JSA_PYTHON:-python}"
vgg_name="2.3_semantic_spatial_decoupled_slot_vggss_144k"
flickr_name="2.3_semantic_spatial_decoupled_slot_flickr_144k_frame8_center5"

cd "${project_root}"
mkdir -p "${script_dir}/smoke_results/logs"

echo "Stage 1/2: parallel smoke and full frozen-AUD reproduction"
"${python_bin}" "${script_dir}/train.py" \
  --experiment vggss_144k --gpu 0 --smoke-only \
  >"${script_dir}/smoke_results/logs/vggss_144k.log" 2>&1 &
vgg_smoke_pid=$!
"${python_bin}" "${script_dir}/train.py" \
  --experiment flickr_144k --gpu 1 --smoke-only \
  >"${script_dir}/smoke_results/logs/flickr_144k.log" 2>&1 &
flickr_smoke_pid=$!
vgg_smoke_status=0
flickr_smoke_status=0
wait "${vgg_smoke_pid}" || vgg_smoke_status=$?
wait "${flickr_smoke_pid}" || flickr_smoke_status=$?
if (( vgg_smoke_status != 0 || flickr_smoke_status != 0 )); then
  echo "Smoke failed: VGG=${vgg_smoke_status}, Flickr=${flickr_smoke_status}" >&2
  tail -n 100 "${script_dir}/smoke_results/logs/vggss_144k.log" || true
  tail -n 100 "${script_dir}/smoke_results/logs/flickr_144k.log" || true
  exit 1
fi
echo "Both smoke audits passed."

for experiment_name in "${vgg_name}" "${flickr_name}"; do
  experiment_dir="${project_root}/checkpoints/${experiment_name}"
  if [[ -f "${experiment_dir}/latest.pth" || -f "${experiment_dir}/final.pth" ]]; then
    echo "Refusing to overwrite existing experiment: ${experiment_dir}" >&2
    exit 1
  fi
  mkdir -p "${experiment_dir}/logs"
done

timestamp="$(date +%Y%m%d_%H%M%S)"
vgg_log="${project_root}/checkpoints/${vgg_name}/logs/${vgg_name}_train_${timestamp}.log"
flickr_log="${project_root}/checkpoints/${flickr_name}/logs/${flickr_name}_train_${timestamp}.log"

echo "Stage 2/2: formal 50-epoch training"
echo "VGG log: ${vgg_log}"
echo "Flickr log: ${flickr_log}"
"${python_bin}" "${script_dir}/train.py" \
  --experiment vggss_144k --gpu 0 --epochs 50 --experiment-name "${vgg_name}" \
  2>&1 | tee "${vgg_log}" &
vgg_pid=$!
"${python_bin}" "${script_dir}/train.py" \
  --experiment flickr_144k --gpu 1 --epochs 50 --experiment-name "${flickr_name}" \
  2>&1 | tee "${flickr_log}" &
flickr_pid=$!
vgg_status=0
flickr_status=0
wait "${vgg_pid}" || vgg_status=$?
wait "${flickr_pid}" || flickr_status=$?
if (( vgg_status != 0 || flickr_status != 0 )); then
  echo "Formal training failed: VGG=${vgg_status}, Flickr=${flickr_status}" >&2
  exit 1
fi

"${python_bin}" "${script_dir}/aggregate_results.py"


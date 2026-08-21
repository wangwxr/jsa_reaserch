#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
python_bin="${JSA_PYTHON:-python}"
vgg_name="2.4_object_aware_multigeom_spatial_specialization_vggss_144k"
flickr_name="2.4_object_aware_multigeom_spatial_specialization_flickr_144k_frame8_center5"

cd "${project_root}"
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

echo "Experiment 2.4 starts from formal Stage1 checkpoints; smoke automatically precedes training."
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
  echo "Experiment 2.4 failed: VGG=${vgg_status}, Flickr=${flickr_status}" >&2
  exit 1
fi

"${python_bin}" "${script_dir}/aggregate_results.py"

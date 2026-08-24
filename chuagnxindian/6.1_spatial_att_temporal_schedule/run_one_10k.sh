#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <early_high_late_low|early_low_late_high> <vggss|flickr> <gpu>" >&2
  exit 2
fi

schedule="$1"
dataset="$2"
gpu="$3"
repo_root="/data/wxr/audio_video/JSA"
experiment_root="${repo_root}/chuagnxindian/6.1_spatial_att_temporal_schedule"

case "${schedule}:${dataset}" in
  early_high_late_low:vggss) experiment="6.1_early_high_late_low_vggss_10k" ;;
  early_high_late_low:flickr) experiment="6.1_early_high_late_low_flickr_10k_frame8_center5" ;;
  early_low_late_high:vggss) experiment="6.1_early_low_late_high_vggss_10k" ;;
  early_low_late_high:flickr) experiment="6.1_early_low_late_high_flickr_10k_frame8_center5" ;;
  *) echo "Unsupported schedule/dataset: ${schedule}/${dataset}" >&2; exit 2 ;;
esac

checkpoint_dir="${repo_root}/checkpoints/${experiment}"
mkdir -p "${checkpoint_dir}/logs"
if compgen -G "${checkpoint_dir}/*.pth" >/dev/null; then
  echo "Refusing to overwrite checkpoint(s) in ${checkpoint_dir}" >&2
  exit 3
fi
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="${checkpoint_dir}/logs/${experiment}_train_${timestamp}.log"
workers=16
if [[ "${dataset}" == "flickr" ]]; then workers=12; fi

cd "${repo_root}"
echo "Experiment: ${experiment}; schedule: ${schedule}; dataset: ${dataset}; GPU: ${gpu}; log: ${log_path}"
python -u "${experiment_root}/train.py" \
  --schedule "${schedule}" \
  --dataset "${dataset}" \
  --gpu "${gpu}" \
  --workers "${workers}" \
  --experiment-name "${experiment}" 2>&1 | tee "${log_path}"

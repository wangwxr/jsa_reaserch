#!/usr/bin/env bash
set -euo pipefail

script_root=$(cd "$(dirname "$0")" && pwd)

export JSA_PYTHON=${JSA_PYTHON:-/home/wxr/miniconda3/envs/wwww/bin/python}
export VGG_PREP_ROOT=${VGG_PREP_ROOT:-/home/wxr/datasets/JSA/VGGSound_144k_npy}
export FLICKR_PREP_ROOT=${FLICKR_PREP_ROOT:-/home/wxr/datasets/JSA/FlickrSoundNet_144k_frame8_center5_npy}

run_queue() {
    local dataset=$1
    local gpu=$2
    local experiment_name

    for split in 10k 144k; do
        if [ "$dataset" = "vggss" ]; then
            experiment_name="mufasa_jsa_v1_vggss_${split}"
        else
            experiment_name="mufasa_jsa_v1_flickr_${split}_frame8_center5"
        fi
        echo "Starting ${experiment_name} on GPU ${gpu}"
        bash "$script_root/train_${dataset}_mufasa.sh" \
            "$split" "$gpu" "$experiment_name"
    done
}

echo "JSA Python: $JSA_PYTHON"
echo "VGG prepared root: $VGG_PREP_ROOT"
echo "Flickr prepared root: $FLICKR_PREP_ROOT"

vgg_status=0
flickr_status=0

run_queue vggss 0 &
vgg_pid=$!
run_queue flickr 1 &
flickr_pid=$!

wait "$vgg_pid" || vgg_status=$?
wait "$flickr_pid" || flickr_status=$?

if [ "$vgg_status" -ne 0 ] || [ "$flickr_status" -ne 0 ]; then
    echo "Queue failed: VGGSS=${vgg_status}, Flickr=${flickr_status}"
    exit 1
fi

echo "All four MUFASA-JSA v1 experiments completed successfully."

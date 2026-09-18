#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wxr/audio_video/JSA/chuagnxindian/8.7_10k_mean_hard_cram
PY=/home/wxr/miniconda3/envs/wwww/bin/python
GPU=${1:-0}
mkdir -p "$ROOT/logs"
"$PY" "$ROOT/run_stage1.py" --method mean --dataset flickr --gpu "$GPU" >>"$ROOT/logs/mean_flickr_stage1.log" 2>&1
"$PY" "$ROOT/run_stage2.py" --method mean --dataset flickr --gpu "$GPU" >>"$ROOT/logs/mean_flickr_stage2.log" 2>&1
"$PY" "$ROOT/run_stage1.py" --method hard_default --dataset flickr --gpu "$GPU" >>"$ROOT/logs/hard_default_flickr_stage1.log" 2>&1
"$PY" "$ROOT/run_stage2.py" --method hard_default --dataset flickr --gpu "$GPU" >>"$ROOT/logs/hard_default_flickr_stage2.log" 2>&1

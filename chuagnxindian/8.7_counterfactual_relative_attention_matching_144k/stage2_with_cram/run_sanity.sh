#!/usr/bin/env bash
set -euo pipefail
PY=/home/wxr/miniconda3/envs/wwww/bin/python
ROOT=/data/wxr/audio_video/JSA
cd "$ROOT"
"$PY" chuagnxindian/8.7_counterfactual_relative_attention_matching/stage2_with_cram/train.py --dataset vggss --variant lambda050k --gpu 1 --sanity-only
"$PY" chuagnxindian/8.7_counterfactual_relative_attention_matching/stage2_with_cram/train.py --dataset flickr --variant lambda050k --gpu 1 --sanity-only

# 8.7 C3 Stage-2 with additive CRAM

This directory is isolated from the successful C3 Stage-2 run. It uses the C3
Stage-1 selected teacher and leaves all original Stage-2 losses and architecture
intact. Read `PROTOCOL.md` before launch.

The completed pre-training sanity checks are stored under `sanity/` for
`lambda050k`; no training checkpoint exists yet.

Commands, once training is authorized:

```bash
PY=/home/wxr/miniconda3/envs/wwww/bin/python
cd /data/wxr/audio_video/JSA

# Required no-write preflight
$PY chuagnxindian/8.7_counterfactual_relative_attention_matching/stage2_with_cram/train.py \
  --dataset vggss --variant lambda050k --gpu 0 --sanity-only

# One independent variant per dataset; use separate GPUs.
$PY chuagnxindian/8.7_counterfactual_relative_attention_matching/stage2_with_cram/train.py \
  --dataset vggss --variant lambda050k --gpu 0 --epochs 50 --run-train
$PY chuagnxindian/8.7_counterfactual_relative_attention_matching/stage2_with_cram/train.py \
  --dataset flickr --variant lambda050k --gpu 1 --epochs 50 --run-train
```

The pre-registered variants are `lambda025k`, `lambda050k`, and `lambda100k`.
They are gradient-scale variants, not changes to CRAM itself.

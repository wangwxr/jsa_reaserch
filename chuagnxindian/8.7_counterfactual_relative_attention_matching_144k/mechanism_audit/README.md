# Experiment 8.7 Mechanism Audit

Frozen audit of audio counterfactual sensitivity and matched-audio object-level
spatial discrimination for original 1.3G final, 8.7 C3 Stage-1, and 8.7 C3
Stage-2.

The authoritative protocol is in `PROTOCOL.md`. Generated artifacts are kept
under `results/{vggss,flickr}` and `summary`; existing 1.3G and 8.7 outputs are
read-only inputs.

Run order:

```bash
PY=/home/wxr/miniconda3/envs/wwww/bin/python
$PY scripts/extract_stage2_maps.py --dataset vggss --model original_1.3g_final --gpu 1 --limit 8 --tag sanity
$PY scripts/extract_stage2_maps.py --dataset vggss --model cram_c3_stage2 --gpu 1 --limit 8 --tag sanity
$PY scripts/sanity_check.py
$PY scripts/run_full.py --gpu 1
```


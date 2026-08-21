# Experiment 5.0 - Agreement-Disagreement Pseudo-Label Purity Probe

Zero-training diagnostic of sparse pseudo-label purity from frozen Stage1
`AUD_L4` / `IMG_L4` agreement and disagreement. Stage2 maps are diagnostic only.

Run both formal 144k evaluations on two GPUs:

```bash
bash run_two_gpus.sh
python aggregate_results.py
```

Smoke test examples:

```bash
python probe.py --experiment vggss_144k --gpu 0 --max-batches 1 --skip-qualitative --output-root smoke_results
python probe.py --experiment flickr_144k --gpu 1 --max-batches 1 --skip-qualitative --output-root smoke_results
```

The probe creates no optimizer, calls no backward pass, adds no trainable
parameters, and verifies checkpoint SHA256 and mtime before/after evaluation.

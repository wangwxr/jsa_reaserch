# Experiment 3.2 - A4 Temporal Grounding Probe

Zero-training evaluation of temporal evidence inside the 16 frozen A4 audio tokens of the
formal original 1.3G checkpoints.

The experiment evaluates fixed two-chunk and four-chunk partitions, raw and evaluator-normalized
arithmetic/geometric consensus, temporal query semantics, map similarity, false-positive and
over-expansion stability, OGL-rescue capture, and deterministic qualitative cases.

Run both formal 144k evaluations on two GPUs:

```bash
bash chuagnxindian/3.2_a4_temporal_grounding_probe/run_two_gpus.sh
```

The probe creates no optimizer, calls no backward pass, freezes every loaded parameter, and
verifies checkpoint SHA256/mtime values before and after evaluation.


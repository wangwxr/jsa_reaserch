# Experiment 2.5 - Dual-Path Decision Probe

Pure inference experiment with two independent probes:

- Part A keeps pooled 7x7 L3 Slot updates unchanged and reads native 14x14 L3 keys with the final pooled semantic query.
- Part B fuses original 1.3G audio maps with 2.4 ownership maps at inference time only.

No optimizer, backward pass, trainable parameter, checkpoint edit, weight merge, or weight averaging is used.

Run all four settings on two GPUs:

```bash
bash chuagnxindian/2.5_dual_path_decision_probe/run_two_gpus.sh
```

Results are written under `results/`; the combined report is `REPORT.md`.


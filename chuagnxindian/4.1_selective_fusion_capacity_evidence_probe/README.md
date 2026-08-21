# Experiment 4.1 - Selective Fusion Capacity & Evidence Probe

Zero-training evaluation of sample, connected-region, and pixel fusion capacity,
plus fixed label-free selector evidence from the formal original 1.3G model.

```bash
bash chuagnxindian/4.1_selective_fusion_capacity_evidence_probe/run_two_gpus.sh
```

No optimizer, backward pass, parameter update, GT/OGL selector input, or learned
threshold is used.

Outputs:

- `REPORT.md`: complete cross-dataset result and decision.
- `results/<setting>/summary.json`: formal metrics and zero-training audit.
- `results/<setting>/per_sample_metrics.csv`: sample-level oracle/evidence data.
- `results/<setting>/qualitative/`: deterministic qualitative panels.
- `results/transfer_threshold_diagnostic.json`: non-official post-hoc transfer audit.

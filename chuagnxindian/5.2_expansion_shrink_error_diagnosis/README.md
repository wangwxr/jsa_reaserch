# Experiment 5.2 - Expansion-Shrink Error Diagnosis

Zero-training diagnosis of whether frozen JSA/MUFASA localization errors admit
distinct expansion, shrink, and keep regimes. The only learned objects created
are analysis-only scikit-learn linear probes evaluated out of fold; no model
parameter, prediction path, or checkpoint is changed.

```bash
bash run_two_gpus.sh
python aggregate_results.py
```

Formal oracle operations are fixed before evaluation:

- `EXPAND = max(AUD, PROP_F34)` or `max(AUD, PROP_K34)`.
- `SHRINK = binary(AUD >= .6) intersect binary(IMG >= .6)`.
- beneficial gain and strict-class dominance margin are both `0.01` IoU.

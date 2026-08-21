# Experiment 5.3 - AUD-Only Leakage Cue Probe

Candidate-independent intrinsic AUD error diagnosis and frozen pixel-level
probing of `TRUE_EXTENT` versus `CONTEXT_LEAKAGE` inside the official
AUD-only prediction region.

```bash
bash run_two_gpus.sh
```

The dataset jobs perform frozen extraction only. `aggregate_results.py` fits
analysis-only linear probes with fixed sample-hash folds and performs direct
VGG-to-Flickr and Flickr-to-VGG transfer without target normalization fitting.
No localization model or checkpoint is trained or modified.

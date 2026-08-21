# Experiment 5.1 - Agreement-Seeded Visual Propagation Probe

Zero-training propagation of the frozen Stage1 Top20 agreement seed through
the original 1.3G `F34` and `K34` visual spaces.

```bash
bash run_two_gpus.sh
python aggregate_results.py
```

The formal seed is always `P = top10(AUD_L4) intersect top10(IMG_L4)` at
`7x7`, nearest-resized to `14x14`. GT and OGL are analysis-only. No optimizer,
backward pass, trainable projection, or new model is created.

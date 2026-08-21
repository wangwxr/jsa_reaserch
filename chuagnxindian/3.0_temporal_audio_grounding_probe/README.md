# Experiment 3.0 - Temporal Audio Grounding Probe

Zero-training inference probe using the formal original 1.3G checkpoints.

- Primary: four contiguous temporal chunks.
- Diagnostic: two contiguous halves.
- Every chunk uses the unchanged frozen Audio Slot Branch and the same G `K34` readout.
- Temporal mean/geometric consensus is computed in raw attention probability space.
- OGL is evaluation-only and is never used to construct a temporal map.

Run both formal 144k settings:

```bash
bash chuagnxindian/3.0_temporal_audio_grounding_probe/run_two_gpus.sh
```

Results are written under `results/`; the detailed report is `REPORT.md`.


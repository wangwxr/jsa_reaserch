# Experiment 3.1 - Hierarchical Audio Representation

Stage1-only experiment derived from the formal L3+L4 Two-Level Stage1.

## Method boundary

- Visual L3/L4 branches, visual semantic fusion, evaluator, losses, and saver are unchanged.
- A3 and A4 use independent Audio Slot branches with the same shared initial slots.
- InfoNCE and audio diversity use the A4-anchored A3+A4 fused audio slots.
- Localization, attention alignment, and audio reconstruction remain A4-only.
- No G, Stage2, object ownership, temporal chunking, gate, or extra loss is used.

## Formal four-setting run

The script runs VGGSS on GPU0 and Flickr on GPU1. Each GPU runs 10k and then 144k sequentially, while the two dataset chains run in parallel.

```bash
bash chuagnxindian/3.1_hierarchical_audio_representation/run_four_two_gpus.sh
```

All checkpoints, logs, results, qualitative panels, and the final report remain inside this directory.

# Experiment 4.2 - Counterfactual Cross-Modal Reliability Probe

Zero-training keep/remove interventions over equal-area AUD and IMG candidate
regions, evaluated with the formal Stage1 InfoNCE semantic metric.

```bash
bash chuagnxindian/4.2_counterfactual_crossmodal_reliability_probe/run_two_gpus.sh
```

The experiment creates no optimizer, backward pass, trainable parameter,
learned threshold, or selector. GT/OGL are used only for post-hoc analysis.

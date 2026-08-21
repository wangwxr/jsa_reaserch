# Experiment 4.0 - AUD-IMG Redundancy Probe

Zero-training diagnostic for formal L3+L4 Stage1 and original 1.3G Stage2.

Run both formal 144k settings:

```bash
bash chuagnxindian/4.0_aud_img_redundancy_probe/run_two_gpus.sh
```

The probe uses `model.eval()` and `torch.inference_mode()`, creates no optimizer,
calls no backward pass, freezes every loaded parameter, and verifies checkpoint
SHA256/mtime before and after evaluation.


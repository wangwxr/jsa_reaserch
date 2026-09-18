# 8.7 Hard-negative-aware CRAM

This directory is isolated from Mean-CRAM. It will retain the C3 Stage-1 and
vanilla Stage-2 architecture/training recipe, changing only negative distance
aggregation after the recorded distance and gradient-scale audits pass.

## Vanilla Stage-2 handoff

Stage-2 is deliberately not auto-launched by Stage-1. After both Stage-1 runs
write `final.pth`, launch the unchanged refinement recipe on two GPUs:

```bash
/home/wxr/miniconda3/envs/wwww/bin/python chuagnxindian/8.7_counterfactual_relative_attention_matching/8.7_hard_negative_cram/stage2/run.py --dataset vggss --gpu 0
/home/wxr/miniconda3/envs/wwww/bin/python chuagnxindian/8.7_counterfactual_relative_attention_matching/8.7_hard_negative_cram/stage2/run.py --dataset flickr --gpu 1
```

The runner refuses to start until Stage-1 `final.pth` exists and writes only
under this Hard-CRAM directory. It passes the selected Hard-CRAM teacher to the
original 1.3G Stage-2 architecture, optimizer, data pipeline and losses.

To automatically hand off a currently running Stage-1 process, use
`stage2/auto_after_stage1.py` with that process's parent PID. The supervisor
waits for `final.pth`, aborts if the parent exits without it, then launches the
same `stage2/run.py` command and records its state beside the Stage-2 outputs.

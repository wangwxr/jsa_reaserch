# Experiment 6.1 — Spatial Attention Temporal Schedule Study

本目录仅改变 `L_att_space` 的 epoch 权重；模型、数据、其余损失、优化器和 IQR 选模协议均复用 6.0 Full。

- `early_high_late_low`：1–20 为 100，21–40 每轮下降 4.5，41–100 为 10。
- `early_low_late_high`：1–20 为 0，21–40 每轮上升 5，41–100 为 100。

从仓库根目录运行：

```bash
bash chuagnxindian/6.1_spatial_att_temporal_schedule/run_10k_two_gpus.sh
```

脚本先执行 schedule/loss 回归测试，再按两轮双卡并行运行四个 10k 实验；任一进程失败会阻止下一轮。

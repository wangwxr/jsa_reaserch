# 2.1R Fixed-Slot Reliability Recheck

完全零训练地重新计算 2.1 reliability。Rescue/Hurt 和所有 spatial feature 均改为
正式保留的 fixed slot0 candidate：`0.6*AUD_FINE + 0.4*SLOT_L4_slot0`。

同时输出两类 seed containment：

- `eval_seed_top10/top20`：按 evaluator resize 到 224、min-max 后计算；
- `raw_seed_top10/top20`：将真实 `[B,2,7,7]` ownership 成对 bilinear 到 14×14，
  不做 min-max，在原始 AUD_FINE top tokens 上读取 slot0 概率。

运行：

```bash
bash chuagnxindian/2.1R_fixed_slot_reliability/run_144k_two_gpus.sh
```

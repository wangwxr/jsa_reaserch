# Experiment 6.0 — Base Loss Causality Study

本目录只包装正式 `mufasa_ablation2_l3_l4_ablation`，不修改根目录 JSA、正式
L3+L4 或 1.3G。实验把旧接口中的三个聚合辅助损失拆成六项：

```text
L = L_info
  + λ_rec_img L_rec_img + λ_rec_aud L_rec_aud
  + λ_div_img L_div_img + λ_div_aud L_div_aud
  + λ_att_space L_att_space + λ_att_time L_att_time
```

Full 模式使用旧训练器完全相同的分组加法顺序，以便进行 FP32 数值回归。Stage A
只计算 forward/gradient 与 decoder-alpha 诊断，不创建 optimizer、不 step、不保存模型；
Stage A 失败时调度器禁止进入 Stage B。

Stage B 只训练 `NO_SPATIAL_ATT`、`NO_IMAGE_REC`、`NO_BOTH` 的 VGGSS/Flickr 10k，
同配置双卡并行，三个配置依次运行。best checkpoint 仍按 IQR cIoU。

运行：

```bash
bash chuagnxindian/6.0_loss_causality_study/run_10k_two_gpus.sh
```

`SHRINK_BASE` 与 `EXPAND_BASE` 仅是基础 AUD/IMG 分支交互诊断，不是 Experiment 5.2
的 PROP_F34/K34 oracle，不能混用数值。

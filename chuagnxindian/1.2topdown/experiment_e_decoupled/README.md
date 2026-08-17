# Experiment E：Decoupled Fine Spatial Learning

本目录是 Experiment D 的严格单变量消融。D 的模型、联合训练状态、base loss、优化器、学习率、epoch、batch、数据与 checkpoint 选择均保持不变；唯一变化是完全删除 14×14 `AUD_FINE -> IMG_FINE.detach()` 像素级 imitation。

## 唯一算法变化

Experiment D：

```text
L_refine_D = MSE(AUD_FINE, detach(IMG_FINE))
           + MSE(SumPool(AUD_FINE), detach(AUD_L4))
           + MSE(SumPool(IMG_FINE), detach(IMG_L4))
```

Experiment E：

```text
loss_fine_match = 0（常量，没有计算图）

L_fine_E = MSE(SumPool(AUD_FINE), detach(AUD_L4))
         + MSE(SumPool(IMG_FINE), detach(IMG_L4))

L_total = L_base + 1.0 * L_fine_E
```

两个 Fine branch 只向各自 coarse map 对齐，不交叉模仿。L4-level 原始 `att_loss` 完全保留。

## 代码复用

`model_e.py` 直接继承 Experiment D 的 `JointL3L4TopDownModel`，只覆盖 `refinement_losses()`。因此以下路径与 D 是同一份实现：

- L3+L4 backbone、Slot Attention、M-Fusion；
- `F34=Up(F4)+Adapter(F3)` 与 zero-init；
- `Qa/Q4/K4/K34`；
- `AUD_FINE/IMG_FINE`；
- sum-pooling、coarse teacher detach；
- 原始四项 base loss；
- 参数冻结模板、AdamW、AMP、DataLoader、checkpoint。

`train_e.py` 以只读方式加载 D 的训练工具，新增 E 的 evaluator 诊断和独立输出目录，不修改 D 文件。

## 新诊断

每轮和 best checkpoint 额外保存：

```text
aud_img_map_cosine
    = mean_batch cosine(flatten(AUD_FINE), flatten(IMG_FINE))

fusion_gain_ciou
    = IQR_FINE cIoU - AUD_FINE cIoU
```

它们写入 `epoch_metrics.csv`、checkpoint metrics、`best_test_metrics.json` 和训练曲线，并打印到控制台。

## 不变设置

- 初始化：原 L3+L4 best checkpoint，新 adapter 仍为 zero-init；
- trainable base：36,417,282；refinement head：1,311,488；总 trainable：37,728,770；
- AdamW，lr `5e-5`，weight decay `0.01`，无 scheduler；
- batch 256，100 epochs，seed 12345；
- `lam1=0.1, lam2=0.1, lam3=100, lambda_f=1`；
- best 仍按 `AUD_FINE cIoU`；
- 不使用 OGL、OBJ_PRIOR 或 GT localization supervision；
- 第一轮仅支持 VGGSS-10k、Flickr-10k，没有144k launcher。

## 运行

从 repo root 执行两卡 sanity-only（无 optimizer step）：

```bash
bash chuagnxindian/1.2topdown/experiment_e_decoupled/sanity_check_10k.sh
```

两卡同时启动两个10k：

```bash
bash chuagnxindian/1.2topdown/experiment_e_decoupled/run_10k_two_gpus.sh
```

分别运行：

```bash
bash chuagnxindian/1.2topdown/experiment_e_decoupled/train_vggss_10k.sh 0
bash chuagnxindian/1.2topdown/experiment_e_decoupled/train_flickr_10k.sh 1
```

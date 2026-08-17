# Experiment C：L3+L4 Two-Level Ablation

## 结构

```text
L3 -> 256→512 projection -> AdaptiveAvgPool(7,7) -> Visual SA3 -> S3
L4 -> 512→512 projection -> 7x7                    -> Visual SA4 -> S4
P34 = (S3 + S4) / 2
P34 -> Linear(512,1024) -> GELU -> Linear(1024,512) -> Sf
```

不注册 L2 projection，也不创建 L2 visual SA。Visual SA3/SA4 参数独立，使用同一组
learnable initial slots；audio encoder 和 single audio SA 不变且只运行一次。

Sf 用于 InfoNCE、image reconstruction 和 visual divergence；reconstruction target 仍是
L4。`att_loss`、`AUD`、`IMG_QUERY` 只使用 L4 attention，不使用 L3 spatial attention。

## 二层/三层 M-Fusion 泛化

v1.1 的 M-Fusion 写死为三个 levels。本目录的局部实现只做如下最小泛化：对相邻 levels
分别求平均，拼接全部 adjacent-pair representation，再使用

```text
Linear((N-1)*D, N*D) -> GELU -> Linear(N*D,D)
```

其中 `N` 只能为 2 或 3。`N=3` 时参数 shape、公式和 v1.1 完全相同；`N=2` 时只有
`P34`，对应 `512→1024→512` 的 learned projection。没有 direct concat、residual 或其他
fusion。

## 命令

```bash
bash chuagnxindian/mufasa_ablation2_l3_l4_ablation/train_vggss.sh \
10k 0 mufasa_ablation2_l3_l4_ablation_vggss_10k

bash chuagnxindian/mufasa_ablation2_l3_l4_ablation/train_flickr.sh \
10k 1 mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5
```

# 8.7 Stage-specific Audit — 结论

本次支持把 **C3 Stage-1 CRAM + vanilla Stage-2** 作为主方法，支持把当前
Stage-2 追加 CRAM 的负结果作为有信息量的设计消融。它不能证明所有可能的
Stage-2 CRAM 都无效，也不能唯一归因于“阶段”本身。

已确认两个独立问题：

1. 在有限的共享 refinement 参数上，CRAM 与原 Stage-2 objective 很快出现强烈
   梯度冲突；这个结论在 FP32 中成立，且局部参数干预直接提高了 base loss。
2. 旧 `grad_norm=inf` 的停止机制存在 AMP loss scaling 溢出解释。相同晚期
   checkpoint 在 FP32 和较低 scale 下有限；性能退化则早于溢出，不能归为
   纯数值问题。此前把 inf 直接称为“真实梯度爆炸”、把所有失败唯一归为权重过强，
   证据不足，现予以修正。

## 范围和验证

- 每数据集 64 batch、batch size 256、seed 12345、lambda=25,000；原始 C3
  initialization、144k NPY、AdamW 5e-5/.01、两视图损失。不是重新跑 50 epoch。
- 逐 batch 分别测 base、weighted CRAM、total 梯度；AMP 先乘 lambda 再求导，
  测后除 scale。没有用已经下溢成零的 raw CRAM gradient 估算夹角。
- step 0/16/63 用同 batch FP32 复核。梯度加和残差、零梯度与冻结分组均落盘。
- 旧 25k epoch1、VGG epoch39/Flickr epoch46 各四个固定训练 batch 做只读 FP32。
- 完整测试集 5,158 VGG-SS / 250 Flickr；固定既有 GT/NearFP indices、same-image
  eight-wrong-audio mapping、原插值/归一化/threshold=.6。保留所有样本结果。
- wrapper 与原 Stage-2 在相同状态和输入上的 shared output 差异为 0；三个
  原机制模型 per-image IoU 回归通过。没有修改旧实验。
- 旧 50k checkpoint 是 VGG epoch1/32、Flickr epoch2/39；旧25k 是两者 epoch1
  加 VGG39/Flickr46。不存在独立 epoch5/10/20，未冒充这些轨迹点。

## TASK 1：冲突在什么时候、什么地方出现

下面是全部可训练 student 的 `cos(g_base, lambda*g_cram)`。旧状态为四个
固定训练 batch 的 FP32 结果，少量 batch 统计不能当作跨 seed 泛化置信区间。

| 数据集 / 状态 | cosine mean | median | P25 | P75 | cos<0 | cos<−.25 | cos<−.5 | weighted/base norm 比均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VGG：短程64 batch | −.2664 | −.6093 | −.7335 | .4509 | 67.19% | 65.63% | 59.38% | .7243 |
| Flickr：短程64 batch | .0138 | .0154 | −.0973 | .1409 | 43.75% | 9.38% | 0% | .2461 |
| VGG：旧 epoch1 | −.8256 | −.8237 | −.8807 | −.7686 | 100% | 100% | 100% | 1.0264 |
| Flickr：旧 epoch1 | −.9491 | −.9575 | −.9640 | −.9426 | 100% | 100% | 100% | 1.2430 |
| VGG：旧 epoch39 | −.9885 | −.9901 | −.9910 | −.9876 | 100% | 100% | 100% | .9530 |
| Flickr：旧 epoch46 | −.9774 | −.9792 | −.9823 | −.9743 | 100% | 100% | 100% | 1.1038 |

不是第一步就普遍强烈反向。VGG 前16步 mean cosine +.6299，后32步 −.7092；
Flickr 对应 +.1850 / −.1043。旧 epoch1 已有两数据集共同的强冲突。
初始 .35/.14 的梯度比并不保持：epoch1 已约 1.03/1.24，说明单个初始化 batch
不能保证整个训练过程中权重安全。

| 旧状态 | ||base|| | ||weighted CRAM|| | ||total|| |
|---|---:|---:|---:|
| VGG epoch1 | .000585 | .000601 | .000345 |
| VGG epoch39 | .015378 | .014631 | .002997 |
| Flickr epoch1 | .000817 | .000939 | .000370 |
| Flickr epoch46 | .007912 | .008577 | .002106 |

total 梯度比两项单独更小，是近反向相互抵消的结果；仅监控 total norm
可能漏掉这种目标冲突。全部组的 norm、mean/median/P25/P75 和负夹角比例见
`gradient_group_summary.csv`。

参数空间与激活空间要区分：

| 分组 | VGG epoch1 | Flickr epoch1 | VGG late | Flickr late |
|---|---:|---:|---:|---:|
| L3 projection 参数 | −.8265 | −.9427 | −.9884 | −.9855 |
| adapter 参数 | −.8276 | −.9535 | −.9892 | −.9736 |
| adapter 最后卷积 | −.7555 | −.9636 | −.9880 | −.9641 |
| K34_A 激活 | +.0752 | +.0899 | +.0140 | +.0528 |
| F34_A 激活 | +.0924 | +.1011 | +.0715 | +.0218 |

因此不支持“原始 K34 激活梯度强烈反向”，支持“梯度经共享空间网络传回参数后
强烈冲突”。base 还有 view B 的贡献；分开后 late view A/B 的参数 cosine
在 VGG 为 −.9765/−.9820、Flickr 为 −.9615/−.8999，说明不是简单的视图相加假象。
音频分支与 Teacher projection 是冻结的，统计为 N/A；AUD_FINE 是 CRAM 的
兄弟读出节点，直接对此 tensor 求 CRAM 导数不是正确的共同路径诊断。

进一步分解发现：late CRAM 与 coarse/equiv 梯度分别为 VGG −.9638/−.9518、
Flickr −.9465/−.8349。CRAM 与 `−lambda/2 * D_neg` 梯度 cosine 达到
.9968/.9842；其与正项 `lambda/2 * D_pos` 为 −.9486/−.8395。
在该小 gap 区间 softplus 导数约 .5，negative repulsion 方向主导了更新。

对相同晚期 checkpoint 做 L2=0.001 的 FP32 可恢复参数干预：

- VGG：沿 −g_CRAM，base loss +1.489e−5；沿 −g_base，−1.527e−5。
- Flickr：沿 −g_CRAM，base loss +8.284e−6；沿 −g_base，−8.386e−6。

这直接支持局部优化目标冲突；没有用 GT 选择干预方向，且参数随后恢复。

## 数值溢出与目标冲突是两回事

同一旧25k晚期 checkpoint / batch：AMP scale=2^16 或 2^24，total norm 均有限，
VGG 约 .002186、Flickr 约 .001357。scale=2^28 时 VGG total 非有限；
Flickr 独立 base/CRAM 已非有限而 total 仍可有限，scale=2^32 时 total 非有限。
FP32 独立两项有限，且保留强冲突。

旧 trainer 的 `GradScaler` 默认会增长 scale；一旦 unscale 后遇到 inf，
保护检查在 `scaler.step/update` 前直接 raise，不给 scaler 降低 scale 并跳步的
机会。历史没有保存 scaler/optimizer state 或失败 batch，故不能精确复现
历史失败时刻；本次验证了溢出易感性，不能宣称唯一确定的 inf 根因。

日志中 weighted CRAM 约 17,328/34,657 主要是 lambda*ln(2) 常数，
常数大小本身不能证明压制 base objective。应看实测导数、夹角及原损失变化。
训练没有梯度累积配置，不能把结果解释成 gradient accumulation bug。

## TASK 2：matched-audio 退化路径

下列全测试集结果来自统一冻结评测。`late` 为旧25k VGG39/Flickr46，
不是完整50轮 final；短程 step64 与旧run 是不同固定 seed 轨迹。

| 数据集 / 状态 | GT response | NearFP | GT−Near gap | GT/Near AUROC | coverage | precision | area | cIoU | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VGG 1.3G | .7979 | .7862 | .0150 | .5920 | .8677 | .4964 | .5078 | .4269 | .4230 |
| VGG 8.7 S1 | .7309 | .7034 | .0304 | .5901 | .7577 | .5442 | .3914 | .3912 | .4132 |
| VGG vanilla S2 | .7631 | .7203 | .0455 | .6256 | .8110 | .5376 | .4303 | .4343 | .4296 |
| VGG init | .7306 | .7023 | .0313 | .5936 | .7520 | .5456 | .3867 | .3881 | .4126 |
| VGG short64 | .7508 | .7177 | .0359 | .6066 | .7862 | .5309 | .4194 | .3990 | .4163 |
| VGG 25k epoch1 | .7377 | .7036 | .0366 | .5979 | .7764 | .5276 | .4194 | .3790 | .4081 |
| VGG 25k late | .4310 | .4190 | .0129 | .5809 | .2134 | .2241 | .1501 | .0727 | .1271 |
| Flickr 1.3G | .8060 | .7640 | .0427 | .6600 | .9114 | .6449 | .6475 | .8120 | .6356 |
| Flickr 8.7 S1 | .7132 | .6254 | .0889 | .6732 | .7655 | .7492 | .4631 | .8640 | .6184 |
| Flickr vanilla S2 | .7563 | .6590 | .0981 | .7078 | .8369 | .7256 | .5275 | .8960 | .6532 |
| Flickr init | .7213 | .6292 | .0933 | .6782 | .7732 | .7479 | .4687 | .8720 | .6236 |
| Flickr short64 | .7306 | .6383 | .0934 | .6844 | .7902 | .7393 | .4855 | .8720 | .6296 |
| Flickr 25k epoch1 | .6706 | .5544 | .1179 | .7120 | .6836 | .8065 | .3877 | .7560 | .5826 |
| Flickr 25k late | .5604 | .4291 | .1329 | .7150 | .4709 | .8411 | .2626 | .3880 | .4216 |

gap 是有 GT 和 NearFP 的有效样本平均，两个 response 各有其有效样本集合，
所以表中两个宏均值之差不必恰等于 gap。Flickr 旧日志的 late AUC=.4214、
50k best=.3442，统一缓存 query replay 为 .4216/.3440，涉及极少阈值边界差异；
本表使用统一 replay，保留旧日志数值，不将差异解释成性能变化。

**VGG：Case A+C，伴随明显面积收缩。** 相对 vanilla，GT response −.3320、
NearFP −.3013；gap −.0326，配对95%CI [−.0350,−.0302]；AUROC −.0447
[−.0491,−.0404]。coverage −.5976，precision −.3135。既过度抑制目标，也
损坏了目标/近背景区分，不是更干净但更小的正确预测。

**Flickr：主要 Case B，过度收缩损伤覆盖。** gap +.0348，95%CI [.0248,.0447]；
AUROC 点估计 +.0072，但区间 [−.0058,.0213] 跨零。coverage −.3661
[−.3959,−.3353]、precision +.1155。更高 gap 不足以保证完整目标定位。

相对同一个原始1.3G基准的像素变更均值：

| 数据集 / 模型 | RemoveFP | AddTP | RemoveTP | AddFP |
|---|---:|---:|---:|---:|
| VGG vanilla S2 | 3887.7 | 441.6 | 1497.0 | 1053.7 |
| VGG 25k epoch1 | 4458.8 | 523.3 | 2156.9 | 1655.8 |
| VGG 25k late | 10521.3 | 307.9 | 9858.2 | 2122.2 |
| Flickr vanilla S2 | 4183.7 | 367.0 | 2734.8 | 529.0 |
| Flickr 25k epoch1 | 6491.3 | 134.7 | 6810.7 | 129.6 |
| Flickr 25k late | 7997.7 | 220.3 | 11890.1 | 351.2 |

RemoveFP 大量增加时 RemoveTP 也大幅增加。不能只报背景抑制。
25k late normalized entropy 仍为 VGG .9981、Flickr .9974，接近最大熵；
不支持“概率图熵趋近零”的极端塌缩。阈值面积却明显减小，说明概率熵和
经过 min-max 的 activated area 是不同性质的量。早期64步还可有小幅正向变化，
不能说所有第一步就失败；也不能说只有最后出现 inf 才退化。

## OGL 和其他 readout（测试用，不参与诊断训练）

原始 readout 定义：先将 AUD、IMG_QUERY、OBJ_PRIOR 各自 bicubic resize 到224
并 min-max，再融合再 min-max；OGL 固定 .6 AUD + .4 ImageNet ResNet18 prior。

| 数据集 / 模型 | AUD cIoU/AUC | OGL cIoU/AUC |
|---|---|---|
| VGG 1.3G | .4269/.4230 | .4570/.4401 |
| VGG 8.7 S1 | .3912/.4132 | .4242/.4277 |
| VGG vanilla S2 | .4343/.4296 | .4523/.4401 |
| VGG 25k best(epoch1) | .3790/.4081 | .4329/.4312 |
| VGG 25k late(epoch39) | .0727/.1271 | .2819/.3622 |
| VGG 50k best(epoch1) | .2076/.3009 | .3404/.3900 |
| VGG 50k late(epoch32) | .0520/.0950 | .1929/.3093 |
| Flickr 1.3G | .8120/.6356 | .8680/.6592 |
| Flickr 8.7 S1 | .8640/.6184 | .8480/.6160 |
| Flickr vanilla S2 | .8960/.6532 | .8880/.6430 |
| Flickr 25k best(epoch1) | .7560/.5826 | .7800/.5972 |
| Flickr 25k late(epoch46) | .3880/.4216 | .5720/.5184 |
| Flickr 50k best(epoch2) | .2480/.3440 | .3880/.4424 |
| Flickr 50k late(epoch39) | .1120/.2964 | .3960/.4472 |

IQR、IMG_QUERY、OBJ_PRIOR、EXTRA_IQR_OGL 的逐模型/逐样本结果完整保存。
OGL 不能挽救追加 CRAM 的晚期退化。有效的 vanilla8.7 主结果是 AUD，两数据集
四个 AUD 指标点估计都提高；不能把它扩大为“所有 readout 都提高”，尤其 Flickr
固定 OGL AUC 对1.3G为 −.0162，配对95%CI [−.0258,−.0066]。

## TASK 3：为什么 S1 有益而当前 S2 移植有害

详细 tensor/shape/computation graph 见 `../representation_location/COMPUTATION_GRAPH.md`。
已观察到的链条是：

```text
S1: trainable audio+visual paths + full positive matching
    + relative wrong-audio constraint
    -> reshaped coarse representation / teacher
    -> vanilla S2 retains coarse structure + geometric refinement
    -> stronger final target/NearFP discrimination and AUD localization

Current S2 transplant: frozen queries + trainable fine K34 only
    + moving detached visual target + strong relative repulsion
    -> increasing conflict with coarse/equivariance in shared student parameters
    -> Flickr: better contrast but lost target coverage
       VGG: lost coverage AND discrimination
    -> poor localization well before AMP overflow
```

证据支持 stage-specific design 的解释。不能直接证明 Stage-1 内部已学到
完整 object-level ownership：例如 VGG S1 GT/Near AUROC .5901，未超过 baseline
.5920，最终提升在 vanilla S2 后更明确。Stage-1 的作用宜表述为在完整空间正锚定
下塑造有利于后续 refinement 的表示，而非直接宣称每一步语义排序都已改善。

## TASK 4：只保留两个后续方向

1. **优先级最高：固定 CRAM 只在 S1，继续打磨与验证现有 vanilla S2 主路线。**
   当前所有可训练层均存在追加 CRAM 的晚期冲突，没有证据支持挑某个现有层继续塞
   CRAM。先做配对多 seed、完整消融和论文整理，收益在于确认可复现方法，而不是
   再耗费预算猜 lambda。
2. **仅作为候选：若继续追求提点，优先研究覆盖保持条件下的选择性 CRAM。**
   Flickr 提供了 contrast 提升而 coverage 丢失的明确动机；但 VGG 同时损失区分度，
   因此不能假定 gating 一定有效。需要先有训练侧可定义、无 GT 的置信/覆盖依据，
   并在固定训练 batch 验证它是否缓解参数冲突，再决定是否训练。本次未实现。

不推荐把它简单移到完全冻结的 coarse 输入；那样没有可训练梯度。解冻则引入
新的实验变量。也不建议继续单纯搜索12.5k/6.25k/100k。

## TASK 5：论文成熟度

1. **核心问题得到实证改善，但没有被彻底解决。** 最终8.7的 GT/NearFP gap、
   AUROC 和 AUD localization 改善；最终 heatmap 仍高度 audio-invariant，不应声称
   已解决普遍 audio-conditioned target ranking 或完整 sounding-object ownership。
2. **diagnosis → method → mechanism → localization gain 的主体证据链已形成。**
   机制图表、固定区域量化、两阶段输出和本次负例可以共同支撑。具体内部因果路径
   仍是受证据支持的解释，不是排除所有 representation/optimization 替代解释的证明。
3. **Stage-2 负结果适合作为 design ablation，而不是已成功主方法的缺陷。**
   要写成“所测试的 fine-K34 transplant 失败”，并说明冻结、temperature、scale
   和正锚定差异，不能宣称所有 Stage-2 relative supervision 都有害。
4. **可以现在开始正式写主方法论文，但实验尚非投稿定稿状态。** 当前 C3 只发现
   seed12345。原8.7-S2对1.3G的 AUD 配对95%CI：VGG cIoU +.00737
   [−.00175,.01687]、AUC +.00665 [.00429,.00915]；Flickr cIoU +.084
   [.044,.124]、AUC +.0176 [.00580,.02861]。这是固定checkpoint的样本 bootstrap，
   不含训练seed方差或选模偏差。

必须补：

- 至少第二个配对 seed（条件允许三个），baseline与S1-CRAM+vanilla-S2使用相同
  预算和选模口径；四个AUD指标报告均值/方差，不能只挑最好seed。
- 整理/补齐同初始化、同数据预算的原Stage1→vanillaS2 与 CRAM Stage1→同S2
  消融和已有强度对照。现有可复用checkpoint先审计，不重复训练已等价的实验。
- 明确报告目前每epoch在 benchmark test 上选best的做法及限制，最终确认实验
  预先固定checkpoint/选择规则，避免继续基于测试集选lambda或设计；不能悄悄
  改数据划分来美化结果。补训练时间、参数量、K=4额外计算成本。
- 主方法的claim限定到AUD；如要宣称兼容OGL或所有readout都提升，当前证据不支持，
  需要额外验证。OGL现有负结果应照实列出，不凭测试集调alpha。

锦上添花：更多可视化、更多数据集、额外negative策略、confidence-gated候选，
以及从稳定数值配置进一步验证Stage-2负例；这些不是开始写论文的前置条件。

## 限制与记录

- 只进行了64步新训练；旧epoch1/late梯度各四个固定训练batch。足以复现局部
  冲突和历史退化，不代表完整训练分布、多seed或所有层配置的普适定理。
- 本次小步干预验证base objective局部上升，不是对测试cIoU做可微因果干预。
- entropy与activated area不能混同；NearFP来自固定诊断区域，不是完整多物体GT。
- 历史optimizer/scaler/RNG没有保存，无法精确恢复失败batch；没有生成假的final。
- 所有代码、短程student、逐batch梯度、逐sample响应、映射与hash均在本新目录。
  原1.3G、8.7-S1、vanilla-S2、旧mechanism_audit和25k/50k结果未改动。

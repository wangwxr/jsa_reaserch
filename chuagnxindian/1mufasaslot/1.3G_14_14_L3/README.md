# 1.3G_14_14_L3

本目录是原 `1.3G-multigeom_equivariant_l3_refine` 的独立复制实验。geometry、multi-view augmentation、inverse warp、VALID_MASK、KL、coarse anchor、optimizer、lr、epoch、batch、best checkpoint criterion 均保持正式原实现。

唯一实验变量是 frozen semantic teacher 来源：

```text
1.1.1_14_14_L3 best checkpoint -> fully frozen teacher
```

Student 数据流保持：

```text
native layer3 [B,256,14,14] -> trainable copied proj3_spatial -> F3 [B,512,14,14]
F3 -> adapter -> delta_F3
frozen F4 [B,512,7,7] -> bilinear upsample -> F4_up [B,512,14,14]
F34 = F4_up + delta_F3 -> [B,512,14,14]
fine_tokens -> frozen L4 norm/key head -> [B,196,512]
frozen Qa [B,2,512] x fine_keys -> AUD_FINE [B,1,14,14]
```

Teacher 全参数冻结且始终 `eval()`；optimizer 仅包含 `student.proj3_spatial` 与 `student.adapter`。训练启动前自动保存 teacher checkpoint 来源、参数名/数量、shape、几何、梯度、初始化 copy 与完整 test split coarse reproduction audit。

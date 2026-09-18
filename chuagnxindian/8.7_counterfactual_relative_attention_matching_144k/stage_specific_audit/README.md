# 8.7 Stage-specific Audit

已完成：两数据集各 64 个 batch 的 25k 短程梯度诊断；old25 epoch1/late
各四个固定训练 batch 的 FP32 审计；loss-scale 扫描；原 wrapper 行为回归；
可恢复的局部梯度方向检查；完整 5,158 VGG-SS / 250 Flickr 样本的定位、
audio-swap 与六 readout（含 OGL）评测。没有新开完整训练。

主要入口：

- [综合报告](summary/stage_specific_summary.md)
- [全部实测表与置信区间](summary/measured_data.md)
- [计算图](representation_location/COMPUTATION_GRAPH.md)
- [梯度趋势](summary/gradient_conflict_trends.png)
- [定位退化图](summary/degradation_path.png)
- [机器可读结果](summary/stage_specific_summary.json)
- [六 readout 性能](summary/six_readout_performance.csv)

每样本数据保存在 `degradation_path/{dataset}/`，每 batch 梯度和负音频 IDs
保存在 `gradient_conflict/{dataset}/`。源 checkpoint 路径、epoch、SHA256
与回归检查保存在对应 manifest / validation / source JSON。

复现：在仓库根目录用 `/home/wxr/miniconda3/envs/wwww/bin/python` 运行：

```bash
python stage_specific_audit/gradient_conflict/run.py --dataset vggss --gpu 0
python stage_specific_audit/gradient_conflict/component_audit.py --dataset vggss --gpu 0
python stage_specific_audit/degradation_path/evaluate.py --dataset vggss --gpu 0
python stage_specific_audit/summary/summarize.py
```

上述脚本前缀需展开为本目录完整路径；Flickr 替换 dataset / GPU 参数。
训练诊断和提取脚本拒绝覆盖已有目录。`--analyze-only` 仅重新分析本 audit 的
已提取 maps。`*_preflight_failed` 保留首次 GradScaler 初始化接口检查失败记录，
该尝试在任何 optimizer update 前停止，未用于结论。

历史 25k/50k best checkpoint 只是提前终止 run 的已保存点，不是完整预算 final。
这些点依然可以做冻结诊断与 OGL 评测；此前对话说“不能评测”不准确。

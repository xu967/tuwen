# BARV 实现说明（独立于原 YOLOVO 训练路径）

本目录是对两份 BARV 方案文档的**可执行最小实现**，不改动 `ultralytics/` 下现有
YOLOVO/关系头代码。原关系头保留为参考基线；由于压缩包缺少 MobileCLIP 与 YOLOVO
权重，主实验使用冻结 ImageNet ResNet-18 作为公开骨干（方案 v2 允许的同规模冻结对照）。

## 任务

给定图像、主体/客体 panoptic mask 和谓词，输出二元关系概率；在预算内可选择一次
局部高分辨率 ROI（本版主配置 `max_steps=1`），并报告真实执行的 fine-encoder 次数。

## 标签协议（重要）

- **positive**：PSG 官方 relation triple（已核验正例）。
- **verified negative**：仅对“物理接触是必要条件”的谓词，用同图实体对 mask 间距
  超过训练集正例 **max gap × 1.25 + 6px** 来构造；这是**几何可证负例**，不是人工核验。
- **unknown**：接触/间距不足但未标注的实体对；不进入 BCE。
- 每谓词负例/正例比例上限 4:1；负例过少的谓词（<50）整体丢弃（全量数据上没有触发）。
- 训练集按图像分为 train_core（关系头）/ train_teacher（fine updater 70% /
  value-teacher 30%）；dev 早停；calib 只做温度/阈值/abstain；test 与官方 val2017
  的 1,000 张 psgval 只做最终评估。图像级 split 无重叠。

## 视觉骨干

`model.backbone` 可选 `resnet18`（默认）或 `mobileclip_s0`。MobileCLIP 使用
`ultralytics/nn/textencoder/mobileclip/mobileclip_s0.pt` 的 image tower（丢弃 text
tower），粗特征为 stride-16/32 特征图，fine 为 512 维全局 CLIP 嵌入。所有入口都把
letterbox 后的 0..255 张量交给 `backbone.preprocess()`：ResNet 做 `/255` + ImageNet
归一化，MobileCLIP 只做 `/255`（其官方 transform 无 mean/std）。

## 模块

| 文件 | 职责 |
|---|---|
| `common.py` | letterbox、label grid、设备/种子 |
| `psg.py` | PSG schema、panoptic 解码、mask 几何、正/负/unknown 查询构造、六种动作 |
| `prepare.py` | 图像级 split、阈值校准、冻结骨干特征缓存、负例平衡 |
| `model.py` | ResNet-18 粗/细编码、mask pooling、关系头、fine gate、MVE 价值头 |
| `train.py` | 关系头训练、fine updater、离线 one-step teacher、价值头、λ/停止阈值 |
| `engine.py` | 惰性 ROI 执行、固定/随机/熵/MVE 基线、成本与延迟统计、校准工具 |
| `experiment.py` | 全量/40% 运行编排、指标与 bootstrap、对比 |
| `metrics.py` | AUROC/AUPRC/校准/风险覆盖/image-clustered bootstrap |

## 关键防泄漏

- 粗模型几何特征**不含** mask gap / overlap / contact 标志（它们正是负例规则）。
- 价值/策略动作特征只含动作类型 one-hot、seen、cost；不含 crop 坐标/IoU/接触距离。
  执行器仍使用坐标裁剪，但学习到的策略看不到这些几何量。
- fine updater 只在 train_teacher 的 70% 图像上训练；teacher gain/value 只用其余 30%。
- 全部 fine encoder 调用都在策略选中 crop 后发生；coarse/full 特征缓存不计入“按需”调用。
- `full` 基线沿用粗尺度训练的关系头，是**分布外参考**，不是严格同预算上界。

## 命令

```bash
python -m barv.prepare --config configs/barv_psg.yaml
python scripts/audit_data.py --config configs/barv_psg.yaml
python scripts/benchmark_hw.py
python scripts/run_barv.py --config configs/barv_psg.yaml --fracs 1.0,0.4 --seeds 101,202,303
python scripts/make_report.py --runs runs/barv/main
python -m pytest tests/test_barv.py -q
```

## 已知限制

- 负例是规则可证而非人工标注；对 PSG 标注不完整/panoptic 边界噪声仍可能含少量假负例。
- 仅 8 个谓词进入最终二元任务（standing on 在全量数据上有负例，但阈值很保守）。
- 未做官方 PSG PredCls、未做预测 mask（P2）、未训练开放词汇文本塔。
- 未单独训练 full-res 关系头；full 只作诊断。
- 关系头只跑一个 seed/子集；论文级结论需要 3 个训练种子。

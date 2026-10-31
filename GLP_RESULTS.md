# GLP 实验结果（RA-4M smoke，2026-11-01）

- 机器：RTX 2080 Ti 22GB，torch 2.5.1+cu124
- 统一超参：`imgsz=640, batch=16, workers=8, lr0=1e-3, epochs=5`
  （batch/workers 为用户指定；未改其他训练超参）
- 数据：`RA-4M/glp/`（600 train / 41 val；从 RA-4M 图像取 20 类 DIOR 对齐子集；
  RelGen 生成 4 个谓词：beside / on / to the left of / to the right of；1591/90 条关系）
- 主模型：DIOR 20 类 `best.pt`，冻结；关系头 `YOLOVORelate` 约 26.6M 可训练
- checkpoint：`runs/test/glp_v1_ra4m_e5/weights/last.pt`、`runs/test/glp_v2_ra4m_fixed_e5/weights/last.pt`

复现命令（仓库根目录）：

```bash
export LD_LIBRARY_PATH=/root/rivermind-data/cudnn9/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export YOLOVO_RELATE_DATA=RA-4M/data_glp.yaml
export YOLOVO_PRIMARY=/root/rivermind-fs/.Trash-0/files/code/best.pt
export YOLOVO_RELATE_INPUTS=P4
YOLOVO_FIELD=glp YOLOVO_EPOCHS=5 YOLOVO_NAME=glp_v1_ra4m_e5 python train.py
YOLOVO_FIELD=v2  YOLOVO_EPOCHS=5 YOLOVO_NAME=glp_v2_ra4m_fixed_e5 python train.py
```

## 1. 公开指标：SGDet（41 张 val）

| 指标 | V1 5ep | V2 5ep（修复后） |
|---|---:|---:|
| R@20 / R@50 / R@100 | 0 / 0 / 0 | 0 / 0 / 0 |
| mR@50 / mR@100 | 0 / 0 | 0 / 0 |
| F1@50 | 0 | 0 |
| predicate_acc | 0 | 0 |
| PairRecall | 0 | 0 |
| DetPairRecall | 0.1220 | 0.1220 |
| GT_MRR | 0.0099 | 0.0099 |

说明：SGDet 用主模型 NMS 框。主模型是 DIOR 域，RA-4M 域差使 `DetPairRecall=0.122`；
`F1@50=0` 主要反映检测/域差，不能归因于关系头，也不是公开 SGG 结论。

## 2. GT 框 / SGCls 口径（20 张 val，`field_metrics`）

| 指标 | V1 5ep | V2 5ep（修复后） | 说明 |
|---|---:|---:|---|
| top1 | 0.5111 | 0.3556 | 4 类，随机 0.25 |
| top5 | 1.000 | 1.000 | V=4，top5 无意义 |
| macro-F1 | 0.1691 | 0.2032 | 4 类 |
| existence P/R/F1 | .608/1.000/.756 | .608/1.000/.756 | 阈值 0.5 |
| FMA | -0.8485 | -0.9315 | `E[2·mass_in_obj−1]`，越高越好 |
| WBG gap | n/a | 6.69e-5 | 正=GT 背景质量更低；接近 0 |
| AURC | n/a | 0.6832 | 越低越好 |
| EE（训练后） | n/a | 6.73e-8 | 闭式反变换误差，约 0 |
| SFC（单测） | n/a | <1e-4 | 主体平移/缩放后局部核不变 |

### 2.1 SGCls 公开指标（5 张 val sanity；不是统计结果）

`rel.val(protocol='sgcls')` 已按 GT 框重打分；完整 41 图的两模型评测过慢（双重打分）已中止，
下表只验证接口/数值方向：

| 指标 | V1 5ep | V2 5ep（修复后） |
|---|---:|---:|
| R@20/50/100 | 0.5 | 0.4 |
| mR@50/100 | 0.5 | 0.4 |
| F1@50 | 0.5 | 0.4 |
| predicate_acc | 0.5 | 0.4 |
| pair_existence_AP | 0.600 | 0.443 |
| GT_MRR | 0.322 | 0.306 |

5 张图 / 10 条 GT，只说明接口可用；不能作为结论。

## 3. 训练曲线（5 epoch）

| epoch | V1 rel_geo | V1 rel_pred | V2 rel_geo | V2 rel_pred | F1@50 (both) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5600 | 1.5107 | 0.5596 | 1.5136 | 0 |
| 2 | 0.5421 | 1.4736 | 0.5433 | 1.4679 | 0 |
| 3 | 0.5349 | 1.4751 | 0.5338 | 1.4702 | 0 |
| 4 | 0.5264 | 1.4679 | 0.5238 | 1.4609 | 0 |
| 5 | 0.5250 | 1.4546 | 0.5221 | 1.4449 | 0 |

V1 ep5 诊断：`gamma_l1=1.383`，`delta_norm=0.053`。
V2 ep5 诊断：`w_bg_mean=0.482`，`delta_norm=30.32`。

## 4. 结论与边界

- V1/V2 都能用 `train.py` 跑通；batch=16、workers=8 按要求固定；23 项单测通过。
- V2 的闭式等变/主体系一致性成立：EE≈6.7e-8（fp32），SFC<1e-4；
  但场没有真正对齐：FMA 为负且比 V1 更差，WBG gap≈0，`w_bg` 仍约 0.48。
  因此**这份 smoke 不能证明 V2 更好**；“等变成立”与“场被有效使用”是两件事。
- V1 top1 0.511 vs V2 0.356；5 图 SGCls F1@50 V1 0.5 vs V2 0.4：
  V2 去掉对象/锚点几何后，在 4 个几何谓词上反而更难学。这与 V2 文档 §8 的
  “去掉绝对捷径短期可能掉点”一致，但样本太小，需要更多数据/训练才能定论。
- 公开 SGG 指标为 0 的主因是主模型/数据域差；GT 框口径的 top1/macro-F1 更适合评估关系头本身。
- 关系是 RelGen 几何生成的 4 谓词，不是 RA-4M 真实 `rel_annotations`；不能外推到开放谓词。
- 1 epoch 的旧 V2 数字（`w_bg≈0.796`、FMA=-0.971）是背景桶 LayerNorm bug 下的结果，已作废。
- 本表是管线验证，不是收敛结果，也不是论文数字。

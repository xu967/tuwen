# 与论文基线（RelateAnything, 2609.12552）的同口径对比

同一 1,024 张 RA-4M val 图（`RA-4M/data_full_monitor.yaml`，标签 = 原始 id），GT 框，
同一 10,370 条谓词词表，graph-constrained（每个有序对只出 argmax 谓词），同一评测代码
`tools/sgg_protocol_eval.py`（指标定义按论文 §5.1：R@K 逐图平均，mR@K 按谓词平均，桶按该
split 内谓词 GT 数 <50 / 50–500 / >500）。

| 模型 | 输入 | R@50 | mR@50 | F1@50 | R@50 micro | mR@50 rare | mR@50 common | mR@50 frequent | pair_acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RelateAnything 官方 ViT-S/16+（论文发布权重） | 框 | 26.61 | **13.11** | **17.56** | 26.71 | **11.96** | 34.69 | 21.25 | 34.26 |
| 本仓库 `glp_full_e4` epoch 1（EMA） | 框 + GT 类别 | **44.81** | 4.07 | 7.45 | **43.03** | 1.29 | **47.69** | **58.05** | **54.27** |
| 本仓库 `glp_full_e4` epoch 1（EMA） | 框 | 23.23 | 1.04 | 1.99 | 22.74 | 0.00 | 12.01 | 42.84 | 29.27 |

1,018 张图有 GT 关系，9,194 条 GT 三元组，436 个谓词（rare/common/frequent = 411/20/5）。

## 怎么读

- 论文的设定是「只给框、不给物体类别」（§3.1）。同一设定下我们的模型（1 epoch 全量、冻结 YOLO11n-DIOR 颈部、
  MobileCLIP 文本）在所有指标上低于官方模型：R@50 23.2 vs 26.6，mR@50 1.0 vs 13.1。
- 给 GT 类别后我们的 micro recall 反超（44.8 vs 26.6），但 mR@50 仍只有官方的 1/3：
  我们的头主要靠类别先验命中高频谓词，尾部谓词（411 个 rare）几乎为 0（1.29）。官方模型 rare 桶 11.96。
- 这不是论文表 4 的数字：论文报的是 VG150/PSG 等 50 类词表的跨数据集迁移；这里是 10,370 路、RA-4M 域内。
  两边都在同一口径下跑，才可比。
- 官方模型 12 epoch × 4 A100 × DINOv3 全量微调（448 px）；我们是 1 epoch、单卡、冻结骨干。
  `glp_full_e4_resume` 跑完 4 epoch 后用同一脚本重跑即可更新本表。

## 复现

```bash
# 官方模型（需要 git@github.com:Maelic/RelateAnything.git 与 transformers>=4.56 overlay）
USE_TF=0 PYTHONPATH=/root/rivermind-data/ra_overlay:/tmp/yolovo-audit-numpy2:/root/rivermind-data/RelateAnything \
  python tools/run_ra_official_predcls.py --release assets/relsgg-vits16plus --data RA-4M/data_full_monitor.yaml \
  --out reports/baseline_cmp/ra_official_vits16plus.pt
# 本仓库模型（走训练内同一个验证器）
PYTHONPATH=/tmp/yolovo-audit-numpy2 python tools/dump_predcls_ours.py <ckpt> RA-4M/data_full_monitor.yaml out.pt [--no-cls]
# 同口径评测
python tools/sgg_protocol_eval.py ra_official_vits16plus.pt ours.pt --names RA ours
```

官方权重细节：`assets/relsgg-vits16plus/model.pth`（EMA，含 DINOv3 backbone 与 19,103 条训练词表的 W/alpha）；
我们的 10,370 条谓词里 10,120 条直接取自该词表，250 条用随附的 text student 编码（已验证 text student 对已知串的
输出与 W 的 cos=1.0）。采样器预算按官方 `eval_zeroshot.py` 默认（eval_budget 500 → min(500, geo 400) = 400 对）；
分数 = sigmoid(a·(pred+pair)+b)，a/b 取 `calibration.json`（单调变换，不影响排序）。

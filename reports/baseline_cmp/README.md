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
| 本仓库 `glp_full_e4_resume` epoch 2（EMA，训练中途快照） | 框 + GT 类别 | **45.34** | 6.63 | 11.56 | **43.60** | 3.84 | **52.29** | 53.26 | **54.93** |
| 本仓库 `glp_full_e4_resume` epoch 2（EMA，= 4 轮跑完后的 `best.pt`） | 框 | 24.35 | 1.45 | 2.74 | 23.74 | 0.34 | 13.41 | 44.59 | 30.59 |
| 本仓库 `glp_full_e4_resume` epoch 4（EMA，`last.pt`） | 框 + GT 类别 | 42.92 | **8.64** | **14.38** | 41.18 | **6.21** | 48.11 | 50.59 | 52.26 |
| 本仓库 `glp_full_e4_resume` epoch 4（EMA，`last.pt`） | 框 | 22.71 | 1.73 | 3.22 | 21.72 | 0.67 | 13.68 | 41.16 | 28.29 |

1,018 张图有 GT 关系，9,194 条 GT 三元组，436 个谓词（rare/common/frequent = 411/20/5）。
全表 `protocol_eval_full_monitor_e4.json`（2026-11-07 12:12 由 `tools/post_full_e4.sh` 产出，13:40 补 epoch 4 只给框一行）。
`best.pt` 按训练内 fitness = pc_R@50 选，4 轮跑完仍是 epoch 2 的权重，所以「e4 best」与「epoch 2」两行相同。

### 4 轮全程（训练内验证器，同 1,024 图；`runs/test/glp_full_e4{,_resume}/results.csv`）

| epoch | lr(末) | train nce | pc_R@50 | pc_mR@50 | pc_F1@50 | pc_pair_acc | pair_existence_AP | mR@50 支持 2–4 / 5–19 / ≥20 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3.3e-4 | 3.079 | 42.62 | 3.55 | 6.56 | 53.70 | 54.32 | 0.0 / 0.75 / 32.58 |
| 2 | 5.0e-4 | 2.119 | **43.59** | 6.63 | 11.50 | **54.91** | 61.13 | 2.08 / 12.73 / 41.37 |
| 3 | 5.0e-4 | 2.022 | 41.98 | 7.74 | 13.07 | 53.17 | 62.27 | 3.65 / 16.37 / 42.65 |
| 4 | 2.6e-4 | 1.958 | 41.18 | **8.64** | **14.28** | 52.26 | **63.19** | 4.17 / 17.87 / 43.35 |

## 怎么读

- 论文的设定是「只给框、不给物体类别」（§3.1）。同一设定下我们最好的检查点（epoch 2）R@50 24.35 vs 官方 26.61，
  mR@50 1.45 vs 13.11，F1@50 2.74 vs 17.56，pair_acc 30.59 vs 34.26：全面低于官方，差距主要在尾部谓词。
- 给 GT 类别后 R@50 反超（45.3 vs 26.6），F1@50 最高的检查点是 epoch 4（14.38，官方 17.56）：靠的是 mR@50 从
  6.63 涨到 8.64、rare 桶 3.84→6.21。
- **4 轮曲线是「头部换尾部」**：pc_R@50 在 epoch 2 见顶后每轮掉 0.8~1.6 点，pc_mR@50、pc_F1@50、pair_existence_AP
  每轮都在涨，训练损失也一直在降。不是过拟合；是 10,370 路 InfoNCE 继续训练时，argmax 从高频谓词（on / has / wearing
  一类，占 GT 大头，决定 R@50）向中低频谓词转移。图约束下每对只出一个谓词，头部让出的每一个 argmax 都直接扣 R@50。
- 按 README「实验流程」的过关标准（以 pc_R@50 为「还在涨」的判据）**不过关**（`runs/test/glp_full_e4_resume/post/gate.json`：
  c1 第 4 轮 < 第 3 轮；c2 只给框 R@50 24.346 vs 参照 24.35，差 0.004，等于持平；c3 pair_acc 通过），20 轮未自动启动。
  若改以论文主指标 F1@50 作判据，则 4 轮单调上升、未饱和，结论相反。这是指标选择问题，见 progress.md 待确认。
- 这不是论文表 4 的数字：论文报的是 VG150/PSG 等 50 类词表的跨数据集迁移；这里是 10,370 路、RA-4M 域内。
  两边都在同一口径下跑，才可比。
- 官方模型 12 epoch × 4 A100 × DINOv3 全量微调（448 px）；我们是 4 epoch、单卡、冻结 YOLO11n-DIOR 颈部 + MobileCLIP 文本。

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

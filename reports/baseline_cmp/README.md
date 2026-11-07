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

## 差距拆解（2026-11-07 14:00，`tools/gap_analysis.py` → `gap_analysis.json`）

同一 9,194 条 GT，按谓词类型分组（按谓词名关键字粗分；directional = 左右/上下/前后类，proximity = 靠近类，
contact/support = on/in/holding/wearing 等接触支撑类，其余 = action/other）。表内为该组 GT 的 micro R@50。

| 组（GT 数 / 谓词数） | 官方 RA（框） | 我们 ep2（框） | 我们 ep4（框） | 我们 ep2（框+类别） | 我们 ep4（框+类别） |
|---|---:|---:|---:|---:|---:|
| directional（3,503 / 42） | 23.7 | **52.4** | 49.4 | 44.3 | 42.0 |
| contact/support（3,012 / 87） | **26.4** | 9.5 | 5.5 | 51.6 | 46.1 |
| proximity（925 / 89） | **23.7** | 4.1 | 3.5 | 13.6 | 14.7 |
| action/other（1,754 / 218） | **35.0** | 1.3 | 4.1 | 44.4 | 45.0 |
| rare 桶 micro R@50（1,814 / 411） | **20.5** | 1.4 | 2.4 | 16.3 | 21.6 |
| 至少命中 1 次的谓词数（/436） | **133** | 27 | 34 | 67 | 87 |

读法：

1. **只给框时我们是一台几何机器。** 左/右/上/下/前/后这类能从框坐标判定的谓词，我们 52.4 比官方 23.7 高一倍
   （`to the right of` 0.73 vs 0.32，`above` 0.63 vs 0.33）；凡是要看像素的谓词全部塌掉：`wearing` 0.001、
   `holding` 0.07、`on` 0.014、`part of` 0.00，218 个动作谓词合计 1.3%。436 个谓词里只有 27 个命中过。
2. **给了 GT 类别后，塌掉的那些靠类别先验补回来**（`wearing` 0.82、`part of` 0.86、`holding` 0.39），
   说明关系头本身能学会「人 + 衬衫 → wearing」这类规则；补不回来的是像素才能分的（`beside` 0.04、`next to` 0.07）。
3. 两条合起来：**冻结的 YOLO11n-DIOR 颈部特征（P4，40×40）几乎不携带物体身份和接触语义。** 官方模型从 DINOv3
   （全量微调）的像素特征里直接读出 `holding` 0.39、动作 35%，不需要类别。这是第一位的差距，不在关系头里。
4. 官方的头部数字被同义词压低：`wearing` 的 GT 对上它 71% 答 `wearing clothing`，`on` 答 `worn by / covering /
   resting on`。按精确匹配这些都算错（论文自己也说明了这一点）。所以官方的「真实」语义召回比 26.6 还高，
   差距比表面大。我们按单字符串训练，没有这个折损，这也是我们 directional 数字高的原因之一。
5. **谓词文本空间**（我们 MobileCLIP-S0 vs 官方蒸馏 student，同一批谓词）：

   | | 词表均值 off-diag cos | 中位谓词 cos≥0.9 的邻居数 | 反义对均值 cos | 同义对均值 cos | above/below | left/right | in front of/behind |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | 我们 MobileCLIP-S0 | 0.858 | **1,441** | 0.917 | 0.903 | 0.96 | 0.98 | 0.90 |
   | 官方 student | 0.142 | 1 | 0.24 | 0.704 | 0.09 | 0.15 | 0.05 |

   我们的 10,370 个谓词向量挤在一个 cos≈0.86 的团里，反义词和同义词分不开。方向类谓词没被它拖垮是因为几何
   特征（`q_spa`）能直接判左右；但 `holding` 要在 1,441 个几乎同向的列里被 InfoNCE 挑出来，长尾谓词在这样的
   空间里没有可分的方向。论文 §3.3 的原话：「No visual encoder trained against such targets can reliably
   separate above from below」。
6. 训练预算：官方 12 epoch × 4 A100、backbone 一起微调（lr 5e-5）、InfoNCE 正样本是同义词组加权均值、负例按
   共标注概率 P̂(q|p) 折扣；我们 4 epoch × 1×4080、backbone 冻结、单字符串正样本、负例折扣只按文本余弦
   （`ra/ontology.py`，而文本余弦本身已失真，见第 5 条）。

按影响排序的下一步（每项先 4 轮短跑，见 README「实验流程」）：

- A. 给关系分支一份可训练的 YOLO 颈部副本（lr 5e-5，不动一级检测器），直接对准第 3 条。
- B. 把 W 换成官方 student 文本空间（10,120 条已在其词表，250 条用随附 student 编码），零结构改动，对准第 5 条。
- C. 同义词组正样本 + 共标注负例折扣（论文 Eq. 3），对准长尾。
- D. 场头候选列排序监督（消融 A3，已在跑）：决定创新点本身的贡献上限。

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

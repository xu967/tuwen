# 进度（新开对话先读这里）

最后更新：2026-11-07 13:45

## 正在跑的守护进程（新会话先核对这里）

**当前没有本仓库的进程在跑，GPU 空闲**（13:30 时卡上有一个 1 GB、别的容器的进程 PID 444063，不是本仓库的）。

| 任务 | 状态 | 产物 |
|---|---|---|
| `glp_full_e4_resume` 全量训练 epoch 2~4 | 12:06 正常结束（3 轮 13.5 h） | `runs/test/glp_full_e4_resume/{results.csv,weights/best.pt(=epoch 2),weights/last.pt(=epoch 4)}` |
| 收尾流水线 `tools/post_full_e4.sh` v2 | 12:25 全部结束，`post/DONE` 各步 rc=0；**过关判断 rc=3 不过关，20 轮未启动** | `post/gate.json`、`post/protocol_eval.txt`、`reports/baseline_cmp/protocol_eval_full_monitor_e4.json` |
| BARV 难负例 cache + 对照 | 12:25 结束 | `data/barv_cache_contact/`（119,525 查询 / 29,772 图）、`runs/barv_contact/seed42/report.md` |
| PSG parquet → OpenPSG 转换 | 03:51 结束 | `/root/rivermind-data/psg_hf/{psg_train_from_hf.json,images/}` |

### 4 轮全量结果一句话

pc_R@50 在 epoch 2 见顶（43.59）后每轮下降（41.98 → 41.18）；pc_mR@50（6.63 → 7.74 → 8.64）、pc_F1@50（11.50 → 13.07 → 14.28）、pair_existence_AP 每轮上升，训练损失一直在降。头部谓词换尾部谓词，不是过拟合。与官方 RA 同口径：只给框时最好的 epoch 2 R@50 24.35 / mR@50 1.45 / F1 2.74（官方 26.61 / 13.11 / 17.56）；给 GT 类别时 epoch 4 F1@50 14.38 为历次最高。全表 `reports/baseline_cmp/README.md`。

### BARV 难负例对照一句话

去掉几何捷径后（负例 = 几何接触、类别合法、未标注），test AUROC：几何 LR 0.680 / 几何 MLP（像素置零）0.750 / 视觉 0.777；对 contact_other 负例 0.706 / 0.782 / 0.810。视觉特征在 9 个谓词里 8 个高于几何 LR（touching 例外 0.57 vs 0.67）。pilot 上 0.996 vs 0.994 的「无差别」是几何捷径造成的，现在差别出来了，但绝对水平只有 0.78。

## 已完成的事项

- GLP 关系头（RelateAnything 移植 + 谓词高斯场）在 RA-4M 上的 50k 代理实验 v5→v8 全部跑完，结论见 `WORKLOG.md`（1284 行起）：v6 是主线（pc_R@50≈0.42），v8 三项机制整体负收益。
- 全量 RA-4M（472,344 图）4 epoch 实验：`glp_full_e4` 跑完 epoch 1 后中断；2026-11-06 已修好断点续训并以 `glp_full_e4_resume` 续跑 epoch 2~4（val 换成标签正确的 `full_monitor`，原 run 的 val 标签 11% 错位）。日志 `logs/full_e4_resume.log`，结果 `runs/test/glp_full_e4_resume/results.csv`，预计 2026-11-07 10:00 前后结束。
- 代码与交接文档全面审计（含 BARV）：`worklog/2026-11-06-2130-glp-audit-resume-full-run.md`。训练步速 1.21→0.90 s/it 的几项无数值影响的优化已合入工作树。
- BARV pilot（`runs/barv/pilot_seed42`）：视觉模型 AUROC 0.996 vs 几何 LR 0.994 vs 单个 IoU 标量 0.9655，标签由几何规则决定，现有分数无法证明视觉证据价值。

- 2026-11-07：代码与文档已提交推送（`2e802bf`、`c198a8c`）。论文基线同口径对比完成：`reports/baseline_cmp/README.md`（官方 RelateAnything 权重 vs 我们，同 val/同词表/同评测）。
- 2026-11-07 03:50：PSG train 源数据到齐并转换完成：46 片 parquet（22 GB）→ `/root/rivermind-data/psg_hf/psg_train_from_hf.json`（46,267 条记录 / 46,042 张 COCO train2017 图 / 261,666 条关系，全部物体与 panoptic 段 IoU≥0.99，0 张未匹配）+ `/root/rivermind-data/psg_hf/images/`（46,042 张 JPEG，22 GB）。`barv.prepare` 用 `configs/barv_psg_contact.yaml` 在前 300 条记录上冒烟通过（1,290 条查询，0 张缺图）。

## 关键决策

- 全量实验**不重头跑**，从 epoch 1 断点续跑，保持 lr/warmup 调度连续；只把 val 从错标的 `full_monitor_v5` 换成 `full_monitor`。epoch 1 在正确 val 上的复评：pc_R@50=0.4303（`reports/glp_full_e4_epoch1_predcls_full_monitor.json`）。
- 需要重训才能体现的 bug（场头在 `field_fuse=max` 下只靠 0.25×sigmoid 拿梯度、rel focal 只训 128 对、cap 对 q_spa 不对称、geo BCE 把反向对标 0 等）**不在实验途中修**，等全量结果出来后分别消融。
- 工作树里 v8 三项机制（cfa / box_token_dropout / geo_res_proj）保持开启：全量实验是用它们训的，代码必须与结果一起定版。
- 一级模型权重已从回收站路径备份到 `weights/primary_dior20_best.pt`；`train.py` 默认路径未改。
- 2026-11-07 05:12 用户接受三条过关阈值并要求开始执行。`glp_full_e4_resume` 的过关判断已自动化（`tools/post_full_e4.sh` 第 4 步，参照 = epoch 2 快照）：c1 epoch 4 的 pc_R@50 > epoch 3；c2 只给框 R@50 ≥ 24.35；c3 框+类别 pair_acc ≥ 54.93 − 0.5。三条全过 → 立即起 `glp_full_e20`（同配置、从头起），BARV 的 GPU 步骤与它并行（resnet18 特征抽取显存小）。
- 2026-11-07 04:30 用户定下实验流程：**每个配置先跑 4 轮看效果，效果好再跑 20 轮**。流程、启动命令、过关标准（3 条默认阈值）、20 轮从头起而不是续 4 轮权重的理由，都写在 `README.md`「实验流程」一节。正在跑的 `glp_full_e4_resume` 就是第一个 4 轮短跑，它跑完后按该标准决定是否起 `glp_full_e20`（约 78 h）。
- 2026-11-07 04:10 用户确认：无用文件可删、数据盘会扩容。已删 46 片 PSG parquet（22 GB，JSON + JPEG 已替代）和 `logs/error_report_*.txt`（112 个旧崩溃报告）。保留未动：`images.tar.gz`（1.7 GB，flickr/nyu 图，来源不明）、`mobileclip_s0_pt_7z_parts/`（400 MB，用户上传的分卷包，`assets/` 下已有解出的 `.pt`）、已结束代理实验的 `runs/test/*/weights/train-checkpoint.pt`（每个 1.1 GB，共约 10 GB，续训状态）。

## 未完成的待办

0. **待用户拍板（GPU 现在空着）**：4 轮短跑按约定的 pc_R@50 判据不过关，20 轮没起。两条路：
   (a) 维持 pc_R@50 判据 → 不起 20 轮，先做消融 A3（场头监督）/ 修已知训练缺陷，再按流程重跑 4 轮；
   (b) 改以论文主指标 F1@50 为「还在涨」的判据（4 轮单调上升 11.5→14.3，未饱和）→ 立即起 `glp_full_e20`（README 命令，约 78 h）。
   建议 (a)：20 轮会把已知缺陷（场头在 `field_fuse=max` 下只靠 0.25×sigmoid 拿梯度、rel focal 只训 128 对等）一起固化 3 天；且 R@50 与 pair_acc 在掉，不是单纯「还没收敛」。
1. BARV 下一步（看完 `runs/barv_contact/seed42/report.md` 后）：多种子（3 个）确认视觉 +2.7 AUROC 不是噪声；`contact_unlabeled` 标注噪声单独消融；再决定做不做 ROI 增益 / teacher-value 实验。
2. 对 best.pt / last.pt 跑全量 val（24,964 图，`RA-4M/data_full.yaml`，`rel.val(protocol="predcls")`），确认 1,024 图上的「头部换尾部」在全量 val 上成立。
3. 消融 A3：让场头的候选列拿到排序监督（M 路 CE 或 NCE 读 fused），再比较 field/max/raw。按 README 流程：每个变体先 4 轮，过关的再 20 轮；GPU 一次只跑一个，排队。
3b. 把 README 里的「训练 → dump → 评测 → 过关判断」收成一个入口脚本 `tools/run_exp.sh`（配置区在顶部，守护启动，结束写 DONE）。GPU 已空，可用 50k 代理 1 轮冒烟。
4. `mobileclip_s0_pt_7z_parts/` 应 ignore 或移出。
5. 重写或标注过期文档：`GLP_INNOVATIONS_IMPLEMENTATION.md`、`新手指南.txt`、`barv/README.md`。

## 环境速查

```bash
export LD_LIBRARY_PATH=/root/rivermind-data/cudnn9/nvidia/cudnn/lib:$LD_LIBRARY_PATH   # train.py 自带
export PYTHONPATH=/tmp/yolovo-audit-numpy2                                            # numpy 2.2.6 overlay，必须
/opt/conda/bin/python train.py        # 默认会起 proxy50k 旧词表 100 轮，先看 train.py 顶部的环境变量
# 续训：YOLOVO_RESUME=runs/test/<run>/weights/train-checkpoint.pt YOLOVO_EPOCHS=<总轮数> YOLOVO_NAME=<run> ...
# 单测（无 pytest）：python /root/rivermind-data/scratch_audit/run_tests.py
```

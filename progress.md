# 进度（新开对话先读这里）

最后更新：2026-11-07 05:30

## 正在跑的守护进程（新会话先核对这里）

| 任务 | 进程 | 日志 / 完成标记 | 停止 |
|---|---|---|---|
| `glp_full_e4_resume` 全量训练 epoch 2~4 | python PID 82761，PGID 82753（setsid，忽略 SIGHUP，与 IDE 无关） | `logs/full_e4_resume.log`；`runs/test/glp_full_e4_resume/results.csv` 每 epoch 一行 | `kill -- -82753` |
| 训练收尾流水线 `tools/post_full_e4.sh` v2（等训练退出 → best/last PredCls dump → 与官方 RA 同口径评测 → **过关判断，过关就自动起 20 轮 `glp_full_e20`** → BARV 难负例 cache（`barv.prepare`）→ 几何/视觉对照（`barv.run_contact`）） | PGID 509167（`runs/test/glp_full_e4_resume/post/run.pid`） | `runs/test/glp_full_e4_resume/post/run.log`；过关结果 `post/gate.json`；结束写 `post/DONE`（每步退出码） | `kill -- -509167` |
| `glp_full_e20`（**仅当过关时由上一行自动启动**，预计 11:00 后，约 78 h） | pid 在 `logs/glp_full_e20.pid`（自成 PGID） | `logs/glp_full_e20.log`；`runs/test/glp_full_e20/results.csv` | `kill -- -$(cat logs/glp_full_e20.pid)` |
| PSG parquet → OpenPSG 转换 `/root/rivermind-data/psg_hf/convert.sh` | 已结束（03:43 起，8 分钟） | `/root/rivermind-data/psg_hf/convert.log`；`CONVERT_DONE` = `exit=0` | 无 |

状态快照 2026-11-07 03:50：训练在 epoch 3/4 的 25%，1.06 it/s，显存 15.4 G；epoch 2 的 pc_R@50=0.4359（epoch 1 同 val 复评 0.4303）。预计 epoch 4 结束 + 终评 ≈ 11:00。

epoch 2 中途快照同口径评测（04:07，`reports/baseline_cmp/README.md` 已加两行）：框+GT 类别 R@50 45.34 / mR@50 6.63 / pair_acc 54.93（epoch 1：44.81 / 4.07 / 54.27）；只给框 R@50 24.35 / mR@50 1.45 / pair_acc 30.59（epoch 1：23.23 / 1.04 / 29.27；官方 RA：26.61 / 13.11 / 34.26）。头部在涨，尾部谓词仍接近 0。

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

0. BARV 续做：难负例规则已实现并在第 0 片验证（几何 LR 0.97→0.63，`configs/barv_psg_contact.yaml`）。数据已转换完（见上）。`tools/post_full_e4.sh` 第 5 步会自动跑：`barv.prepare` 全量重建 cache（`data/barv_cache_contact/`，`source` 列现在区分 contact_other/contact_unlabeled）→ `barv.run_contact`（几何 LR / no_visual MLP / 视觉 scorer，dev 选 epoch、test 报总体与按负例来源的 AUROC，输出 `runs/barv_contact/seed42/report.md`）。看完 report 再决定 BARV 下一步（多种子、ROI 增益实验）。

1. 等 `glp_full_e4_resume` 跑完（≈ 11:00）：过关判断与 `glp_full_e20` 启动已自动化（见守护进程表与 `post/gate.json`）；人工要做的是核对 `gate.json`、若未过关分析原因。`tools/post_full_e4.sh` 会自动产出 `reports/baseline_cmp/ours_full_e4_{best_cls,best_nocls,last_cls}.pt`、`protocol_eval_full_monitor_e4.json`、`post/protocol_eval.txt`；人工要做的是把结果写进 `reports/baseline_cmp/README.md`，并对比 epoch 2~4 的 `pc_R@50 / pc_mR@50_sup_ge20 / pc_pair_acc` 与 v8 同轮（`runs/test/glp_50k_v8_cfa_e14/results.csv`），注意两者词表不同（10,370 vs 2,809），不是单变量对照。
2. 对 best.pt 跑全量 val（24,964 图，`RA-4M/data_full.yaml`，`rel.val(protocol="predcls")`）。
3. 消融 A3：让场头的候选列拿到排序监督（M 路 CE 或 NCE 读 fused），再比较 field/max/raw。按 README 流程：每个变体先 4 轮，过关的再 20 轮；GPU 一次只跑一个，排队。
3b. 把 README 里的「训练 → dump → 评测 → 过关判断」收成一个入口脚本 `tools/run_exp.sh`（配置区在顶部，守护启动，结束写 DONE）。现在 GPU 被全量训练占着没法自测，等 11:00 后再写并用 50k 代理 1 轮冒烟。
4. 决定是否提交当前工作树（含 v8 改动 + 本次改动）；`mobileclip_s0_pt_7z_parts/` 应 ignore 或移出。
5. BARV：重建难负例后再决定是否继续。
6. 重写或标注过期文档：`GLP_INNOVATIONS_IMPLEMENTATION.md`、`新手指南.txt`、`barv/README.md`。

## 环境速查

```bash
export LD_LIBRARY_PATH=/root/rivermind-data/cudnn9/nvidia/cudnn/lib:$LD_LIBRARY_PATH   # train.py 自带
export PYTHONPATH=/tmp/yolovo-audit-numpy2                                            # numpy 2.2.6 overlay，必须
/opt/conda/bin/python train.py        # 默认会起 proxy50k 旧词表 100 轮，先看 train.py 顶部的环境变量
# 续训：YOLOVO_RESUME=runs/test/<run>/weights/train-checkpoint.pt YOLOVO_EPOCHS=<总轮数> YOLOVO_NAME=<run> ...
# 单测（无 pytest）：python /root/rivermind-data/scratch_audit/run_tests.py
```

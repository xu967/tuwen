# 进度（新开对话先读这里）

最后更新：2026-11-07 01:00

## 已完成的事项

- GLP 关系头（RelateAnything 移植 + 谓词高斯场）在 RA-4M 上的 50k 代理实验 v5→v8 全部跑完，结论见 `WORKLOG.md`（1284 行起）：v6 是主线（pc_R@50≈0.42），v8 三项机制整体负收益。
- 全量 RA-4M（472,344 图）4 epoch 实验：`glp_full_e4` 跑完 epoch 1 后中断；2026-11-06 已修好断点续训并以 `glp_full_e4_resume` 续跑 epoch 2~4（val 换成标签正确的 `full_monitor`，原 run 的 val 标签 11% 错位）。日志 `logs/full_e4_resume.log`，结果 `runs/test/glp_full_e4_resume/results.csv`，预计 2026-11-07 10:00 前后结束。
- 代码与交接文档全面审计（含 BARV）：`worklog/2026-11-06-2130-glp-audit-resume-full-run.md`。训练步速 1.21→0.90 s/it 的几项无数值影响的优化已合入工作树。
- BARV pilot（`runs/barv/pilot_seed42`）：视觉模型 AUROC 0.996 vs 几何 LR 0.994 vs 单个 IoU 标量 0.9655，标签由几何规则决定，现有分数无法证明视觉证据价值。

- 2026-11-07：代码与文档已提交推送（`2e802bf`、`c198a8c`）。论文基线同口径对比完成：`reports/baseline_cmp/README.md`（官方 RelateAnything 权重 vs 我们，同 val/同词表/同评测）。BARV 源数据（PSG train，HF `JosephZ/psg_train_sg`）正在后台下载到 `/root/rivermind-data/psg_hf/`。

## 关键决策

- 全量实验**不重头跑**，从 epoch 1 断点续跑，保持 lr/warmup 调度连续；只把 val 从错标的 `full_monitor_v5` 换成 `full_monitor`。epoch 1 在正确 val 上的复评：pc_R@50=0.4303（`reports/glp_full_e4_epoch1_predcls_full_monitor.json`）。
- 需要重训才能体现的 bug（场头在 `field_fuse=max` 下只靠 0.25×sigmoid 拿梯度、rel focal 只训 128 对、cap 对 q_spa 不对称、geo BCE 把反向对标 0 等）**不在实验途中修**，等全量结果出来后分别消融。
- 工作树里 v8 三项机制（cfa / box_token_dropout / geo_res_proj）保持开启：全量实验是用它们训的，代码必须与结果一起定版。
- 一级模型权重已从回收站路径备份到 `weights/primary_dior20_best.pt`；`train.py` 默认路径未改。

## 未完成的待办

0. BARV 续做（方案见 worklog 11-07 01:00 更新）：等 `/root/rivermind-data/psg_hf/download.log` 出现 ALL_DONE → 写难负例规则 → 重建 cache → 重跑几何 LR / 视觉 scorer。

1. 等 `glp_full_e4_resume` 跑完：对比 epoch 2~4 的 `pc_R@50 / pc_mR@50_sup_ge20 / pc_pair_acc` 与 v8 同轮（`runs/test/glp_50k_v8_cfa_e14/results.csv`），注意两者词表不同（10,370 vs 2,809），不是单变量对照。
2. 对 best.pt 跑全量 val（24,964 图，`RA-4M/data_full.yaml`，`rel.val(protocol="predcls")`）。
3. 消融 A3：让场头的候选列拿到排序监督（M 路 CE 或 NCE 读 fused），再比较 field/max/raw。
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

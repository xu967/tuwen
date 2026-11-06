# tuwen

Ultralytics 魔改版 YOLOVO（开放词表检测 + MobileCLIP 文本塔）上的两条研究线。先读 `progress.md`。

## 研究线

| 线 | 内容 | 代码 | 入口 | 状态 / 结果 |
|---|---|---|---|---|
| GLP | RelateAnything 风格关系头 + 谓词高斯场，RA-4M 数据，PredCls 口径 | `ultralytics/models/yolovo/relate/` | `train.py`（环境变量见文件顶部） | `WORKLOG.md`（1284 行起最新）、`GLP_RESULTS.md`、`runs/test/*/results.csv` |
| BARV | PSG 关系的主动 ROI 验证 | `barv/`、`configs/barv_*.yaml` | 无 CLI，见 `barv/RESULTS.md` | `runs/barv/pilot_seed42`，几何捷径问题见 `worklog/` |

## 必读文档

- `progress.md`：当前进度、关键决策、待办。
- `worklog/2026-11-06-2130-glp-audit-resume-full-run.md`：最新审计（bug 清单、评测口径、文档勘误）与全量实验续跑记录。
- `worklog/2026-11-06-code-audit-barv-pilot.md`：BARV pilot 审计。
- `WORKLOG.md`：GLP 线逐日日志（历史，部分早期结论已被后文推翻，以最新小节为准）。
- `GLP_INNOVATIONS_IMPLEMENTATION.md` / `GLP_V2_RELATIONAL_FIELD.md`：11-01 的方案稿，引用的 `scripts/`、`reports/*.md`、SUN/SpatialSense 数据本机不存在。
- `新手指南.txt`：一级 YOLOVO 的入门，不含 GLP/BARV，环境章节已过期（见 progress.md 环境速查）。

## 环境

RTX 4080 SUPER；`/opt/conda/bin/python` 3.11 + torch 2.5.1+cu124。两个必需变量：
`LD_LIBRARY_PATH=/root/rivermind-data/cudnn9/nvidia/cudnn/lib`（train.py 自动设置）和
`PYTHONPATH=/tmp/yolovo-audit-numpy2`（numpy 2.2.6；默认环境的 numpy 1.26.4 已损坏）。pytest 未安装，单测用 `scratch_audit/run_tests.py` 风格的直接调用。

## 数据与权重

- RA-4M：`RA-4M/`（`images/train` 472k 图；`data_full_monitor.yaml` 全量词表 + 1,024 图正确标签监控集；`data_proxy50k_v5.yaml` 50k 子集 + 2,809 截断词表）。`full_monitor_v5/` 的标签 id 是 v5 词表的，只能配 v5 词表。
- 一级模型（DIOR 20 类，冻结）：`weights/primary_dior20_best.pt`（`train.py` 默认仍指向 `/root/rivermind-fs/.Trash-0/files/code/best.pt`）。
- MobileCLIP s0：`ultralytics/nn/textencoder/mobileclip/mobileclip_s0.pt`。

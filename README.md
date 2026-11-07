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

## 实验流程：先 4 轮，效果好再 20 轮（2026-11-07 起执行）

每个新配置（消融、改损失、换数据、换词表）都走两段，不直接起长实验：

| 段 | 轮数 | 全量 RA-4M（472k 图，1 epoch ≈ 3.9 h） | 50k 代理（`data_proxy50k_v5.yaml`，1 epoch ≈ 25 min） | 目的 |
|---|---|---|---|---|
| 短跑 | 4 | ≈ 16 h | ≈ 1.7 h | 看方向对不对、曲线有没有在涨 |
| 长跑 | 20 | ≈ 78 h（3 天多） | ≈ 8.5 h | 只给短跑过关的配置 |

### 短跑（4 轮）

一条命令跑完整段（训练 → dump → 同口径评测 → 过关判断 → `post/DONE`），配置区在脚本顶部：

```bash
cd /root/rivermind-data/YOLOVOmodal_3871538535
NAME=<实验名>_e4 DATA=RA-4M/data_proxy50k.yaml EPOCHS=4 REF_RUN=runs/test/<参照 run> <YOLOVO_* 消融开关> \
  setsid nohup bash tools/run_exp.sh > logs/<实验名>_e4.queue.log 2>&1 < /dev/null &
# DATA：消融先用 50k 子集（全词表、同一 val，1.7 h）；过关的再换 RA-4M/data_full_monitor.yaml（16 h）
# 排队：在一个 bash -c 里顺序调用两次 run_exp.sh（例见 logs/queue_a3.log 第一行）
```

手动分步（与脚本等价）：

```bash
cd /root/rivermind-data/YOLOVOmodal_3871538535
export PYTHONPATH=/tmp/yolovo-audit-numpy2
NAME=<实验名>_e4          # 命名：<实验名>_e4 / <实验名>_e20，同一配置只差轮数
# 基线配置 = 正在跑的 glp_full_e4_resume：全量训练集 + 1,024 图监控 val、field=glp(v1)、batch 32、不缓存。
# YOLOVO_PRIMARY 用仓库内备份（与 train.py 默认的回收站路径 md5 相同）。消融在这组变量基础上只改一项。
YOLOVO_RELATE_DATA=RA-4M/data_full_monitor.yaml YOLOVO_PRIMARY=weights/primary_dior20_best.pt \
YOLOVO_EPOCHS=4 YOLOVO_NAME=$NAME YOLOVO_CACHE=0 <其他环境变量> \
  setsid nohup bash -c 'echo $$ > logs/'$NAME'.pid; exec /opt/conda/bin/python train.py' > logs/$NAME.log 2>&1 < /dev/null &
sleep 5; ps -o pid,pgid,etime,cmd -p $(cat logs/$NAME.pid)   # 确认活着；停：kill -- -<PGID>
```

（pid 在子 shell 里用 `$$` 记：交互终端下 `setsid` 会再 fork 一次，`$!` 记到的是已经退出的父进程。）

跑完后用同一口径评测（与 `reports/baseline_cmp/README.md` 同一 1,024 张 val、同一词表、同一脚本）：

```bash
python tools/dump_predcls_ours.py runs/test/$NAME/weights/best.pt RA-4M/data_full_monitor.yaml reports/baseline_cmp/$NAME.cls.pt
python tools/dump_predcls_ours.py runs/test/$NAME/weights/best.pt RA-4M/data_full_monitor.yaml reports/baseline_cmp/$NAME.nocls.pt --no-cls
python tools/sgg_protocol_eval.py reports/baseline_cmp/ra_official_vits16plus.pt reports/baseline_cmp/<参照 e4>.cls.pt reports/baseline_cmp/$NAME.cls.pt reports/baseline_cmp/$NAME.nocls.pt --names RA ref ours ours_nocls --json reports/baseline_cmp/$NAME.json
```

### 过关标准（「效果好」的默认定义，阈值可改）

参照 = 同数据、同轮数的上一个基线（当前基线是 `glp_full_e4` + `glp_full_e4_resume` 的 4 轮结果，见 `reports/baseline_cmp/README.md`）。
满足下面 1 和 2，并且 3 不触发，才起 20 轮：

1. 还在涨：`results.csv` 里 epoch 4 的 `pc_R@50` 高于 epoch 3（4 轮里 3 轮是 warmup，第 4 轮还在涨说明没饱和；已经平了的配置加到 20 轮也不会好）。
2. 有提升：对参照 `pc_R@50` ≥ +0.5 点，或 `pc_mR@50` ≥ +1.0 点（1,024 张监控集上 ±0.3 点以内当噪声）。
3. 没有退化：`pc_pair_acc` 不比参照低 0.5 点以上；只给框（`--no-cls`）的 `R@50` 不比参照低。

三条都写进该实验的 worklog，再决定起不起长跑。第一次用这套标准的对象就是正在跑的 `glp_full_e4_resume`：它没有同轮数参照，只看第 1 条和第 3 条里「只给框 R@50 是否继续逼近官方 26.61」。

### 长跑（20 轮）

- 从头起，不从 4 轮权重续：学习率是「3 轮 warmup + 线性衰减到 lrf=0.01×lr0」，衰减按总轮数算，4 轮末 lr 已经掉到底，接着跑 20 轮会让 lr 跳回去，结果和干净的 20 轮不可比。
- 命令同短跑，只改 `YOLOVO_EPOCHS=20 YOLOVO_NAME=<实验名>_e20`；想省那 16 小时可以 `YOLOVO_RESUME=runs/test/<实验名>_e4/weights/train-checkpoint.pt YOLOVO_EPOCHS=20` 续，但 worklog 里要写明是续跑。
- 20 轮全量要 3 天多，期间 GPU 不能给别的训练；短跑验证过的配置排队跑，一次一个。
- 跑完同样过一遍 dump + `sgg_protocol_eval.py`，把 e4 / e20 两行一起写进 `reports/baseline_cmp/README.md`。

### 每次实验必留的东西

`runs/test/<name>/results.csv`（逐轮指标）、`logs/<name>.log`、`reports/baseline_cmp/<name>.json`（同口径评测）、worklog 小节（配置差异、三条过关判断、是否起长跑）。进程一律 `setsid nohup` 起，pid/PGID 记进 `progress.md` 的守护进程表。

## 环境

RTX 4080 SUPER；`/opt/conda/bin/python` 3.11 + torch 2.5.1+cu124。两个必需变量：
`LD_LIBRARY_PATH=/root/rivermind-data/cudnn9/nvidia/cudnn/lib`（train.py 自动设置）和
`PYTHONPATH=/tmp/yolovo-audit-numpy2`（numpy 2.2.6；默认环境的 numpy 1.26.4 已损坏）。pytest 未安装，单测用 `scratch_audit/run_tests.py` 风格的直接调用。

## 数据与权重

- RA-4M：`RA-4M/`（`images/train` 472k 图；`data_full_monitor.yaml` 全量词表 + 1,024 图正确标签监控集；`data_proxy50k_v5.yaml` 50k 子集 + 2,809 截断词表）。`full_monitor_v5/` 的标签 id 是 v5 词表的，只能配 v5 词表。
- 一级模型（DIOR 20 类，冻结）：`weights/primary_dior20_best.pt`（`train.py` 默认仍指向 `/root/rivermind-fs/.Trash-0/files/code/best.pt`）。
- MobileCLIP s0：`ultralytics/nn/textencoder/mobileclip/mobileclip_s0.pt`。

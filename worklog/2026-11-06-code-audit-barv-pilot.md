# 代码与交接文档审计：BARV baseline pilot

- 时间：2026-11-06 05:45 Asia/Shanghai
- 范围：仓库交接文档、GLP/RA-4M 关系训练路径、BARV 数据构建与训练路径；按单机研究训练、单个 GPU 进程评估。
- 结论：部分完成。完成重点路径审计及 BARV coarse/geometry-only pilot；整仓 Ultralytics 通用检测代码未逐文件审完。

## 仓库用途

仓库当前有两条研究线：GLP/RA-4M 用冻结一级检测器训练谓词关系头；BARV/PSG 研究用按需 ROI 证据验证关系。根 README 只有标题，交接状态需依赖多份方案、WORKLOG 和结果文件。

## Must Fix：阻断结论或复现

### 1. BARV 分数主要可由几何规则预测

- **位置**：`barv/psg.py:316-355`、`barv/psg.py:221-253`、pilot 输出 `runs/barv/pilot_seed42/`。
- **问题**：正例来自 PSG triple，负例由 `mask gap > threshold` 构造；输入几何虽未直接包含 gap，但包含主体/客体/union 框、IoU、中心偏移和面积。30 轮 visual scorer 在 held-out test 上 AUROC=0.9960，geometry-only LogisticRegression 已达 0.9942；balanced accuracy 分别为 0.9820 和 0.9756。两者差距很小，说明当前任务标签可被框几何近似恢复，尚不能证明模型学会了视觉关系证据，更不能证明主动 ROI 有价值。
- **建议修复**：先补按类别与框几何匹配的人工核验难负例，重建独立测试集并重跑 geometry-only 基线；只有视觉/局部证据对该基线有稳定增益后，再做 WP3 ROI oracle gap 与 latency，最后才训练 teacher/MVE。
- **若跳过**：后续策略很可能只学会“查询标签生成规则”，会得到高指标但没有有效视觉证据增量。

### 2. GLP 全量训练状态在交接文档中不准确

- **位置**：`GLP_RESULTS.md:94-109`、`WORKLOG.md:1730-1768`、`runs/test/glp_full_e4/results.csv`、`/tmp/full_e4.log`。
- **问题**：`GLP_RESULTS.md` 仍称 11 月 1 日的 100 epoch 实验进行中；WORKLOG 记录 `glp_full_e4` 启动及预计完成时间，但产物仅有 epoch 1 的 results 行，日志到 epoch 2 的 3,130/14,760 steps（约 21%）后结束。不能把它当 4 epoch 完成结果。该配置本身 warmup 占 4 epoch 总步数约 75%，`close_mosaic=10` 又大于 epoch 数，即使跑完也只是难以解释的 pilot。
- **建议修复**：更新单一权威状态文档，记录 epoch 1 完成、epoch 2 中断、实际指标和 checkpoint；重设计足够训练视界与 warmup 后再决定是否重跑，不复用这份 4 epoch 对照做论文结论。
- **若跳过**：接手者会按过期计划误以为实验已经完成或仍在运行，并可能基于不可比指标继续扩训练。

### 3. BARV README 的复现入口和源数据不成立

- **位置**：`barv/README.md:53-61`、`barv/experiment.py:60-73`、`configs/barv_psg.yaml:3-8`。
- **问题**：README 引用的 `scripts/audit_data.py`、`scripts/benchmark_hw.py`、`scripts/run_barv.py`、`scripts/make_report.py` 和 `tests/test_barv.py` 均不存在；`barv/experiment.py` 有函数但没有命令行入口。已有 `data/barv_cache/queries.npz` 可供缓存级训练使用，但配置指向的 `data/psg/psg_train_val.json` 和 `data/coco/images/` 当前缺失，无法从源数据重建缓存或运行完整 ROI 流程。
- **建议修复**：先恢复并登记 PSG/COCO 源数据及许可、版本和 SHA256，再补实际 CLI/测试入口；README 只保留已验证可运行的命令。
- **若跳过**：现有实验不能由新环境复现，缓存结果也无法追溯到源数据。

### 4. GLP 中断 checkpoint 当前不能在此环境加载

- **位置**：`runs/test/glp_full_e4/weights/train-checkpoint.pt`、`ultralytics/models/yolovo/relate/trainer.py:406-429`。
- **问题**：checkpoint pickle 引用了多个顶层 `mobileclip.*` 类；当前仓库没有该 Python 包，尝试加载 `last.pt` 报 `ModuleNotFoundError: No module named 'mobileclip'`。所以本轮没有续训，也不能确认 `train-checkpoint.pt` 可恢复。
- **建议修复**：为训练 checkpoint 固定并随项目声明完整的 MobileCLIP 代码依赖；把 checkpoint 加载/恢复 smoke 纳入训练前检查，避免保存出只能在原进程环境加载的权重。
- **若跳过**：训练被中断时无法续跑，旧模型资产也可能随环境迁移失效。

## Should Fix：结果解释与回归保护

### 5. GLP 的 SGDet 指标受一级检测器域差主导

- **位置**：`runs/test/glp_full_e4/results.csv`、`GLP_RESULTS.md:96-100`。
- **问题**：RA-4M 有 497 个对象类，记录中的冻结一级模型是 DIOR 20 类；full-e4 SGDet `DetPairRecall=0.0193`、`R@50=0.00349`，而 PredCls/GT 框口径 `pc_R@50=0.42623`。两种协议回答不同问题，当前 SGDet 数字不能用于判断关系头质量或端到端性能。
- **建议修复**：明确将当前结论限定为 oracle-box/PredCls 关系分类；若要声称端到端 SGG，需使用 RA-4M 类别域的检测器并单独评测。
- **若跳过**：会把检测召回失败误归因于关系头，或把 oracle-box 成绩误报为端到端结果。

### 6. BARV 谓词数量文档和 cache 不一致

- **位置**：`barv/README.md:67`、`configs/barv_psg.yaml:10-19`、`data/barv_cache/queries_stats.json:8-19`。
- **问题**：README 说最终任务有 8 个谓词；配置列出 9 个，已有 cache 也有 0–8 共 9 个谓词且 `dropped_predicates` 为空。
- **建议修复**：以实际训练词表和 cache manifest 为准，同步文档并把谓词名称/id 写入结果元数据。
- **若跳过**：不同实验可能按不同谓词集合比较，结果无法正确复核。

### 7. CFA 的测试没有验证 box-token dropout 的效果

- **位置**：`tests/test_glp_field.py:430-455`、`ultralytics/models/yolovo/relate/ra/transformer.py:97-110`。
- **问题**：新增测试把 `box_token_dropout=1.0` 后只检查 loss 有限及反向可运行；即使 transformer 忽略 `box_token_drop`，该测试仍会通过。
- **建议修复**：单独断言全隐藏 box tokens 与保留 tokens 的确定性前向存在预期差异，并确认 padding slots 仍被屏蔽。
- **若跳过**：后续改动可能静默断开 dropout 接线，训练仍通过当前回归测试。

## Nice to Have：维护成本与速度

### 8. 项目入口和实现文档陈旧

- **位置**：根 `README.md` 仅有 `# tuwen`；`GLP_INNOVATIONS_IMPLEMENTATION.md:3-64` 日期为 11 月 1 日，列出的 `scripts/` 资产在当前仓库不存在，且仍写 GPU 不可用。
- **问题**：新接手者无法从仓库入口判断主线、有效命令、最近结果和未完成项。
- **建议修复**：增加简短 README 索引到当前 GLP/BARV 状态、可运行命令、结果与数据依赖。
- **若跳过**：重复启动已经作废的实验、使用不存在脚本的风险会继续存在。

### 9. CFA 每个训练 batch 都触发一次 GPU 同步

- **位置**：`ultralytics/models/yolovo/relate/head.py:365-382`。
- **问题**：`bool(keep.any())` 把 CUDA 张量转换为 Python bool，会在 CFA 热路径同步 CPU/GPU；全量训练会重复大量 batch。
- **建议修复**：避免在该路径读取 CUDA tensor 的 Python bool，或用无同步的空索引逻辑处理。
- **若跳过**：结果不变，但每步可能产生不必要的同步等待。

## 本轮训练

- `BARV coarse visual scorer`：seed 42，30 epoch，train_core 59,992 queries，dev 6,657，test 4,998；best dev AUROC 0.9974（epoch 29），test AUROC 0.9960。
- `geometry-only` 对照：相同 train/dev/test，test AUROC 0.9942、balanced accuracy 0.9756。
- 所有六个图像 split 的两两交集均为 0；cache 统计为 28,545 张图、85,373 queries、44,923 正例和 40,450 规则负例。
- 训练与对照结果、命令细节和限制见 `barv/RESULTS.md`。这只是 cache 上的 baseline pilot，不是 BARV 完整策略训练或论文结果。

## 做了什么

- 只读检查重点交接文档、GLP 关系头/训练验证路径、BARV 数据与训练路径；保留开始时已存在的三个未提交文件改动。
- 在临时 NumPy 2.2.6 overlay 下运行 `tests/test_glp_field.py` 中 25 个测试函数，25/25 通过；默认环境缺少 pytest，因此使用直接函数调用。
- 训练并评估 cache 上的 BARV coarse visual scorer 和 geometry-only 对照，保存到 `runs/barv/pilot_seed42/`。
- `git diff --check` 通过；没有改动源码、配置或用户已有工作区文件，没有 commit/push。

## 没做什么

- 没有续训 `glp_full_e4`：旧 checkpoint 引用了当前不可导入的 `mobileclip.*` 模块；且原训练配置 warmup/增强视界不适合作为可解释对照。
- 没有训练 BARV teacher/MVE 或跑 ROI benchmark：当前源 PSG JSON 与 COCO 图像目录缺失，完整 ROI 流程不能复现；geometry-only pilot 已显示应先处理标签捷径。
- 没有在当前仓库对所有 Ultralytics 通用检测、分割、姿态和部署代码逐文件审计；本次聚焦研究主线和交接风险。
- 没有安装/修改全局 Python 包；训练使用临时 overlay。当前默认环境的 NumPy 1.26.4 与 OpenCV 4.13.0.92 不兼容，标准 pytest runner 也未安装。

## 待确认

- 是否能恢复 PSG JSON、COCO 图片与许可信息，以便从源数据重建 BARV cache。
- 是否把 BARV 定为主研究线；需先修正 geometry-only shortcut，再过 WP3 的 ROI 信息增益和真实延迟门槛。
- GLP 是否继续：先解决 MobileCLIP checkpoint 加载依赖，并决定是否有适配 RA-4M 类别域的一级检测器。

## 验证到哪一步

| 项 | 结果 |
|---|---|
| `nvidia-smi -L` | RTX 4080 SUPER |
| `tests/test_glp_field.py` 的直接函数运行 | 25/25 通过；默认环境没有 `pytest` |
| BARV coarse scorer | 30 epoch 完成；test AUROC 0.9960 |
| BARV geometry-only 对照 | 完成；test AUROC 0.9942；六个 split 图像交集为 0 |
| GLP checkpoint load | 失败：`ModuleNotFoundError: No module named 'mobileclip'` |
| `git diff --check` | 通过 |

## 风险与遗留

- BARV test 指标高度可能由几何规则决定；恢复原始标注后应做几何匹配人工审计，不能将当前高分写成视觉关系性能。
- 缓存只支持当前特征级训练；原始 PSG/COCO 数据缺失，无法完整复现或评测 ROI 执行。
- 工作区开始时已有 `WORKLOG.md`、`tests/test_glp_field.py`、`ultralytics/models/yolovo/relate/head.py` 未提交修改；本轮未改写它们。

## 更新 2026-11-06 23:30（复核纠正）

- **Must Fix #4「GLP 中断 checkpoint 当前不能在此环境加载」不成立**：`ModuleNotFoundError: No module named 'mobileclip'` 只在裸 `torch.load` 时出现。仓库自身的加载路径会先 import `ultralytics.models.yolovo.text_tower.mobileclip.factory`（它把 `_vendor/` 注入 `sys.path`），之后 `runs/test/glp_full_e4/weights/{last,train-checkpoint}.pt` 均可正常加载（实测 epoch=0、V=10,370、含 cls_proj/geo_res_proj）。续训已于 11-06 实际完成（`glp_full_e4_resume`），方法与结果见 `worklog/2026-11-06-2130-glp-audit-resume-full-run.md`。
- **Must Fix #3 中「`data/coco/images/` 当前缺失」不成立**：目录存在，7,153 张图（1.2 GB）；`data/psg/psg_train_val.json` 缺失属实（`data/psg/` 为空）。
- **Must Fix #2 补充**：`glp_full_e4` 的 epoch 1 val 还有一个更根本的问题：训练用 10,370 词表，而 `full_monitor_v5/labels` 的 R 行 id 是 v5 的 2,809 词表重映射过的（1,042/9,171 行 id 不同），该行 results.csv 数字按错标签计算。正确标签下 epoch 1 的 pc_R@50 为 0.4303（`reports/glp_full_e4_epoch1_predcls_full_monitor.json`）。
- **#1 几何捷径补充量化**（test 4,998 条）：单个标量 IoU 的 AUROC=0.9655、框间距 0.9650、「框是否重叠」0.9519；P(框间距=0｜正)=0.961、P(…｜负)=0.054。另：contact 动作的裁剪框尺寸直接由 gap 构造（`barv/engine.py`），第 4 个 mask 通道对其池化，视觉特征层面也编码了标签。pilot 未应用 `balance_positives`（n_train=59,992 为全量 train_core）。

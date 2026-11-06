# GLP/BARV 代码与交接文档审计 + 全量实验续跑

- 时间：2026-11-06 21:00 起（本文档写于 23:20，全量实验仍在跑，结果见「更新」小节）
- 范围：`ultralytics/models/yolovo/relate/*`（head/field/trainer/dataset/val/wiring/field_metrics）、`barv/*`、交接文档（WORKLOG.md、GLP_*.md、新手指南.txt、barv/README.md、worklog/2026-11-06-code-audit-barv-pilot.md）、全量 RA-4M 实验 `glp_full_e4`
- 结论：部分完成。审计完成；全量实验已从 epoch 1 的断点续跑到 epoch 4（`runs/test/glp_full_e4_resume`，预计 2026-11-07 10:00 前后结束）；发现的「需重训」类 bug 未修（会改变实验口径）。
- 环境：RTX 4080 SUPER 32 GB（`nvidia-smi -L` 实测）；`/opt/conda/bin/python` 3.11 + torch 2.5.1+cu124；必须 `LD_LIBRARY_PATH=/root/rivermind-data/cudnn9/nvidia/cudnn/lib`（train.py 自带）且 `PYTHONPATH=/tmp/yolovo-audit-numpy2`（numpy 2.2.6 overlay；默认环境 numpy 1.26.4 已损坏：`torch.from_numpy` 报 "Numpy is not available"，cv2 导不进）。pytest 未安装。

## 做了什么

### 1. 审计方式

- 5 路并行只读子代理（head/field、trainer/dataset、val/metrics、交接文档、BARV），其中 4 路在约 20 分钟后被平台以「no progress」中止，但其中间推理已完整留存，我逐条复核；head/field 一路重新派发（精简版）并成功返回。
- 我自己做的实测：25 项单测（无 pytest，直接函数调用）、`_mix_partners`/`box_token_drop`/`geo_res_proj`/空图 NaN/`torch.maximum` 梯度/NCE 多谓词分母 等 10 项 CPU 小实验（`/root/rivermind-data/scratch_audit/audit1.py`）、全量数据 700 步 timedebug + py-spy 采样剖析、BARV cache 几何捷径复算、监控集标签 id 全量比对。

### 2. 代码改动（全部在全量实验启动前完成，实验用的就是这份代码）

| 文件 | 改动 | 是否改变训练数值 |
|---|---|---|
| `relate/trainer.py` | 新增断点续训：`YOLOVO_RESUME=<train-checkpoint.pt>` → `setup_model` 返回 ckpt，父类 `resume_training` 恢复 optimizer/start_epoch/best_fitness；兼容旧格式（整模型 half，只取关系层参数）与新格式 `rel_state`（关系层 fp32 state_dict）；`YOLOVO_RESUME_RESET_BEST=1` 清零 best_fitness；续训时备份/恢复 results.csv | 否 |
| `relate/trainer.py` | `train-checkpoint.pt` 改存关系层 fp32 `rel_state` + optimizer + `ema_updates`（351 MB，原 1.1 GB 整模型 half） | 否 |
| `relate/trainer.py` | EMA 改为只跟踪关系层的 `_RelateOnlyEMA`（原 ModelEMA 每步遍历整模型 state_dict，py-spy 占主线程 7.9%） | 否（冻结层 EMA 恒等） |
| `relate/trainer.py` | 训练期 `FieldMetricAccumulator` 改为每 `YOLOVO_FIELD_METRICS_EVERY`（默认 8）个 batch 采样一次（原逐图全量，占 7.4%，纯日志） | 否（只影响 `[FieldMetrics]` 日志行的样本量） |
| `relate/head.py` | swap hinge 由逐三元组 Python 循环改为向量化（原每条 GT 约 5~7 次 CUDA 同步，batch 32 约 2,000 次/步） | 否：60 组随机用例与原实现最大差 1.5e-8（`scratch_audit/swap_equiv.py`） |
| `relate/field_metrics.py` | 去掉每图一次的 `int(slot_has.sum())` 同步（`n_pred` 恒等于 GT 有序对数，直接加 Python int） | 否 |
| `relate/wiring.py` | 谓词词表编码时把文本塔临时搬到 GPU（31k 条文本原在 CPU 跑 6~7 分钟/次启动） | 浮点舍入级差异（tripwire 容差 1e-3） |
| `relate/val.py:407` | SGDet 预测框高度归一化 `wh[:,1]/iw` → `/ih`（640 方图下两者相等，不影响历史数字） | 否（rect=False） |
| `weights/primary_dior20_best.pt` | 一级模型从 `/root/rivermind-fs/.Trash-0/files/code/best.pt` 备份一份（原路径在回收站，随时可能消失；train.py 默认路径未改） | — |

实测效果：全量数据训练步速 1.52 s/it（原 2080 Ti）→ 1.21 s/it（4080，改前）→ 0.90~0.95 s/it（改后），GPU 利用率 32% → 60% 左右。

### 3. 全量实验续跑（用户要求「跑全量实验」）

- 原 `glp_full_e4`（WORKLOG 1704 行起）：epoch 1 完成（11-05 19:22），epoch 2 跑到 3,130/14,760 步后进程消失（`/tmp/full_e4.log` 止于 11-05 20:46，无报错记录；疑与机器从 2080 Ti 换到 4080 SUPER 有关），没有续训机制。
- **发现并绕开一个口径错误**：`RA-4M/data_full_train_monitor_v5.yaml` 用全量 10,370 谓词词表训练，但其 val `full_monitor_v5/labels` 的 R 行 id 是按 v5 的 2,809 词表重映射过的（只有前 84 个 id 两套词表同名；1,042/9,171 行 id 不同，另有 23 行被 v5 丢弃）。原 epoch 1 的 val 指标是对约 11% 错标签算的。
- 续跑配置：`YOLOVO_RESUME=runs/test/glp_full_e4/weights/train-checkpoint.pt YOLOVO_RESUME_RESET_BEST=1 YOLOVO_EPOCHS=4 YOLOVO_NAME=glp_full_e4_resume YOLOVO_CACHE=0 YOLOVO_RELATE_DATA=RA-4M/data_full_monitor.yaml python train.py`，日志 `logs/full_e4_resume.log`。
  - train 仍是 `images/train`（472,344 图，原词表 id），词表与原 run 逐字相同 → `labels/train/labels.relate.cache` 命中，无需重扫。
  - val 换成 `full_monitor/images`（同 1,024 张图，标签 = `labels/val` 原始 id，9,194 行 / 436 谓词）。
  - 关系头参数从旧 ckpt（half）恢复，optimizer 状态恢复，从 epoch 2 继续，lr/warmup 调度与原 run 连续（warmup 按全局步数计）。
- 旧 ckpt（epoch 1 EMA，`glp_full_e4/weights/last.pt`）在两套 val 上的 PredCls 复评（`reports/glp_full_e4_epoch1_predcls_*.json`）：

| val 标签 | pc_R@50 | pc_mR@50 | pc_F1@50 | pc_predicate_acc | pc_pair_acc | pc_pair_acc@3 | n_gt |
|---|---:|---:|---:|---:|---:|---:|---:|
| monitor_v5（错标 11%，= 原 results.csv 口径） | 0.42623 | 0.03551 | 0.06557 | 0.43518 | 0.53700 | 0.74193 | 9171 |
| full_monitor（正确标签，续跑用） | **0.43028** | **0.04065** | **0.07428** | **0.43920** | **0.54267** | **0.74990** | 9194 |

monitor_v5 行与原 `results.csv` 逐位一致，证明复评管线与训练内验证器同口径。

### 4. 文档改动

- 本文档；根目录 `progress.md`（新建，三段式进度）；`README.md` 加 30 行索引；`GLP_RESULTS.md` §5 的「进行中」改为实际状态；`worklog/2026-11-06-code-audit-barv-pilot.md` 追加「更新」小节纠正其 #3/#4 两条结论。

## 审计结论

### A. 确定的 bug（按严重度）

| # | 位置 | 问题 | 影响 | 是否需重训 |
|---|---|---|---|---|
| A1 | `RA-4M/data_full_train_monitor_v5.yaml` + `full_monitor_v5/labels` | 全量词表训练、v5 重映射 id 的 val（见上） | `glp_full_e4` epoch 1 的 val 行按错标签算；WORKLOG 1724 行「与 v6/v8 监控口径完全一致」不成立；且 2,809 vs 10,370 词表本身就是第二个变量，「单一变量对照」不成立 | 已绕开（续跑换 val） |
| A2 | `head.py:844-855` | rel focal 损失只在第二阶段留下的 128 对上算，`loss.py:7` 与 RA 论文 §B.1 都是几何阶段的 400 对；被第二阶段淘汰的 129~400 名永远拿不到负监督，128 槽里负例全是它自己选进来的（自我确认回路） | 采样器 relatedness 训练目标偏 | 是 |
| A3 | `head.py:630` + `734/780/816` | `field_fuse='max'` 下 GLP 场头的梯度只来自 sigmoid×0.25（bg×0.05 只在 field>raw 的格子）；nce/swap 读 `logits_raw`，不经过场头。CPU 逐项反传实测：`field_read/q_res_proj` 只在 sigmoid 项有梯度；logit≈5 时正样本梯度 0.25×(1−σ)≈0.0017 | **场头在当前训练配方下几乎拿不到排序监督**，与 WORKLOG 1536 行实测「raw 比 field 更高」一致：GLP 的主创新点目前没有被有效训练，而不是「场没用」 | 是 |
| A4 | `head.py:99-100` vs `114-115` | `torch.maximum(logits, fused)` 两边精度不同：全词表 cos 是 autocast fp16 matmul，候选列 cos 是 fp32 逐元素；场头零初始化时 field==raw，胜负由 2.4e-4 的舍入噪声决定，梯度随机路由 | 训练早期两头各随机丢一半格子的 sigmoid 梯度 | 否（口径变化很小）；修复：matmul 用 fp32 |
| A5 | `head.py:617-620` | 场残差 cap 以 ‖q‖（LayerNorm 后 ≈22.6）为基准，却同样加到无 LayerNorm 的 `q_spa` 上；实测 cap=2.26 → Δcos≤0.0996、Δlogit≤1.67；若 ‖q_spa‖≪22，同一 res 会主导空间专家 | 场头对两个专家的影响不对称且不可控 | 是 |
| A6 | `head.py:838` vs `467` | geo BCE 把强制入候选的反向对标成 0，同时又要求采样器保留它们 | 几何预打分器目标自相矛盾（n=20 时正例仅 2.4%） | 是 |
| A7 | `ontology.py:33-36, 21-25` | 10,370 词表只有 15 个对称词、8 对反义词；其余近万谓词被 swap hinge 强加方向；NCE 里 cos≥0.9 的同义列（中位 240 个/谓词）仍以 0.25 权重进分母 ≈ 60 个满权负例 | 同义词得分与 GT 相当时 NCE 下限 ≈ ln61≈4.1 nat；精确字符串匹配指标系统性偏低 | 是 |
| A8 | `head.py:757/775/786/820/839/855` + `trainer.py` | 七项损失都是「图内均值再图间均值」，GT 多的图每条三元组权重按 1/G 稀释；RA 按 anchor 平均；`head.py:691` docstring 写「sum form」 | 与 RA 配方不一致 | 是 |
| A9 | `ontology.py:56-57,80` + `head.py:699` | `pos_mask` 建了从不使用；`neg_lw`(430 MB fp32)+`inverse_mask`+`pos_mask`(各 107 MB) 全部常驻 GPU，且是普通属性而非 buffer：被 deepcopy 进 EMA、pickle 进每个 ckpt（这就是 best/last.pt 近 1 GB 的原因） | 显存 ≈645 MB×2、ckpt 体积、每轮保存时显存尖峰 | 否 |
| A10 | `relate/loss.py` + `tasks.py:2916-2931` | 不可达死代码（训练态 head 恒返回 `rel: None`），被触发即崩（引用不存在的 `head.replay_image`/`predicate_bank`） | 误导阅读 | 否 |
| A11 | `val.py:407` | SGDet 预测框高度除以 `iw`（已修） | rect=True 或非方图时才会出错 | 否 |
| A12 | `train.py:45` | 默认 `YOLOVO_RELATE_DATA=RA-4M/data_proxy50k.yaml`（旧 10,370 词表的 proxy50k）、`YOLOVO_EPOCHS=100`：新人不带环境变量直接 `python train.py` 会起一个 100 轮的旧口径实验；else 分支的 `/root/datasets/M3FD_split` 不存在 | 交接风险 | 否 |
| A13 | 工作树 `head.py`（未提交） | v8 三项机制（`cfa_prob=0.5`、`box_token_dropout=0.3`、`geo_res_proj`）默认开启，而 WORKLOG 1720 行结论是 v8 整体负收益、不作主线；`glp_full_e4` 却是用这份代码训的 | 若用已提交的 HEAD 代码评估 full_e4 ckpt，`geo_res_proj` 不会被调用，结果不同；这份 diff 必须与 full_e4 结果一起定版 | — |
| A14 | `trainer.py`（原 save_model）/ `validator` | 验证时 `model.half()` 原地把 EMA 转 fp16 再 `.float()`，只恢复 dtype 不恢复精度，W/W_obj/alpha 每轮被量化一次；原 save_model 里「W 还原 FP32」的循环因此无效 | 估算 logit 误差 ≤0.03，低 | 否 |
| A15 | `barv/psg.py`（负例构造） | 负例 = mask gap > 阈值（如 holding 451 px），正例几乎都接触：test 集上单个标量 IoU 的 AUROC=0.9655、框间距 0.9650、「框是否重叠」0.9519；P(框间距=0｜正)=0.961 vs P(…｜负)=0.054；几何 LR 0.9942 vs 视觉模型 0.9960 | BARV 现有分数基本由标签构造规则决定，不能证明视觉证据或主动 ROI 的价值 | 需重建负例 |
| A16 | `barv/engine.py`（contact action） | contact 动作的裁剪框尺寸由 gap 构造，第 4 个 mask 通道对该框池化：特征里直接编码了标签 | 同上，视觉特征层面的泄漏 | 需改 |
| A17 | `barv/train.py` `train_evidence_update` | 早停用训练张量上的 AUROC 自选；BCE 缺另一路径有的 `pos_weight` | 该阶段尚未跑 | — |
| A18 | `barv/prepare.py` | `n_missing_images` 从不递增，`_collate` 静默丢缺图 | 统计失真 | 否 |
| A19 | `barv/experiment.py` | pilot 未应用 `balance_positives`（n_train=59,992 = 全量 train_core；应用会丢约 6,685 个正例）；`run_one` 依赖的 `teacher_fine.npz` 无产出入口；无 CLI | RESULTS.md 的复现命令对应不到实际入口 | — |
| A20 | `ontology.py:57-78` | `inverse_mask`/`sym` 固定建在 CPU，`W` 在 CUDA 上时 `masked_fill` 报 device mismatch（子代理脚本实测触发）；正常路径 attach 发生在 `.to(device)` 之前所以没暴露 | 模型搬到 GPU 后再 `install_vocabulary` 会崩 | 否（`device=W.device` 两处即可） |

### B. 优化点（速度 / 显存）

py-spy 对全量训练主线程采样（改前）：`_compute_loss` 46%（其中 `_losses` 27%，`_forward_pairs` 9%，`field_metrics.update` 7.4%）、EMA `update` 7.9%、backward 0.46 s/步、forward 0.63 s/步，GPU 利用率 32%。

| # | 位置 | 内容 | 状态 |
|---|---|---|---|
| B1 | `head.py` swap hinge | 逐三元组 CUDA 同步 | 已向量化 |
| B2 | `trainer.py` EMA | 整模型 state_dict 遍历 | 已改关系层子集 |
| B3 | `trainer.py` field metrics | 逐图全量统计 + 每图一次同步 | 已采样 + 去同步 |
| B4 | `wiring.py` | 31k 文本 CPU 编码 6~7 分钟/启动 | 已搬 GPU；进一步可按词表哈希缓存 W |
| B5 | `trainer._compute_loss` | 逐图 Python 循环调 `_losses`（32 次/步）：NCE/sigmoid/bg/geo/rel 全部可整批算，预计再省 20~30% 主线程 | 未做（数值等价性验证成本高，不在实验途中改） |
| B6 | `head.py:199-205` `_gt_grid` | 逐 GT 两次逐元素 CUDA 写（约 600 kernel/步）→ 一次 `index_put_` | 未做 |
| B7 | `head.py:713-720` | `gt_hot` 在 `_losses` 重建，`_forward_pairs` 已算出 `gt_slots` | 未做 |
| B8 | `field.py` / `deformable.py` | 填充槽位全程参与计算，可按 valid 压缩 | 未做 |
| B9 | `ontology.py` | `neg_lw` 行按需现算（`W[g]@W.T`）、`pos_mask` 不上 GPU、tables 不进 ckpt：省 ≈645 MB 显存 ×2、ckpt 从 ~1 GB 降到 ~300 MB | 未做 |
| B10 | `val.py` | 验证逐图调 `score()`（batch=1，SGDet+PredCls 各一次）、每次 `self.float()` 遍历参数、`fmap_proj` 重算；可按图批处理 | 未做（验证 1,024 图约 2 分钟，可接受） |
| B11 | `head.py:783` sigmoid aux | V=10,370 时每个负例权重 1e-4，被易负例主导；可只在 top-512 hard 列上算（RA 口径） | 未做（需重训） |
| B12 | `trainer.py` dataloader | `pin_memory=False`、`workers=8`（88 核）；数据等待在剖析中不显著（<2%），暂不改 | 未做 |

### C. 评测口径（可对外 / 仅内部）

- `pc_R@K` 是全图总命中/总 GT（micro），文献是逐图 recall 再平均（macro）；本 val 61.8% 的图 pair 数 ≤20，K 基本不截断，`pc_R@50≈pc_predicate_acc`。→ 仅内部监控。
- `pc_mR@K` 排除 val 中不出现的谓词类（SGB 计 0）；`sup_*` 分桶是本仓库自定义。→ 仅内部。
- `pc_pair_acc@k` 是逐对 top-k 命中率，不是 ngR@K（WORKLOG 1524 行标注有误）。
- `pair_existence_AP`：正例取 fused 排序首个匹配行而非最高 exist 分（仅 SGDet 受影响）；相等分数下 argsort 不稳定。
- SGDet 三元组分数不乘检测置信度；一级模型是 DIOR 20 类、数据是 RA-4M 497 类，SGDet 数字（R@50≈0.003）反映检测域差，不能评价关系头。
- `predicate_acc` 结构上限 0.8104（多谓词对只出一个 argmax）。
- 自定义场指标 FMA 被框面积支配（均匀场基线 ≈−0.67），不能读成对齐好坏。

### D. 交接文档审核

| 文档 | 问题 | 处理 |
|---|---|---|
| `README.md` | 只有 `# tuwen`，新人无入口 | 已加索引 |
| `GLP_RESULTS.md` §5 | 写 100 epoch 全量实验「进行中」，实际 `glp_full_v1_e100` 当日即被放弃（results 空）；`logs/full_v2.log` 不存在；机器写 2080 Ti；「23 项单测」现为 25 | §5 已改为实际状态并指向本文档 |
| `GLP_INNOVATIONS_IMPLEMENTATION.md` | 引用的 `scripts/*.py`、`reports/*.md`、`weights/yolovo11n_glp.pt`、`runs/glp/*`、SUN/SpatialSense 数据均不存在；同一文档写「9 项单测」又写「23 项」；Phase 0 的 C-AUC 数字无任何产物支撑；仍写 GPU 不可用 | 未改（整份是 11-01 的方案稿，建议标注「历史方案，资产已不存在」） |
| `GLP_V2_RELATIONAL_FIELD.md` §6 | 说删除 `FieldConfig.mode/lam`，代码仍保留 | 未改 |
| `WORKLOG.md` | 1724 行「监控口径与 v6/v8 完全一致」不成立（A1）；1710 行「同轮对齐 v6（ep12，两者都是各自峰值）」：v8 的 pc_R@50 峰值在 ep3 不是 ep12；1757 行 close_mosaic 风险为伪风险（relate 数据集本来就关掉 mosaic/翻转/仿射，args.yaml 里的 `mosaic=1.0 fliplr=0.5` 不生效）；11-02 某节写「工作区不是 git 仓库」与 11-01 的提交/推送记录矛盾；文件开头的总结论自 11-01 后未更新 | 未改正文（历史日志）；本文档记录 |
| `新手指南.txt` | 写 numpy<2.0、Python 3.10，实际需 numpy 2.2.6 overlay、Python 3.11；完全不提 GLP/BARV 两条研究线；默认 `python train.py` 行为与指南描述不符（A12） | 未改 |
| `barv/README.md` | 引用的 `scripts/audit_data.py` 等 4 个脚本与 `tests/test_barv.py` 不存在；「8 个谓词」与配置/cache 的 9 个不一致（8 来自 `barv_cache_2k_new2` 丢了 standing on）；8 份 `configs/barv_*.yaml` 的差异与用途无说明；`barv_psg_long40.yaml` 名含 seed 303 实为 42；配置键 `p99` 已不是代码口径 | 未改 |
| `worklog/2026-11-06-code-audit-barv-pilot.md` | #4「checkpoint 不能加载（ModuleNotFoundError: mobileclip）」不成立：仓库自身加载路径（先 import `text_tower.mobileclip.factory` 注入 `_vendor`）可正常加载，裸 `torch.load` 才会报错；#3「`data/coco/images/` 缺失」不成立：目录存在，7,153 张图；`data/psg/` 为空属实 | 已追加更新小节 |

### E. git 工作区

- 未提交改动：`WORKLOG.md`、`tests/test_glp_field.py`、`head.py`（v8 三项 + 本次 swap 向量化）、`trainer.py`、`field_metrics.py`、`val.py`、`wiring.py`。`glp_full_e4(_resume)` 用的就是这份工作树，**建议作为一个 commit 定版**（待确认）。
- 未跟踪：`barv/RESULTS.md`、`worklog/`、`progress.md`（应提交）；`mobileclip_s0_pt_7z_parts/`（400 MB zip，应 ignore 或移出）。
- 被 ignore 的大文件：`images.tar.gz` 1.8 GB（根目录）、`2609.12552v1.pdf` 11 MB、`runs/` 34 GB、`logs/error_report_*.txt` 100 多个（仓库自带的异常上报钩子产物）。未删任何东西。

## 没做什么

- 没修 A2~A8、B11（需重训才能体现，且会改变正在跑的全量实验口径）。建议在全量实验出结果后，按「A3 → A2 → A5 → A4」顺序分别消融。
- 没修 A9/A10/B5~B10（不改数值的重构）：全量实验进行中，不在实验途中改训练代码；可在实验结束后做。
- 没改 `train.py` 默认值（A12）、没改一级模型路径：只备份了权重到 `weights/primary_dior20_best.pt`。
- 没重写 `GLP_INNOVATIONS_IMPLEMENTATION.md`、`新手指南.txt`、`barv/README.md`、WORKLOG 正文：属历史文档，只在本文档登记问题，是否重写待决策。
- 没 commit / push：工作树含用户之前未提交的 v8 改动，是否一并提交需用户拍板。
- 没跑全量 val（24,964 图）：仍用 1,024 图监控集；实验结束后可对 best.pt 单独跑。
- 没重跑 v6/v8 以消除词表混淆：全量 run 用 10,370 词表，v6/v8 用 2,809 词表，两者不是单变量对照；若要严格对照需用 `data_proxy50k.yaml`（原词表）重跑 50k 基线（约 10 h）。
- 没验证 SGDet 口径下 `val.py:407` 修复的数值影响（当前 rect=False、方图，影响为 0）。
- BARV：没重建负例、没跑 teacher/MVE、没恢复 PSG JSON（`data/psg/` 为空）。
- 子代理 4 路被平台中止，其「最终报告」缺失；本文档的相应结论来自其留存的中间推理 + 我的复核，复核不到的（例如 val 侧 `pair_existence_AP` 的 tie-break 细节）已按「疑似」口径写。

## 待确认

1. 是否把当前工作树（v8 三项 + 本次改动）作为 `glp_full_e4_resume` 的代码定版提交？
2. 全量实验结束后，是否按 A3（场头梯度饥饿）优先做消融？这是 GLP 主创新点是否成立的关键，当前所有「场头无效/有害」的结论都建立在场头几乎没被训练的前提上。
3. v6/v8 vs 全量 的对照是否需要补一个「原词表 50k」基线以消除词表变量？
4. BARV 是否继续：先重建难负例（同类别配对 + IoU>0 分层采样，或 PSG 同图未标注对），再谈 ROI 价值。
5. `GLP_INNOVATIONS_IMPLEMENTATION.md`/`新手指南.txt`/`barv/README.md` 是重写还是加「历史文档」标注？

## 验证到哪一步

| 项 | 结果 |
|---|---|
| `nvidia-smi -L` | RTX 4080 SUPER（32 GB） |
| `tests/test_glp_field.py` 25 项（直接函数调用，CPU） | 改动前 25/25；改动后 25/25 |
| swap 向量化等价性（60 组随机用例，37 组非零 hinge） | max \|diff\| = 1.49e-8 |
| `scratch_audit/audit1.py` 10 项 CPU 实验 | 空图无 NaN；`_mix_partners` 配对正确但 64% 伙伴跨图；`box_token_drop` 生效；`geo_res_proj` 末层零初始化；V1 mean-shift 为恒等（`q_center==mu`，`log_omega` 不是混合密度）；`torch.maximum` 平局各 0.5 梯度；多谓词槽位 NCE 分母含另一 GT 列 |
| 1,024 图 1 epoch 冒烟（改后代码） | pc_R@50=0.0773、pc_predicate_acc=0.0855，与改前冒烟逐位一致 |
| 续训冒烟（1 epoch → `YOLOVO_RESUME` 续第 2 epoch） | 从 epoch 2/2 起跑，results.csv 两行连续，nce 7.03→6.38、pc_R@50 0.0773→0.0935 |
| 旧格式 ckpt（`glp_full_e4/train-checkpoint.pt`，整模型 half）续训 | 正常：参数恢复、optimizer 恢复、「Resuming training … from epoch 2 to 4」，损失曲线接上原 epoch 2 的数值（geo 0.605 / rel 0.0743 / nce 2.18） |
| epoch-1 ckpt 复评 | monitor_v5：pc_R@50 0.42623（= results.csv）；full_monitor：0.43028 |
| 监控集标签比对 | 1,024 图同名；`full_monitor/labels` = `labels/val`（0 文件差异）；monitor_v5 有 548 个文件不同，1,042/9,171 行 id 不同 |
| BARV 几何捷径复算（test 4,998 条） | AUROC(IoU)=0.9655、AUROC(−box gap)=0.9650、AUROC(IoU>0)=0.9519 |
| 全量实验 `glp_full_e4_resume` | 23:17 时 epoch 2/4 进度 19%，1.06 it/s，无报错；预计 11-07 02:20 出 epoch 2 行 |

## 风险与遗留

- 全量实验仍在跑；若中断，`runs/test/glp_full_e4_resume/weights/train-checkpoint.pt`（新格式）可用同一命令 + `YOLOVO_RESUME=<该文件>` 续跑；注意 `YOLOVO_RESUME_RESET_BEST` 续跑新格式时不要再加。
- 全量 run 的 lr 调度（4 轮、warmup 3 轮）使最大 lr 只到 0.505e-3（epoch 2 末），epoch 3 平台 0.2575e-3：这是原 run 既定的、可解释性差的调度，续跑未改以保持连续。
- 本次改动的 swap/EMA/field-metrics 等价性在 CPU 单测与 1,024 图冒烟上验证，全量 GPU 上的训练曲线与原 run 的 epoch 2 前 3,000 步数值吻合（0.6012/0.07432/2.173/0.00725 vs 原 0.6013/0.07408/2.17/0.00725），但没有做逐位对比。
- 原 `glp_full_e4` 目录保留未动（epoch 1 的 best/last/train-checkpoint 与错标 val 的 results.csv）。
- `scratch_audit/`（仓库外，`/root/rivermind-data/scratch_audit/`）保留了本次所有实验脚本与 py-spy 原始采样，可删。

## 更新 2026-11-07 01:00（提交 GitHub / 论文基线同口径对比 / BARV 续做准备）

### 做了什么

- **提交并推送**：`2e802bf feat(relate)…`、`c198a8c docs…` 已推到 `xu967/tuwen` main（SSH 可用；HTTPS 到 github.com/huggingface.co 被墙，`hf-mirror.com`、pypi 可达；环境变量里的代理 127.0.0.1:7897 没有进程在听）。`mobileclip_s0_pt_7z_parts/` 加入 .gitignore。
- **论文基线直接对比**（用户要求 2）：把论文发布的 RelateAnything ViT-S/16+ 权重（`assets/relsgg-vits16plus/`，含 DINOv3 backbone）在本机跑起来，与我们的模型在同一 1,024 张 RA-4M val 图、GT 框、同一 10,370 词表、同一评测代码下对比。结果与解读见 `reports/baseline_cmp/README.md`；关键数字：只给框时 R@50 官方 26.6 vs 我们 23.2，mR@50 13.1 vs 1.0；给 GT 类别后我们 R@50 44.8 但 mR@50 仍只有 4.1。
  - 新增 `tools/sgg_protocol_eval.py`（论文 §5.1 口径：逐图 R@K、按谓词 mR@K、LVIS 式三桶）、`tools/dump_predcls_ours.py`（钩住训练内验证器导出逐图预测，与 results.csv 同口径，校验 pc_R@50 逐位相同）、`tools/run_ra_official_predcls.py`（官方模型推理）。
  - 官方代码 `git@github.com:Maelic/RelateAnything.git` 克隆到 `/root/rivermind-data/RelateAnything`（SSH 克隆 ~45 分钟）；transformers 4.57.1 装在 `/root/rivermind-data/ra_overlay`（不动 conda 主环境；需 `USE_TF=0`，否则 transformers 会去 import 损坏的 tensorflow）。发布权重是 transformers 5.x 导出的，DINOv3 的 state_dict 多一层 `model.`，脚本里重映射后 0 missing / 0 unexpected / 0 shape mismatch；tokenizer 只有 fast 格式，`PredicateTextStudent._ensure_tokenizer` 在脚本内 patch 成 `CLIPTokenizerFast`（官方 backbone 加载会重新 import transformers，模块级 patch 会丢）。
  - 验证：text student 对词表已知串的编码与 ckpt 的 W cos=1.0（8/8）；20 图冒烟后跑全量 1,024 图 82 s。
- **BARV 续做准备**（用户要求 3）：源数据路线已打通：`JosephZ/psg_train_sg`（HF，PSG train 46,563 图，含图像字节 + 物体 id/类别/框 + 关系三元组，22.8 GB/46 片）可从 hf-mirror 下载（~3 MB/s），`/root/rivermind-data/psg_hf/download.sh` 正在后台顺序下载（日志 `download.log`）；第 0 片已核对 schema。COCO panoptic PNG 在本地 `data/coco/panoptic_annotations_trainval2017.zip`。

### 没做什么

- 没有重跑 v6/v8 与官方模型的对比（它们是 2,809 词表）；没有在 VG150/PSG test 上复现论文表 4（本机无这两个 test 集；下载 PSG 后可补 PSG test 的 GT 框协议）。
- 没有评估官方模型的 SGDet（无检测器对齐）；没有做 `pair_weight=0` 变体。
- BARV 难负例的构造代码未写、cache 未重建、模型未重训：需要先下完 22.8 GB（约 2~3 h）并等 GPU（全量 GLP 训练到 11-07 10:00 左右）。方案已定（见下）。
- `tools/run_ra_official_predcls.py` 依赖仓库外的 `/root/rivermind-data/RelateAnything` 与 `ra_overlay`，未纳入仓库（第三方代码 + 164 MB 依赖）。

### BARV 续做方案（待数据下完、GPU 空出后执行）

1. **难负例规则**（`barv/psg.py` 新增 source=`contact_other` / `contact_unlabeled`）：对谓词 p 的候选 (s,o)：类别组合在 RULES[p] 允许集合内；**框接触**（IoU>0 或框间距 ≤ 8 px，替代现有 `gap > 451 px` 的「远离」规则）；(s,o,p) 未标注；优先取 (s,o) 标注了其他谓词 q≠p 的对（`contact_other`，几何与正例同分布且确定有关系），其次取同图未标注的接触对（`contact_unlabeled`，带 PSG 稀疏标注噪声，单独打 source 以便消融）。按类别组合与 IoU 分层配额，使正负在 (IoU, 框间距, 面积比) 上的边缘分布匹配（分层后重算几何 LR 的 AUROC 作为「捷径残留」指标，目标 <0.6）。
2. **特征层面去捷径**：contact 动作框尺寸不再由 gap 决定（`build_action_boxes` 第 0 个动作改为固定 `0.45·min(side)`），第 4 个 mask 通道改为接触点固定窗口。
3. 从 HF parquet 重建 `psg_train_val.json`（OpenPSG 格式：image_id、segments（类别/框/与 panoptic PNG 的 seg_id 匹配）、relations），走现有 `barv/prepare.py` 重建 cache；train/dev/test 仍按图划分。
4. 重跑 geometry-only LR、视觉 scorer（30 轮）、再做 WP3 的 ROI oracle gap；只有视觉对几何的增益稳定（≥+3 AUROC 点且 bootstrap CI 不含 0）才继续 teacher/MVE。

# GLP V1/V2 实现与训练（RA-4M）

- 时间：2026-11-01
- 范围：`ultralytics/models/yolovo/relate/`、`tests/test_glp_field.py`、`train.py`、`RA-4M/`
- 结论：进行中（V1/V2 已按全量 RA-4M + 100 epoch 启动；单测 24/24 通过）

## 做了什么

- 工作目录原先 `ultralytics/`、`tools/` 为空；从回收站
  `/root/rivermind-fs/.Trash-0/files/code/YOLOVOmodal_3871538535` 恢复代码，未覆盖现有根脚本。
- 新增 `ultralytics/models/yolovo/relate/field.py`：V1 绝对系谓词高斯场 + V2 主体参照系关系算子场
  （`local_frame/use_null_bin/use_geo`、5 sigma 点读取、背景桶、`compose_fields` AND/OR/传递）。
- `relate/head.py`：新增 `score_candidates`、`_select_candidates`、`score_all(return_field=True)`、
  field 分支与 NCE 候选列 scatter；`field_ablate` 支持 agnostic/shuffled。
- `relate/wiring.py` / `namespace.py`：`field_cfg` 透传并写入 `relate_info`；已有关系层时换配置 fail-fast。
- `relate/trainer.py`：记录 `field_diag`（gamma/delta/w_bg）+ `FieldMetrics` 日志。
- 新增 `relate/field_metrics.py`：公开 SGG 指标累计（top1/top5/macro-F1/existence P/R/F1）
  + 自定义 FMA/WBG gap/AURC/SFC/EE。
- 新增 `tests/test_glp_field.py`：23 项单测（identity、null 守恒、平移/尺度等变、V2 对象独立、
  候选打分、代数、V1/V2 head 前向反向、指标累计器、gate）。
- `train.py`：新增 `YOLOVO_RELATE_DATA` 环境变量分支，复用 `model.rel.train`；默认一级训练行为不变。
- 数据：从 `RA-4M` 的 20 类 DIOR 对齐子集（600 train / 41 val）生成 YOLO 框 + RelGen R 行，
  放在 `RA-4M/glp/`，配置 `RA-4M/data_glp.yaml`；未复制大图，只建软链。
- 环境：`pip install einops`；复制 `pytorch_grad_cam` 占位模块；复制 MobileCLIP 权重到
  `ultralytics/nn/textencoder/mobileclip/`；补 `ultralytics/data/dataset.py::DATASET_CACHE_VERSION` 别名。
- 个人 skill：在 `/root/.claude/skills/personal-skill/SKILL.md` 强调临时测试/训练脚本用完即删。
- 独立代码审核后修复 7 个真实问题：
  - V2 背景桶缩放被 LayerNorm 抵消（β 无梯度、`z` 不衰减）→ 先归一化空间权重，LayerNorm 后再乘 `1-w_bg`；
  - V2 绝对 σ clamp 破坏尺度等变 → 去掉绝对 clamp，只保留局部 σ̃ 的有界参数化；
  - WBG gap 符号与文档相反 → 改为 `neg-pos`；
  - `eval_budget/topk>128` 被 `final_budget` 截断 → eval 时同时设置两个 budget；
  - `fmap_proj` 通道数硬编码 128 → 从一级头 `cv2` 推断 P3/P4/P5 通道，失败时不再猜；
  - `compose_fields` 保留左操作数的 `w_bg/mu_local/subject_box` 陈旧元数据 → 只返回复合后的 `mu/sigma/logw`；
  - `negate` 单核空操作 → 改为显式 `mass` 的近似诊断，不再伪称高斯混合；
  - `rel.val(data=subset)` 被 `setdefault` 忽略 → 显式 data 覆盖 checkpoint 的 train args；
  - `protocol='sgcls'` 只存不用 → 在 `update_metrics` 用 GT 框重打分；`init_metrics` 不再覆盖外部设置的 protocol。

## 没做什么

- 未实现 RES/RefCOCO/CRIS/LAVT、official RA adapter、SUNRGBD/SpatialSense 评测
  - 原因：本机没有这些数据/权重，V2 文档也明确本轮搁置。
- 未训练 50 epoch、未做 3 seed、未调超参
  - 原因：先验证管线；按用户要求只固定 `batch=16, workers=8`，未改其他训练超参。
- 未实现学习风险头
  - 原因：V2 用背景桶 `w_bg` 替代，不新增 `risk_head.py`。
- 未把 RA-4M 全量 497 类/10370 谓词接进来
  - 原因：主模型 `best.pt` 是 20 类 DIOR；当前用 20 类对齐子集，避免 `W_obj`/label 越界。
- 已推送 GitHub：`git@github.com:xu967/tuwen.git`，分支 `main`
  - 独立本地仓库，未触碰 ALVLN/uav_rgb/CF-Drone；使用用户新建的 `~/.ssh/tuwen_deploy`
    （`GIT_SSH_COMMAND` 临时指定，未改 `~/.ssh/config`）。
  - 远端已有 `Initial commit`（README），已 rebase 到其之上；当前提交
    `e47cbd8`（代码）+ `b264315`（worklog）。

## 待确认

- GitHub 目标仓库 URL 与可见性；提交身份 `user.name/user.email`。
- 是否改用 RA-4M 全量 497 类（需换/重训主模型）与真实 `rel_annotations`，而不是 RelGen 几何关系。
- 是否接受 V2 背景桶 `w_bg≈0.48` 作为待研究问题，或加反塌缩正则/校准。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `tests/test_glp_field.py`（23 项，CPU） | 23/23 通过 |
| V1 `train.py` 1 epoch（batch 16 / workers 8 / 640） | 完成；`rel_geo=0.5700, rel_exist=0.0757, rel_pred=1.5113`；val F1@50=0 |
| V2 `train.py` 1 epoch（同上） | 完成；`rel_geo=0.5829, rel_exist=0.0739, rel_pred=1.5106`；val F1@50=0 |
| V1 5 epoch（batch 16 / workers 8 / 640） | 完成；epoch5 `rel_geo=0.5250, rel_exist=0.0675, rel_pred=1.4546`；val F1@50=0 |
| V2 5 epoch（修复后） | 完成；epoch5 `rel_geo=0.5221, rel_exist=0.0676, rel_pred=1.4449`；val F1@50=0 |
| 自定义指标（20 张 GT 框） | V1 top1=0.511、FMA=-0.849；V2 top1=0.356、FMA=-0.932、WBG gap=6.7e-5、AURC=0.683、EE=6.7e-8 |
| SGCls（5 张 sanity） | V1 F1@50=0.5、pred_acc=0.5；V2 F1@50=0.4、pred_acc=0.4；仅接口验证 |
| 完整 41 图 SGCls | 未跑完（双重打分过慢，已中止）；SGDet 41 图已完成 |

## 风险与遗留

- 1 epoch / 600 图远未收敛；`F1@50=0` 主要受 DIOR 主模型与 RA-4M 域差（DetPairRecall=0.122）影响，
  不能当作公开 SGG 结论。
- 1 epoch 的 V2 结果是在背景桶 LayerNorm bug 下得到的，已作废；以修复后的 5 epoch 为准。
- 修复后 V2 `w_bg_mean≈0.48`（1 epoch 0.537 → 5 epoch 0.482），不再是全弃权塌缩，但仍占约一半质量；
  FMA 为负且差于 V1，说明场没有真正对齐目标框。
- V2 `delta_norm≈30.3`（5 epoch），`read_out` 无界；后续可加输出范数约束或更小 field lr。
- `rel_swap` 恒 0（当前数据/采样无 swap 对），swap 损失未验证。
- 完整 41 图 SGCls 未跑完；5 图结果只验证接口，不能当统计结论。
- 当前 RA-4M 关系是 RelGen 几何生成（4 谓词），不是 RA-4M 真实 `rel_annotations`；结论不能外推到开放谓词。
- `ultralytics` 代码来自回收站，与当前根脚本版本需持续核对；`DATASET_CACHE_VERSION` 别名是兼容补丁。

## 更新 2026-11-01 全量 RA-4M + 100 epoch

### 做了什么

- **个人 skill**：新增硬规则「禁止多 seed」——所有实验固定 `seed=0`，不跑 seed sweep、
  不用多 seed 均值±方差。
- **关系头性能改造**（`relate/head.py`、`relate/trainer.py`）：
  - 新增 `_forward_pairs()`：整批一次前向，图像按最大框数右填充 `boxes[Nmax]`，
    用 `box_counts` 标记有效前缀；`_compute()` 保留为单图视图（eval 路径不变）。
  - `_Sampler` 已支持 `box_counts`（原本就有，只是没有传）。
  - `_select_candidates` 改为整批 `[B,K,V]`。
  - 训练器 `_compute_loss`：先整批前向，再逐图算七项损失；不再逐图调用 `_compute`。
  - `_losses` 热点向量化：`gt_hot` 用 scatter；swap 用 `sv/ov` 掩码查找；
    NCE 的 `hot @ [V,V]` 换成稀疏行 `index_add_`（10k 谓词时每图少读 430MB）。
- **全量数据接入**：
  - COCO → YOLO 框 + `R sub obj rel_id` 行，写入 `RA-4M/labels/{train,val}`。
  - 统计：train 472,344 图 / 2,668,852 框 / 4,282,531 关系；val 24,964 图 /
    140,912 框 / 226,203 关系；skipped_rels=0，missing_images=0。
  - 谓词 10,370（`rel_categories` 全量）、类别 497（`categories` 全量）。
  - `RA-4M/data_full.yaml`（train/val 全量）；`data_full_monitor.yaml`
    （train 全量 + val 取 1,024 张固定子集，用于逐 epoch 监控）；`data_full_smoke.yaml`（64 图）。
- **`W_obj` 支持 data yaml 覆盖**（`wiring.py` / `namespace.py`）：`resolve_objnames()`
  把 data yaml 的 `names` 传给 `attach_relate_layer`，解决一级模型 20/80 类与 RA-4M 497 类不一致。
- **文本编码分批**（`encode_predicate_bank`）：10,370×3 模板分 512 一批，避免 OOM。
- **单测**：新增 `test_batched_forward_matches_per_image_losses`（填充批前向 vs 单图损失，
  误差 <1e-4），共 24/24 通过。

### 实测速度

| 场景 | 结果 |
| --- | --- |
| 600 图（V=4）改造前 | 96.3 s/epoch，2.36 s/it |
| 600 图（V=4）改造后 | 38.8 s/epoch |
| 2,048 图全量口径（V=10,370） | 112.6 s/epoch（含 64 图 val），约 0.56 s/it / 35 ms 每图 |
| 推算全量 | 约 4.6 h/epoch；100 epoch 约 19 天/变体（加监控 val 约 20 天）|

### 启动

- 顺序后台任务：`glp_full_v1_e100` → `glp_full_v2_e100`，均 `batch=16, workers=8,
  imgsz=640, lr0=1e-3, seed=0, amp=true`。
- 日志：`logs/full_v1.log`、`logs/full_v2.log`、`logs/full_runs.log`。
- 首次启动要扫描 472k 标签建 `labels.relate.cache`（约 30 分钟，只发生一次）。

### 没做什么

- 没有跑完任何全量 epoch（刚启动），没有全量结果。
- 没有做最终全量 val（24,964 图）；训练期 val 用 1,024 图子集，避免每 epoch 2 小时验证。
- 没有多 seed（按用户新规则，固定 seed=0）。
- 没有改训练超参（batch/workers/imgsz/lr0/AMP 均沿用上一轮）。
- 没有提交/推送 Git（等用户确认）。
- 没有给 `build_ontology` 的 `inverse_mask` 补 device 推断（当前流程在 CPU 上 attach，
  不受影响；模型已在 GPU 上再 attach 会报 device mismatch）。

### 待确认

- 全量 100 轮约 20 天/变体、共约 40 天，是否接受这个时长。
- 训练期 val 用 1,024 图子集、最终再跑全量 24,964 图验证，是否接受这个口径。

## 更新 2026-11-01 训练进度条不显示

### 根因

- Jupyter/网页终端把 pty 尺寸报成 `0x0`；tqdm 的 `_screen_shape` 得到 `(0,0)`，
  `nrows = -1`，`display()` 里 `pos >= nrows - 1` 直接 `return False`，
  所以**每次刷新都被静默丢弃**，整条进度条完全不渲染。
- 用真 pty 复现：`tqdm(total=3)` 的 `nrows=-1`、`display()` 返回 False；
  `ioctl(TIOCSWINSZ, 40x120)` 后 `nrows=39`、进度条正常。
- 与训练代码无关：`TQDM` 的 `disable` 一直是 False，`VERBOSE=True`。

### 修复

- `train.py` 顶部新增 `_ensure_tty_size()`：检测到 stdout/stderr 的终端尺寸 `<2` 时，
  用 `fcntl.ioctl(TIOCSWINSZ)` 设成 40 行 × 120 列，并补 `COLUMNS/LINES` 环境变量。
- 终端尺寸正常时什么都不做；非 TTY（重定向到日志）不受影响。

### 验证

| 项 | 结果 |
| --- | --- |
| 真 pty 下原始 tqdm | `nrows=-1`，无输出 |
| 真 pty 下 `_ensure_tty_size()` + tqdm | 尺寸 120x40，进度条正常渲染 |
| 非 TTY 管道 | 仍输出 `\r` 分隔的条形（日志照常） |

### 没做什么

- 没有改 ultralytics 的全局 `TQDM` 类（避免影响验证/预测等其他入口）。
- 日志重定向到文件时，进度条仍是 `\r` 覆盖式，`tail -f` 看不到动态刷新；
  要看刷新用 `tail -f logs/xxx.log | tr '\r' '\n'`。

## 更新 2026-11-01 改用官方 proxy50k 子集

### 决定

- 全量 472k × 100 epoch 约 20 天/变体、两版约 40 天；用户拍板先用官方 development proxy。

### 做了什么

- 数据：读 `RA-4M/packs/megasg_proxy50k/train/file_names.json`，在
  `RA-4M/proxy50k/{images,labels}` 建 50,000 对软链（不复制大文件）。
  - 50,000 图 / 276,790 框 / 495,470 关系（9.91 关系/图，全量 9.07）。
- 配置：`RA-4M/data_proxy50k.yaml`
  - `train: proxy50k/images`，`val: full_monitor/images`（1,024 图）
  - 词表保持 497 类 / 10,370 谓词，checkpoint 与全量配置兼容
- `train.py` 默认改为 `data_proxy50k.yaml`，默认 run 名
  `glp_proxy50k_v1_e100` / `glp_proxy50k_v2_e100`。
- 全量配置保留：`data_full.yaml`（全量 val）、`data_full_monitor.yaml`；
  需要全量训练时用 `YOLOVO_RELATE_DATA=RA-4M/data_full_monitor.yaml python train.py` 覆盖。

### 验证

- `YOLOVO_EPOCHS=1 YOLOVO_NAME=tmp_proxy50k_e1 python train.py` 正在跑，
  用来确认真实 epoch 时间和全链路（启动、扫描 50k 标签、训练、监控 val）。

### 没做什么

- 没有重采样/分层：proxy50k 是官方给定的随机子集，不做二次挑选。
- 没有为子集裁剪谓词词表（保持 10,370，和全量口径一致）。

### 实测（proxy50k）

- 1 epoch 验证跑到 366/3125 iter：`1.22 it/s`（0.82 s/it），对应 **约 43 分钟/epoch**；
  加 1,024 图监控 val（约 10 分钟）≈ **53 分钟/epoch**，100 轮约 **3.7 天/变体**，
  V1+V2 约 **7.5 天**（比预估的 2.7 天慢，原因是 proxy 的关系/图更多、GPU 与另一进程共享）。
- proxy50k 的 `labels.relate.cache` 已生成（36.8 MB），用户正式跑时无需再等 7 分钟扫描。
- 50,000 张里有 **1 张 0 字节坏图**（`objects365_v2_01228848.jpg`，源数据就是空文件），
  扫描时自动跳过，不影响训练。

## 更新 2026-11-01 训练速度审核（只测不改）

### 测量方法

临时脚本放 `/tmp` 并已删除：dataloader 单独计时、整步 CUDA event 分段、
`torch.profiler`（CPU+CUDA kernel）、逐项拆 `_losses`、A/B 对比。
GPU 与用户自己的训练进程共享，绝对值偏高，比例可信。

### 结论

| 环节 | 实测 |
| --- | --- |
| dataloader（8 workers） | 59 ms/batch，270 img/s → **不是瓶颈** |
| 骨干前向（冻结，AMP） | ~80 ms/step |
| 关系头前向 `_forward_pairs` | ~176 ms/step（单个最大 kernel：`grid_sampler_2d_backward` 68 ms） |
| 逐图 `_losses` × 16 | 350–430 ms/step；A/B 跳过它：1100 → 380 ms/step（**~50%**） |
| 逐图指标 / field_diag 同步 | ~200 ms/step；A/B 去掉：1.20–1.26× |
| 反向 + 优化器 | ~380 ms/step |
| GPU 利用率 | 31–41%（CPU/launch 受限，不是 GPU 打满） |

原因（按影响排序）：

1. `_compute_loss` 逐图调 16 次 `_losses`：每次独立跑 `[K,512]@[512,V=10370]`
   的 NCE/swap 余弦、`[K,V]` sigmoid、`topk(10370)`、对象对齐 CE → 小 GEMM + 反向，launch 开销和低占用叠加。
2. 逐图 GPU→CPU 同步：`int(counts[i])`、`float(v[i])`（field_diag）、
   `FieldMetricAccumulator.update` 的 `.tolist()/.argmax/.cpu()`，一步约 130 次强制同步。
3. 槽位浪费：sampler 的 `K2` 固定 `final_budget=128`，但每图有效对只有
   `min(128, n(n-1))`——proxy50k 平均 29.6、中位 12，**槽位利用率 ~23%**。
   后续 transformer / deformable / field(4 候选×32 核×5 点 grid_sample) /
   `[K,V]` 打分全按 128 算，这是 `grid_sampler` 反向 68 ms 的主因。
4. `_losses` 每图重算一遍 N×N 几何特征 + `geo_scorer` + `f_sub/f_obj`，
   `_Sampler` 在整批前向里刚算过。
5. swap 为了几个 GT 对的 cos 先算了两遍 `[K=128,V]` 全词表余弦（4.9 ms/图，`_losses` 内最大单项）。

### 顺手发现的一个正确性问题（非速度）

`YOLOVOTrainer.preprocess_batch` 会把 `batch["cls"]` 重映射为 batch 内采样文本的局部 id
（`remap_cls`），而 `_losses` 的对象对齐损失用这个 id 索引 `W_obj`（RA-4M 全局类名，
497 行）——索引错位，obj 损失（λ=0.1）在学错的目标。

### 没做什么

- 没有改任何训练/模型代码，没有动超参，没有重跑实验。
- 没有测 solo（独占 GPU）下的绝对耗时；用户进程全程在跑。
- 没有实现上面任何一条优化。

## 更新 2026-11-01 训练加速（已改代码）+ batch 32

### 做了什么

1. `relate/head.py::_losses`：全词表 cosine 不再每图重算。
   NCE 与 swap 改为从 `out["logits"]` 反算 `cos = (logit - bias) / scale`
   （dual-score 本来就是 `cos*scale + bias`，已含 field 候选列的 scatter）；
   `logit_scale/logit_bias` 取 `detach()`，保证 NCE 不把梯度回传到仿射参数。
   删掉原来每图的两次 `[K,512]@[512,V]`（4.9 ms/图，`_losses` 内最大单项）。
2. `relate/head.py::_forward_pairs`：整批算一次 N×N 几何打分与 `f_sub/f_obj`，
   通过 `out["geo_scores"] / out["zs"] / out["zo"]` 传给逐图 `_losses`
   （原来每图重算一遍，`_losses` 缺这些键时仍回退到重算，接口不变）。
   `_compute` 同步透传。
3. `relate/trainer.py::_compute_loss`：`counts.tolist()` 一次代替逐图 `int(counts[i])`；
   `field_diag` 用张量累加、末尾一次取值；`n_pred_pairs` 末尾一次 stack 统计。
4. `relate/field_metrics.py`：`FieldMetricAccumulator` 全程张量累计，`compute()`
   才取值；新增 `slot_labels` 入参（trainer 直接给每槽位 GT 谓词），免掉逐图
   `.tolist()`；`_macro_f1` 从「按类循环 10,370 次 + 每次同步」改为 `bincount` 向量化。
5. `train.py`：关系训练 `batch=16 → 32`（`nbs=64`，内部 accumulate=2，
   等效优化器步长仍是 64 图）。

### 验证

| 项 | 结果 |
| --- | --- |
| `tests/test_glp_field.py` | 24/24 通过（含 batched vs 单图损失 <1e-4） |
| 新旧 cosine 等价性 | 最大绝对差 6.3e-5（fp16 matmul 舍入级） |
| 单步（含前向+反向+优化器）batch 16 | 1100 → **535 ms** |
| 单步 batch 32 | 1027 ms（31.2 img/s，峰值显存 9.1 G） |
| 真实 trainer：2,048 图 1 epoch | **74 s**（1.08 s/it，28 img/s）+ 64 图 val 11 s |
| 1 epoch 冒烟（`data_full_smoke.yaml`） | 通过；loss/FieldMetrics/保存/val 全链路正常 |

按实测推算 proxy50k 50,000 图：约 **30 min/epoch**（原来实测 43 min），
100 轮约 2.1 天/变体（原 3.7 天）。

### 没做什么

- 没做「七项损失整批向量化」和「按框数分桶的 batch sampler」——那是下一档 2-3×，
  改动大，先不动。
- 没修 `batch["cls"]` 被 `remap_cls` 重映射导致 `W_obj` 索引错位的问题（正确性问题，
  改它会改变 obj 损失的目标，等用户决定）。
- 没跑多 seed、没调其他超参；没有重训已有 run。

## 更新 2026-11-01 打开图像 RAM 缓存（用户要求）+ batch 可降级

### 做了什么

- `train.py`：`cache=True`（`YOLOVO_CACHE=0` 可关）；`batch` 改为
  `YOLOVO_BATCH` 环境变量可覆盖（默认 32）。
- 原因：用户扩容内存后要求开启。实测数据等待本来只占 1.6%（29 ms/batch，
  workers=16 时 0.5%），开缓存后这部分归零，但**不是**主要瓶颈。

### 实测

| 项 | 结果 |
| --- | --- |
| 缓存容量（proxy50k 10% 抽样外推） | 50,000 图 ≈ **41 GiB**（不是 91 GiB：该 fork 的 `load_image` 先把图拉伸成 640×640 再缓存，每图 ~0.88 MB） |
| 首轮装载时间 | ~2 分钟（56 线程 ThreadPool） |
| 端到端（batch 16 + cache，1 epoch 冒烟） | 通过 |
| batch 32 + cache | **CUDA OOM**：卡上另有别人的 11.6 GiB 进程，21.49 GiB 卡放不下我们的 10–11 GiB |

### 没做什么

- 没有为 cache 改 collate/augment；缓存数组只读，worker 通过 fork COW 共享，不会翻倍。
- 没有把 val 的缓存单独关掉（1,024 图约 1.2 GiB，可忽略）。

### 风险

- batch 32 峰值显存 10–11 GiB；和别人的大进程同卡会 OOM，用 `YOLOVO_BATCH=16` 降级
  （峰值约 5 GiB）。

## 更新 2026-11-01 指标链路审核 + 验证表头修正

### `train.py` 里 `data="/root/datasets/M3FD_split/data.yaml"` 是什么

- 它在 `else:` 分支里，即 `YOLOVO_RELATE_DATA` 为空时才走——那是**仓库原有的一级
  YOLOVO 多模态检测训练入口**（M3FD 数据集），不是关系训练配置。git 里 `e47cbd8`
  已存在，我们没有改过。
- 该路径在本机不存在（`/root/datasets` 整个目录都没有），所以 `YOLOVO_RELATE_DATA= python train.py`
  会直接 FileNotFoundError。默认路径（`data_proxy50k.yaml`）不受影响。

### 评价指标链路（逐段核对）

| 环节 | 用的是关系指标？ |
| --- | --- |
| best.pt 选择 / 早停 fitness | **是**：`val.py::get_stats` 写 `d["fitness"]=F1@50`，`engine/trainer.py:814` pop 出来 |
| results.csv 列 | **是**：R@20/50/100、mR@50/100、F1@50、pair_existence_AP、predicate_acc、DetPairRecall、PairRecall、GT_MRR |
| 每轮日志 | **是**：`[RelateVal] ...` 一行 |
| 验证进度条表头 | **原来不是**：继承 DetectionValidator 的 “Box(P R mAP50 mAP50-95)”，这些列关系验证从不填充 → 已改 |
| results.png（训练结束自动画图） | **不是**：`plot_results` 仍按 `metrics/mAP50-95(B)` 等列名找列，找不到就 warning + 按默认下标乱画 |

### 做了什么

- `relate/val.py`：新增 `RelateMetricsMixin.get_desc()`，进度条表头改为
  `RelationVal (R@K / mR@K / F1@50 / ExAP)`，不再显示从不计算的 Box 列。

### 没做什么

- 没改 `results.png` 的列映射（改的是 `utils/plotting.py` 的公共代码，影响其他任务）；
  训练结束那张图仍是无意义的，看 `results.csv` 即可。
- 没动 `train.py` 的 else 分支（仓库原有代码），要不要删等用户定。

## 更新 2026-11-01 关系链路审计（ponytail-audit）+ cache 默认关闭

### 做了什么

- 审计 `ultralytics/models/yolovo/relate/` 全链路：`head.py` / `field.py` / `dataset.py` / `gen.py` /
  `trainer.py` / `wiring.py` / `namespace.py` / `val.py` / `predict.py` / `field_metrics.py`。
- 实测验证了两件事：
  - 验证集的 GT 框与预测框**同处 letterbox 画布坐标系**（`ds[0]` 的 `bboxes` 已被变换：
    `0.117676→0.13588`，`ratio_pad=((0.6253,0.625),(16,21))`）——验证链路坐标系是对的。
  - 单测 24/24 通过（`tests/test_glp_field.py`，无 pytest，用脚本逐函数调用）。
- `train.py`：`cache` 默认 `1` → `0`。原因（实测口径）：
  - `cache_images` 把 5 万张图按长边缩放到 640 的 uint8 数组常驻，平均 0.84 MiB/张 →
    跑满 **41.0 GiB**（日志 38.7 GiB @ 47,340/49,999 外推吻合）。
  - 同一 cgroup 里还有别的进程 ~12–13 GiB anon、训练进程自身 ~3.8 GiB、读原图的
    page cache ≤4.2 GiB（可回收）、内核开销 ~0.4 GiB，95% 时合计 ≈58.5–60.5 GiB，
    越过 `memory.max` = 58.0 GiB（`memory.events` 里 oom_kill=6，swap 为 0）。
  - `check_cache_ram` 用 `psutil.virtual_memory()`，读到的是**宿主机** 251 GiB
    （available 222 GiB），看不到 cgroup 上限，所以放行。
  - 数据加载本来就有 9× 余量（270 vs 30 img/s），缓存对 epoch 时长没有收益；
    `cache="disk"` 也不可行：落盘的是原图解码数组（87.6 GiB），`check_cache_disk`
    要求 131 GiB 而卷上只剩 114 GiB，会直接放弃缓存。

### 没做什么

- 审计发现的问题**一个都没有代码改动**（除上面 cache 默认值），等确认后再逐条修。
- 没有改 `results.png` 的列映射、没有删 `train.py` 的 `else:` 分支。

### 审计发现（按严重度）

| # | 位置 | 问题 |
| --- | --- | --- |
| 1 | `trainer.py:247` + `head.py:598-609` | `batch["cls"]` 已被 `YOLOVOTrainer.preprocess_batch` 重映射成**局部文本 id**（0..K-1，K≈80），却被用来索引 497 行的 `W_obj`（按全局类 id 排）。obj 损失（λ=0.1）在教错类别，梯度还经 `sub_text_proj/obj_text_proj` 污染主查询。 |
| 2 | `predict.py:88` | `postprocess` 逐图循环里 `rel.score(boxes)` 没传 `img_idx`，默认 0。batch>1 时第 2 张及以后的图都用**第 1 张图**的特征打分。 |
| 3 | `dataset.py:250` | 扫描期 `except Exception` 吞掉 `parse_relate_lines` 的 ValueError（越界框索引、非法谓词 id），只丢图不中止，与 docstring 承诺相反。 |
| 4 | `dataset.py:78-82` | `parse_rnames` 忽略原始 id，排序后重新 `enumerate`。1-based 或非连续 `rnames` 会被静默重编号，谓词整体错位。 |
| 5 | `dataset.py:324/337` | 缓存 hash 只含路径+字节数，`rnames` 变了仍命中旧 cache，关系 id 静默错位。 |
| 6 | `dataset.py:459-467` | `jitter_boxes_xywhn` 的 `generator` 参数没被使用（`rand_like` 不接 generator）；w/h 放大后 `clamp(min=1-w/2)` 上下界倒置。 |
| 7 | `dataset.py` + `collate` | `rswap` / `rel_weights` / `spatial_flags` 解析、缓存、collate 一路传递，但全仓无消费者（head/loss 从不读）。 |
| 8 | `field.py` | `compose_fields` / `gate` / `FieldConfig.compose` 只有单测在用，训练/验证路径从不调用。 |
| 9 | `head.py:32` | `field_grid` / `field_stats` 只 import 未使用。 |
| 10 | `field.py:92-96` | `field_grid(field, k, m)` 实际索引是 `field["mu"][batch, pair]`，k/m 语义与文档相反（仅诊断用）。 |
| 11 | `dataset.py:394-397` | `build_relate_dataset` 原样透传 `task` 给 segment/obb/pose 的 Format，关系标签没有 segments/keypoints，会被清空或重排，relations 索引随之失效。 |
| 12 | `trainer.py:341` | 日志读 `_rel_log["n_swap_pairs"]`，但从未被写入，恒 0。 |
| 13 | `val.py:307` | 归一化尺寸用 `rel.feat_strides[0]`，但 capture 存的是 `base_idx`（P4）特征。`trainOVrel.py` 接线 P3/P4/P5 时 ih=40×8=320（应为 640），框坐标整体放大 2 倍再打分 → 该入口的 R@K/F1 全部失真。本仓 `train.py` 只用 P4 不受影响。 |
| 14 | `val.py:358` + `val.py:97-105` | 检测为空时 `pred_boxes = labels`（GT 框），early-return 拿 GT 框算 IoU → DetPairRecall 恒 1.0；该分支也不累加 `pair_recall_total`，零检测图被 PairRecall 剔除。 |
| 15 | `val.py:349` vs `314` | `update_metrics` 硬编码 60，`postprocess` 用 `rel.max_objects`；两者不等时 `pred_rels` 的框索引与 `pred_boxes` 对不上。 |
| 16 | `val.py:141-151` | `_match_triplets` 的返回值从未使用，R@K 又在循环里重写同一匹配；IoU 矩阵整块重算 3 次（111/145/162 行）。 |
| 17 | `val.py:159-168` | `pair_existence_AP` 没有一对一匹配，同一 GT 的近重复框会产生多个正样本，AP 偏高。 |
| 18 | `val.py:208-210` | `F1@50` 实为 `2·R@50·mR@50/(R@50+mR@50)`，两项都是召回，没有精度项。`fitness=F1@50` 的语义与名字不符（注释里已说明是"惩罚偏科"，但对外报告别写成 F1）。 |
| 19 | `val.py:98/150/218` | 未召回 GT 的 rank 哨兵是 `topk+1=101`，GT_MRR 每条未命中贡献 0.0099 而非 0。 |
| 20 | `data/base.py:394` | `check_cache_ram` 用 `psutil.virtual_memory()`（宿主机 251 GiB 口径），看不到 cgroup 的 `memory.max`=58 GiB，因此任何 `cache=True` 都会被误放行。要根治需改读 cgroup v2 的 max/current。 |
| 21 | `relate/trainer.py:188-195` + `yolovo/train.py:80,599-604` | **最严重**：排除 `set_vcte_alpha` 的 `add_default_callbacks` 全仓无人调用（且 `super()` 上没有该方法），回调照常每 epoch 执行。它遍历所有模块做 `m.alpha.fill_(factor)`，把关系头 `vocab_head.alpha`（10370 维、本应每谓词不同）覆写成常数；`factor` 在 epoch 0 = 0 → `cos=(1-α)cos_sem+αcos_spa` 退化成纯语义专家，`spa_proj/q_spa` 整条支路零梯度。同时冻结骨干的 VCTE alpha 也每轮被改（0→0.15→约 0.32），"冻结"的一级特征在漂移。实测 `runs/test/glp_proxy50k_v1_e100/weights/last.pt`：`model.31.vocab_head.alpha` 全 0。 |
| 22 | `relate/trainer.py:173-177` + `field_metrics.py:156-177` | `self._field_metrics = None` 只写在同样死掉的 `on_train_epoch_start` 里，`compute()` 也不清空 list → 累计器从第 1 轮涨到最后一轮：日志里 FieldMetrics 是全程累计值，且每 epoch 约 30 万个 GPU 张量常驻。 |
| 23 | `relate/trainer.py:377-388` | ontology 三张 [10370,10370] 稠密表（615 MiB）是模块上的普通 dict，随模块 pickle 进每个 checkpoint：`last.pt` 948 MB（state_dict 只有 150 MB）、`train-checkpoint.pt` 1055 MB，而后者全仓无读者。每 epoch 写 ~2.9 GB。 |
| 24 | `relate/head.py:541-542` | 对 `logit_scale/logit_bias` 取 `detach` 想把 NCE 的梯度切断，实际留下一个非零梯度（分子仍带图、分母被当常数）。实测 detach 时 `logit_scale.grad=-5.4e-2`，不 detach 时 ≈0。应删掉这两个 detach。 |
| 25 | `relate/loss.py` 全文件 + `nn/tasks.py:2892-2931` | v3 的四项损失实现，与 v4 七项完全对不上，还引用了不存在的 `self.rel.predicate_bank`。当前训练路径不走它（死代码），但被调用即 AttributeError。 |
| 26 | `relate/trainer.py:94-114` | 训练 DataLoader 未传 `drop_last`，最后一个不满 batch 照常参与均值，每 epoch 影响 1 个 step。 |
| 27 | `relate/trainer.py:325-326` | `loss_names` 是 (rel_geo, rel_exist, rel_pred, rel_swap)，`items` 顺序是 [geo, rel, nce, swap] → results.csv 的 `rel_exist` 列其实是 rel focal 项、`rel_pred` 列其实是 NCE；obj/sigmoid/bg 从不落盘。 |

## 更新 2026-11-01 按审计结论修复

### 做了什么（按发现编号）

| # | 文件 | 改动 |
| --- | --- | --- |
| 21 | `relate/trainer.py::__init__` | 在 `super().__init__()` 之后直接过滤 `self.callbacks["on_train_epoch_start"]` 里 qualname 含 `vcte` 的回调；删除从未被调用的 `add_default_callbacks`。`set_vcte_alpha` 不再每轮覆写 `vocab_head.alpha` 与冻结骨干的 VCTE alpha。 |
| 22 | `relate/trainer.py::_model_train` | `_field_metrics = None` 移到 `_model_train`（每 epoch 必走）；删除死掉的 `on_train_epoch_start`。场指标改为按 epoch 统计。 |
| 1 | `relate/trainer.py::preprocess_batch/_compute_loss` | 预处理前把原始 cls 存成 `batch["cls_global"]`，obj 对齐损失改用它（全局类 id）索引 `W_obj`。 |
| 2 | `relate/predict.py` | 逐图打分传入 `img_idx=bi`，batch>1 时不再共用第 0 张图的特征。 |
| 3 | `relate/val.py::update_metrics` | 无检测时 `pred_boxes` 传空张量（原来是 GT 框，导致 DetPairRecall 恒 1.0）；evaluator 的空预测分支补记 `pair_recall_total`。 |
| 4 | `relate/val.py::postprocess` | 归一化尺寸改用 `rel.feat_strides[rel.base_idx]`。 |
| 8 | `relate/val.py` | `postprocess` 把 `rel.max_objects` 存到 `self._rel_max_objects`，`update_metrics` 的截断域与之一致（不再硬编码 60）。 |
| 9 | `relate/field_metrics.py` | 无 GT 关系的图不再整张早退：存在性 tp/fp/fn 对所有有效槽位统计；`compute()` 的 top-k 与存在性指标拆成各自的 gate。 |
| 10 | `relate/val.py::stats` | 未召回 GT 的 rank 哨兵由 `topk+1` 改为 `inf`，GT_MRR 不再给未命中贡献分。 |
| 16 | `relate/val.py` | 删除 `_match_triplets`（返回值从未使用）与其调用；IoU 矩阵整块只算一次、`ps/po/pp` 提到循环外。 |
| 24 | `relate/head.py::_losses` | 去掉 `logit_scale/logit_bias` 的 `.detach()`（detach 反而留下非零梯度，持续压低 logit_scale）。 |
| 26 | `relate/trainer.py::get_dataloader` | 训练 loader 加 `drop_last=True`。 |
| 27 | `relate/trainer.py::get_validator` | `loss_names` 改为 `("geo","rel","nce","swap")`，与 `_compute_loss` 返回的 items 顺序一致；日志删掉恒 0 的 `pairs(swap)`。 |
| 5 | `relate/dataset.py::scan_relate_label` | `ValueError/SyntaxError` 原样上抛（框索引越界、谓词 id 非法不再被当 corrupt 图吞掉）。 |
| 6 | `relate/dataset.py::parse_rnames` | 要求 id 必须是 `0..V-1` 连续，否则 SyntaxError；不再静默重编号。 |
| 7 | `relate/dataset.py::_cache_hash` | 标签缓存指纹混入 `rnames` 的 sha256 前 16 位，改词表不再命中旧 cache。 |
| 20 | `data/base.py::_mem_available_total` | `check_cache_ram` 改读 cgroup 限额（v2 `memory.max` / v1 `limit_in_bytes`），可用量按 `limit - anon` 估；容器内不再用宿主机口径放行。 |

### 验证

- `py_compile` 全部改动文件通过。
- `tests/test_glp_field.py` 24/24 通过。
- 新写 `/tmp/t_eval.py` 三个用例验证 `RelateEvaluator`：无检测（DetPairRecall=0、PairRecall 分母计入）、正确命中（全 1.0）、谓词错（R@K=0、GT_MRR=0）。用完已删。
- 实测 `_mem_available_total()` 返回 42.8/58.0 GiB（宿主机口径是 222/251），缓存 41 GiB × 1.5 现在会被拒绝。

### 没做什么

- 没修 #11（非 detect 任务透传）、#12（`rswap/rel_weights/spatial_flags` 死字段）、#13/#14/#17/#18/#19/#23/#25（见下表说明）。
- 没动 `train-checkpoint.pt` 与 ontology 入库（#23）：三个权重文件是固定路径、每轮覆盖，磁盘峰值只有约 2.9 GB，不是累计上百 GB。
- 没有跑真实训练验证，只做了导入/单测/评测器冒烟。

### 已知遗留

- `F1@50` 仍是 R@50 与 mR@50 的调和平均（无精度项），`fitness` 语义未改。
- `pair_existence_AP` 仍无一对一匹配，近重复框会抬高 AP。
- `relate/loss.py` 与 `nn/tasks.py` 的关系分支仍是 v3 死实现。

## 更新 2026-11-01 GFLOPs 摸底（只测不改模型）

### 做了什么

- 新增 `tools/gflops_report.py`：加载 checkpoint → 设文本提示 → 跑检测器一次 → hook 拿到 P4 →
  用 `YOLOVORelate._forward_pairs` 跑关系头，FLOPs 用 torch 官方 `FlopCounterMode`
  （按 aten 算子累加 conv/mm/addmm/bmm，含 attention，不受 thop 只认 Conv/Linear 的限制）。
- 口径：batch=1、fp32、CPU（GFLOPs 与设备无关，避免和正在跑的 `train.py` 抢 2080 Ti），
  检测器 imgsz=640；关系头 N=60 框、V=10370 谓词、eval budget=500 / train budget=400-128。
- 被测 checkpoint：`runs/test/glp_proxy50k_v1_e100/weights/last.pt`（即当前 `train.py` 正在写的那个，
  73.18 M 参数 = 检测器 46.50 M + 关系头 26.68 M）。

结果（GFLOPs）：

| 组件 | 口径 | GFLOPs |
| --- | --- | ---: |
| 检测器 640，1 个提示 | 仅检测前向 | 8.31 |
| 检测器 640，80 个提示（= max_texts） | 仅检测前向 | 9.02 |
| 检测器 640，497 个提示 | 仅检测前向 | 12.71 |
| 关系头，N=20 | 推理 budget=500（40 对截到 400 对） | 41.57 |
| 关系头，N=40 | 推理 budget=500 | 48.24 |
| 关系头，N=60 | 推理 budget=500 | 48.44 |
| 关系头，N=60 | 训练 budget=128 对 | 24.39 |

- 推理合计（640 + 80 提示 + N=60）：**57.45 GFLOPs**；训练前向合计：**33.40 GFLOPs**。
- 关系头内部占比（N=60, budget=500）：SoftSpatialPool 15.78（32.6%）、RelationTransformer 12.06（24.9%）、
  谓词库打分约 10.6（500×10370×512 两次 matmul）、RelationInteractionBlock 1.36、PredicateFieldRead 0.86。
- 交叉核对：thop（`compute_model_gflops`，仅 Conv/Linear）给检测器 8.44，比 FlopCounter 少 0.75，
  差的就是注意力里的 bmm；两者在检测器上一致到 8%。

### 没做什么

- 没改任何模型结构/训练代码；没有为了“降 GFLOPs”动 budget、V、候选数。
- 没把 MobileCLIP 文本塔算进去（它在检测图之外，每套提示向量化一次，不是逐帧成本）。
- 没测 GPU 上的实际时延；2080 Ti 上 FLOPs 不等于 ms，尤其关系头在 batch=1 是 dispatch-bound。
- 没把 FlopCounter 与 torch.profiler(with_flops) 的差异写死成结论：同一次关系头前向 profiler 报 75.6 GF
  （einsum/expand 视图按逻辑尺寸重复计），FlopCounter 报 48.4 GF 且与单模块隔离测试（SoftSpatialPool
  N=1000 两边都是 11.4 GF）一致，故采用 FlopCounter 口径。

## 更新 2026-11-01 关系指标恒 0 的链路审计（只读，未改代码）

现象：`glp_proxy50k_v1_e100` 第 1–4 轮 R@20/50/100、mR@50/100、predicate_acc、GT_MRR 全 0，
DetPairRecall 恒 1.40–1.45%，而训练日志 `[FieldMetrics] rel_top1` 从 0.979 升到 1.000。

### 做了什么（测量）

| 检验 | 口径 | 结果 |
| --- | --- | --- |
| GT/pred 坐标系 | 代码事实 + 16 图实测 | 同帧（都是 letterbox 画布归一化）。`postprocess` 由 P4×stride 反推的 ih/iw 与 `batch["img"]` 逐像素相等；val 强制 rect=True，画布实际是 672×384/672×416，两侧一致 |
| 一级检测器 | 64 图，conf≥0.001 / ≥0.25 | val GT 框单端 IoU≥0.5 召回 **3.3% / 0.00%**；train **9.7% / 1.1%**。权重本体健康（DIOR 自带 val mAP50=0.802） |
| 一级权重身份 | checkpoint 元数据 | `nc=20`，DIOR 遥感 20 类；被冻结用在 COCO/Objects365 自然图、497 类标注上 |
| SGCls（GT 框 oracle）全量 val | 1024 图 | DetPairRecall=1.0000、PairRecall=1.0000、pair_existence_AP=0.526，但 **predicate_acc=0.0003**、R@K=0 |
| 同上，换 256 张**训练集**图 | 同一评测器 | **predicate_acc=0.0000**（2590 条 GT），DetPairRecall=1.0000 |
| 谓词候选 | 真实 ckpt 单图 | `n_candidates=4`；72 个有效对的 top-4 候选**完全相同** {7877,6575,5654,2111}；全部 72 对输出同一个 id 2111 |
| GT 谓词在 raw logit 的排名 | 同一图 17 条关系 | 排名 **332–9112**，top-4 命中 0/17 |
| 训练路径 vs 评测路径 | 同图同 GT | 训练强制 GT 入候选后 top1 = 11/11 = **1.0**，这正是日志里那个 1.0000 |

### 根因（三条，互相独立）

1. **一级检测器与数据域/词表完全不匹配**（`YOLOVO_PRIMARY` 指向 DIOR 20 类航拍权重）。
   DetPairRecall 只由检测框决定，冻结权重 → 各轮恒定 1.4%，和关系头怎么训无关。
2. **检索头从来没被正样本训练过**：`head.py:454` 用 `logits.scatter(-1, cand, logit_m)` 把 GT 列的
   raw logit 整个覆盖成 field logit，`_losses` 的 NCE/sigmoid/bg 都读这份被覆盖的 logits。于是
   `q·W_gt` 永远收不到正样本梯度，raw 检索退化到"所有对 top-4 相同"。训练时 GT 又被
   `_select_candidates(training=True)` 强行塞进 4 个候选，field 只需在 4 选 1 上取胜 → nce 两轮降到
   3.5e-4、rel_top1 到 1.0，全是构造出来的假象。
3. **训练期指标与评测口径不一致**：`FieldMetrics` 用强制入候选的 logits，评测用 raw top-4，
   两者测的不是同一件事，所以 1.0 和 0 能同时存在。

附带：`_gt_grid` 对同一有向对 last-wins，17 条关系只留 11 条（丢多谓词标注）；train/训练期 val/独立
val 三处文本 prompt 词表不一致（497 / 20 / 497）；`plot_relations` 的 GT panel 没做逆 letterbox。

### 没做什么

- 没改任何代码、没动权重和正在跑的进程；所有测量脚本放 /tmp，已删除。
- 没换一级权重（这台机器上没有 COCO/RA-4M 域的一级 checkpoint）。
- 没验证修复后的训练曲线——修复必须重启训练，旧 4 轮权重不可复用。

### 结论

在当前配置下，无论训多少轮，SGDet 关系指标都会停在 0：检测端给不出可用框，关系端即使给 GT 框也
答不对谓词。100 轮（约 2.2 h/轮）跑完约 9 天，不会产出可用的数。

## 更新 2026-11-01 修复：检索头正样本梯度 + 训练指标口径

### 做了什么

- **`relate/head.py`**：`_forward_pairs` 在 `scatter` 前保存 `out["logits_raw"] = logits`。
  `_losses` 的 NCE / swap 改用 `logits_raw`（检索头），sigmoid / bg 仍用散列后的 `logits`（场头）。
  之前 GT 列被 field 分数整列覆盖，`q·W_gt` 拿不到任何正样本梯度，推理时 raw top-4 必然
  召回不到 GT（实测排名 332~9112、top-4 命中 0/17）。
- **`relate/trainer.py`**：`_compute_loss` 构造的逐图 `out_i` 透传 `logits_raw`。
- **`relate/field_metrics.py`**：`rel_top1/top5/macro_f1` 改用 `logits_raw` 计算（原来用强制
  GT 入候选的散列分数，恒≈1.0 是构造出来的），并新增 `rel_gt_in_cand`（GT 谓词落在 field
  候选中比例）——这个数直接反映场头还有没有重排的机会。

### 没做什么

- 没改 `n_candidates=4`（场头仍只在 4 个候选上重排）、没动 `_gt_grid` 的多谓词 last-wins
  （同一有向对多条关系仍只保留最后一条）、没换一级检测器、没动 `train.py`/`data/base.py`。
- 没有动已停掉的 100 轮实验产物；旧 4 轮权重不可复用。

### 验证

跑的是 **5000 图子集 × 15 轮**（`runs/test/glp_fix5k_e15`，只有全量的 1/10 数据），不是正式训练。

训练侧（逐轮，FieldMetrics 现在算的是未散列的检索分数）：

| 轮次 | nce | rel_top1 | rel_top5 |
| ---: | ---: | ---: | ---: |
| 1 | 7.540 | 0.0915 | 0.2837 |
| 8 | 7.147 | — | — |
| 15 | 7.075 | 0.2855 | 0.4310 |

`nce` 从修复前的 0.00035 变成 7.54→7.07。这不是变差，是**换成了真正的全词表检索目标**
（`ln 10370 ≈ 9.25` 是随机水平）；以前那个 0.0003 是 GT 被强制入候选造出来的。

SGCls（GT 框 oracle，full_monitor 全量 1024 图 / 9194 条）：

| | 修复前（50k 图 / 4 轮） | 修复后（5k 图 / 15 轮） |
| --- | ---: | ---: |
| R@20 | 0.0003 | **0.1684** |
| R@50 | 0.0003 | **0.1886** |
| R@100 | 0.0003 | **0.1957** |
| predicate_acc | 0.0003 | **0.1850** |
| GT_MRR | 0.0001 | **0.0592** |
| pair_existence_AP | 0.5256 | 0.2597 |

结论：根因判断成立，检索头一旦拿到正样本梯度，谓词检索就从"GT 排名 332~9112"变成
"top-1 ≈ 28%、top-5 ≈ 43%"，SGCls R@50 从 0.03% 到 18.9%。数据只有 1/10，这个数是下限。

副作用：`pair_existence_AP` 0.526 → 0.260。存在性头（`rel` 项）没改，但检索分数分布变了，
两者共用同一个 `q`；也可能是数据量/轮次差。未单独归因。

### 还没做

- `rel_gt_in_cand` 第一版实现有误（读的是强制 GT 之后的 `cand_idx`，恒为 1.0），已改成用
  未散列分数重新取 raw top-M；这个改动本身只跑了编译和单测，没再跑一轮训练确认。
- 没解决一级检测器（SGDet 的 R@K 仍会是 0）。

## 更新 2026-11-02 field 候选数 4 → 8

- `FieldConfig.n_candidates` 默认值 4 → 8（`relate/field.py`）。理由是测到 top-5 召回 0.43，
  M=4 会把场头能重排的上限卡在四成。
- **16 试过，不行**：batch=32 下 backward OOM（已用 19.25 GiB 仍要再要 4 GiB）。8 是这张
  22 GiB 卡在 batch 32 下的上限；再大只能砍 batch。
- 5000 图 × 2 轮冒烟：无 OOM，显存峰值 15.4 GB，5.95 min/轮（M=4 时 4.85 min/轮，+23%）。
  `rel_gt_in_cand`（现在按 raw top-M 算，不再是强制入候选的假 1.0）= 0.369 → 0.371。
- 全量 50k × 100 轮已启动：`runs/test/glp_proxy50k_v1_fix_e100`，按 +23% 推算约 **41 min/轮、
  总计约 69 小时**。
- 已知：训练内建的验证仍是 SGDet 口径，在 DIOR 检测器下恒为 0，所以 `best.pt` 的 fitness
  （F1@50）不具区分度；选权重请用 `last.pt` 或单独的 SGCls 评测。

## 更新 2026-11-02 全量 run 发散（M=8）与残差截断

### 发生了什么

`glp_proxy50k_v1_fix_e100`（50k × 100 轮，M=8）跑到：

| 轮 | geo | rel | nce | predicate_acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6463 | 0.0912 | 7.379 | 0.1205 |
| 5 | 0.6226 | 0.0893 | 7.088 | 0.0714 |
| 6 | 0.6221 | 0.0903 ↑ | 7.201 ↑ | 0.0244 ↓ |
| 7 | 0.6225 | **nan** | **nan** | **0** |

发散前两轮 `delta_norm` 24.7 → 26.3 → 35.9 → **49.8**，`gamma_l1` 25 → 35 → 50 → **59.5**。
第 8 轮仍在跑但损失已 NaN。已按要求停止（含残留 worker 进程组）。

SGCls 全量评测（1024 图 / 9194 条）：

| 权重 | R@50 | predicate_acc | GT_MRR | ExAP |
| --- | ---: | ---: | ---: | ---: |
| `best.pt`（实际是第 1 轮，`epoch=-1`） | 0.1494 | 0.1461 | 0.0402 | 0.4136 |
| `last.pt`（第 8 轮） | **0.0207** | **0.0212** | 0.0035 | 0.2708 |
| 对照 5k×15 轮（M=4） | 0.1886 | 0.1850 | 0.0592 | 0.2597 |

`last.pt` 的权重张量本身没有 NaN（AMP 跳过了一部分坏 step），但已经不可用。

### 修了什么

- `relate/head.py`：`_forward_pairs` 里对场残差做**投影式范数截断**，`delta` 的范数上限
  `self.field_delta_max = 16.0`（健康期观测值 8~25）。超出的部分等比缩回，不新增参数。
  诊断量 `delta_norm` 仍按截断前的值记录，便于继续观察。

### 没做什么

- 没有定位到 NaN 的**第一现场**（是 forward 溢出还是梯度爆炸）：只确认了残差范数在发散前
  两轮显著抬升，截断是针对这个观测量的止血，不是根因证明。
- 没有回退 M=4，也没有调场的学习率/`logit_scale`。
- 没有重跑验证截断是否真的防住了——新 run `glp_proxy50k_v1_fix2_e100` 已启动，需观察到
  epoch 8 以后才作数。


---

## 更新 2026-11-02 五路并行审计（关系头 / 数据 / 评测 / 检测 / 训练动力学）

### 做了什么

5 个只读审计 agent 并行跑了关系头数值链路、数据与标签、评测口径、一级检测干扰、
训练动力学，交叉结论如下（全部有 `文件:行号` 或 checkpoint 实测）：

1. **日志里「指标掉到 0」大部分是测量假象**：训练内验证走 SGDet，冻结的 DIOR 20 类
   航拍检测器在 COCO/Objects365 val 上 `DetPairRecall≈1.4%`，`pa_denominator` 只有
   76–84 对、分子 0–10。epoch7 的 `predicate_acc=0` 是 0/76，同一权重离线 SGCls 全量
   是 195/9194=2.12%，落在 0/76 的置信区间内。同一 e3 权重 SGDet R@50=0.00098，
   SGCls R@50=0.1494（152×）。
2. **真实退化是场头塌成「与候选无关的常量残差」**：50k 第 1 轮 `field_read.out_norm.bias`
   norm 已达 475（5k 跑满 15 轮是 298），`q_res_proj.2.bias` norm 140（5k 72）；
   探针测得 `‖q_res_proj(delta)‖/‖q‖=228`、`delta` 在候选维度的相对变化 **0.000000**。
   散列后 8 个候选列变成近似常量，argmax 退化成噪声；场发生器梯度被压到 1e-9~1e-22，
   自己救不回来。
3. **触发条件**：`warmup_bias_lr=0.1` 作用在随机初始化的关系头 bias 组上
   （`relate/trainer.py` 把 bias 放进第一个 AdamW 组，`engine/trainer.py:467` 对该组
   从 0.1 插值）。warmup 步数 = 3×每轮步数：50k 4686 步 vs 5k 468 步，剂量 10×。
   基类在 AdamW 下会强制 `warmup_bias_lr=0`，但 mixin 覆写 `build_optimizer` 绕过了。
4. **昨天的 `field_delta_max=16` 打偏了**：它限制 delta 的输入，但 `q_res_proj` 的输出
   仍可达 `||q||` 的 100 倍，候选变化仍是 0；只能防 fp16 溢出，防不住退化。
5. **检测头没有拖累训练**：训练用 GT 框，检测 criterion 被删除，检测输出无人读取，
   46.50M 一级参数全冻结。检测只拖累 SGDet 读数和 best.pt 选择（fitness=F1@50≈7e-5，
   best.pt 实际是 epoch3）。
6. 数据侧真实缺陷（压低上限、不解释崩溃）：多谓词 last-wins 覆盖 23.0% 训练行
   （`head.py:178-189`）；train 方形 letterbox / val 矩形 letterbox 坐标帧不同
   （`dataset.py:536`）；`text_encoder` 未进 eval，BN 统计量每步更新；缓存 hash 写读不一致
   （`dataset.py:342` vs `:355`）。
7. 5k 与 50k 不是同配置：`n_candidates=4` vs `8`，且 100 epoch 的 LR 视界更长，
   两者与「数据量」混淆。

### 改了什么（两处）

- `ultralytics/models/yolovo/relate/trainer.py::build_optimizer`：显式
  `self.args.warmup_bias_lr = 0.0`，对齐基类 AdamW 分支；关系头是随机初始化，0.1 的
  bias warmup 会把 `out_norm.bias` 从 0 推到 475。
- `ultralytics/models/yolovo/relate/head.py::_forward_pairs`：`q_res_proj` 只算一次，
  输出按 `cap = ||q||/10` 做投影式范数截断（`cap` detach），保证场头只能「轻推」排序；
  原来的 `delta` 截断保留，作为 fp16 溢出保护。

### 验证

- `py_compile` 通过；`tests/test_glp_field.py` 24/24 通过。
- 输出限幅 sanity：把 `q_res_proj` 换成输出恒为 1000 的模块，候选列 logit 与 raw 检索
  分数最大差 0.228（scale=5），不加限幅会完全盖掉 raw；`CLAMP_OK`。

### 没做什么

- 没有改 `n_candidates`（仍 8），没有改 lr0，没有多 seed（seed=0）。
- 没有修数据侧那 4 个缺陷（last-wins / rect / text_encoder BN / cache hash），
  它们会压低上限，但不是本次退化的原因。
- 没有定位第 7 轮 NaN 的第一现场；`tloss` 是整体滑动平均，任一项 NaN 会让四个记录项
  全部显示 nan，所以「nce=nan」不能单独证明 nce 自身溢出。
- 停掉了 `glp_proxy50k_v1_fix2_e100`（带错误 clamp 的 run，epoch1 跑到 48%）。
- 没有 commit / push。

### 追加（同一轮次）

- 用户要求看 5k 20 轮效果：停掉刚启动的 `glp_5kfix2_e15`（当时还在建缓存，无训练损失），
  改为 `glp_5kfix2_e20`（20 epoch，其余不变：fraction=0.1、M=8、batch 32、lr0=1e-3、seed=0）。
- 注意：20 epoch 会改变 `lf` 的衰减视界，前 15 轮与 15-epoch run 不完全等价。
- `runs/test/glp_5kfix2_e15` 是中断留下的空壳目录（只有 labels 图和空 weights），未删除。

---

## 更新 2026-11-02 20 轮 5k 结果 + 四路底层审计

跑完 `runs/test/glp_5kfix2_e20`（20 epoch，2.823 h，无 NaN），并对 `last.pt` / `best.pt`
做 SGCls 全量验证。随后派 4 个 agent 从底层（数据词表 / 文本嵌入 / 损失数学 / 评测协议）
并行审计「为什么指标绝对值还是低」。

### 做了什么

- 训练：`glp_5kfix2_e20` 20 轮完成。训练损失单调下降（nce 7.830→6.830、rel 0.0914→0.0821、
  swap 0.0215→0.0095），`rel_top1` 0.0893→0.3439、`rel_gt_in_cand` 0.332→0.477，
  `delta_norm` 收在 1.33。**没有再出现退化或 NaN**，两处修复（warmup_bias_lr=0、
  q_res_proj 输出限幅）有效。
- 离线 SGCls（GT 框 oracle，9194 对）：
  | 指标 | 旧 5k 15ep | 新 5k 20ep last | 新 5k 20ep best | 旧 50k 7ep |
  |---|---:|---:|---:|---:|
  | R@50 | 0.1886 | 0.2378 | 0.2123 | 0.1494 |
  | predicate_acc | 0.1850 | 0.2384 | 0.2162 | 0.1461 |
  | GT_MRR | 0.0592 | 0.0648 | 0.0565 | 0.0402 |
  → 三个可比指标全部超过旧 5k 与旧 50k。**best.pt 全面劣于 last.pt**（best 按 SGDet F1@50 选，
  该指标分母只有 86，等于用噪声挑权重）。
- 审计（4 个 agent，只读，未改代码）关键结论：
  1. **5k 子集 = 0% Objects365**：`fraction` 取排序后前 5000 张（`data/base.py:222-229`），
     第一张 Objects365 在排序第 20402 位；5k 训练行是 78.28% COCO / 21.72% OpenImages /
     0% Objects365，而 val 的 GT 行 61.84% 是 Objects365。**5k 的绝对值不能用来判断模型**，
     必须跑 50k。
  2. **词表几何是主因**：10370 个 MobileCLIP 文本嵌入，off-diagonal 平均 cos=0.858，
     89.9% 的类对 cos≥0.8，16.5% ≥0.9；最近邻 cos 中位 0.967；86% 的 SVD 能量集中在
     一个公共方向（中心化后参与率 37、90% 能量只需 131 维）。「above」↔「below」cos=0.9604。
  3. **双专家是同一个排序器**：`cos(q_sem,q_spa)`=0.9835；分数向量 cosine 0.992；
     top-1 一致率 96.8%。**alpha 冻结在 0.50322±0.0047**，`gate_mlp` grad 全为 None、
     optimizer state 里没有它的形状、bias 精确为 0.0；`compose_gate` 退到 [0.0198,0.0084]
     （角色文本投影只贡献 2%/0.8%）。
  4. **场是净负面**：净 top-1 +0.3~0.6pt，但 GT MRR 从 0.2628→0.1680（best，-36%）/
     0.2595→0.1996（last，-23%）；rescue 23 次 vs break 21 次 ≈ 抛硬币；8 候选预算把
     top-1 上限卡在 raw R@8≈33~36%，而 raw R@50 有 44~47%。GT 有 63.7~67.1% 落在
     raw top-8 之外，场永远救不回。
  5. **NCE 与 top-1 解耦**：`nce=6.830` ⇔ p(GT)≈0.1~0.2%，只比均匀好 2.4 nat；
     模型普通 NLL=6.14，**比常数先验的 4.46 还差**，但 top-1 是先验的 4.4 倍。
     加权梯度预算 nce 74.1%、geo 8.0%（geo 梯度只进 `sampler.geo_scorer`，对 VocabHead 为 0）。
  6. **`neg_lw` 没有把近义类踢出分母**：它在 log 空间加偏置，近义列被设成偏置 0 但仍以
     权重 1 留在 softmax 分母里（注释说「由 mask 处理」，消费端只 mask 了偏置）。
  7. **last-wins 结构性上限**：5k 有 12322/59228 = 20.80% 的关系行从未作为正样本；
     val 结构上限 7441/9194 = 80.93%。
  8. **训练/验证 letterbox 不一致**：`dataset.py:536` 验证强制 rect=True；1024×768 下图
     归一化坐标差 0.0586（30~38 px），59% 的 GT 框中心偏移 >0.01，23.7% >0.05。
  9. **文本塔 BN 被训练更新**（`num_batches_tracked` 377866→380986 = 20×156），但打分用的是
     ckpt 里冻结的 W（与初始塔 cos=1.000000），所以排序未直接受污染；污染的是每 batch 的
     class-text 条件与重新编码路径。
  10. **朴素基线**（用 GT 类别 id，模型拿不到）：5k 类对查表 20.67%、50k 类对查表 32.42%、
      恒预测众数 5k 6.79% / 50k 8.53%。模型 23.84% = 比 5k 查表高 3.17pp，比 50k 查表低 8.58pp。
  11. `_compute` 不返回 `logits_raw`，走它的 loss 路径会静默 fallback 到 scattered logits，
      NCE 凭空低 0.602 nat。主训练路径在 `trainer.py:306-307` 手拷 raw 绕过。
  12. `sigmoid` + `bg` 在膨胀 `logit_scale`（5.0→9.87），而 NCE 用固定 `cos/0.07` 绕过它，
      该参数是保序变换，对指标无影响。

### 没做什么

- 没有改任何代码（本轮只有审计 + 训练 + 评测）。
- 没有跑 50k 全量。
- 没做消融（关 field / 单专家 / rect=False / 合并同义词），所以各项对指标的因果贡献
  只有方向与上限，没有精确数字。
- 没有修 last-wins / rect / text_encoder.eval() / cache hash / `_compute` logits_raw /
  死掉的 gate_mlp —— 等用户决定优先级。
- 没有 commit / push。
- 临时脚本 `/tmp/run_5kfix.py`、`/tmp/eval_5kfix2.py` 已删除。
- 注：审计过程中有 agent 短暂触发仓库的 error-report 钩子生成 `logs/error_report_*.txt`，
  其中一个 agent 自行删除了 `26.11.02_6..10` 这几份；这些是自动生成的报错记录，不影响代码。

---

## 更新 2026-11-02 三处一行级修复 + 全量 50k × 20 轮

用户选定：先修 3 个确定性 bug，再跑全量 50k 20 轮（约 28h）。

### 做了什么

1. `relate/dataset.py:536`：`rect=cfg.rect or mode == "val"` → `rect=bool(getattr(cfg, "rect", False))`。
   验证集不再被强制矩形 letterbox，与训练的方形 640 对齐。
2. `relate/namespace.py:140`：`rect=True` → `rect=False`。离线 `m.rel.val()` 走的是这条独立
   路径，上面那处管不到它。
3. `relate/head.py::_compute`：向外传 `logits_raw`。原来 `_losses` 用
   `out.get("logits_raw", out["logits"])` 兜底，走 `_compute` 的 loss 路径会静默 fallback 到场
   散列后的 logits（NCE 凭空低 0.602 nat），主训练路径在 `trainer.py:306-307` 手拷 raw 绕过了。
4. `relate/trainer.py::_freeze_primary`：`_find_relate_module` 的返回值不再丢弃，并在冻结后补
   `text_encoder.eval()`。文本塔不在 `m.model` 里，原循环管不到；`build_optimizer` 里设的
   eval() 每个 epoch 都被 `_model_train` 翻回 train，20 轮后 `num_batches_tracked` 涨了 20×156、
   `running_mean` 最大偏移 0.38。

验证：`py_compile` 全过；`tests/test_glp_field.py` **24/24 通过**。

### 启动

- `/tmp/run_50k.py` → `runs/test/glp_50k_fix3_e20`，全量 50k（fraction=1.0）、20 epoch、
  batch 32、lr0=1e-3、seed=0、M=8、其余超参不变。日志 `/tmp/quick50k.log`。
- 选 fraction=1.0 的原因：5k 子集是排序前 5000 张，含 **0% Objects365**，而 val 的 GT 行
  61.84% 是 Objects365；第一张 Objects365 在排序第 20402 位，任何 fraction<0.41 都取不到。

### 没做什么

- 没改 `_gt_grid` last-wins（需要 multi-hot 目标，会改变方法）、没改 field 的 8 候选预算、
  没改 `neg_lw` 的近义类分母、没合并同义词 family。这些都是架构级改动，等 50k 基线出来再定。
- 没改任何超参（epochs/imgsz/batch/lr0/M/seed 与 5k 那次一致）。
- 没有 commit / push。
- `/tmp/run_50k.py` 在跑，跑完删除。

---

## 更新 2026-11-02 50k 中途 SGCls 评测：三处修复有效，但 50k 数据基本没换来收益

### 做了什么

1. **同一协议重测两个 checkpoint**（`/tmp/ev_sgcls.py`，用完已删）。协议统一为
   `protocol="sgcls"`、val=`RA-4M/full_monitor`（1024 图 / 9194 GT 对）、GT 框 oracle、
   `rect=False`（与训练一致的方形 640 letterbox）、batch 4。评测进程用
   `torch.cuda.set_per_process_memory_fraction(0.20)` 硬顶显存，避免挤爆同时在跑的训练。
   - `runs/test/glp_5kfix2_e20/weights/last.pt`（5k、20 轮，已跑完）
   - `runs/test/glp_50k_fix3_e20/weights/last.pt`（50k 全量、训练到第 15/20 轮）

   | 指标 | 5k e20 | 50k e15 | 差 |
   |---|---:|---:|---:|
   | R@20 | 0.21340 | 0.21808 | +0.47pp |
   | R@50 | 0.24190 | 0.24690 | +0.50pp |
   | R@100 | 0.25092 | 0.25506 | +0.41pp |
   | predicate_acc | 0.24070 | 0.23842 | −0.23pp |
   | GT_MRR | 0.06808 | 0.05902 | −13.3% |
   | pair_existence_AP | 0.46015 | 0.28533 | −38.0% |
   | mR@50 | 0.01181 | 0.01034 | −12.4% |

2. **rect 修复的单独效果（同一 checkpoint，只改预处理）**：
   5k e20 在 `rect=True` 下 R@50=0.2378，`rect=False` 下 R@50=0.2419，**+0.41pp**。
   修复方向正确，但量级很小。

3. **验证 text_encoder.eval() 修复**：把 50k e15 checkpoint 的 BN buffer 与一级权重
   `best.pt` 逐项比对，`text` 的 8 个 `num_batches_tracked` 和 8 个 `running_mean`
   **全部逐位相同**（max|d|=0.0000）。旧 20 轮 5k 跑完时 `num_batches_tracked` 涨了 20×156、
   `running_mean` 最大偏移 0.38。修复生效。

4. **数据侧统计（解释 pair_existence_AP 崩塌）**：
   | 数据集 | 图数 | 框/图 | 关系/图 | 有序对正例率 |
   |---|---:|---:|---:|---:|
   | 训练 50k 全量 | 49999 | 5.54 | 9.91 | **0.608** |
   | 训练 5k 子集（排序前 5000） | 5000 | 7.82 | 11.85 | **0.509** |
   | 验证 full_monitor | 1024 | 5.57 | 8.98 | **0.517** |

   5k 子集与验证集的正例率几乎一致（0.509 vs 0.517），50k 全量高出 9pp。
   训练集先验与验证集错配，与 `train/rel` 从 0.0881 升到 0.0920（违反下降趋势）
   和 `pair_existence_AP` 腰斩互为印证。

### 结论

- 三处修复都有效，但**修 bug 不等于指标能上去**：rect 只值 +0.4pp。
- **数据量 5k→50k（10 倍图、10 倍步数）只换来 R@50 +0.5pp**，同时
  `pair_existence_AP` −17.5pp、`GT_MRR` −13.3%。**50k 这一轮基本没有效果。**
- 模型 R@50=0.2469 仍低于「用 GT 类 id 查 50k 共现表」的朴素基线 0.3242。
  也就是说模型并没有把 50k 的共现统计吃进去。
- 与四路底层审计一致：瓶颈不在数据量。文本嵌入近乎退化（非对角余弦均值 0.858，
  "above"↔"below" 0.9604）、场头对 MRR 是净负、`alpha`/`gate_mlp` 是死代码。
  这些不解决，加数据不会有收益。

### 没做什么

- 没有停 50k 训练（还剩 5 轮，约 3h），也没决定是否停。
- 没有改任何代码、超参、数据。
- 没有跑多 seed，因此 +0.5pp 不能区分是数据收益还是训练随机性。
- 没有对 `_gt_grid` last-wins、field 8 候选预算、`neg_lw` 分母、同义词 family 做改动。
- 没有 commit / push。

### 环境风险

- **`/` 卷 30G 只剩 1.1G（97%）**。中途把 checkpoint 拷到 `/tmp` 时一度写满磁盘，
  已 `rm` 复原。后续所有大文件放到 `/root/rivermind-data`（101G 可用）。
  Claude Code 自己的临时目录也在这个卷上，写满会让工具直接失效。

---

## 更新 2026-11-02 数据集适配性诊断：标签空间形态和方案不匹配（不是数据量的问题）

前提动作：按用户「我同意你的想法」停掉了全量 50k 训练（`pkill -f /tmp/run_50k.py`），
停在 15/20 轮，`runs/test/glp_50k_fix3_e20/weights/{last,best}.pt` 保留。`/tmp/run_50k.py` 已删。

### 做了什么（只读诊断，`/tmp/ds_diag.py`、`/tmp/ds_diag2.py`，用完即删）

**A. 谓词词表的真实形态**

| 量 | 值 |
|---|---:|
| 名义词表 V | 10,370 |
| 训练关系实例 | 495,464 |
| 频次分布的指数熵 → **有效词表** | **148.6** |
| 训练实例 ≤4 的谓词 | 7,039 个（**67.8% of V**） |
| 训练实例 ≤1 的谓词 | 4,391 个（42.3% of V） |
| top-10 / top-50 / top-1000 谓词覆盖实例 | 48.5% / 72.5% / 93.4% |
| top-149 覆盖 **val** 实例 | **94.45%** |

**B. 谓词是开放式改写短语，不是受控词表**

- 平均 2.89 词/谓词。长尾样本：`sharing the bathroom environment with`、
  `making shushing gesture towards`、`holding the reins of`、`covered in seasoning from`。
- 归一化 token 集合完全相同的组 **1,133 组**（`in front of`/`front of`；
  `to the left of`/`on the left of`/`left of`；`wearing`/`wearing by`/`wearing on`），
  覆盖 **55.18%** 的关系实例。平均每组 2.6 个名字。
- 纯虚词谓词 5 个（`on`/`in`/`is`/`a`/`of`），覆盖 4.29% 的训练实例。

**C. 标签稠密自动生成，且 23% 被训练管线丢掉**

- 每图 R 行 / 有序对数 = 0.608（中位 0.583），18.6% 的图 ≥100%（满密度）。
- 381,316 个有标注的 (sub,obj) 对中，**25.5% 带 ≥2 个谓词**
  （例：同一对同时标 `above` 和 `to the right of`；`riding`/`jumping`/`above`）。
- `_gt_grid` last-wins 因此直接丢弃 **114,148 条 R 行 = 23.04%** 的监督信号。
- 长尾不是单一数据源的问题：o365 有 6.1% 实例落在排名>1000 的长尾，其余来源 7.6%。

**D. train/val 覆盖没问题**

- val 9,194 实例中，谓词在 train 出现过的占 **99.90%**（仅 9 个未见过）。
- val 实例对应的类对在 train 完全没见过的仅 2.18%。
- val 实例的 train 支撑中位数 = 13,971。
- ⇒ 不是「val 词表没在 train 出现」的问题。

**E. 类对条件熵与朴素基线（同一 9,194 实例）**

- train 出现过的 (sub_cls, obj_cls) 对 = 19,393；H(pred | class_pair) = **2.664 nat**
  ⇒ 给定两个物体的类别，平均还剩约 **14.35 个等效谓词**要选。任务本身不是退化的。
- 类对查表 top-1 = **0.3223**；常量众数基线 = 0.0853。
- 模型（50k e15，SGCls）= **0.2384** ⇒ 比「查类对众数」低 **8.39pp**，
  比常量众数高 15.3pp。
  注意口径：查表用的是 GT 类 id，模型拿到的是冻结检测器的视觉特征，信息不完全对等。

### 结论

1. **数据本身可用**：覆盖、来源、框、train/val 重叠都正常。
2. **不匹配的是标签空间形态**：我们把它当成「10,370 类闭集谓词分类」，
   但它的实际形态是「约 149 个有效谓词的开放式改写集合 + 一个统计上不可学的长尾」。
   67.8% 的类只有 ≤4 个样本却占满 softmax 分母。
3. **三条可量化的机制**解释了之前观察到的现象：
   - 同义改写各自成类、只标一个 ⇒ NCE 分母里全是假负例，这解释了
     「NCE 卡在 7.09（p(GT)≈0.08%）而 top-1 还有 24%」的脱耦。
   - 有效词表只有 ~149 ⇒ 5k 子集（49.5k 实例 ≈ 每类 33 个）**已经够**；
     50k 加到 495k 实例对 149 类的覆盖提升很小，所以只 +0.5pp。
   - 50k 训练集正例率 0.608 vs val 0.517 ⇒ 先验错配，
     这最可能是 `pair_existence_AP` 从 0.460 腰斩到 0.285 的原因。
   - 模型打不过类对查表（23.84% vs 32.23%）⇒ 它连 P(pred|类对) 这个
     19,393 项查找表都没学到，说明 10,370 类 softmax 的判别难度把它压住了。
4. **last-wins 丢 23% 监督**是纯管线缺陷，不是模型能力问题。

### 没做什么

- 没有改任何代码、超参、数据。没有合并同义族、没有截断词表、没有改 multi-hot 目标。
- 没有做「把词表限制到 top-K 后重测」的模型侧探针（需要 dump 逐实例预测，
  或屏蔽 `vocab_head.W` 的长尾行；后者会改变 top-M 候选选择，结果需要小心解读）。
- 没有重跑 50k 的 16–20 轮。
- 没有 commit / push。

## 更新 2026-11-02 代码 + 数据集全面审核（ponytail-audit）：数据集是结构错配，同时代码有 3 处静默丢监督

范围：`ultralytics/models/yolovo/relate/`（14 个文件，4256 行）+ `RA-4M` 标签缓存 + 数据来源 README。
只读审计，未改任何代码/数据/超参。

### 做了什么

1. 逐行读完 head.py / dataset.py / trainer.py / field.py / val.py / wiring.py /
   field_metrics.py / loss.py / ra/ontology.py / gen.py，并 grep 了调用点。
2. 重写只读脚本重算标签统计（脚本在 /tmp，已删）。**修正了上一次报告里的三个数**：
   - `top-149 覆盖` 是 **val 95.78%**（train 79.73%），上次口径混了。
   - `H(pred|类对)` = **2.01 nat ≈ 7.46 个等效谓词**（只统计类对在 train 出现的 97.82% 行）；
     上次报的 2.664/14.35 是加了平滑并含未见类对的口径。查表 top-1 复现为 0.3210（上次 0.3223）。
   - 同义族：**按「去停用词后的词袋」归并是 870 组 / 2180 谓词 / 覆盖 52.84% 实例**；
     按「原样词集」只有 4 组。上次写 1133 组/55.18% 归一化口径不可复现，作废。
3. 发现 RA-4M 不是 `gen.py` 生成的：它是外部发布语料（VLM gemma-4-26B 标注 + 几何门过滤 11.3%），
   README 明确写「synonyms are never collapsed；surface-form diversity is the label space」
   「tail strings 是词表覆盖，不是平衡标签集」「无人工精度审计」。
4. 发现 `full_monitor` 只是 1024 图子集，而 README 的预期 val 是 `packs/megasg/val`
   （24,964 图 / 226,203 关系 / 2,222 个谓词）。

### 代码侧确认的缺陷（按严重度）

1. **swap 方向损失实际被关闭**：`ra/ontology.py:59` 把 `pos_mask` 对角线填 True，
   `:65` 的 `sym = pos_mask.any(-1) & ~inverse_mask.any(-1)` 因此对所有非反义词恒 True；
   `head.py:595` 直接 `continue`。results.csv 里 swap 恒 ≈0.0128 即此原因。
2. **`_gt_grid` last-wins**（`head.py:185-188`）丢 114,148 条 R 行 = 23.04% 训练监督
   （val 19.07%），且使 SGCls 的 R@K 有 0.8093 的硬上限（一个有序对只能命中一条）。
3. **`neg_lw` 口径不一致**：`ontology.py:61-63` 造的是乘性权重 ∈[0.25,1]，
   `head.py:584` 当 log 权重直接加到 logits 上。远处负列被 +1.0 nat（2.72×），
   同义列只到 +0.25，同义折扣实际只有设计值的 1/2。该表 float32 [10370,10370] = 430 MB 显存。
4. `_VocabHead.gate_mlp` 从不进计算图（`alpha` 只在 `set_vocabulary` 算一次），
   「W-routed 双专家路由」实为**初始化时冻结的随机向量**；65k 参数白占优化器。
5. `n_candidates=8`：场头只能重排 raw top-8，且 scatter 会把 top-8 分数替换成场分数——
   场头能把 top-8 压低、让 rank≥9 的列胜出，但**永远无法把 rank 9 的 GT 提到第一**。
6. `dataset.py:285` 写入 `get_hash(...)+fp[:16]`（32 字符），`:355` 用 `get_hash(...)`（16 字符）比对，
   **永不相等** ⇒ relate cache 从来不会被复用，每个进程重扫全部标签（安全但白写）。
7. `val.py:349` `fitness = F1@50`，训练期验证器协议仍是 `sgdet`；且 F1@50 =
   R@50 与 mR@50 的调和平均，**没有精度项**。best.pt 因此由检测质量 + 一个噪声 macro 决定。
8. `mR@50` 分母是 val 出现过的 436 个谓词，其中 **272 个（62%）只有 ≤2 条实例**；
   F1@50 实际被这些不可学的谓词主导（≈0.022）。
9. 死代码：`loss.py` 173 行（`tasks.py:2922` 那条路径在 `rel=None` 时就 return，
   构造不到 criterion）、`field.py:381-471` 的 compose/gate ≈90 行无调用、
   `head.py:32` import 的 `field_grid/field_stats` 未使用、`RelateEvaluator.pred_correct` 未使用。

### 数据集侧确认的数字（今天重算）

- V=10,370；train 49,999 图 / 495,464 R 行 / 381,316 个不同有序对。
- 有效词表 exp(H)=148.6；出现过的谓词 9,848；≤4 例 7,039 个（67.8% of V）；≤1 例 4,391 个（42.3%）。
- 覆盖率：train top-10 48.51% / top-100 77.38% / top-149 79.73%；val top-100 93.84% / top-149 95.78%。
- 同义族（去停用词词袋）870 组覆盖 52.84% 实例；纯虚词谓词 5 个覆盖 4.29%。
- 一对多谓词：25.55% 的有序对带 ≥2 谓词；last-wins 丢 23.04%。
- 密度（R 行 / 有序对）：train 0.6082、val 0.5196（这是 `pair_existence_AP` 下降的一个可能来源）。
- val 99.90% 的行其谓词在 train 出现过；10.53% 的行其「谓词×类对」组合在 train 没出现过。
- H(pred|类对)=2.01 nat；查表 top-1 0.3210；常量众数 0.0853；模型 0.2384。

### 结论

数据集不是「脏」或「太少」，而是**为开放词表设计的语料被当成了封闭 10,370 类分类集**：
同义不合并、长尾是覆盖不是标签、val 预期是 2.4 万图而我们用 1024 图。这是方案与数据的结构性错配。
同时代码侧有 3 处会静默吃掉监督/让损失失效的缺陷，所以「指标上不去」不能全算在数据头上。

### 没做什么

- 没改任何代码、超参、数据；没重训；没跑修复后的对照实验。
- 没有把 `pos_mask`/`neg_lw` 的实际矩阵打出来（需要加载 checkpoint 的 W 走 MobileCLIP 路径）。
- 没有审 `predict.py` / `results.py` / `ra/*.py` 里除 ontology 之外的模块（只按调用面看过）。
- 没有 commit / push。

## 更新 2026-11-02 修复审核确认的三处缺陷（sym / gt_hot 多热 / cache 指纹）

### 做了什么

1. **`ra/ontology.py`：`sym` 修正（1 行）**
   `sym = pos_mask.any(-1) & ~inverse_mask.any(-1)` 因为 `pos_mask.fill_diagonal_(True)`
   恒为 True，导致 `head._losses` 对几乎全部谓词 `continue`，swap 方向损失被整体跳过。
   改为 `sym = (pos_mask.sum(-1) > 1) & ~inverse_mask.any(-1)`（恢复作者原意：
   「除自己外还有同义兄弟」才算对称谓词）。CPU 冒烟里 swap 项从 0 变为 0.1788。
   注意：按这个既有启发式，同义族内的谓词仍被豁免；swap 实际覆盖约 47% 的实例。

2. **`head.py` `_losses`：GT 目标改为全量多热**
   原实现从 `_gt_grid` 的 last-wins 单标签建 `gt_hot`，同一有序对的第 2..k 个谓词
   被覆盖（训练集 23.04% 的关系行）。改为按 `gt_list` 直接建 `[K,V]` 多热
   （一次 `nonzero`，无逐 GT 同步）。sigmoid/nce/bg/rel 四项同时生效多标签。
   同时删掉 `_losses` 的 4 个死参数：`q_spa` / `geo19` / `labels` / `force`
   （函数体内从未使用），并同步 `_compute` 与 `trainer._compute_loss` 两个调用点。

3. **`dataset.py`：relate cache 指纹比对修正（1 行）**
   写入用 `get_hash()+fp[:16]`（32 字符），读取用 `get_hash()`（16 字符），永不相等
   ⇒ 缓存从未被复用。改为 `assert cache["hash"] == self._cache_hash()`。
   实测：首次构建 1.05 s（重建缓存），第二次 0.21 s（命中）。

4. `_gt_grid` 补 6 行 docstring，说明它仍是 last-wins，只用于候选强制与诊断。

### 验证

- `py_compile` 四个改动文件通过。
- CPU 冒烟（不建模型，直接调 `YOLOVORelate._losses`）：七项损失全部有限，
  `swap` 非零，`sum(terms).backward()` 通过。
- `build_ontology` 断言：有同义兄弟/无反义的谓词 sym=False，反义词 sym=False。
- relate cache 复用实测见上。

### 没做什么（含原因）

- **没改 `neg_lw`**：进一步分析后发现，`ontology.py` 造的是乘性权重 ∈[0.25,1]，
  head 当加性 bias 用。虽然单位不一致，但当前效果恰好是「远处负列 +1.0、同义列 +0.25、
  同义掩码列 0」，**相对排序是对的**，只是整体多了一个常数偏移。改成 `log()` 会把
  同义列的折扣反过来抹平（掩码把 pos_mask 列置 0，在 log 口径下等于不降权）。
  所以这项按「不改」处理，只记录。
- 没改 `n_candidates=8`、没改 `fitness`/验证协议、没删 `loss.py`/`field.py` 的死代码
  （删除会和既有 checkpoint 的 state_dict/导入路径耦合，收益小于风险）。
- **没有重训**。这三处都会改变损失数值口径：swap 从 ≈0.013 变成非零，
  sigmoid/nce 的 GT 变多热。修复前后的 results.csv 曲线**不可直接对比**。
- 没有 commit / push（工作区不是 git 仓库）。

## 更新 2026-11-03 三路并行深审：bug 为何反复出现 + 指标为何不动

### 做了什么

- 50k × 20 轮（`glp_50k_fix4_e20`）由用户手动 kill，停在**第 8 轮**。8 轮指标无趋势：
  `nce` 8.08→7.92、`R@50` 3e-4~1.5e-3、`mR@50` 2e-5~6e-5、`pair_existence_AP` 0.05~0.19、
  `DetPairRecall` 恒等于 0.01925（177/9194）。
- 三个 subagent 并行审计：① 代码根因，② 开源 SGG/PSG 实现调研，③ 数据与评测链路。
- 自己复核了其中两条最关键、且互相矛盾的结论（见下）。

### 复核结果（与 agent 报告有出入的地方以本节为准）

1. **检测器确认是错的**：`best.pt` 的 `names` 是 **DIOR 20 类航拍**
   （airplane / airport / bridge / harbor / ship / storagetank / windmill / vehicle…）。
   数据集是 Objects365 街景 + 497 类。`DetPairRecall` 恒为 0.01925 是因为检测器冻结、
   输出逐 epoch 完全一致，不是巧合。SGDet 下所有指标都只在 1.9% 的样本上有定义。

2. **坐标系：评测路径没有 bug（推翻 agent 的说法）**。实测非方形图（768×1024，pad 80）：
   标注 `cy=0.587891, h=0.803385`，`batch["bboxes"]` 里是 `cy=0.565918, h=0.602539`
   = `0.75·cy+0.125` 与 `0.75·h`，说明 `LetterBox._update_labels` 后又被 `Format`
   归一化到 letterbox 尺寸；预测框 `b/640` 同空间。**GT 与预测同坐标系。**
   真正的不一致在推理侧：`predict.py:75` 用 `r.boxes.xywhn`（原图归一化），
   与训练/验证的 letterbox 归一化不同，非方形图上 `score()` 的几何特征分布偏移。

3. **`F1@50` 不是"缺精度项"**：`F@K = 2·R@K·mR@K/(R@K+mR@K)` 是 SGG 文献里的标准 F@K。
   问题不在公式，在它被用作 best.pt 选择指标时 R 与 mR 都被检测噪声支配。

4. **last-wins 仍有三处消费确认**：`_gt_grid`（`head.py:178-194`）只保留同一有序对的
   最后一个谓词，仍被强制候选（`head.py:449` → `_select_candidates` 的 `gtp`）、
   `trainer.py:311` 的 `slot_labels`、`field_metrics.py:113` 消费；
   上次的多热修复只落在 `_losses`（`head.py:547-558`）。训练集 23.04% 的关系行、
   验证集 35.90% 的行受影响，可达上限 80.93%。

### 关键数字（本轮新增）

- `ln(10370) = 9.25`；加 `neg_lw` 偏置后的随机基线约 9.75；8 轮训练后 NCE = 7.92。
- `mR@50` 的分母是验证子集里实际出现的 **436** 个谓词（不是 10,370）；
  其中 62% 只有 1–2 个样本。
- `predicate_acc` 的分母 `pa_denominator` 只有 117~132，先被 `DetPairRecall`（177 行）
  限制，再被候选采样限制。
- 文献侧：VG=50 类、PSG=56、GQA=100，最大的 curated 基准 VG-1800 也只有 1,807 类
  且要求每类测试样本 ≥5；**10k 类 closed-set softmax 无先例**。

### 没做什么（含原因）

- **没改任何代码、没动超参、没重训**：本轮是审计，方向未定。
- 没换检测器：换掉后所有既有数字不可比，需要用户拍板。
- 没有把 `n_candidates=8`、last-wins 三处消费、`predict.py` 坐标一并修掉：
  这些都是「改完必须重跑」的项，等方向确定后一次做完。
- 没有 commit / push。

### 补充：proxy50k 谓词长尾实测（50,000 张、495,470 条关系行）

| 阈值 | 谓词数 | 覆盖关系行 |
|---|---|---|
| ≥1 | 9,848 | 100.0% |
| ≥2 | 5,457 | 99.1% |
| ≥5 | 2,809 | 97.7% |
| ≥10 | 1,883 | 96.5% |
| ≥20 | 1,308 | 94.9% |
| ≥50 | 788 | 91.6% |
| ≥100 | 378 | 85.6% |

- 每个谓词出现次数的**中位数是 2**；52% 的谓词（5,457 / 9,848）只出现 1–2 次。
- top-10 谓词各约 1.3–3.7 万次。
- 多谓词有序对 97,424 个。
- 结论：10,370 路 softmax 里有一半以上是「几乎无法学习」的类；截断到 ≥5 样本
  （2,809 类，保留 97.7% 的关系行）与文献最大的 VG-1800（1,807 类、要求 ≥5 测试样本）
  是同一量级。

## 更新 2026-11-03 方案 A：修测量（PredCls 评测 + last-wins 三处消费）

### 做了什么

1. **`head.py`：多谓词 GT 的候选强制（真源合并）**
   新增 `gt_slots`：按 `(slot, pred)` 的稀疏索引，由 `sub_idx/obj_idx` 与 `gt_list` 向量化
   比对得到（同 `_losses` 里 gt_hot 的写法），放进 `out` 供两处复用。
   `_select_candidates` 增加 `gt_slots` 参数，把同一有序对的**全部** GT 谓词置为 `+inf`
   再 topk；原来只强制 `_gt_grid` last-wins 留下的那一个。

2. **`trainer.py`：`slot_labels` → `slot_gt`**
   场指标的输入从「每槽位一个谓词」改为「每槽位的全部 GT 谓词」。

3. **`field_metrics.py`：按 GT 行统计**
   - `_topk_correct` 拆成 `_topk_hit`（GT 行级）+ `_slot_hit`（槽位级：命中任一 GT 谓词即算对）；
   - `top1/top5/cand_hit` 分母改为「有 GT 的槽位数」，与旧口径一致；
   - FMA / WBG / AURC 改为按 `(slot, pred)` 行取；
   - 存在性 P/R/F1 不再因为「本图没有 GT」而整块跳过（fp 照常计）。

4. **`val.py`：PredCls 并列评测**
   - `init_metrics` 里多建一个 `RelatorEvaluator` 和一个 `_rel_head` 缓存
     （原来每图重扫 `model.modules()`，只在 sgcls 模式下才跑，现在每图都跑，必须缓存）；
   - `update_metrics` 在 SGDet 主指标之外，用 GT 框再打一次分，写入 `pc_*`；
     `protocol=sgcls` 时这份取代主指标；
   - `print_results` 现在打印合并后的 `rel_stats`，训练时能直接看到 `pc_*`。

5. **`val.py`：`fitness` 换成 `pc_R@50`**
   原为 `F1@50`。SGDet 下它恒 ≈0（冻结检测器压死），等于随机挑 checkpoint；
   `pc_F1@50` 里 `mR@50` 的分母 62% 是 ≤2 样本的类，同样由噪声主导。
   `pc_R@50` 是文献标准头条指标，稳定可解释。

6. **`namespace.py`**：`protocol` 接受 `predcls` 别名（映射到 `sgcls`）。
   本实现只喂框、不预测物体类别，文献里的 SGCls 与 PredCls 在这里同义。

### 验证

- `py_compile` 五个改动文件通过。
- 单元冒烟（/tmp/smoke_a.py，已删）：多谓词全部进候选、槽位级命中、
  场指标多谓词口径、空 GT 不崩，四项全过。
- **PredCls 端到端**（`glp_50k_fix4_e20/weights/best.pt`，8 轮 checkpoint）：

  | 口径 | 数据 | R@50 | mR@50 | F1@50 | ExAP | pa_denominator |
  |---|---|---|---|---|---|---|
  | SGDet | full_smoke 64 图 | 0.0000 | 0.0000 | 0.0000 | 0.1115 | 5 |
  | pc_（同一次运行） | full_smoke 64 图 | 0.2486 | 0.0607 | 0.0976 | 0.4219 | 543 |
  | PredCls | full_monitor 1024 图 | **0.2796** | 0.0124 | 0.0238 | 0.4691 | **9194** |

  SGDet 与 PredCls 差 **300 倍**，`pa_denominator` 从 5 → 9194。
- **训练 1 轮冒烟**（full_smoke 64 图，batch 4）：跑通，CSV 里出现
  `pc_n_imgs / pc_R@20 / pc_R@50 / pc_mR@50 / pc_F1@50 …` 共 36 列，
  `R@50=0.0018` 而 `pc_R@50=0.0755`，`best.pt` 正常保存。

### 没做什么（含原因）

- **没有重训**：本轮只改测量，不产生新的精度数字。
- 没动词表（10,370）、没动 `n_candidates=8`、没换检测器：属于方案 B/C。
- 没把 `predict.py:75` 的原图归一化框改成 letterbox 归一化：只影响推理，且要
  和 capture 的 `ratio_pad` 对齐，单独一次改动做。
- 没修 `tests/test_glp_field.py` 的旧签名（仍是 15 参，跑不起来）：改动面超出本轮。
- 没有 commit / push。

## 更新 2026-11-03 诊断闭环 + 词表截断（v5）

### 1. 诊断（全部只读，未改代码）

- **过拟合测试**：100 张图 × 200 轮（2600 步），训练集谓词 top-1 从 0.024 涨到 **0.856**、
  top-5 0.961，`nce` 9.29→5.48。架构能拟合。
  另一组 205 图 × 30 轮（780 步）只到 0.187 —— 说明是步数不够，不是容量不足
  （先前据此判断「架构不能拟合」是错的，已推翻）。
- **场头消融**：把 `rel.field_read = None` 后重评 PredCls，
  `R@50 = 0.2753` vs 有场头 `0.2815`、`pair_existence_AP` 完全相同 0.4876。
  → 场头既不帮忙也不捣乱，是个惰性模块（`field_fma` 200 轮恒为 −0.66）。
- **谓词嵌入**：`_VocabHead.W` 是 `register_buffer`，`requires_grad=False`，
  全程冻结（只在 `set_vocabulary` 里 `F.normalize` 一次）。最近邻余弦中位数 **0.9668**，
  80.3% 的谓词 ≥0.95；0.95 阈值下连通分量退化成含 7,448 个谓词的巨块。
- **词表频次**（proxy50k 50,000 图 / 495,470 行）：9,848 个谓词中
  **7,039 个（71%）样本数 <5**，它们只贡献 2.3% 的关系行；每类出现次数中位数是 **2**。
- **训练曲线**：`glp_50k_fix5_predcls_e20` 5 轮（第 5 轮后手动停）：
  `pc_R@50` = 0.2540 / 0.2350 / 0.2470 / 0.2815 / 0.2459，无趋势；
  `nce` 8.078→7.878。旧 run 15 轮 nce 也只到 7.09 —— 延长训练不是解。

### 2. 词表截断（方案 1）

按训练集出现次数 ≥5 截断：9,848 → **2,809** 个谓词，保持原顺序、id 重映射。

生成的是**新数据视图，原数据未改动**：

- `RA-4M/proxy50k_v5/{images,labels}`：图像符号链接、标签 R 行重写（未保留的行删除）；
- `RA-4M/full_monitor_v5/{images,labels}`：同上；
- `RA-4M/data_proxy50k_v5.yaml`：`names` 不变，`rnames` 为 2,809 条。

实测保留率：

| | 保留 | 丢弃 | 保留率 |
|---|---|---|---|
| 训练关系行 | 484,217 | 11,253 | 97.73% |
| 验证关系行 | 9,171 | **23** | **99.75%** |

验证集几乎不受影响（9,194 → 9,171）。`W` 由 `attach_relate_layer` 从新 `rnames`
重新编码，运行时零改动。

### 3. 验证

- yaml `names=497 / rnames=2809`；`full_monitor_v5` 标签最大 rel id = 2,748 < 2,809。
- `build_relate_dataset` 加载成功：样例 10 个框、9 条关系。

### 4. 启动

`glp_50k_v5_e20`：`YOLOVO_RELATE_DATA=RA-4M/data_proxy50k_v5.yaml`，20 轮，
seed 0、batch 32、lr0 1e-3 不变，`fitness = pc_R@50`。

### 没做什么（含原因）

- 没动损失权重、没把 `W` 改成可训练参数、没改任何超参：本轮只测「输出空间」这一个变量。
- 没在 `full`（24,964 张）验证集上跑基线：太慢，先用 `full_monitor` 做每轮监控。
- 没修 `field_fma` / 场头：已证明它与主指标无关（去掉后无差别），不属于本轮。
- 没改 `predict.py:75` 的坐标不一致。
- 没有 commit / push。

---

## 4 路并行代码审计（2026-11-04）：v5 平台期的根因

四个子代理分别只读审计 head 前向、损失/训练循环、评测口径、数据管线；我另行做了
纯计数基线和 ckpt 数值复核。**未改任何源码**。

### 1. 主因（证据最硬）：关系头从不接收物体类别 id

- `grep -c "\bcls\b" ultralytics/models/yolovo/relate/head.py` = **0**。
- `score(boxes, ...)` 只吃 `[N,4]` 框（head.py:739）；`_compute(..., box_cls=None)` 的
  `box_cls` 是死参数（head.py:509），`_forward_pairs` 签名里没有它，eval 两个调用点都不传
  （head.py:754/791）。
- 类别 id 唯一被用到的位置是辅助 obj 对齐损失（trainer.py:323 → head.py:647），只训
  `W_obj`/`sub_text_proj`，不进入谓词打分。
- 与文献 PredCls（GT 框 **+ GT 类别**）口径不同；本实现比文献更严格。

**对照实验**（同一批 9,171 个 val GT 对，训练集建表）：

| 方法 | 谓词 top-1 | 宏平均召回(417 类) |
|---|---|---|
| 全局最高频谓词 | 8.55% | — |
| 只按 subj_cls | 21.97% | — |
| 只按 obj_cls | 21.82% | — |
| `(subj_cls,obj_cls)` 查表 | **32.32%** | **6.69%** |
| 当前模型 (`pc_predicate_acc`/`pc_mR@50`) | 27.46% | 1.08% |

模型在同一批 GT 对上低于一张两行 Counter，说明瓶颈不是容量、不是长尾，而是
谓词打分缺少类别先验。

**原型的可分性复核**（`last.pt` 的 W，2809×512）：两两 cos 中位 0.861；
cos≥0.90 近邻数中位 240、均值 392；**cos≥0.95 近邻数中位仅 2（均值 7.4）**，
0.97 以上均值 1.3。→ 0.9 的"密集"是宽相似，真正近重复很少，查询向量可分；
不需要换可训练分类头。顺带确认 `vocab_head.logit_bias` 是**标量** -0.7432，冻结头里
没有任何逐类可调偏置。

### 2. 损失口径缺陷（会主动压住 top-1）

- `neg_lw`（ontology.py:59-63，值域 [0.25,1]）被当作**加性 logit 偏置**直接加到分母
  （head.py:607-610）：cos<0.8 的远负例 +1.0，cos≥0.9 的近义列被清零 →
  最难做的负例几乎不参与，softmax 由容易的远负例主导。多谓词槽位 `index_add_`
  把偏置累加 k 次（训练集 25.55% 的槽位是多谓词）。
- 后果：`train/nce` 被系统性抬高 0.3–1.0 nat，与 ln(2809)=7.93 不可比；
  NCE 下降主要来自压远负例 + 旋转 q，而非学会近义判别 —— 与
  "nce 7.02→6.83 而降，pc_predicate_acc 0.2746→0.2700 不升反降"吻合。
- 最小修复：`lw.index_add_(0, a_idx, torch.log(neg_lw[g_idx]))`；多 GT 用 min/mean；
  反义列不要清零（它们才是最难的负例）。

### 3. `alpha` 是 buffer，`gate_mlp` 永远零梯度

- `_update_alpha` 仅在 `set_vocabulary` 里 no_grad 调一次（head.py:77-80），
  打分只读 buffer（head.py:86-89）；`gate_mlp` 仍被收进 AdamW（trainer.py:223-225）。
- 实测 ckpt：alpha mean 0.4917 / std 0.0034 / [0.478,0.506]，`gate_mlp` 范数与 xavier
  初值吻合 → 一次都没更新。"W 路由双专家"实际是全体谓词固定 0.51/0.49。
- 最小修复：forward 里现算 `sigmoid(gate_mlp(self.W))`（可导），或删掉 `gate_mlp`。

### 4. 场头候选：训练强制 GT / 评测 raw top-M

- 训练用 `inf` 把 GT 强行塞进候选（head.py:726-731），评测不强制（head.py:493）。
  本轮日志 `rel_gt_in_cand=0.4803`：推理时候选约一半不含 GT，场学的是推理时
  不存在的问题。`n_candidates=8` 同时卡上限。
- 最小修复：训练候选也按未散列 raw top-M 构造；GT 只在 top-M 内才算正例。

### 5. 训练/评测预算不一致

- 训练 `geo_budget=400 → final_budget=128`；评测把两者都设成 500（head.py:367-383）。
  按框数分布只有 5.4% 的图 pair 数 >128，**影响有界**，不是主因。

### 6. 评测口径（对外数字不可信）

1. **`pc_R@50` 在本 val 上退化成逐 GT 行 top-1**：61.8% 的图 pair 数 ≤20、93.3% ≤100，
   K=50 基本不做筛选。实测 `pc_R@50≈pc_predicate_acc`（0.2735 vs 0.2746）。与文献
   VG R@50（约 380 pair/图）不可比。
2. **`pc_mR@50`/`pc_F1@50` 由尾类主导**：417 个 val 谓词中 185 个只有 1 行、254 个 ≤2 行；
   宏平均标准误 ≈0.022，而实测值 0.0108 → **1σ 内与 0 无法区分**。K 几乎不起作用
   （mR@50 0.0108 vs mR@100 0.0112）。
3. **`_macro_f1` 把 predicted-only 类计入分母**（field_metrics.py:78-82）→ `rel_macro_f1`
   恒 ≈0.0009，不可跨 ckpt 比。
4. `pc_DetPairRecall`/`pc_PairRecall` 恒 1.0000 是定义使然（PredCls 下框就是 GT 框、
   候选枚举全部有序对），对 PredCls 零信息。
5. `predicate_acc` 结构上限 ≈81.0%：19.0% 的 GT 行是同一 pair 的第 2+ 谓词，而每 pair
   只出一个 argmax。
6. 当前 val 是 1,024 图 monitor 子集（417 谓词，按训练词表重映射丢了 19 个 val-only
   谓词 / 23 行），不是 24,964 图全量 val，也不是开放词表结论。
7. `field_fma` 被框面积支配（-0.6424 而均匀场基线 ≈-0.673），不是符号 bug，不能读成
   对齐好坏。

### 7. 数据侧（当前 run 不触发）

- 词表一致性全程无校验（namespace.py:42-51、trainer.py:97-101）：v5 与原词表只有前
  84 个 id 同名，旧 ckpt 续训/评 v5 会静默错 97% 谓词。本次 fresh attach，不受影响。
- v5 是符号链接数据集，`scan_relate_label` 对损坏 JPEG 会 `save(im_file)` 写穿 symlink
  改到原始图（dataset.py:243）。本次 0 次触发。
- 无自环 / 重复 R 行守卫；无"索引列必须为整数"校验（实测数据全干净）。
- `train.py` 未提交改动把 `YOLOVO_CACHE` 默认从 0 改成 1，下次不设环境变量会尝试
  把 5 万图缓存进 RAM。
- 已确认：v5 重映射全量核对 0 处不一致；train/val 零泄漏（文件名/realpath/md5 交集均 0）；
  GT 与 PRED 坐标空间一致。

### 8. 当前 run 状态

`glp_50k_v5_e20` 跑到第 3 轮（epoch 2 已出）：

| epoch | train/nce | pc_R@50 | pc_predicate_acc | pc_GT_MRR | pc_mR@50 |
|---|---|---|---|---|---|
| 1 | 7.024 | 0.2735 | 0.2746 | 0.0833 | 0.0108 |
| 2 | 6.826 | 0.2697 | 0.2700 | 0.0744 | 0.0140 |

nce 降、各项不升反降，与第 2 节"在拟合一个被钝化的目标"一致。

### 做了什么 / 没做什么

- 做了：4 路只读审计、计数基线对照、ckpt 数值复核（W 近邻分布、alpha、logit_bias）。
- 没做：**没有改任何源码、没有停训练、没有 commit/push**。`logs/error_report_26.11.04_*`
  是加载 ckpt 触发的仓库报错钩子产物，未删。
- 没做：没有做 `neg_lw`/`alpha`/类别输入的消融训练，所以第 2、3 节的因果幅度只有
  方向和静态统计，没有消融数字。

---

## 修复三项主因 + 评测口径（2026-11-04）

### 做了什么

**1. 关系头补类别输入（`head.py` / `trainer.py` / `val.py`）**

- `YOLOVORelate` 新增 `cls_proj: Linear(512→768)`（xavier，393k 参数）。
- `_forward_pairs(..., box_cls=None)`：在 `v_obj = spatial_pool(...)` 之后
  `v_obj = v_obj + cls_proj(W_obj[box_cls])`。`W_obj` 仍是冻结的类名文本嵌入，
  `cls_proj` 可学；`W_obj` 设备不一致时就地迁移。
- `_compute(..., box_cls=...)` 透传；`score()` / `score_all()` 新增 `cls=` 参数。
- `trainer._compute_loss` 构造 `cls_pad [B,N]` 并传入；padding 槽填 0（不会被任何
  pair 选中）。
- `val.update_metrics` 取 `batch.get("cls_global", batch.get("cls"))` 切出本图 GT
  类别传给 PredCls 打分；SGDet 路径若检测结果里有 `cls` 也一并传（当前 postprocess
  不产出该字段，SGDet 因此仍是无类别输入）。

**2. NCE 的 `neg_lw` 单位（`head.py`）**

`neg_lw` 是乘性权重 ∈[0.25,1]（`ontology.py` 文件头明写 "down-weight ... floor at
min_lw"）。原实现直接相加 → 远负例被抬高 +0.25~+1.0、近义列零偏置。改为
`lw.index_add_(0, a_idx, neg_lw[g_idx].clamp_min(1e-8).log())`，同一槽位多个 GT 时
log 相加 = 权重连乘。`pos_mask`/`inverse_mask` 列仍置 0（GT 列必须留在分母）。

**3. `alpha` 改可学习（`head.py`）**

`register_buffer("alpha", ...)` → `register_parameter("alpha", None)`；
`_update_alpha` 赋值 `nn.Parameter`（`gate_mlp` 只提供初值）。optimizer 照常覆盖
（`build_optimizer` 遍历 `rel.named_modules()`），`_freeze_primary` 每 epoch 把它
重新设为 requires_grad=True。

**4. 评测口径（`val.py` / `field_metrics.py`）**

- 新增 `pair_acc` / `pair_acc_denominator`：按 (sub,obj) 去重，命中该对任一 GT
  谓词即算对。修掉行级 `predicate_acc` 的 19% 结构性上限。
- 新增 `mR@{50,100}_sup_{le1,2_4,5_19,ge20}`：按 support 分桶的宏平均，暴露尾类噪声。
- `field_metrics._macro_f1` 分母从 `pred+label` 并集改为**只按 GT 类**，避免
  predicted-only 类被恒记 F1=0 把宏平均压成常数。

### 验证

- `py_compile` 四个文件通过。
- **冒烟训练**：`/tmp/smoke_v5.yaml`（train=full_monitor_v5, 1,024 图）跑 1 轮、
  `YOLOVO_EPOCHS=1`，无报错。日志：
  - `关系层可训练 26.68M → 27.07M`（+393k = `cls_proj` 768×512）。
  - 新字段全部进入 results.csv：`pc_pair_acc=0.1134`、`pc_pair_acc_denominator=7432`
    （=val 唯一有序对数）、`pc_mR@50_sup_ge20=0.0257`（其余桶为 0）。
  - ckpt 里 `vocab_head.alpha` 为 2809 维张量，均值 0.4917→0.4976（1 轮 32 步的
    AdamW 量级）；`cls_proj.weight` 范数 24.80→24.83，确认在优化器里。
- 冒烟指标本身（`pc_R@50=0.085`）无意义：只训了 32 步、且训练集=验证集。

### 没做什么（含原因）

- 没改场候选（训练强制 GT / 评测 raw top-8）、`n_candidates=8`、训练/评测 budget
  128 vs 500：这些不在本轮批准的三项内，且已证明有界。
- 没加词表一致性断言、没修 symlink 写穿、没改 `train.py` 的 `YOLOVO_CACHE` 默认 1。
- 没改损失权重、lr、batch、seed（仍 seed 0）。
- SGDet 推理路径拿不到预测类别，会以「无类别输入」打分；PredCls 不受影响。
- `logs/error_report_26.11.04_*` 未清理。
- 没有 commit / push。

### 正式 run

`glp_50k_v6_cls_e20`：v5 词表（2,809 类）、seed 0、batch 32、lr0 1e-3、20 轮，
数据 `RA-4M/data_proxy50k_v5.yaml`。

---

## 更新 2026-11-04 五路并行代码审计（只读，未改源码、未停训练）

### 做了什么

5 个代理并行，分工：head+field（前向/候选/七项损失）、trainer+dataset+train.py
（padding/缓存/freeze/rect）、val+field_metrics+namespace（指标口径）、
wiring+ontology+加载（词表/alpha/ckpt）、实测验证（CPU-only，真实 val 数据）。

### 已验证的关键数字

- `pc_R@50`/`pc_mR@50`/`pc_F1@50`/`pc_GT_MRR` 的理论上限 = 7432/9171 = **0.8104**
  （`score()` 每对只返回 argmax 谓词，多谓词对的第 2+ 行 rank 恒 ∞）。
- `pair_existence_AP` 分母是"预测出的正例数"而非 GT 正例数，SGDet 下放大约 1/PairRecall。
- `sym` 在真实 2809 词表上仍 **96.97% 为 True**（`pos_mask.sum(-1)>1` 形同虚设）。
- `neg_lw` 的 cos≥0.90 近义列被 `masked_fill(...,1.0)` 设回满权重 → 近义负例从未降权。
- `gt_slots` 与 `cls_pad` 实测无漏无多；val 的 `batch["cls"]` 是全局类 id；
  整批 `_forward_pairs` 与逐图 `_compute` 的七项损失最大差 4.8e-6。
- `YOLOVO_CACHE=1` 在 58 GiB cgroup 下永远通不过 `check_cache_ram`（need≈61.5 GiB，
  avail≈42.8 GiB），实际是"静默不缓存"，不是 OOM。

### 没做什么（含原因）

- 没改任何源码、没停 `glp_50k_v6_cls_e20`、没 commit/push。
- 没有做 P1 类问题（neg_lw 豁免、sym、swap 反向对、cls_proj 泄漏、weight decay）
  的消融训练，因果幅度只有静态统计。
- 没修 `tests/test_glp_field.py:416` 的旧 `_losses` 签名，测试套件目前跑不起来。
- 代理脚本报错触发的 `logs/error_report_26.11.04_8/15/16/17.txt` 未删。

---

## 更新 2026-11-04 评测口径修正 + 推理期优化调研

### 做了什么

**1. PredCls 按索引配对（`val.py`）**

`RelateEvaluator(match_by_index=True)`：PredCls 下 pred 框就是 GT 框，IoU 矩阵用单位阵，
等价于按 (sub,obj) 索引配对。原来走 IoU 首匹配，重叠框会把预测行配到另一对。
实测 `predicate_acc` 0.4018 → 0.4219。

**2. `pair_existence_AP` 分母（`val.py`）**

正例改为按唯一有序对去重（取该对最高存在性分），未被任何预测行匹配的 GT 对记入
`exist_missed` 留在分母。原实现除以"预测出的正例数"，SGDet 下被放大约 1/PairRecall：
SGDet 0.1675 → **0.0001**，PredCls 0.4852 → 0.5484。

**3. `pair_acc` 分母（`val.py`）**

early-return 分支（无预测行）补 `pa_pair_total` / `pair_topk_total` / `exist_missed`，
SGDet 的 pair_acc 不再只对"至少产出一条候选"的图求值。

**4. 新增 `pair_acc@k`（`val.py` + `head.py`）**

`score(pred_topk=5)` 额外返回每对 top-k 谓词，评测器算"该对 top-k 命中任一 GT 谓词"。
这是 **ngR@K 口径**的补充指标；标准 graph-constrained R@K 仍是每对 argmax，不能混比。

**5. `field_fuse` 开关（`head.py` + `namespace.py`）**

`field`（覆写，原行为）| `max`（只允许场加分）| `raw`（不用场，消融）。
旧 ckpt 没有该属性，`getattr` 兜底。训练路径默认仍是 `field`，行为不变。

**6. 修复 `tests/test_glp_field.py:416`** 的旧 `_losses` 签名（含 `box_cls` 需 unsqueeze）。

### 实测（`glp_50k_v6_cls_e20/best.pt`，PredCls，1,024 图）

| field_fuse | R@50 | predicate_acc | pair_acc | pair_acc@3 | mR@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| field | 0.4121 | 0.4219 | 0.5206 | 0.7667 | **0.0866** |
| max | **0.4344** | 0.4452 | 0.5494 | 0.7748 | 0.0776 |
| raw | 0.4406 | **0.4509** | **0.5564** | 0.7691 | 0.0365 |

- 场头**对长尾有用、对整体精度有害**：raw 的 R@50 最高但 mR@50 只有 0.0365（field 的 42%）。
- `max` 是折中：R@50 +2.2pp、mR 保留 90%。
- `pair_acc@3` 约 0.77，远高于 (subj_cls,obj_cls) 查表 top-3 的 0.53。

**其余调研结论（未改代码）**

- 提高候选数 M 是负收益：M=8→16/32/64 时 `predicate_acc` 0.4018→0.377/0.336/0.241，
  因为散列覆写会压掉 raw 赢家。
- `logit_scale.exp()=16.78` 但实测 `sigmoid==1.0` 占比 0，不存在排序饱和。
- v6 超过查表基线 7.6pp（0.4018 vs 0.3257）；v5 低于基线 5.1pp。
- 训练第 11–13 轮是平台，之后过拟合；`mR` 在 1,024 图子集上限 100%，全量 val 上限 72.28%，
  不能外推。

### 没做什么（含原因）

- 没改训练目标（neg_lw、sym、swap 反向对、cls_proj 泄漏、alpha 约束、weight decay 分组），
  这些需要重训才能体现，留到下一轮。
- 没把 `field_fuse` 默认改成 `max`：它会同时改变训练期 sigmoid/bg 的输入，属于训练侧改动。
- 没修 `train.py` 的 `YOLOVO_CACHE` 默认 1。
- 没做 M<8 与"完全关闭场散列"之外的更多消融。
- 未清理 `logs/error_report_26.11.04_*`。

---

## 更新 2026-11-04 对 d8a1d3f 的 diff 复审（3 路并行 + 自检）

### 做了什么

三个代理分别审 `val.py/field_metrics.py`、`head.py`、`trainer.py/dataset.py/namespace.py/train.py`
的 diff，逐个用合成张量或真实 ckpt 复现；我另做了 `_ap`/pair_acc 的手算单元校验。修了 12 处：

**行为 bug**
1. `_forward_pairs` 的 `cls_proj` 改为 `getattr`：旧 ckpt（无该层）做 val/续训会 AttributeError。
   `field_delta_max` 同样兜底。
2. `_losses` 里 `box_cls.reshape(-1)`：`_compute` 传 [1,N]、trainer 传 [N]，原先 assert 会炸。
3. `score(pred_topk)` 的 <2 框分支宽度改为 `min(k, V)`，与正常分支一致。
4. `alpha` 设备迁移改用 `.data` 原地搬，避免 `Parameter.to()` 返回 Tensor 导致 TypeError
   和 optimizer 持有旧对象。
5. `gt_slots` 增加框索引/谓词 id 边界过滤，与 `_gt_grid` 一致。
6. `pair_acc@k` 列数固定为 `pair_topk_k`，某轮低 k 无命中时不再整列消失（否则 results.csv 错位）。
7. `pred_topk` 入口搬到 CPU，去掉逐对 `.tolist()` 的 CUDA 同步。
8. `match_by_index` 的守卫从「形状相等」改为「内容相等」（`torch.equal`），避免检测框数
   恰好等于 GT 框数时按索引错配。
9. `namespace.val` 的 `field_fuse` 用 try/finally 还原，不再污染后续 train/predict。
10. `namespace.val` 的 rect 改为取 `overrides["rect"]`，不再写死 False。
11. `_cache_hash` 混入 `rcoords`。
12. `build_optimizer` 分组对齐基类：bias / Norm.weight / logit_scale / alpha 不衰减。

**其它**
- `predict.py` 补传 `cls`，与训练/验证口径一致。
- `train.py` 的 `YOLOVO_CACHE` 默认值改回 0，注释改为实测口径（41 GiB，cgroup 会拒绝）。
- 修正 `trainer.py` 与 `ontology.py` 两处与事实不符的注释。
- `tools/gflops_report.py` 删未用 import，特征取 `base_idx` 而不是第 0 个接线层。

### 未修（记录理由）

- `cls_proj` 让 obj 对齐损失自答（实测 CE 2.05 vs 5.96）：需重训，留到下一轮。
- `sym` 仍 96.97% 为 True（`SYM_COS_THRESHOLD=0.90` 太低）：改阈值会改变训练行为，留到重训。
- `neg_lw` 近义列未降权（吃掉 41.9% NCE 质量）：同上。
- `install_vocabulary` 重建 alpha 会重置学习值：当前路径只装一次，不触发。
- `YOLOVO_CACHE=disk` 仍会被当成 RAM：仓库内无使用方，未改。

### 验证

- `py_compile` 全部通过。
- `tests/test_glp_field.py` **24/24 通过**。
- 重跑 `field_fuse=max` 的 PredCls 验证，指标与修复前逐位一致
  （R@50=0.4344、predicate_acc=0.4452、pair_acc=0.5494、pair_acc@3=0.7748），
  且验证结束后 `field_fuse` 已还原为 `field`。

---

## 更新 2026-11-04 训练目标优化（第二轮）

### 做了什么

基于上一轮的量化证据，改掉五处训练目标缺陷：

1. **`cls_proj` 泄漏**：`_forward_pairs` 保留未注入类别的 `v_obj_base`，`_losses` 的 obj
   对齐损失改用它；trainer 的 `out_i` 同步透传。原实现里 CE 可以直接由注入的
   `W_obj[cid]` 解出（实测 CE 2.05 vs 5.96，base-only top-1 0/55）。
2. **`neg_lw` 近义列**：`ontology.py` 不再把 `pos_mask` 列设回 1.0；`head.py` 只豁免反义列，
   并把 GT 列显式置 0。近义列现在保留 closeness 折扣（实测 88.8% 的列 <1，中位 0.757），
   原先它们吃掉 NCE 分母 41.9% 的 softmax 质量。
3. **`sym` 改显式对称表**：新增 `SYMMETRIC_PREDICATES`，实测 2809 个谓词里只有 8 个
   （0.28%）被判为对称，swap 方向损失不再被 96.97% 的豁免吞掉。
4. **对称谓词的反向对不再强制入候选**：`force = is_gt | swapped` 改为跳过 `sym[p]=True`
   的反向对，避免它们被 sigmoid/bg/geo/rel 当负例（原先覆盖 70% 的关系行）。
5. **`alpha` 直通式截断到 [0,1]**：旧 ckpt 实测跑到 [−0.55, 1.52]，超出混合权重的定义域。

另：`field_fuse` 默认值改为 `max`（训练与评测一致，场头只能加分不能覆盖检索头）。

### 验证

- `py_compile` 通过；`tests/test_glp_field.py` 24/24。
- 真实词表上复核：sym 8/2809；neg_lw 中位 0.757。
- 1 轮冒烟（1024 图）：无报错，`train/swap` 从 0.0139 升到 0.0259（swap 首次覆盖全词表）。

### 没做什么

- 没有改 `n_candidates`（实测 M=8 最优）、budget、lr、batch、seed。
- 没有改 `SYMMETRIC_PREDICATES` 之外的对称判定逻辑：该表是按 VG 常见谓词手写的，
  不是从标注统计学出来的。
- 没有重跑全量 val 做交叉验证。

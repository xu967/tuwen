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

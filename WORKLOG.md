# GLP V1/V2 实现与训练（RA-4M）

- 时间：2026-11-01
- 范围：`ultralytics/models/yolovo/relate/`、`tests/test_glp_field.py`、`train.py`、`RA-4M/glp/`
- 结论：部分完成（代码+单测完成；RA-4M smoke 训练完成，指标未达可用水平，5 epoch 对照结果见 `GLP_RESULTS.md`）

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
- 未推送 GitHub
  - 原因：本机只有 `xu967` 的 SSH key，没有 GitHub token/目标仓库 URL；已初始化独立本地仓库并暂存。

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

# 调研：把关系模型的视觉权重换成 DINOv3 值不值得做

- 时间：2026-11-07 15:10
- 性质：只读调研，未改仓库任何文件，未起 GPU 进程
- 证据来源：论文 `2609.12552v1.md`（行号为该 md 文件行号）、本仓库代码、`assets/relsgg-vits16plus/model.pth`（CPU 只读键名）、`RA-4M/`、curl 探测
- 写法约定：标「事实」的有文件/命令证据；标「估算」「推测」的是本人推算；拿不到证据的写「未验证」

## 1. 结论

1. **值得做，而且是现在排序最高的实验**。差距拆解（`reports/baseline_cmp/README.md:47-68`）已经指出第一位差距是冻结的 YOLO11n-DIOR 颈部 P4 特征不带物体身份与接触语义；论文自己的消融给出了同一方向的证据：去掉像素，官方模型 micro Acc@1 掉 57.5%/68.3%/44.3%（VG150/PSG/IndoorVG），去掉框几何只掉 16.8%/23.6%/6.0%（Tab. 10，标题 `2609.12552v1.md:309`，数字 `:313`）。我们现在的模型只给框时方向类 52.4、动作类 1.3，恰好是一台「只有框几何、没有像素」的机器。换视觉特征对准的就是这个缺口。
2. **推荐做法是 (a) 双骨干，先冻结**：YOLO 继续做检测，DINOv3 ViT-S/16+ 只给关系头供 F_map。改动集中在 `head.py` 的 `_fmap`/`forward` 和 `trainer.py`/`val.py` 的特征获取点，关系头其余部分、损失、评测全部不动，与现有 4 轮基线可背靠背比较。(b) 整体换骨干等于复现 RA，丢掉本项目的前提；(c) 蒸馏进 YOLO 颈部是 (a) 成功之后的部署优化，不是第一步。
3. **权重可以离线拿到**：`model.pth` 内含完整的微调后 DINOv3 ViT-S/16+（`backbone.*` 236 个键，28.69M 参数，hidden 384，12 层，4 个 register token，SwiGLU MLP），且顶层带 `backbone_config`，无需联网即可按 `relsgg/checkpoint.py:50-61` 的方式构建。原始预训练权重：facebook 官方仓库是 gated（HTTP 401），但 timm 转换版 `timm/vit_small_plus_patch16_dinov3.lvd1689m`（114.75 MB）经本机代理可公开下载（HEAD 返回 302→200）。
4. **无泄漏**：`full_monitor` 的 1,024 张图全部在 `RA-4M/images/val`，0 张在 `images/train`；`images/train ∩ images/val = 0`；官方训练 args 里 `val_root = runs/packed/megasg` 只用作 checkpoint 验证损失，发布的是 `checkpoint_last`（epoch 11/12 的 EMA），不是按 val 选的 best。
5. **代价**：ViT-S/16+ 前向估算 448 px 约 57 GFLOPs、640 px 约 139 GFLOPs（2×MAC 口径），对比我们检测器 640 px 实测 8.3–9.0 GFLOPs、关系头 48 GFLOPs（`WORKLOG.md:477-487`）。冻结用法只多前向，不多反向，显存增量估算 < 2 GiB；全量微调估算再加 6–12 GiB（batch 32，448 px）。
6. **对结果的预期（推测）**：接触/支撑类、动作类、靠近类会从个位数明显上来；方向类不会掉（几何分支 `q_spa` 还在）；但 mR@50 要从 1.45 追到两位数还需要同时修文本空间（差距拆解第 5 条），单换视觉特征不够。所以第一个实验的过关标准建议盯「只给框时的 contact/support 与 action/other 组 micro R@50」，而不只是总 R@50。

## 2. 论文证据

### 2.1 骨干阶梯 Tab. 29（`2609.12552v1.md:761-765`，事实）

| 骨干 | OVS-F1 | A1 | A2 | A3 | A4 | A6 | HICO F1 | 参数 | img/s（4×A100 训练） | 峰值显存 GB | 墙钟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| released family（RA-4M + 5% HICO-DET，source-aware negatives） | | | | | | | | | | | |
| ViT-S/16 | 34.6 | 27.4 | 59.6 | 30.6 | 24.3 | 35.1 | 36.4 | 46.1M | 114 | 25.1 | 4:52 |
| **ViT-S/16+（发布）** | **37.2** | 28.6 | 60.8 | 32.5 | 27.4 | 37.9 | 37.1 | 53.2M | 109 | 30.0 | 5:04 |
| ViT-B/16 | 37.2 | 28.9 | 64.4 | 33.4 | 26.8 | 37.2 | 37.5 | 113.8M | 87 | 39.8 | 6:09 |
| zero-shot trio（RA-4M + raw VG，无 HICO） | | | | | | | | | | | |
| ViT-S/16 | 34.7 | 23.2 | 53.0 | 31.1 | 27.0 | 38.7 | 18.5 | 46.1M | 115 | 25.0 | 3:58 |
| ViT-S/16+ | 34.2 | 23.4 | 53.1 | 32.5 | 24.7 | 40.8 | 18.7 | 53.2M | 108 | 29.9 | 4:12 |
| ViT-B/16 | 35.3 | 24.7 | 56.0 | 33.4 | 25.8 | 39.8 | 20.0 | 113.8M | 87 | 39.8 | 5:10 |

参数列含关系头（53.2M 中 backbone 28.7M，见 §4.1）。Tab. 29 没有单独的 R/mR/F1 列，只有六轴分数；OVS-F1 是 A1/A2/A4/A6 的调和均值。

推理成本 Tab. 33（`2609.12552v1.md:839-843`，关系模型整体含骨干，448 px，11 框，bf16）：ViT-S/16 80 GFLOPs、ViT-S/16+ 91 GFLOPs、ViT-B/16 200 GFLOPs；batch 1 在 A40 上三者都是 19.3–20.0 ms（dispatch 受限）；batch 32 吞吐 201/188/130 img/s（A40）。

§G.1（`:787`）与 §8.1（`:345`）的读法：容量不涨综合分（S→S+ 参数 +15% 得 +7.4%，S+→B 参数 +114% 得 −0.1%）；骨干越大越依赖像素、越不依赖框几何（去框几何的损失从 −35.2% 降到 −9.8%，去像素的损失从 −53.0% 升到 −58.1%，VG150）；作者据此发布 ViT-S/16+，并写明「下一个该改进的部件是检测器」。

### 2.2 relation fine-tuning 改变了骨干什么（§7.3、Tab. 28、Fig. 26，事实）

- 架构（§3.2 `:71`，Tab. 14 `:425`）：DINOv3 ViT-S/16+（LVD-1689M），448 px 多尺度 [0.5,1.5]×448 七档，taps −6/−3/−1 各自 LayerNorm 后按学习的 softmax 权重融合，骨干全量微调，lr = 头的 1/8（头 4e-4，骨干 5e-5），12 epoch，batch 32×4 GPU，EMA 0.9998。
- 微调改了多少（`:303`、Fig. 26 `:757`）：与预训练模型的 linear CKA 随深度单调下降到最后一层 0.58；两次不同配方的微调彼此 CKA ≥ 0.94（终点由任务决定）；权重相对漂移每块 6–9%，只有最后一块的 q/k 投影动了 23%/18%；三个 tap 的融合权重从「86% 在最后一层」变成 51%，中层被调用起来。
- 改的是什么（Tab. 28 `:739-743`）：class selectivity 0.242→0.281，relation selectivity 0.133→0.138（基本不变）；「关系伙伴比同类无关物体更像主语」的概率 0.469→0.385（低于 0.5 的随机水平，微调后更差）；接触区特征的 contact advantage 0.356→0.438，但从该特征跨图 1-NN 检索动词只有 0.269（多数类随机 0.424，低于随机），检索物体类别 0.761（随机 0.068）。
- 冻结探针（`:749`）：在**冻结的 box-pooled token** 上做线性探针，predicate-free 的 relatedness（同类对 AUC）预训练就有 0.73（几何基线 0.70），微调不增加（0.73→0.72）；增加的是谓词的线性可解码性（PSG macro-F1 0.445→0.475，VG150 0.264→0.305）。作者解释为「头的梯度够到了它读的特征」，不是骨干内部形成了关系表示。
- 结论句（`:313`）：按现在的方式训出来的场景图骨干「holds no relational representation for a detector to share」。

对我们的含义：**骨干给关系头的主要是物体身份与接触区的外观，不是「关系」本身**。这正是我们缺的东西（`wearing` 0.001、`holding` 0.07、给 GT 类别后分别回到 0.82、0.39，`reports/baseline_cmp/README.md:62-66`）。冻结的预训练 DINOv3 已经携带大部分身份信息（object 1-NN 0.713 vs 微调后 0.761），所以「先冻结」是合理的第一步。

### 2.3 frozen / LoRA / 分辨率消融（§A.1、§H，事实）

- §A.1（`:399`）原话：全量微调（lr 低头 8×）「outperforms both a frozen backbone and LoRA adapters, although the ordering of those two alternatives depends on the checkpoint selection rule and inverts under best-epoch selection」。即 frozen 与 LoRA 都比全量微调差，但二者谁更好取决于选 checkpoint 的规则。论文没有给 frozen 的具体数字（Tab. 30 里没有 frozen 行）。
- Tab. 30（`:799`）可读到的相邻臂：released recipe（全量微调 lr 5e-5）OVS-dev 37.1；「full fine-tune, backbone lr 1e-5」34.1；「LoRA backbone, 8 epochs」33.8；「LoRA backbone, 12 epochs」（50k 代理）30.2 vs 同为 50k 代理的「full fine-tune, backbone lr 5e-5」31.1。即骨干学习率和适配方式对综合分的影响在 3–4 点量级。
- 分辨率：Tab. 31（`:827`）发布模型不重训，在 448/560/672 评测 R@50 52.2/52.3/52.3（VG150），mR@50 28.1/27.7/28.5，基本平；Tab. 32（`:833`）「1.5× resolution yields +10% rare-bucket recall and nothing on micro recall or the spatial axis」。§H.2（`:811`）多尺度训练 +3.7% 是正则化效应不是分辨率效应。
- 没有 224 px 的消融（未在论文中找到）。224 px 下 ViT-S/16 只有 14×14 网格，对 SoftSpatialPool 的小物体不利（推测）。

### 2.4 本仓库与论文配方的对应（事实）

| 项 | 论文/官方 | 本仓库 |
|---|---|---|
| F_map 来源 | DINOv3 ViT-S/16+ 三 tap LN 融合，d=384（`relsgg/model/relsgg.py:47-49,269`） | YOLO11n-DIOR 颈部 P4 经 1×1 conv + GELU 到 768 维（`head.py:5`、`head.py:346-354`），冻结且 `detach()`（`head.py:408`） |
| 输入分辨率 | 448 方形拉伸（`relsgg/data/dataset.py:262`），多尺度训练 | 640 letterbox（`dataset.py:406-411`），P4 网格 40×40 |
| 骨干是否训练 | 全量微调 lr 5e-5（model.pth `args.backbone_lr=5e-05`） | 完全冻结（`trainer.py:244-275 _freeze_primary`），优化器只含关系层参数（`trainer.py:332-359`） |
| 关系头宽度 | d_model 512，scene_dim 384 | d_model 512，backbone_dim 768（`head.py:224`） |
| DINOv3 Backbone 类 | `relsgg/model/backbone.py:39-148` | 已移植在 `ra/backbone_block.py:20-78`（文件头注明「DINOv3 parts unused」） |

## 3. 三种做法对比表

算力口径说明：ViT 数字为按结构解析估算（d=384、12 层、6 头、SwiGLU 隐层 1536、4 register + 1 cls token，2 FLOPs/MAC），参数量用同一公式算得 28.69M，与 `model.pth` 中 `backbone.*` 的 28.69M 一致，所以结构假设可信；GFLOPs 本身是估算。YOLO 侧为 `WORKLOG.md:477-487` 的 FlopCounter 实测（同 2×MAC 口径）：检测器 640 px 8.31（1 提示）/9.02（80 提示）GFLOPs，关系头 N=60、budget 500 为 48.44 GFLOPs。任务说明里的「YOLO11n 6.5 GFLOPs」是 Ultralytics 官方数字（无文本头）。

| | (a) 双骨干：YOLO 检测 + DINOv3 只供关系特征 | (b) 整体换成 DINOv3 | (c) DINOv3 特征蒸馏进 YOLO 颈部 |
|---|---|---|---|
| 推理算力增量 | ViT-S/16+ 前向：224 px ≈ 12 GFLOPs（201 token）、448 px ≈ 57 GFLOPs（789 token）、640 px ≈ 139 GFLOPs（1,605 token）。按 448 算，系统从 57 GFLOPs 变成约 114 GFLOPs（+100%） | 与官方相同：关系模型 91 GFLOPs（Tab. 33）+ 外部检测器；YOLO 只剩出框 | 部署仍是 YOLO + 关系头 57 GFLOPs，零增量；训练期多一份教师前向 |
| 训练显存（估算） | 冻结：只有前向，不存激活，448 px batch 32 增量 < 2 GiB，batch 16 更少；在现有 10–11 GiB（`train.py` 注释）/ 15.4 GiB（`run_exp.sh:16`）基础上可行。微调：激活约每图 190 MB（bf16）/380 MB（fp32），batch 32 增 6–12 GiB，batch 16 增 3–6 GiB；官方 batch 32 多尺度至 672 px 峰值 30 GB（Tab. 29） | 同官方：batch 32 峰值 30 GB，本卡 32 GB 总量且现被占 15.5 GB，只能 batch 16 或固定 448 不做多尺度 | 学生是 2.6M 参数颈部副本（progress.md 待办 1 的 A 方案），显存增量小；教师冻结前向同 (a) |
| 训练时间（估算） | 50k 子集 4 轮现在 1.7 h（README 表）；当前 GPU 利用率只有 30%、34 img/s，数据管线是瓶颈（progress.md 待办 7），冻结 ViT 前向在 640 px 需约 4.7 TFLOPS，卡能吃下；估 2–2.5 h。微调版估 3–4 h | 官方 12 轮 5h04m 在 4×A100；单张 4080 SUPER 粗估 40–60 h，且要改 batch | 蒸馏阶段 50k 子集若干轮（估 2–4 h）+ 再训关系头 1.7 h；先要确定蒸馏目标与损失 |
| 本代码库改动点 | 1) `head.py:405-419 forward`：`base` 改为由外部 F_map 提供者给出（新增 `self.aux_backbone`，接收图像）；2) `head.py:346-354 _fmap`：输入通道从 C4 改为 384（`_ensure_fmap_proj(ch4)` 已按输入通道懒建，几乎不用改）；3) `trainer.py:135-138 compute_batch_loss`：在 `self.model.predict(batch["img"])` 之后用 `batch["img"]` 跑 DINOv3，覆盖 `rel._batch_feats[0]`；4) `val.py:306-345`：capture 时同样替换 `_captured["feats"]`，并把 `ih, iw = fh*stride` 的反推改为直接取 batch 图像尺寸（若 DINOv3 跑 640 px，网格 40×40×stride 16 = 640，这行可以不改）；5) `wiring.py:117-178 attach_relate_layer`：`relate_info` 记录 `vis_backbone` 字段，保证 ckpt 自描述；6) 骨干构建复用 `ra/backbone_block.py:20-78 Backbone` + `tools/run_ra_official_predcls.py:62-71` 的键名重映射（`backbone.model.model.*`→`backbone.model.*`），需 `PYTHONPATH` 含 `ra_overlay`（transformers 4.57.1 带 `DINOv3ViTModel`；默认环境 4.49.0 没有）；7) 微调版再加：`trainer.py:244 _freeze_primary` 放行骨干、`build_optimizer` 加第二参数组 lr 5e-5、`_RelateOnlyEMA` 纳入骨干。工作量：冻结版 1 天内（含冒烟），微调版再 0.5–1 天 | 等于把 `relsgg/` 整套搬过来再加 GLP 场头；`YOLOVORelate` 与 Ultralytics trainer/validator 的耦合（capture、letterbox、P 层接线）全部作废；工作量 1 周以上 | 新建蒸馏损失（P4 副本 40×40×C → 投影到 384 后与 DINOv3 融合图做 cos/L2），新增教师前向与缓存，再决定是否让检测器共享该颈部；工作量 2–3 天；收益上限由 2.6M 颈部的表达力决定 |
| 预期收益（对照差距拆解，推测） | 直接对准第 3 条：contact/support（现 9.5，官方 26.4）、action/other（1.3 vs 35.0）、proximity（4.1 vs 23.7）应有明显恢复；directional 应保持（`q_spa` 几何分支不变）；只给框 R@50 有望超过官方 26.61（我们没有同义词折损，`reports/baseline_cmp/README.md:70-72`）。mR@50 受文本空间塌缩限制（第 5 条），单做 (a) 预计只从 1.45 升到个位数中段 | 理论上能复现官方 26.61/13.11，但这不是本项目的贡献 | 取决于蒸馏保真度；论文 §7.3 说骨干贡献的是物体身份，这类稠密语义蒸进 128 通道小颈部通常只能回收一部分（未验证） |
| 风险 | 用官方微调骨干时「借用」了官方头的监督（见 §6）；letterbox 640 vs 官方方形拉伸 448 的输入分布差异；`backbone_dim=768` 与 DINOv3 384 维的适配层把 LN 过的特征又过了一层 GELU；EMA/续训逻辑默认只跟踪关系层 | 丢掉实时检测前提；单卡训练时间与显存都紧 | 两阶段训练，误差来源多；若蒸馏后颈部被检测器共享，会动一级模型 |

## 4. 权重可得性与泄漏检查

### 4.1 `model.pth` 内的骨干（事实，CPU `torch.load(..., map_location='cpu', weights_only=False)`，只读键名）

- 顶层键：`['args', 'pred_names', 'epoch', 'backbone_config', 'model', 'stripped_weights', 'stripped_from_run']`；`epoch=11`，`stripped_weights='ema'`，来源 run 名含 `vits16plus_r0_lr4e-4_ep12 ... ms0.5-1.5`。
- `model` 共 416 个键、63.04M 参数；`backbone.*` 236 个键、28.69M 参数：
  - `backbone.layer_weights (3,)`：三个 tap 的融合权重；
  - `backbone.model.embeddings.{cls_token (1,1,384), mask_token (1,1,384), register_tokens (1,4,384), patch_embeddings.weight (384,3,16,16), .bias (384,)}`；
  - `backbone.model.model.layer.{0..11}.{norm1, attention.{q_proj(带 bias), k_proj(无 bias), v_proj, o_proj}, layer_scale1.lambda1 (384,), norm2, mlp.{gate_proj (1536,384), up_proj (1536,384), down_proj (384,1536)}, layer_scale2.lambda1}`；
  - `backbone.model.norm.{weight,bias} (384,)`。
- `backbone_config`：`DINOv3ViTModel`，`hidden_size 384`，`num_hidden_layers 12`，`num_attention_heads 6`，`intermediate_size 1536`，`use_gated_mlp true`，`hidden_act silu`，`num_register_tokens 4`，`rope_theta 100`，`patch_size 16`，`_name_or_path facebook/dinov3-vits16plus-pretrain-lvd1689m`。官方 `relsgg/checkpoint.py:50-61 materialize_backbone_config` 就是把它写成临时 `config.json` 后 `AutoModel.from_config`，**不需要联网**。
- 头侧与骨干维度相关的键：`spatial_pool.base_query (1,1,384)`、`rel_transformer.scene_proj (512,384)`、`rel_interaction.scene_proj (512,384)`、`sub_text_proj/obj_text_proj (512,384)`、`vocab_head.W (19103,512)`、`W_obj (2884,512)`。
- 训练 args 摘要：`backbone_lr 5e-05`、`lr 4e-4`、`img_size 448`、`multi_scale 0.5,1.5`（7 档）、`batch_size 32`、`epochs 12`、`amp bf16`、`ema_decay 0.9998`、`lora_rank 0`、`clip_grad 1.0`、`val_root runs/packed/megasg`。

### 4.2 本机有没有原始 DINOv3 预训练权重（事实）

- `find / -iname '*dinov3*'`：只命中 `ra_overlay/transformers/models/dinov3_{vit,convnext}/`（模型代码）和 `RelateAnything/training/convert_dinov3_*.py`（脚本），**没有任何权重文件**；`~/.cache/huggingface` 不存在，`HF_HOME` 未设。
- 本机唯一的 DINOv3 ViT-S/16+ 权重就是 `model.pth` 里微调过的那份。

### 4.3 网络可得性（事实，仅 `curl -sI`，未下载）

- 环境有代理 `https_proxy=http://127.0.0.1:7897`。`hf-mirror.com` 对所有请求返回 `308` 重定向到 `huggingface.co`（含非 gated 的 `bert-base-uncased`），即镜像本身现在不提供内容；但经代理 `huggingface.co` 直连可达（`/api/models/...` 返回 200）。
- `facebook/dinov3-vits16plus-pretrain-lvd1689m/resolve/main/config.json`：`401`，`x-error-code: GatedRepo`。无 HF token + 审批拿不到。
- `timm/vit_small_plus_patch16_dinov3.lvd1689m/resolve/main/model.safetensors`：`302` → CDN `200`，`content-length 114,750,056`（公开可下）。`timm/vit_small_patch16_dinov3.lvd1689m`（S/16 非 plus）86,362,376 字节，同样公开。
- `maelic/relsgg-vits16plus`：api 200（我们已有本地副本）。
- timm 转换版到 HF `DINOv3ViTModel` 需要键名转换（timm 的 DINOv3 走 `eva.py` 实现，RoPE/SwiGLU 命名不同）；本机 timm 1.0.30 已装，是否能直接 `timm.create_model('vit_small_plus_patch16_dinov3.lvd1689m')` 并取中间层：**未验证**（未联网加载）。

### 4.4 transformers 版本（事实）

- `PYTHONPATH=/root/rivermind-data/ra_overlay`：transformers 4.57.1，含 `models/dinov3_vit`、`models/dinov3_convnext`，`from transformers import DINOv3ViTModel, DINOv3ViTConfig` 成功。
- 默认 `/opt/conda` 环境：transformers 4.49.0，`hasattr(transformers, 'DINOv3ViTModel') == False`。所以训练进程必须带 overlay 路径（`tools/run_ra_official_predcls.py` 已是这么跑的）。

### 4.5 泄漏检查（事实）

- `RA-4M/README.md:42-47`：train 472,344 图 / val 24,964 图，是数据集自带的两个 split；`:75-78`：`packs/megasg/val` 用途写明「checkpoint validation」。`export_stats.json`：train 472,344 图、4,282,531 关系；val 24,964 图、226,203 关系。
- `RA-4M/data_full_monitor.yaml:2-3`：`train: images/train`，`val: full_monitor/images`。
- 命令核验：`full_monitor` 1,024 张文件名 **全部**在 `images/val`，**0 张**在 `images/train`；`images/train ∩ images/val = 0`；前 200 张 monitor 标签文件与 `labels/val` 逐字节相同（与 `worklog/2026-11-06-2130-glp-audit-resume-full-run.md:160` 一致）；`proxy50k ∩ images/val = 0`。
- 官方骨干训练用的是 train split（`packs/megasg_clean/train`，README `:78`）；`args.val_root=runs/packed/megasg` 只用于验证损失（`val_loss_batches=100`），发布权重是 `checkpoint_last.pth`（`relateanything.json` 的 `source_checkpoint`），不是按 val 选的。结论：官方骨干没有见过我们的 1,024 张 val 图，**不构成泄漏**。
- 要诚实说明的另一件事（不是泄漏）：RA-4M val 与 train 同一标注器、同一风格，论文 §8.1（`:331`）测得域内斜率比迁移斜率高约 5×。我们所有数字都是域内 PredCls，拿官方微调骨干来用时这一点不变。

## 5. 建议的第一个实验

### 5.1 设计：E1 = 冻结的官方微调 DINOv3 ViT-S/16+ 替换 P4 作为 F_map，其余全同 `p50k_base_e4`

为什么选这个版本而不是 timm 预训练版：零下载、结构由 `backbone_config` 精确给定、加载代码已在 `tools/run_ra_official_predcls.py:62-71` 验证过；它给出的是「视觉通路换掉后的上限」，先回答「值不值得」。若 E1 过关，再做 E1b（timm 预训练、未经 RA 微调的权重）把「DINOv3 预训练」与「官方 RA 微调」的贡献分开；E1b 才是可以写进论文的干净版本。

分辨率选 640（直接吃 letterbox 后的 `batch["img"]`）而不是 448：网格 40×40 与 P4 完全一致，`val.py:338-343` 的 `ih, iw = fh*stride` 反推不用改，`feat_strides` 仍为 16；官方多尺度训练覆盖 224–672，Tab. 31 显示 560/672 不掉点。代价是前向 139 GFLOPs/图而不是 57，在当前 34 img/s 的数据瓶颈下可以接受（估算）。

对照：`runs/test/p50k_base_e4`（A3 队列里的基线，50k 子集 + 全 10,370 词表 + 同一 1,024 图 val，4 轮，约 18:30 跑完），同数据同轮数，正好是 README 过关标准要求的参照。

### 5.2 改动清单（按最小改动）

1. `ultralytics/models/yolovo/relate/head.py`
   - `YOLOVORelate.__init__` 增加可选 `self.vis_backbone: Optional[nn.Module] = None`（不进 `_freeze_primary` 的「关系层参数」判定：E1 冻结，`requires_grad_(False)` + `eval()`）。
   - `forward()`（405-419 行）：若 `self._cur_img is not None and self.vis_backbone is not None`，`base = self.vis_backbone.extract(self.vis_backbone.preprocess(img))` 得到 `[B,40,40,384]`，转成 `[B,384,40,40]` 当作 `base`；`_fmap` 的 `_ensure_fmap_proj(base.shape[1])` 自动懒建 `Conv2d(384,768,1)`。
   - `trainer.py:300-316 _materialize_pool` 现在用 `rel._feat_channels[rel.base_idx]` 预建 `fmap_proj`，E1 需改成 384（否则 AdamW 建在 C4 通道的 conv 上，首次 forward 会因通道不匹配报错或懒建一个不进优化器的层）。
2. `ultralytics/models/yolovo/relate/trainer.py:135-138 compute_batch_loss`：在 `self.model.predict(batch["img"])` 前设 `rel._cur_img = batch["img"]`（0–1 浮点，letterbox 后），之后清空。
3. `ultralytics/models/yolovo/relate/val.py:306-316`：capture 前同样设 `rel._cur_img`；`tools/dump_predcls_ours.py` 走同一个验证器，不用改。
4. 骨干构造：新建一个小函数（放 `ra/backbone_block.py` 或 `wiring.py`）：`torch.load(model.pth)` → 用 `backbone_config` 写临时 `config.json` → `Backbone(model_name=tmpdir, pretrained=False)` → 按 `backbone.model.model.*`→`backbone.model.*` 重映射后 `load_state_dict(strict=True)`，并载入 `backbone.layer_weights`。进程需 `PYTHONPATH=/root/rivermind-data/ra_overlay:/tmp/yolovo-audit-numpy2`，`USE_TF=0`。
5. 开关：环境变量 `YOLOVO_VIS_BACKBONE=assets/relsgg-vits16plus/model.pth`（默认空 = 现状，基线逐位不变），写进 `train.py` 顶部配置区和 `relate_info`。
6. 保存/续训：`vis_backbone` 冻结时不必存进 ckpt（存路径即可），避免 best.pt 多 110 MB；但 `attempt_load` 后要能重建，所以 `relate_info["vis_backbone"]` 必须带路径与 sha。

### 5.3 命令（README「实验流程」口径）

```bash
cd /root/rivermind-data/YOLOVOmodal_3871538535
# 等 A3 队列（PGID 1074560）打印 QUEUE_DONE 后再起；一次只跑一个训练
NAME=p50k_dinov3frz_e4 DATA=RA-4M/data_proxy50k.yaml EPOCHS=4 BATCH=32 \
REF_RUN=runs/test/p50k_base_e4 \
YOLOVO_VIS_BACKBONE=assets/relsgg-vits16plus/model.pth \
PYTHONPATH=/root/rivermind-data/ra_overlay:/tmp/yolovo-audit-numpy2 USE_TF=0 \
  setsid nohup bash tools/run_exp.sh > logs/p50k_dinov3frz_e4.queue.log 2>&1 < /dev/null &
sleep 5; ps -o pid,pgid,etime,cmd -p $(cat logs/p50k_dinov3frz_e4.pid)
# 跑完：runs/test/p50k_dinov3frz_e4/post/{gate.json,DONE}、reports/baseline_cmp/p50k_dinov3frz_e4.json
# 分组拆解（只给框）：
PYTHONPATH=/tmp/yolovo-audit-numpy2 /opt/conda/bin/python tools/gap_analysis.py   # 按其现有参数把 p50k_dinov3frz_e4.nocls.pt 与 p50k_base_e4.nocls.pt 并排
```

显存：若 batch 32 超过可用显存（卡上若仍有其他进程），按 `run_exp.sh:16` 注释降到 `BATCH=16`，并在 worklog 写明。

### 5.4 预计时长（估算）

- 冒烟（CPU 构建骨干 + 1 个 batch 前向对形状）：0.5 h 人工。
- 训练：50k × 4 轮，基线 1.7 h；加冻结 ViT 前向估 2–2.5 h；dump + 同口径评测约 10 min（`run_exp.sh` 自动）。
- 总计一个工作日内出结果。

### 5.5 过关判断（README「过关标准」+ 差距拆解补充）

参照 = `p50k_base_e4`。

1. c1：`results.csv` 第 4 轮 `pc_R@50` > 第 3 轮。
2. c2：对参照 `pc_R@50 ≥ +0.5` 或 `pc_mR@50 ≥ +1.0`。
3. c3：`pc_pair_acc` 不低于参照 0.5 以上；只给框的 `R@50` 不低于参照。
4. 本实验专属（建议作为主判据写进 worklog）：只给框时 `tools/gap_analysis.py` 的 contact/support 与 action/other 组 micro R@50 相对参照翻倍以上（参照全量基线为 9.5 与 1.3；50k 基线的数值等 `p50k_base_e4` 出来后填）；至少命中一次的谓词数从 27–34 明显上升。

### 5.6 成功 / 失败分别说明什么

- **成功**（c1–c3 过关且第 4 条成立）：差距拆解第 3 条被实验证实，视觉通路是第一位瓶颈；接着做 E1b（timm 预训练权重）判断是否必须借官方微调；再叠加 B（官方 student 文本空间）看 mR@50 能否到两位数；之后才讨论 (c) 蒸馏回 YOLO 以保住部署成本。
- **只给框 R@50 涨但 contact/action 组不动**：收益来自更好的几何/上下文而非物体身份，说明问题更多在关系头或文本空间，优先级转向 B/C。
- **失败**（基本持平）：说明 10,370 路 InfoNCE 在塌缩的 MobileCLIP 文本空间里即使有好特征也挑不出长尾谓词，应先做 B（零结构改动、1.7 h）再回头做 E1；同时要排查实现问题（letterbox 灰边、`fmap_proj` 是否进了优化器、`_cur_img` 在验证器里是否被设置）。

## 6. 风险

1. **「借用」官方监督**：`model.pth` 的骨干是用官方关系头在 RA-4M train（含 HICO-DET、raw VG）上微调过的；E1 的收益里混有这部分，不能直接写成「我们的方法」。E1b 用 timm 公开预训练权重消除此项；若 timm 权重的键名转换出问题，退路是 `timm/vit_small_patch16_dinov3.lvd1689m`（S/16 非 plus，86 MB，公开可下，Tab. 29 里 OVS-F1 比 S/16+ 低 2.6 点）；facebook 官方的 `dinov3-vits16-pretrain-lvd1689m` 只测了 api 200，权重文件是否 gated 未验证（facebook 系列大概率同样 gated）；ConvNeXt 版（§G `:777` 提到 tuned tiny 变体在 ViT-S 精度档且 batch-1 最快）可得性未验证。
2. **输入分布差异**：官方训练是方形拉伸到 448（`relsgg/data/dataset.py:262`），我们是 640 letterbox 带灰边（`dataset.py:406-411`）。对 ViT 而言是温和的域偏移（官方多尺度覆盖到 672），但灰边区域的 patch token 会进入 SoftSpatialPool 的全图 cross-attention（推测影响小，未验证）。若想消除，可在送 ViT 前裁掉 letterbox 再方形拉伸，但要同步改框归一化，改动面扩大。
3. **维度适配**：现头 `backbone_dim=768`，DINOv3 融合图是 384 维且已 LayerNorm；经 `Conv2d(384,768,1)+GELU` 再进 `SoftSpatialPool`，与官方「直接 384 进池化」不等价。E1 先保留 768 以便与旧 ckpt/基线可比；若要对齐官方，改 `backbone_dim=384` 会改变 `pair_proj`、`rel_transformer`、`rel_interaction.scene_proj` 的形状，旧权重不可比。
4. **优化器/EMA 边界**：`build_optimizer`（`trainer.py:332-359`）与 `_RelateOnlyEMA`（`trainer.py:298`）都按 `model.{rel_idx}.` 前缀选参数；`vis_backbone` 若作为 `rel` 的子模块会被误当成可训练参数并进 EMA（增加 7.9% 以上的每步开销），需显式排除；微调版则相反，要放行并给第二个 lr 组。
5. **显存与排队**：GPU 现被 A3 队列占用（15.5 GB），E1 必须等 `QUEUE_DONE`；全量微调版在本卡只能 batch 16 或放弃多尺度，与官方配方不等价。
6. **指标判读**：4 轮基线本身呈「头部换尾部」（`reports/baseline_cmp/README.md:36-42`），c1 以 `pc_R@50` 为判据可能把一个 mR 上涨的配置判为不过关；E1 建议把第 4 条分组指标与 F1@50 一起写进 worklog 再定。
7. **蒸馏路线 (c) 的上限未知**：论文没有提供「小 CNN 蒸 DINOv3 稠密特征后做关系」的任何数字；YOLO11n-DIOR 颈部是在 20 类遥感数据上训的，蒸馏必须在 RA-4M 自然图像上做，且 128 通道、2.6M 参数的颈部能否承载人/衣物/手等身份信息，只能实验回答。

## 7. 做了什么 / 没做什么

做了什么：读 `progress.md`、`reports/baseline_cmp/README.md`、`README.md`；grep 定位并读论文 §3.2、§7.3、§8.1、§9、§A.1、§A.4、§F、§G、§H.1/H.2/H.4、§I；读 `wiring.py`、`head.py`（forward/_fmap/score）、`trainer.py`（compute_batch_loss/_freeze_primary/_materialize_pool/build_optimizer）、`val.py`（capture）、`ra/backbone_block.py`、官方 `relsgg/model/backbone.py`、`relsgg/checkpoint.py`、`tools/run_ra_official_predcls.py`；CPU 只读加载 `model.pth` 列键名/形状/`backbone_config`/`args`；`find` 全盘找 DINOv3 权重；curl HEAD 探测 hf-mirror / huggingface.co / gated 状态；核验 `full_monitor` 与 train/val split 的集合关系；解析估算 ViT-S/16+ 在 224/448/640 的 GFLOPs 并用参数量交叉核对。

没做什么：
- 没有实际构建 DINOv3 模型做前向（任务要求不跑模型、不起 GPU 进程）；`Backbone(pretrained=False)` + 键重映射的离线构建路径只在 `run_ra_official_predcls.py` 中被用过，E1 需要再冒烟一次。
- 没有下载 timm 权重，也没验证 timm→HF 键名转换是否可行。
- 显存、吞吐、小时数全部是估算，没有实测（卡被占用）。
- 没有读论文 Tab. 3 / §6 的主结果表，与本任务无关。
- 没有修改仓库任何文件；本报告写在 `/root/rivermind-data/scratch_audit/idea1_dinov3.md`。

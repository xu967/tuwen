# 多模态 YAML 构建指点（YOLO/RT-DETR，RGB+X）

> 版本：v1.0（适配本仓库 MultiModalRouter 机制）
>
> 目标：指导使用者按“配置驱动 + 第5字段路由”的方式，自主编写多模态 YAML，用于 YOLO 系列与 RT-DETR 系列。

---

## 1. 基础理念与适用范围
- 统一范式：通过 YAML 第 5 字段标注输入路由（`'RGB' | 'X' | 'Dual'`），其余沿用原有模块与拓扑。
- 两模态抽象：非可见光模态统一抽象为 `X`（例如 depth/thermal/IR 等），通道数由数据配置 `Xch` 决定。
- 架构无关：同一套 YAML 书写规则同时适用于 YOLO 与 RT-DETR，构网与前向由 `MultiModalRouter` 统一接管。
- 配置分区（推荐）：
  - YOLOMM 配置集中在 `ultralytics/cfg/models/mm/`
  - RTDETRMM 配置集中在 `ultralytics/cfg/models/rtmm/`（再按 backbone 分：`rtmm/r18/`、`rtmm/r50/`）
  - 标准 RT-DETR 配置保留在 `ultralytics/cfg/models/rt-detr/`

---

## 2. YAML 基本语法（含多模态第5字段）
- 常规层定义四元组：`[from, repeats, module, args]`
- 多模态扩展为五元组：`[from, repeats, module, args, input_source]`
  - `input_source ∈ { 'RGB', 'X', 'Dual' }`
  - 语义：该层输入来自哪一路模态；`Dual` 表示 6 通道早期融合输入（`3 + Xch`）。
- 通道规则：
  - `RGB` 固定 3 通道
  - `X` 由数据配置 `Xch` 指定（默认 3）
  - `Dual = 3 + Xch`
- 新起点分支：若以 `'X'` 分支作为“新输入起点”，请将该层 `from` 设为 `-1`，第 5 字段填 `'X'`。Router 会用原始 X 输入重置空间尺寸（无需额外代码）。

### 分支书写规范（必读）
- 当在 backbone 中分别对 RGB 与 X 做独立特征提取（中/晚期融合）时，必须保持“分支连续、清晰分区”的书写，不要交替穿插：
  - 先连续写完 RGB 分支的若干层（如至 P3/P4），再连续写 X 分支（从 `from=-1` + `'X'` 开新起点）。
  - 每个分支内部只引用本分支上一层的索引，直到显式融合（如 `Concat`）为止。
  - 在发生融合之前，不要在两个分支之间来回切换或交叉引用，以免造成通道推导与路由混淆。

错误写法（交替切换，难维护且易错）：
```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2], 'RGB']   # 0
  - [-1, 1, Conv, [64, 3, 2], 'X']     # 1  (新起点未声明 from=-1，且立刻切到 X)
  - [-1, 1, Conv, [128, 3, 2], 'RGB']  # 2  (又切回 RGB，分支不连续)
  - [-1, 1, Conv, [128, 3, 2], 'X']    # 3  (再次切到 X)
```

正确写法（分支清晰，融合点明确）：
```yaml
backbone:
  # RGB 分支（0-3）
  - [-1, 1, Conv, [64, 3, 2], 'RGB']   # 0
  - [-1, 1, Conv, [128, 3, 2]]         # 1
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, SPPF, [512, 5]]            # 3  (RGB P4)

  # X 分支（4-7）
  - [-1, 1, Conv, [64, 3, 2], 'X']     # 4  (from=-1 + 'X' 新起点)
  - [-1, 1, Conv, [128, 3, 2]]         # 5
  - [-1, 2, C3k2, [256, False, 0.25]]  # 6
  - [-1, 1, SPPF, [512, 5]]            # 7  (X P4)

  # 融合
  - [[3, 7], 1, Concat, [1]]           # 8  (在明确的融合层连接两路特征)
  - [-1, 2, C3k2, [1024, True]]        # 9
```

---

## 3. 常见融合范式最小模板（YOLO）

### 3.1 早期融合（Early Fusion）
- 特点：输入直接为 `Dual(6ch)`，后续走单路主干与头部，结构最简单，适合模态配准良好场景。
```yaml
# 仅展示关键片段
backbone:
  - [-1, 1, Conv, [64, 3, 2], 'Dual']  # 6ch 输入起点
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  # ... 按标准 YOLO 堆叠
head:
  # 标准 YOLO 检测头
  - [[P3, P4, P5], 1, Detect, [nc]]
```

### 3.2 中期融合（Mid Fusion）
- 特点：RGB/X 双分支分别至 P3/P4 再 `Concat + C3k2/C2PSA` 融合，兼顾细节与语义，适合互补但差异较大的模态。
```yaml
backbone:
  # RGB 分支
  - [-1, 1, Conv, [64, 3, 2], 'RGB']
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, SPPF, [512, 5]]               # 假设到 P4

  # X 分支（新输入起点）
  - [-1, 1, Conv, [64, 3, 2], 'X']        # from=-1 + 'X'
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, SPPF, [512, 5]]

  # 融合
  - [[rgb_P4, x_P4], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [[P3, P4, P5], 1, Detect, [nc]]
```

### 3.3 晚期融合（Late Fusion）
- 特点：RGB/X 各自独立检测，最终在推理层面做决策融合（本仓库示例中 `DecisionFusion` 为占位注释，需自行实现或外部融合）。
```yaml
# 双分支两套 Detect，推理阶段再融合结果
head:
  - [[rgb_P3, rgb_P4, rgb_P5], 1, Detect, [nc]]
  - [[x_P3,   x_P4,   x_P5],   1, Detect, [nc]]
  # - [[-2, -1], 1, DecisionFusion, []]  # 可选：自定义实现
```

---

## 4. RT-DETR 范式模板

### 4.1 早期融合（rtdetr-*-mm-early.yaml）
```yaml
backbone:
  - [-1, 1, ConvNormLayer, [32, 3, 2, None, False, 'relu'], 'Dual']  # 6ch 输入
  - [-1, 1, ConvNormLayer, [32, 3, 1, None, False, 'relu']]
  - [-1, 1, ConvNormLayer, [64, 3, 1, None, False, 'relu']]
  - [-1, 1, nn.MaxPool2d, [3, 2, 1]]
  - [-1, 1, Blocks, [64, BasicBlock, 2, 2, 'relu']]
  # ... 继续至 P3/P4/P5
head:
  # FPN/PAN + AIFI + RTDETRDecoder
  - [[P3, P4, P5], 1, RTDETRDecoder, [nc, 256, 300, 4, 8, 3]]
```

### 4.2 中期融合（rtdetr-*-mm-mid.yaml）
```yaml
backbone:
  # RGB 分支
  - [-1, 1, ConvNormLayer, [32, 3, 2, None, False, 'relu'], 'RGB']
  - [-1, 1, nn.MaxPool2d, [3, 2, 1]]
  - [.., Blocks, [64,  BasicBlock, 2, 2, 'relu']]
  - [.., Blocks, [128, BasicBlock, 2, 3, 'relu']]  # 至 P3

  # X 分支（新输入起点）
  - [-1, 1, ConvNormLayer, [32, 3, 2, None, False, 'relu'], 'X']
  - [-1, 1, nn.MaxPool2d, [3, 2, 1]]
  - [.., Blocks, [64,  BasicBlock, 2, 2, 'relu']]
  - [.., Blocks, [128, BasicBlock, 2, 3, 'relu']]  # 至 P3

  # P3 融合
  - [[rgb_P3, x_P3], 1, Concat, [1]]
  - [-1, 1, Conv, [256, 1, 1]]
  - [.., Blocks, [256, BasicBlock, 2, 4, 'relu']]  # 至 P4
  - [.., Blocks, [512, BasicBlock, 2, 5, 'relu']]  # 至 P5

head:
  - [[P3, P4, P5], 1, RTDETRDecoder, [nc, 256, 300, 4, 8, 3]]
```

---

## 5. 与数据配置（data.yaml）配合
- 指定模态组合：
```yaml
modality_used: [rgb, depth]          # 推荐（优先级高）
# 向后兼容：models: [rgb, depth]
modality:
  rgb: images                        # RGB 图像目录名
  depth: images_depth                # X 模态目录名
Xch: 3                                # X 模态通道数（默认 3，可按实际数据设置）
```
- 说明：
  - `Xch` 影响 Router 的通道划分与 `Dual=3+Xch`，与 YAML 第 5 字段共同决定输入形状。
  - 训练时可用 `--modality rgb` 或 `--modality X` 进行单模态训练/验证（框架会校验并映射到真实 X）。其中 `rgb/x` token 大小写不敏感，内部统一显示为 `RGB/X`。

---

## 6. 命名与约定
- 文件命名建议：`<family>-<variant>-mm-<strategy>.yaml`
  - 例如：`yolo11n-mm-early.yaml`、`yolo11n-mm-mid.yaml`、`rtdetr-r18-mm-mid.yaml`
- 目录约定（本仓库）：
  - YOLOMM：`ultralytics/cfg/models/mm/`
  - RTDETRMM：`ultralytics/cfg/models/rtmm/`（按 backbone：`rtmm/r18/`、`rtmm/r50/`）
- `scales`：YOLO 支持 `n/s/m/l/x`；RT-DETR 通常使用 `l`（可按实现扩展）。
- 训练/验证时显式选择 `scales`（推荐）：
  - CLI：`yolo train model=xxx.yaml scale=s`（或 `model_scale=s`）。
  - Python：`model.train(..., scale='s')`（或 `model_scale='s'`）。
  - 注意：当 `scale` 以 `n/s/m/l/x` 形式用于“模型尺度选择”时，数据增强的缩放请改用 `img_scale`（否则使用默认 `scale=0.5`）。
- 常用模块：
  - YOLO：`Conv / C3k2 / C2PSA / SPPF / Detect`
  - RT-DETR：`ConvNormLayer / Blocks(BasicBlock) / AIFI / RTDETRDecoder`

---

## 7. 何时选择哪种融合
- 早期融合：模态空间/语义高度对齐，期望最小改造与最快速度。
- 中期融合：模态互补明显但仍希望共享部分高语义表征，兼顾精度与可控算力。
- 多级/残差/非对称：对难场景（尺度变化大、噪声模态、昼夜差异）进一步强化跨尺度与跨模态的稳健性与效率。
- 晚期融合：两路完全独立、鲁棒性最高；需要在推理或后处理阶段自行合并结果。

---

## 8. 验证与排查（高频问题）
- 通道不匹配：
  - 现象：报错提示 `expected X channels but got Y`。
  - 排查：检查 `data.yaml` 的 `Xch` 是否与真实数据一致；`Dual` 处是否只在输入起点；有无错误的第 5 字段标注。
- 模块未导入：
  - 现象：`ImportError: 模块 'XXX' 在YAML中使用，但未找到`。
  - 排查：确认模块在 `ultralytics/nn/modules/__init__.py` 或相应 `extra_modules` 已正确导出。
- 结果融合：
  - 说明：`DecisionFusion` 在示例中为占位注释，需自行实现/替换为外部融合脚本。
- 运行确认：
  - 前向 `profile=True` 可看到 Router 的路由日志（RGB/X/Dual 形状、空间重置信息）。

---

## 9. 进阶提示
- 新起点分支：以 `'X'` + `from=-1` 开新分支（中/晚融合）时无需手动处理尺寸，Router 会从原始 X 输入重置空间大小。
- 非对称深度：按任务复杂度决定 RGB/X 路径深浅；深层仅保留 RGB 语义，P3/P4 做跨模态融合可省算力。
- 多级残差：在 P2→P5 分层构造跨模态残差并级联，进一步提升稳健性与训练梯度流。
- Xch ≠ 3：若 X 为单通道（如原始深度图），建议在数据侧转换为 3 通道或在数据配置中设置 `Xch=1` 并相应适配 Reader。

---

## 10. 最小可用清单（Checklist）
- [ ] data.yaml 配置 `modality_used`、`modality`、`Xch`
- [ ] YAML 第 5 字段正确标注 `'RGB'/'X'/'Dual'`
- [ ] 仅在输入起点使用 `Dual`，中/晚融合处使用 `'RGB'`/`'X'` 分路 + `Concat`
- [ ] X 分支新起点使用 `from=-1` + `'X'`
- [ ] 模块均已在相应 `__init__.py` 正确导出
- [ ] 使用 `model.info()` 或 `profile=True` 检查路由与形状

---

如需根据具体数据集/任务给出推荐范式或审阅 YAML，可在当前文件旁新增自定义方案并标注用途与改动点。

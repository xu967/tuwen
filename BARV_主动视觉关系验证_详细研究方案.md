# BARV：预算约束下的主动视觉关系验证——完整研究方案（v1.0）

> 工作标题：**BARV: Budget-Aware Active Visual Relation Verification via Predicate-Conditioned Evidence Acquisition**  
> 项目定位：计算机视觉 / 场景关系理解 / 自适应计算；**不做目标检测作为论文研究任务**  
> 更新时间：2026-10-26  
> 基础代码：用户提供的 `YOLOVOmodal_3871538535.zip`，重点为 `ultralytics/models/yolovo/relate/`。  
> 本文为**可执行研究设计和待验证假说**，不是已有论文结论；所有给定模型规模、实验预算、验收阈值均属建议配置，未伪称实测结果。

---

## 0. 决策摘要

**主线：** 给定图像、主体/客体的区域掩码，以及待验证的谓词（例如 `person — holding — cup`），模型首先使用低分辨率信息判断；仅在证据不足时，按需读取原始图像中的局部高分辨率片段；每一步决策究竟读取哪里、花费多少计算、何时停止、何时拒绝，均由可学习的策略控制。

**主数据集：** PSG / OpenPSG（全景区域 + 人工标注的关系三元组）；构建单独的 **PSG-Active** 精标评估子集（这是我们计划创建的实验协议，不是官方已有数据集）。补充数据集：VG150 / GQA，做跨数据集验证但不混淆标注体系。**不以 M3FD 伪关系生成结果作为语义关系主真值。**

**核心独特性候选：** 不是「视觉裁剪」「主动注意力」「开放词汇关系」三者简单组合，而是 **谓词条件化的增量证据价值（Marginal Value of Evidence, MVE）估计** + **显式真实计算成本约束下的按需特征获取** + **预测/继续/拒绝三元决策** + **标注可核验的关系真值及证据质量评估**。这些独特性仍须接受全面检索和消融检验；目前不能声称“全球首次”。

**建议交付：** 可复现的关系验证基准、策略模型和推理执行器、PSG-Active 评估协议、精度-成本实验、多种负样本/对照与风险分析。

## 1. 研究定位、边界与核心假说

### 1.1 不是要研究什么

- 不做检测框预测、YOLO 检测头改进或检测 mAP 竞赛；第一阶段使用 PSG 已有**实体掩码**，按官方 PredCls 类似的 oracle-region 条件评价关系。
- 不以“框换成掩码”作为创新；FleVRS（NeurIPS 2024）等已有视觉关系分割先例。
- 不以“局部图像放大”“视觉 token 按需获取”自身作为创新；AdaptVision（2025）已研究粗到细的视觉获取。
- 不把用户现有的 `YOLOVORelate`、`RelationTransformer`、`DeformableRelRead` 或文本谓词头作为原创网络贡献：这些模块明确继承/移植 RelateAnything 的方案。
- 不把普通静态 Top-K 关系筛选中的 `geo_budget/final_budget/eval_budget` 视为“视觉计算按需节约”。

### 1.2 正式任务定义

输入：RGB 图像 `I`，可用实体掩码 `M_s, M_o`，文本谓词 `p`，计算预算 `B`。输出：

1. 关系成立概率 `P(y=1 | I, M_s, M_o, p)`；
2. 状态 `ACCEPT_POS / ACCEPT_NEG / ABSTAIN`；
3. 证据查询轨迹 `T = [(a_1,c_1),...,(a_t,c_t)]`；
4. 每一步及全程实际算力、延迟、显存统计；
5. 可选的证据区域位置（仅为模型预测，不默认是真值）。

第一阶段是**给定实体与谓词的二元关系验证**（Relation Verification, RV），而非输入任意图片自动穷举所有场景关系的 SGDet；这大幅降低候选生成与区域获取的混杂因素，同时为未来自动场景图建立可靠关系评分器。多谓词分类可作为附加任务，但不能用闭集多分类代替“未标注不等于否定”的严谨关系验证。

### 1.3 可证伪的假说

- **H1：** 在相等的端到端算力预算下，谓词条件化的主动读取，优于随机读取、静态固定区域、置信度/熵驱动读取和普通一次性高分辨率编码。
- **H2：** 预测“额外证据的**增量效用**”，比仅依据当前预测熵选择下一个区域更有效。
- **H3：** 联合训练停止/拒绝动作，能在同等覆盖率下减小错误风险；不过不会把「简单阈值校准」误记为新算法贡献。
- **H4：** 在固定主体-客体类别条件下，策略依然能够选取真正与谓词有关的视觉证据，而不是仅靠类别共现。
- **H5（外推）：** 多区域查询优于单步查询的收益仅出现在一定难度的关系子集；若所有场景单步足够，多步决策并不必要，应当如实报告。

### 1.4 初步文献边界（正式写论文时继续检索）

| 论文/工作 | 已有贡献 | 本项目必须做到的实质区分 |
|---|---|---|
| Panoptic Scene Graph Generation / OpenPSG, ECCV 2022 | 区域准确的全景场景图与 56 种谓词 | 不是再做 PSG 模型结构，而是建立可选择追加证据的资源约束验证协议 |
| FleVRS, NeurIPS 2024 | 提示式、开放词汇的视觉关系分割 | 不以掩码级关系本身作为新意 |
| ART, ICCV 2025 | 关系泛化的指令调整与实例选择 | 改进的是**测试时采证策略**，而非仅改谓词学习 |
| RegionFocus, ICCV 2025 | 视觉任务的自适应放大及迭代定位 | 必须新增关系谓词特化的效用估计、有效停止/拒绝及相等成本比较 |
| AdaptVision, arXiv 2512.03794 | 粗到细自适应获取视觉 patch 用于 VQA | 必须用区域-谓词-关系真值和增量证据价值解决不同任务，不能复刻流程 |
| RelateAnything, arXiv 2609.12552 | 开放词汇关系预测与区域输入、轻量计算 | 作为基线/可用模块来源，而非本研究的创新 |
| CAGE-SGG, arXiv 2604.22274 | 关系证据反事实问题 | **已被作者撤稿，官方注明方法与评估缺陷；不能引用其性能结论或作为可靠方法依据**。可作为负面教训，需独立设计无泄漏的验证。 |

**强制检索项：** 进一步按 `active visual relation reasoning`, `adaptive glimpse scene graph`, `relation-conditioned visual token`, `value-of-information visual perception`, `budgeted visual grounding`, `cost-aware selective prediction` 在 arXiv/CVF/NeurIPS/ICLR 检索，并按研究问题、输入监督、动作空间、预算计量、评估标准五个维度建立近邻表。

## 2. 数据集、取得方式与拆分

### 2.1 主数据 PSG

官方：<https://github.com/Jingkang50/OpenPSG>，统计：<https://psgdataset.org/data_stats.html>。

- 48,749 幅图像；133 类实体（80 Thing + 53 Stuff），56 类谓词；关联 COCO panoptic 掩码和 PSG 主客体关系索引。
- 下载**官方完整版本**的 PSG 标注、对应的 COCO 2017 图像/全景掩码；不要误用比赛时删去部分测试标注的 JSON。
- 官方早期比赛版的文件与完整版本划分不同，必须固定 `psg.json`/相应官方发布版本的哈希值、数据清单、各划分 image_id，并记录哪些关系可用于评价。
- 先验证 200 张图像的文件存在、图像解码、segment ID、实体类别映射、关系主体客体索引，再扩大到全部。
- 初期调试可抽取 **2k–5k 张训练图像**作为项目内部小子集；测试集保持官方完整、不用于调参。

### 2.2 不完整关系标签——最重要的统计问题

PSG 的一个实体对**没被标注某个关系，不代表该关系不存在**；更不能将 56 种谓词中除已标注谓词之外的 55 个全部当负例。

训练计划：

- 先用官方正关系做表示预训练 / 正例目标；用明确的 `ignore/unknown` 掩码屏蔽所有未核验的二元关系负标签。
- 使用经过人工核验的正、负关系构成主验证训练集；负样本分为：相同主体/客体类别但谓词不成立、同一主体/客体的易混淆谓词、未见类别组合等。
- 类别互斥/方向性规则只有在逻辑上可靠且经过人工确认时，才能生成高精度弱负例；不得随意用 `on` 与 `beside` 做普遍互斥。
- 对于正式公开实验，可额外报告按传统 PSG PredCls 的 R@K/mR@K 等指标，但与精标二元验证指标分栏呈现。

### 2.3 自建 PSG-Active（先 500–1,000 条精标 pilot 查询，按结果扩展）

建议每条记录包含：

```json
{
  "image_id": "...",
  "subject_segment_id": 101,
  "object_segment_id": 204,
  "predicate": "holding",
  "label": "positive|negative|ambiguous",
  "adjudication": "two_annotators_plus_review",
  "evidence_regions": [[x1, y1, x2, y2]],
  "evidence_tags": ["contact", "object_state"],
  "coarse_image_sufficient": false,
  "difficulty": "hard",
  "split": "dev|test"
}
```

- `evidence_regions` 是人工标记的支持/反驳证据区域，可以为空、多区域、或注明全局证据；不是主体/客体掩码的别称。
- `ambiguous` 或无法观察的样本单独用作拒绝预测评测，不硬转换为负例。
- 两人独立标注，争议由第三人裁决；报告一致率与类别分布，建立标注准则和负样本质控样例。
- **分阶段规模**：先标 500–1,000 条 pilot（包括正例、困难负例、歧义样本）；若 pilot 表明任务成立，再另外建设约 2,000–4,000 条训练核验查询、300–500 条验证查询和 500–1,000 条**最终盲测**查询，实际数量根据类别覆盖和一致性预算调整。pilot 的训练/验证部分来自官方训练图像，不得把看过最终测试标签的 pilot 样本带回训练。
- 开发集来自官方训练图像；最终测试精标来自官方保留测试图像；按照 image_id 去重，避免同图不同关系查询跨训练和测试。
- 官方测试不能用于反复迭代策略/超参；预先固定开发与测试划分，公开标注规范与采样种子。
- 为防止单一“person–riding–horse”频繁出现：谓词宏平均、主体客体类别对均衡、困难关系分层采样。

**数据集贡献的验收标准：** 精标查询必须能揭示“低分辨率无法判断、高分辨率局部可解”的真实比例。如果几乎都能在低分辨率上判断，主动查询意义不足，需要调整可视条件或谓词类型，不能通过测试集筛样本制造收益。

### 2.4 泛化集

- VG150：跨数据集迁移与不同关系类别分布；主要框标注，因此明确为 *box-region* 辅助协议，不与 PSG 的 oracle-mask 结果直接比较。
- GQA：利用场景图与语义组合属性，研究未见主客体组合和语义混淆；不能把其功能程序当“真实视觉查询步骤”。
- PSG 与 VG/GQA 图像来源可能重叠；需要 COCO/VG id 映射及感知哈希去重，确保跨数据集真独立。

## 3. 两类评价协议：杜绝 GT 信息泄露与错报省算力

**协议 P1（方法主论文、oracle-region）：** 给定 PSG 提供的主客体掩码，图像编码只读取预定低分辨率视图，高分辨率原始 RGB 像素只允许由行动选择器按需裁剪并在那时通过**真正运行**的局部编码器编码。实体掩码是任务设定中给定的条件变量，不能把它包装成模型预测；对所有对照方法完全相同。所有掩码在初始阶段只能使用低分辨率栅格版本提取几何特征，避免引入额外的细粒度信息；全精度坐标只能用于裁剪位置。成本表中注明 oracle mask 获取成本不纳入本协议。

**协议 P2（落地性补充）：** 使用冻结的分割器产生候选 mask，再做相同关系验证；分割器和区域 proposal 的计算成本必须计入端到端成本，并报告 GT mask 与预测 mask 之间的性能差距。P2 不以检测框为输出目标，外部分割器不是论文的新检测方法。

**高分辨率完整编码基线：** 每张图像按统一长边/固定大小计算一次高分辨率图像特征，不采用任意 Query 的 ROI 捷径；多查询场景下要区分 per-image amortized cost 和 per-query cost。

**预算口径：** 除 FLOPs，还要记录单幅图像端到端 GPU 时延（包含裁剪/重采样/策略运行、数据移动、所有局部编码、融合、校准），峰值显存，实际查询数量与生成裁剪的像素量。画 `accuracy vs. cost`、`selective risk vs. coverage`。单 query/图像和多 query/图像分别报告。

## 4. 方法模型：PEVA（Predicate-conditioned Evidence Value Acquisition）

以下名称 PEVA 为**拟议模块**，待检索确认命名不冲突。

### 4.1 网络总体数据流

```text
  输入：高分辨率原图 I + 主体 mask + 客体 mask + 谓词文本 p + 预算 B
                               │
                   Resize to low-res (256/320)
                               │
                      Coarse Encoder E_c
                               │
                主体/客体 MaskPool + 联合几何
                               │
                 Base Relation Encoder + text
                               │
                 q_0：状态向量 / 粗预测 / 不确定性
                               │
       ┌───────────────────────┴────────────────────────┐
       │           Predicate-conditioned policy         │
       │   候选动作 a：Contact / Sub / Obj / Union / ... │
       └───────────────────────┬────────────────────────┘
              STOP_POS / STOP_NEG / ABSTAIN   或继续查询
                               │
                    crop original image (lazy)
                               │
                     Fine Encoder E_f
                               │
                 Evidence Memory update q_{t+1}
                               │
                  重复，最多 T 次且成本≤B
                               │
           预测+证据轨迹+预算明细+拒绝标记
```

原图高分辨率仅在某次查询被选中时才经过 Fine Encoder；**不得在开始时先编码全部高分辨率 patch 再以“token 选择”伪装按需省算力**。

### 4.2 模块 A：粗视觉/关系表示

- `E_c` 采用现有 YOLOVO 视觉特征抽取的可冻结子网络，或者强公开冻结 backbone（须保持实验中一致）；只用于区域给定的关系编码，不训练检测头。
- `MaskPool` 以低分辨率 mask 的区域覆盖计算主体/客体表征，联合区域与局部交界几何编码统一输入。
- 直接复用或薄封装关系上下文表示层；不宣称其为新模块。
- 谓词使用现有 `_VocabHead` 或冻结的文本编码器得到向量 `e_p`。对反义/近义谓词需要独立校准与文本语义验证。

建议 `q_0 = F(v_s,v_o,v_union,g,e_p)`，初步输出 `z_0`、`p_0=sigmoid(z_0)`，以及显式的 epistemic/aleatoric 指标（先从廉价置信度和 margin 开始，复杂不确定性后置）。

### 4.3 模块 B：有限候选证据动作集合

首版**不要训练自由生成连续任意 bounding box**；那会使动作空间太大、实验不可控且容易与 GUI grounding 等论文重合。定义固定可复现的 6 种动作：

1. `A_contact`：由两个低分辨率掩码边界的最短距离点或局部交界推导接触候选区域；
2. `A_subject_detail`：主体紧致区域的高分辨率裁剪；
3. `A_object_detail`：客体紧致区域的高分辨率裁剪；
4. `A_union_detail`：两者联合区域；
5. `A_context`：扩张后的联合区域，纳入有限上下文；
6. `A_multi_scale`：同一区域的另一预定义尺度；

所有动作在同一原始分辨率中计算，面积、位置、裁剪放大因子需合法且覆盖边界；初期设高分辨率 ROI 输入 `224×224` 或 `256×256`、最多 2/3 次查询，实际实验通过验证集选。过小或过大的 ROI 由合法性掩码排除。

可选 STOP+ABSTAIN 动作。每一张图的所有候选动作集合只依赖初始合法输入（图像低分辨率特征及掩码几何）和当前已观察状态，**不能使用关系真值或未来 crop 的编码特征**。

### 4.4 模块 C：证据价值估计——建议的主方法贡献

普通策略只读当前不确定性 `H(p_t)`。我们要预测选择动作 `a` 之后的**期望有效损失下降/校准收益**：

\[
V_\phi(a\mid s_t,p)=\mathbb E[\ell(y,\hat p_t)-\ell(y,\hat p_{t+1}^{(a)})\mid s_t,p,a]
-\lambda\,\widehat C(a),
\]

其中 `s_t` 仅包含当前可见信息，未来观察仅在真正选择动作后才可获得；`C(a)` 从真实 profiler 或 benchmark 的离线 lookup 测得（区分输入分辨率、ROI 类型、硬件和 batch）。此处是**期望增量收益的估计量，不是保证能精确计算未来效用**。

给定验证标签的训练数据，可在离线 teacher 阶段枚举少量合法动作，实际运行各动作，计算 `Δloss` 获得监督值：

\[
\tilde V_t(a)=\ell(y,\hat p_t)-\ell(y,\hat p_{t+1}^{(a)})-\lambda C(a).
\]

teacher 所使用的 GT 标签只产生训练目标，绝不能直接馈入线上策略网络；teacher traces 的所有图像必须来自**训练 split**，不得用 PSG 测试标签生成 oracle 行动序列。

建议先用**1-step oracle 枚举 + supervised utility regression** 学会价值估计，再逐步扩展 2-step beam rollout（受计算预算约束），避免一上来使用昂贵且不稳定的纯 RL。

### 4.5 模块 D：证据记忆与关系判断更新

局部 ROI 编码器 `E_f` 输出 `v^a_t`，以 gated residual / cross-attention 更新关系状态：

\[
q_{t+1}=\mathrm{LayerNorm}(q_t+G_\theta(q_t,v_t^a,e_p)\odot U_\theta(v_t^a)).
\]

记忆还应记录已查询区域、代价、视觉尺度，防止重复采集。同一 crop 若被多个关系查询共享，可以 cache，但单 query 与多 query 场景要明确分别核算成本。

### 4.6 模块 E：STOP / ABSTAIN 决策

- 在校准验证集上估计当前预测损失和阈值，不得拿测试集选择阈值。
- 继续动作只有当预测的边际收益大于成本和当前决策风险降低阈值时才允许；预算耗尽时按预定规则预测或 ABSTAIN。
- 使用 `coverage = 已给出非拒绝预测的比例`，`risk = 已接受预测中的错误率` 等风险覆盖指标；拒绝所有困难关系不算优越，应同时报告覆盖率和风险。
- 预测正/负必须对称处理，不能只给正关系补采证据。

## 5. 损失函数与训练日程

总损失建议从可控简化版本开始：

\[
\mathcal L=\mathcal L_{RV}+\alpha\mathcal L_{V}\,+\beta\mathcal L_{stop}
+\gamma\mathcal L_{cons}\,+\eta\mathcal L_{budget}.
\]

- `L_RV`：有明确 binary 标签的关系 BCE / focal；未核验标签忽略；基于官方正例数据的辅助目标单独管理，不能当默认全负分类。
- `L_V`：预测 MVE 与 teacher 的候选动作实际 Δloss 的 SmoothL1 / ranking loss；可混合 pairwise 排序以降低不同样本损失尺度不一致的问题。
- `L_stop`：基于 teacher “继续与停止”的监督，校正误拒/过早结束；如果真实收益已无法超过成本，鼓励停止。
- `L_cons`：对同一关系，局部证据与初始全局判断一致性约束；但在初始判断错误时不能硬逼后续维持原判断，建议只在 teacher/标注认可情形下施加。
- `L_budget`：预算违反惩罚/拉格朗日约束。

**分阶段训练（比端到端直接 RL 更可控）：**

- **S0 数据核验：** 从 PSG 获取图像、掩码、三元组，构建 valid/unlabeled/verified-negative 数据协议与严格分割。
- **S1 关系基线：** 固定 `E_c`，训练关系头；构建 low-res、full-res、fixed ROI 等基线；做类别共现-only/geometry-only 与无图像 oracle sanity check。
- **S2 懒执行局部编码：** 实现 `E_f`，冻结关系头的大部分参数，只训练证据融合层，确保一次局部查询确实提升一部分困难样本。
- **S3 价值 teacher：** 在训练图像上枚举合法单步动作，记录实际损失增益和真实代价；训练 `V_φ` 模仿/排序，严格不在推理时使用 teacher 特征。
- **S4 策略训练：** 先 imitation learning，加入 2-step rollout；视训练稳定性决定是否使用 on-policy REINFORCE/actor-critic 做精调，**不将 RL 当作必要创新**。
- **S5 决策校准：** 在独立验证集拟合停止/拒绝阈值与后验概率校准，固定后才进测试。
- **S6 泛化与效率：** PSG 标准、PSG-Active、VG/GQA 跨域和端到端 profiler。

## 6. 预算设置与预期硬件配置

首版实验配置（均为建议，不是经验最优）：

| 条目 | 建议初值 | 必须验证 |
|---|---|---|
| 输入图片 | 原始 COCO 图像（保留真实源分辨率） | 不把上采样伪装为新细节 |
| 初始 coarse 分辨率 | 256 或 320 短边/标准缩放（具体统一定义） | 与强 low-res baseline 公平 |
| Fine crop | 224×224 / 256×256 | 相同 crop encoder、相同 resize |
| 最多查询步数 | 0、1、2、3 | 绘制准确率随步数变化曲线 |
| 预算 levels | 固定 FLOPs 占全图高分辨率基线预算的 0.25 / 0.5 / 0.75 / 1.0 | 先 profiling，不能预设达到比例 |
| Batch 测试 | bs=1 实时推理 + bs=8 或 16 离线吞吐 | 分别统计，GPU 同步 |
| 可选平台 | RTX 5090 训练；若部署意义成立，再对 Jetson 端单独测 | 端侧 TensorRT、功耗模式需固定 |

第一阶段本机依赖建议：PyTorch + torchvision + COCO panopticapi/pycocotools + 已有 YOLOVO 依赖；单独适配 PSG，**不要为了读取 PSG 强制重装其旧版 MMCV/MMDetection 整个环境**。不得重复制造基础 COCO 解码器、数据增强和 profiler。

## 7. 必做基线与消融（实验矩阵）

### 7.1 强基线（共享相同 mask / backbone / 关系头 / 预算定义）

| 编号 | 方法 | 回答什么问题 |
|---|---|---|
| B0 | 类别对 / 几何先验 only | 是否只靠数据偏差预测关系？ |
| B1 | coarse-only，一次预测 | 不增加视觉查询时的质量下限 |
| B2 | full-res 一次性编码 | 固定全局高成本上限参照 |
| B3 | 固定主体、客体、联合区域 ROI | 仅使用 crop 是否已足够？ |
| B4 | 随机合法 ROI | 策略是否优于随机？ |
| B5 | 预测熵/边际概率驱动 ROI | 是否超越普通不确定性查询？ |
| B6 | Predicate-conditioned attention Top-K | 是否只是简单注意力排序？ |
| B7 | 固定相同查询数（no early stop） | 停止策略本身是否有价值？ |
| B8 | MVE 模型（我们的核心） | 估计增量收益是否有效？ |
| B9 | MVE + STOP/ABSTAIN（完整） | 成本与可靠性如何共同改善？ |

**特别注意：** B3–B9 的局部编码器参数、输入大小、真实特征路径、缓存策略一致；B2 必须报告全部计算，不能只把 ROI 的成本与 B2 的局部子模块成本相比。能耗、吞吐、端到端延迟及 FLOPs 完整报表。

### 7.2 架构消融

- `with/without e_p`：去掉谓词条件，检验是否真正谓词特异。
- `entropy vs MVE`：熵和未来增益的决策差异。
- `single-shot vs two/three-step`：多步是否值得。
- `with/without cost predictor`：只学 accuracy utility 与效用减成本的比较。
- `contact / subject / object / union / context` 每个动作单独删除。
- `mask pooling vs box pooling`：掩码能否提升证据决策；强调不是原创。
- `learned STOP vs calibrated threshold STOP`：证明不是简单阈值就够。
- `with/without abstention`：报告拒绝率、风险覆盖。
- `teacher oracle`：作为**非法部署上界**单独报告，并标注明确使用 GT 标签选择动作，不可作为可执行模型。

### 7.3 评价指标（分成不同协议）

**PSG 官方可比协议：** PredCls R@20/50/100、mR@20/50/100；必要时 additional pair-wise recall，保证完全符合官方定义。

**PSG-Active 二元验证协议：** balanced accuracy、macro-F1、AUROC、AUPRC、Brier/ECE（校准）、正/负分层，PR-AUC 对正负比例敏感，须固定阳性率或报告真实分布；ambiguous 组测拒绝行为，不强行计算准确率。

**预算协议：** 端到端 latency (ms)、FLOPs、峰值显存、ROI 次数、平均成本、不同最大预算下的 accuracy-cost Pareto 与曲线下面积；单查询及多查询共享缓存分别列出。

**证据协议：** 人工证据区域一致性 IoU/pointing、重要性曲线、mask-out causal probe（编辑造成的分布偏移必须有控制组），以及选中区域是否真正带来实测预测收益；不要只依赖 attention map 自证可解释性。

**统计检验：** >=3 个训练随机种子；PSG-Active test 按 image_id 聚类 bootstrap 置信区间；同图关系查询应视为相关观测；匹配运行硬件与 PyTorch/CUDA 版本；给出不确定区间而不是单次分数。

### 7.4 关键负面检查

1. **区域泄漏：** GT 掩码边界是否间接泄露了未观察的高分辨率纹理？
2. **策略作弊：** teacher 的未来 crop 特征/GT label 有没有流入策略输入？
3. **标签缺失：** 未标注关系是不是被错误地当作真负例？
4. **伪省算力：** 是否预先运行全图高分辨率网络或缓存了测试集特征，却只计算 ROI 的 FLOPs？
5. **频率捷径：** 去掉视觉输入后是否仍能得到令人误解的高分？
6. **难样本选择偏差：** PSG-Active 是否人工过度筛选只利于高分辨率裁剪的样本？
7. **过拟合人工证据：** 是否将人工 evidence 标注用于训练和测试并且未区分两者？
8. **多 query 重复：** 共享 crop 缓存后是否出现跨模型预算口径不一致？

## 8. 对现有代码的实质适配清单

代码根据压缩包静态检视，不是已运行验证。

| 文件/模块 | 当前情况 | 保留/新增改造 |
|---|---|---|
| `ultralytics/models/yolovo/relate/head.py` | RA-like 关系头；`_compute()` 基于归一化 boxes；P4 base map，`forward` 中 `.detach()`；`_Sampler` 筛候选；`_VocabHead` 文本谓词 | **保留为基线**；通过外层 adapter 暴露 q、候选和谓词嵌入；勿破坏旧 checkpoint 接口 |
| `.../ra/pooling.py` | `SoftSpatialPool` 已支持可选 mask coverage 参数 `cov`，但当前主 `_compute()` 传的是 boxes，未传 mask coverage | 优先薄封装 `MaskRegionAdapter` 使用 `cov`，避免重写池化；确认 mask 边界/梯度含义 |
| `.../ra/deformable.py` | 已有视觉特征读取，依赖提前生成的 scene map | 可用于 coarse 计算；**不能**直接声称其在 high-res lazy crop 上省算力 |
| `.../dataset.py` | YOLO txt + `R sub obj predicate`，主体客体索引依赖框行 | 新增 `PSGRelationDataset` 和安全的原始 segment_id → contiguous instance_id 映射，保留旧 dataset |
| `.../trainer.py` | 关系二级训练、冻结主干、GT 框输入 | 抽取普通 PyTorch 训练组件或加独立 trainer；**不要**绕过标准优化器/AMP 逻辑 |
| `.../predict.py` | `DetectionPredictor` + NMS + relations；循环 `rel.score(boxes)` 未明确传 `img_idx` | 第一阶段**不经此检测后处理路径**；给验证工具写独立 `RelationQueryPredictor`；若保留旧路径则测试 batch>1 特征索引 |
| `.../results.py` | 关系输出/可视化 | 可复用格式；新增 `queries/actions/costs/abstained` 可视化字段 |
| `GenRelated.py`, `gen.py` | 基于 VG 类别先验、框几何生成弱关系标签 | 不作为 PSG 的主标签生成器；仅用于弱监督抗捷径对照组 |
| `trainOVrel.py` | 使用本地 M3FD 路径和检测 ckpt 的训练入口 | 新增干净独立 `train_barv.py`，config 下分阶段训练；不用硬编码 `/root/...` |

**两个工程先验风险：** 

- 该 ZIP 含轻量 `yolo11n.pt`，但没有提供 `trainOVrel.py` 所指向的 M3FD 权重和完整数据。复现现有关系推理需要确认额外权重；不能声称已有可用的 relation checkpoint。
- 现有 `SoftSpatialPool` 提供 mask coverage 接口，但 `YOLOVORelate._compute()` 暂未在主路径调用它；必须以单元测试验证正确传递，不宜仅因接口存在就认为已有像素级关系支持。

## 9. 推荐的项目目录与测试

```text
barv/
├── configs/
│   ├── psg_coarse.yaml
│   ├── psg_value_teacher.yaml
│   ├── psg_policy.yaml
│   └── eval_cost_curves.yaml
├── data/
│   ├── README.md                # 数据许可和官方链接；原始数据不入 Git
│   ├── adapters/psg.py
│   ├── adapters/vg150.py
│   ├── split_manifests/
│   ├── active_query_protocol/
│   └── validators/
├── barv/
│   ├── backbones/coarse.py
│   ├── backbones/fine_roi.py
│   ├── regions/mask_adapter.py
│   ├── relations/ra_baseline_adapter.py
│   ├── policy/action_candidates.py
│   ├── policy/value_estimator.py
│   ├── policy/stop_policy.py
│   ├── memory/evidence_memory.py
│   ├── training/teacher_rollouts.py
│   ├── training/train_relation.py
│   ├── training/train_policy.py
│   └── evaluation/benchmark.py
├── tests/
│   ├── test_psg_index_mapping.py
│   ├── test_unknown_is_not_negative.py
│   ├── test_action_masking.py
│   ├── test_no_oracle_leakage.py
│   ├── test_lazy_fine_encoder.py
│   ├── test_cost_accounting.py
│   ├── test_batch_feature_alignment.py
│   └── test_crop_original_resolution.py
└── reports/
    ├── audit_dataset.md
    ├── baseline_metrics.json
    ├── utility_regression.json
    ├── ablations.csv
    └── profiling_hw.json
```

**关键单元测试样例：**

- 禁止全部 crop 时，`fine_encoder.forward()` 调用次数必须是零。
- 两幅图 batched 推理时，两个图的特征和 query ID 必须对应，不能均访问 batch 0。
- 同一关系被标为 unknown 时，不应进入负样本 BCE。
- 未授权测试图不允许生成 `teacher_rollout` 的 GT-dependent targets。
- 对同一输入固定种子/预算，动作选择与时延统计可重复；epoch RNG 与 DDP worker 种子明确设置。
- 再采样生成 ROI 时，必须从原图坐标裁剪，不能从已经缩小的 coarse 图像上采样冒充细节。

## 10. 研究工作分期、检查点和止损条件

| 里程碑 | 可验收交付物 | 失败时如何调整 |
|---|---|---|
| M0：任务与数据正确性 | PSG 读写器 + 200 图像可视化 + 分区/关系缺失审计 | 若标签不适合二元验证，先缩小至可靠谓词子集 |
| M1：基线复现 | coarse-only、full-res、固定 ROI、随机 ROI 四条可复现曲线 | 若 full-res 无显著额外收益，主动查询无意义，调整输入分辨率和困难样本定义；不得只筛测试集 |
| M2：按需高分辨率执行 | `fine_encoder` 真正按需调用及 profiler；单 query 成本报告 | 若 ROI 开销不低于全图，只做质量自适应，不夸称省算力 |
| M3：证据价值预测 | Teacher 动作收益与预测值相关性（Spearman）、行动排序指标、oracle gap | 若价值预测不优于 entropy，考虑去掉 MVE 研究主张或重设价值监督 |
| M4：策略模型 | MVE 与所有固定/启发式基线 Pareto 曲线 | 若优势仅来自更多计算，采用公平预算匹配重评 |
| M5：精标与强评价 | PSG-Active 标注协议、QA、盲测、差异分析 | 若只有类别共现能预测，增强相同类别对但不同谓词的难例 |
| M6：论文级稳健结果 | seed/CI/效率、跨数据集、消融、反捷径、失败案例 | 未达到预设贡献阈值则止损，不强行包装为 SOTA |

**预先规定的成功门槛（建议；不是保证结果）：**

- 至少在两个固定预算点上，相比最强同计算预算启发式策略有**显著**改善（置信区间不跨零），且提升在不同谓词类别/难度上不是单一类驱动；
- 至少一组在约 50% 全图高分辨率推理预算下，关系质量能接近 full-res 强基线（例如下降控制在 **1–2 个百分点以内**，作为待验证目标而非数据承诺）；
- MVE 与 entropy-only 的差异可经消融复现，而不只是策略阈值调得不同；
- 端到端 ms 与 FLOPs 的趋势一致或明确解释差异，不能把 FLOPs 降低直接等于真实速度提升；
- 精标集难度/负例/歧义比例公开、无测试泄漏。

## 11. 论文叙事与贡献结构

**Problem：** 在关系验证中，不同谓词需要的视觉证据不同，固定高分辨率计算效率低，而一次性粗预测又易遗漏接触、状态等细节；现有主动视觉工作大多面向 VQA/定位，没有把**关系特定的证据收益与真实按需计算成本、停止与拒绝**统一成可测试的决策问题。

**Method：** Predicate-conditioned Marginal Evidence Value（PCMVE）估计 + lazy ROI feature acquisition + evidence memory + cost-calibrated stop/abstain，支持给定区域的非检测关系验证。

**Benchmark：** PSG 的严格 mask-conditional query protocol；PSG-Active 的独立负例/歧义/证据标注与性能-计算双评估。

**Experiments：** Equal-budget comparisons、oracle value gap、cost–accuracy Pareto、selective prediction、robustness/shortcut controls、real hardware latency。

**可选后续工作**（首篇论文完成之前不要强制加入）：

- 用不依赖标签的视觉 region proposals 替换 oracle masks，转向 open-world relation verification；
- 跨模态 RGB-X 主动证据获取；需独立构建真实跨模态关系标签，不能依赖 M3FD 检测框伪标签；
- 视频/具身视觉主动观察，研究动作产生新的真实传感观测而非仅 crop 已存在的图片。

## 12. 仍待确定的严肃风险

1. **任务重合风险：** AdaptVision、RegionFocus 与预测价值驱动 acquisition 文献可能已包含类似策略。必须在开发前补充精确 nearest-neighbor 文献表，方法写作不能只改应用场景。
2. **数据主张风险：** PSG 没有公认逐关系“关键证据区域”标注；任何由模型 attention 自生产的区域都不能声称是人工证据真值。
3. **标注成本风险：** 最初 500–1,000 个 pilot 查询的精标规模对于56个谓词太小，不足以支持所有谓词独立显著性；应按谓词簇做分层和置信区间，并按第 2.3 节的训练/验证/盲测分期扩充，不得因样本太少便对每个谓词宣称显著性。
4. **算力收益风险：** 如果 coarse backbone 已接近 full-res backbone 的计算量或 fine ROI encoder 极重，策略未必有速度收益。代码应先 Profile 再训练 policy。
5. **训练策略风险：** teacher 枚举所有候选动作会造成离线训练开销；需记录 teacher compute，不将其列为推理成本，但在总训练成本部分如实报告。
6. **可迁移性风险：** 只在 mask-oracle 协议下强表现不等于真实应用有效；P2 端到端仅作为补充实验，不与 P1 混淆。
7. **复用许可风险：** YOLOVO 工程含 Ultralytics AGPL；RelateAnything 官网目前存在 AGPL-3.0 及不同模型/标注许可声明。复用与再分发时必须逐模块核对本地文件版权、上游版本和研究发布要求，尤其关注本地 port 注释与上游现行许可是否冲突。

## 13. 已核验的关键资料

1. PSG 官方代码和下载说明（ECCV 2022）：https://github.com/Jingkang50/OpenPSG
2. PSG 官方数据统计：https://psgdataset.org/data_stats.html
3. FleVRS（NeurIPS 2024）：https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2e065133af98888ab11a549abed2cc3-Abstract-Conference.html
4. ART（ICCV 2025）：https://openaccess.thecvf.com/content/ICCV2025/html/Sudhakaran_ART_Adaptive_Relation_Tuning_for_Generalized_Relation_Prediction_ICCV_2025_paper.html
5. Visual Test-time Scaling for GUI Agent Grounding / RegionFocus（ICCV 2025）：https://openaccess.thecvf.com/content/ICCV2025/html/Luo_Visual_Test-time_Scaling_for_GUI_Agent_Grounding_ICCV_2025_paper.html
6. AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition（2025 arXiv）：https://arxiv.org/abs/2512.03794
7. RelateAnything（2026 arXiv）：https://arxiv.org/abs/2609.12552
8. RelateAnything 官方源码/许可说明：https://github.com/Maelic/RelateAnything
9. CAGE-SGG 官方撤稿状态（**已撤稿，不得引用其实验为已成立结论**）：https://arxiv.org/abs/2604.22274
10. VG150 benchmark dataset说明：https://github.com/Maelic/SGG-Benchmark/blob/main/docs/DATASET.md

---

**最终研究判断：** 当前方案具有清晰的问题、代码适配路径和可证伪评价，但还不是一个已经被实验验证的投稿级成果。优先解决 PSG 数据协议、真实算力计量、低/高分辨率 oracle gap 这三个“是否值得做”的问题，再投入复杂策略与多步训练。任何方法宣称都应以同预算强基线、硬件测量和独立标签验证为前提。

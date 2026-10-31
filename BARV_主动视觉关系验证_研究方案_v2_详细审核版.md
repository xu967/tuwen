# BARV v2.0：预算约束下的主动视觉关系验证——研究方案与严格审核

> **工作论文标题（暂定）**：**BARV: Predicate-Conditioned Marginal Evidence Acquisition for Budgeted Visual Relation Verification**  
> **中文题目**：面向视觉关系验证的谓词条件化边际证据获取与预算决策  
> **候选加强版标题**：**ShareBARV: Joint Evidence Scheduling for Multiple Visual Relations under Compute Budgets**  
> **日期**：2026-10-26  
> **材料**：`YOLOVOmodal_3871538535.zip` 的关系模块；上一版 `BARV_主动视觉关系验证_详细研究方案.md`；截至 2026-10-26 能核验的论文/项目页面。  
> **状态**：研究设计，尚无本研究真实训练结果；数据集未在本轮下载、模型未完成训练/实测。所有资源配置、性能阈值和时程均是**待核验建议**，绝非实验结论。  
> **任务边界**：不研究目标检测；默认给定 subject/object 区域掩码和自然语言谓词；重点研究关系判断所需的视觉证据取得策略。  
> **读者说明**：本文件同时包括**审稿式审核结果**、**经过修订的完整执行方案**、**能复现实验的协议与接口定义**、**失败条件与止损方案**。请先读第 0–3 节再决定投入代码。

---

# 第 0 部分：执行结论、研究范围与审查结论

## 0.1 是否推荐继续？

**条件性推荐；不推荐直接按 v1.0 开始训练策略。** 原版方向是可以检验的工程研究，但 2026 年已出现与其“粗到细采证 + 答/拒/继续”组合非常接近的工作。必须将可声称的研究贡献收缩为**关系谓词特异性的边际证据价值估计**和（更有区分度的）**同图多关系的联合证据调度及跨查询复用**，再通过同预算强基线验证其不可被简单启发式替代。

推荐先执行 M0–M2（数据真值、预算测量、低高分辨率差距）。当它们都过关，才投资 M3–M6（价值学习、多步策略、开放词汇泛化）。

**两个互斥的主线选择**（首篇论文只定一条主线）：

- **主线 A / 最小创新单元**：*Single-query Predicate-conditioned Marginal Evidence Value*：对给定 `(subject mask, predicate, object mask)`，选择哪一种区域-尺度动作可带来最大的真实任务收益 / 新增计算成本，并决定停止；可直接形成方法论文，但与已有预算式视觉证据获取邻近，**创新风险较高**。
- **主线 B / 推荐的加强版**：*Multi-query Joint Evidence Scheduling*：同一图像含多个待检验的关系查询，动作既可改善某条关系，也可能改善多条关系；研究有限预算下如何联合选择、共享和复用证据。这改变了决策单元（从单个命题到整幅图像的关系集合），能提出**关系特异性效用、共享收益、重叠惩罚和条件边际价值**。但是否真正区别于已有共享特征/主动感知，仍需严格检索。

注意：主线 B 不是把主线 A 随意“升级”，而是**在理论定义、算法、成本核算和指标层面都需要独立成立**。本文件用 A 作为可以快速运行的基线，以 B 作为优先验证的研究主张。

## 0.2 对上一版 v1.0 的逐条审查与修改

| 级别 | v1.0 问题 | 为什么是问题 | v2.0 修复 | 验收证据 |
|---|---|---|---|---|
| **P0** | 将“获取证据 / 停止 / 拒绝”表述成主要创新 | AdaptVision 已有动态放大；BCEA 已研究受预算约束的证据获取、回答/拒绝，且含空间关系 claim | 不再宣称上述操作本身新颖；重点转向谓词条件的可验证**增量收益**与**同图多关系联合调度** | 与 AdaptVision/BCEA/RelateAnything 的差异矩阵、强基线与消融 |
| **P0** | PSG 上自动将没标的关系视为负例 | PSG 关系标注稀疏，未标注≠不成立 | 三值标签 `positive/negative/unknown`，另列 `ambiguous` 人类评估标签；经核验的负例优先 | 采样审核、人工一致性、unknown 不反传负标签 |
| **P0** | 直接将已有关系头移到 Mask 条件 | 主关系路径用 box 和完整 scene feature；`cov_lambda` 初始为零 | 外层 `PSGAdapter`、`MaskFeaturePool`、`RelationQueryEncoder`，并行保留现有实现 | 单元测试/逐层数值与形状检查 |
| **P0** | 只优化单 query 计算，忽略一图多关系 | 一个 ROI 可能被多关系复用；完整图 backbone 常常已经提前运行 | 预算分为**per-image** 和 **per-query**；B 主线按全图 query 集合调度，计入缓存与共用成本 | 冷启动 / 缓存命中 / batch=1 / 多 query 延迟 |
| **P0** | 用 `Δloss` 直接作为“最优采证依据” | `Δloss` 是知道 GT 的**离线 hindsight 值**，不是推理期可见的信息；模型更新后 teacher 可能过时 | 冻结 scorer 造 teacher，策略仅看当前状态；在 train 抽轨迹；分轮刷新及阻止未来特征泄漏 | teacher/online 输入 schema 区分，oracle gap |
| **P0** | 忽略选择后校准问题 | 动态选择会改变分数分布；原始阈值的覆盖率或保证不再可靠 | 策略锁定后，独立 calibrate split 再校准；如使用 conformal，按完整采证策略重校准 | post-selection ECE / risk–coverage，严格划分 |
| **P1** | 500–1000 条标注即可报告所有谓词 | 56 个谓词长尾，样本量不足，IoU 等证据指标方差大 | pilot 只做可行性；宏平均按谓词簇，正式 blind test 分层抽样及图像聚类 CI | 样本量统计与抽样登记 |
| **P1** | 关系证据区域概念不清 | 主客体 mask 并不是“为什么关系成立”的证据；反例有时需要全局语境 | 正/负证据、局部/全局/不可定位、多区域标签独立；证据与决策评价分开 | 标注 handbook、人工复核样例 |
| **P1** | 目标函数 `L_cons` 可能抑制纠错 | coarse 错时强行一致会损害 fine 的纠正作用 | 默认关闭，必要时只对可信正确的 teacher 分支启用 | 对比实验而非预设有效 |
| **P1** | ROI 224/256 与吞吐假定缺少硬件依据 | 小裁剪、反复 kernel launch、CPU crop 可能比全图更慢 | 先有 CPU/GPU profiler 与批量 ROI scheduler，再训练策略 | 真实 ms、FLOPs、memory、GPU 活跃率 |
| **P1** | PSG 标注版、拆分数量容易混用 | 官方存在完整版和竞赛版；官网总量与竞赛组合不必相等 | 以文件 SHA256 + image_id manifest 为准，绝不机械相加不同版本 | 数据版本报告与完整性脚本 |
| **P1** | 对原有 checkpoint 的可用性假定不足 | 压缩包提供代码，但特定 relation 权重/真实数据未见 | 做代码基线 + 权重清点 + 自训练可复现路径 | checkpoint audit、训练日志、seed |
| **P2** | 没有区分“主动读旧图”与“主动摄影” | crop 原图不能提供图像本来没有的信息，不能视为机器人主动感知 | 全文统一表述为 **adaptive visual computation / evidence acquisition**，不宣称获得新的真实观测 | 术语、实验和任务定义一致 |

## 0.3 文献近邻审查：必须放在开工之前

截至本轮核验，以下研究是直接威胁 BARV 创新性的工作：

| 论文 | 日期、来源 | 已覆盖内容 | 我们不能再声称什么 | 仍需验证的差异 |
|---|---|---|---|---|
| **AdaptVision** | CVPR 2026（2025-12 arXiv） | 粗到细、低分辨率初看、调用裁剪获取额外视觉信息、按样本动态分配计算 | “首次动态放大”“首次采证减少视觉成本” | 谓词条件化的监督信号；同图关系集合共享收益；分割实体协议 |
| **BCEA: Look Again Before You Abstain** | arXiv 2026-06/07 | 预算内回答/拒绝/获取证据；含空间关系 claim；强调动态选择后校准 | “首次联合 answer/abstain/acquire”“首次将预算采证用于空间关系” | 结构化 subject–predicate–object 的实体配对、多关系联动、真实 ROI 成本及关系证据标注 |
| **RelateAnything** | arXiv 2026-09 | 来自任意来源的 regions、文本谓词库、轻量关系预测、PU 监督与跨集评价 | “首次开放词汇关系预测”“首次无 detector-label 条件关系评分” | 预算策略与关系间复用；也要以其公开模型为强关系评分基线 |
| **FleVRS** | NeurIPS 2024 | 提示式和开放词汇视觉关系分割 | “首次 Mask 关系预测” | 输入给定 Mask，不把 Mask 分割当贡献 |
| **A Fair Ranking and New Model for PSG** | ECCV 2024 | 强调 PSG 负标签不可靠、评价和学习协议偏差 | “首次发现 PSG 缺失负标签问题” | 训练/评测利用已核验三值标签与预算设定 |
| **CAGE-SGG** | arXiv 2026，**作者撤稿** | 涉及反事实关系证据的概念 | 不能以作者撤稿论文的实验数字或理论作为可靠依据 | 可列历史检索注记；不给该方法可靠性背书 |

**必须新增引用**：`https://arxiv.org/abs/2606.16667`（BCEA）；`https://openaccess.thecvf.com/content/CVPR2026/html/Lin_AdaptVision_Efficient_Vision-Language_Models_via_Adaptive_Visual_Acquisition_CVPR_2026_paper.html`；`https://arxiv.org/abs/2609.12552`；`https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07817.pdf`。完整引文表见附录 A。

**文献结论**：目前有机会形成有价值研究，但“没人做过”**尚未证明**。如果主线 A 在同预算比较中只比固定裁剪略好、主线 B 的共享策略不优于简单的 greedy ROI 缓存，那么此项目不应该继续按“重大方法创新”包装。

---

# 第 1 部分：研究问题和形式化定义

## 1.1 研究动机：观察信息的边际价值因谓词而异

给定同一对实体，判断不同谓词可能需要完全不同的细节：

- `person — holding — cup`：往往要检查手部与杯口/杯柄的局部接触细节；
- `person — looking at — cup`：可能需要脸部朝向或场景上下文，接触 ROI 不一定有帮助；
- `person — sitting on — chair`：人体姿态与椅面接触区域；
- `car — parked on — road` 与 `car — driving on — road`：单帧静态图可能不能明确区分运动状态；有些查询应拒绝或被标记不够可观察；
- `person — wearing — backpack`：主体边缘与肩部状态往往比主体中心纹理更有用。

所以，单纯“模型不确定时放大主体”并不是我们要研究的策略。真正的决策是：**哪种证据对当前谓词、当前模态和当前状态具有正的预期边际收益，且该收益值得付出实际计算代价？**

## 1.2 单查询协议（Q-RV）

输入：图像 $I\in\mathbb R^{H\times W\times3}$、给定实体掩码 $M_s,M_o\in\{0,1\}^{H\times W}$、谓词文本 $p$、预算 $B$。

- 监督标签：$y\in\{1,0,\varnothing\}$，其中 $\varnothing$ 是**未知/未核验**。`ambiguous` 是人工额外判断标签，单独评估；
- 模型在 $t=0$ 仅看到缩放图 $I_c$、被缩放的 Masks、实体结构描述、谓词嵌入及已经公开的操作约束；
- 每一步选动作 $a_t\in\mathcal A(s_t)$：获取局部细节 / 提交正判断 / 提交负判断 / 拒绝判断；
- 获取动作会从**原始图像**裁剪 ROI 并**当时才执行** fine encoder，产生 $o_{t+1}$；
- 结束时输出关系概率、决策与可复查的动作历史，**并报告全流程真实成本**。

$\mathcal A(s_t)$ 的合法性只由任务输入、已读取的历史、预算余额决定，不能依赖 GT 标签或未读取 ROI 特征。

## 1.3 同图多查询协议（G-RV，推荐论文主问题）

一张图像同时存在多个待判断关系：

$$\mathcal Q=\{q_j=(M_{s_j},p_j,M_{o_j})\}_{j=1}^m.$$

共享一次 coarse 编码，策略面对的是**整幅图像预算** $B_{\text{img}}$：

$$a_t = \pi_\phi(S_t,\mathcal Q,B_{\mathrm{remain}}), \qquad
S_t=\{z_{j,t},u_{j,t},\mathcal C_t,E_{\rm cache}\}_{j=1}^m.$$

一次查询局部区域 $a$ 可能同时降低几条关系的不确定性；若已经算过高分辨率 crop，可以从 cache 共享，但只有**坐标、尺度、预处理和编码器版本确实一致**时，才允许复用。对重叠但不相同的 crop，先视为不同计算，除非实现和实测了可共享的 patch 编码。

$$\max_\pi\ \mathbb E\!\left[\sum_{j=1}^m w_j U(\hat y_j,y_j,d_j)\right]
\quad\mathrm{s.t.}\quad C_{\rm image}(\pi)\le B_{\rm img}.$$

在单独的 Q-RV 协议中，不能用 G-RV 的共享收益来夸大单查询加速；两项实验要分栏。

## 1.4 三种输出与损失

在一条经核验的二元关系上，最终动作 $d_j\in\{\text{POS},\text{NEG},\text{ABSTAIN}\}$。

可使用可解释的决策损失：

$$L_d(d,y)=\begin{cases}
0,&d\in\{\mathrm{POS},\mathrm{NEG}\},\ d=y\\
c_{\rm err}(y),&d\in\{\mathrm{POS},\mathrm{NEG}\},\ d\ne y\\
c_{\rm abs},&d=\mathrm{ABSTAIN}
\end{cases}.$$

这里正负错误的代价与拒绝代价应在验证集上预设或来自实际应用，**不能用最终 test 搜索使方法最优的代价参数**。对于真值 `ambiguous`，仅按“是否合理拒绝、是否错报确定关系”评价，不参与强制正/负 BCE。

## 1.5 明确区分三种成本

1. **计算理论成本** $C_{\rm FLOPs}$：FLOPs/MACs，统一计量规则；
2. **可部署成本** $C_{\rm latency}$：含预处理、策略、裁剪、编码、特征合并、判断、缓存管理的 p50/p90/p95 毫秒；
3. **总任务成本** $C_{\rm total}$：可能还包含离线训练时 teacher 生成开销、人工作业成本；与推理成本分别报告。

不得让 $C_{\rm FLOPs}$ 的改善自动等于 $C_{\rm latency}$ 的改善，更不得用全部 Query 共用的 $C_c$ 在每个 Query 上重复计费。

---

# 第 2 部分：数据集与负样本构造，完整的真实数据路线

## 2.1 PSG 为主：获取、来源与版本

官方项目：<https://github.com/Jingkang50/OpenPSG>；统计：<https://psgdataset.org/data_stats.html>。官方统计为 **48,749 张图像、133 个实体类别、56 个谓词类别**，全景实体标注与关系标注建立在 COCO 图像/掩码基础上。官方 GitHub README 同时提供了**论文完整版**与**竞赛版**数据链接，并提示竞赛版部分测试标注被删去。

特别注意：官方 README 对竞赛文件提到 `psg_train_val.json` 含 **45,697 训练 + 1,000 验证**，`psg_val_test.json` 含 **1,000 验证 + 1,177 无 GT 测试**。这两份竞赛 JSON 的计数不能简单和官方全量统计等同，**不同版本不能混用**。正式代码必须逐项检查当前下载文件的具体 `image_id` 和标注字段，并记录 SHA256。

首版仅需获取：

1. OpenPSG 官方完整标注 JSON（必要时合法备份下载链接）；
2. 对应 COCO 2017 RGB 图像；
3. 对应 panoptic PNG + `segments_info`（以标注中关联的文件名为准）；
4. 词表、schema 定义与官方 evaluator 的必要源码（尽量不安装旧版 MMCV 整套训练环境）。

原始数据只存本地，目录下记录出处、许可、文件大小、校验和、异常列表；不得把 COCO 图像打包进新代码仓库。

### 2.1.1 为什么不用 M3FD 当主关系真值

现有 `RelGen` 使用 Visual Genome 类别对频率先验和框几何规则创建标签，不是人工核实的视觉关系。它可以作为 **prior-only / pseudo-label** 对照，绝不能因为自动生成三元组，就直接用它作为论文中的语义关系测试 GT。

### 2.1.2 不建议一开始追求开放词汇或跨模态

56 个 PSG 谓词足够验证主动信息获取。第一篇先证明在固定关系词表下真正减少计算且改善证据判断。开放词汇和 RGB-X 可作为后续迁移，不要在未跑通基准前重复扩大问题边界。

## 2.2 PSG 原始样本与适配后样本 schema

**先以官方 JSON 为准做 runtime 探查；下面是我们内部标准化格式，不是假称下载的 PSG 原始 JSON 一定具有完全相同字段。**

```json
{
  "image_uid": "coco_image_id_or_hash",
  "image_path": "coco/train2017/...jpg",
  "panoptic_path": "coco/panoptic_train2017/...png",
  "segments": [
    {"segment_id": 29, "category_id": 0, "mask_key": "...", "bbox_xyxy": [12, 10, 72, 122]}
  ],
  "relations": [
    {"subject_segment_id": 29, "object_segment_id": 31, "predicate_id": 12, "source": "psg_official"}
  ],
  "split": "train",
  "source_sha256": "...",
  "mapping_version": "v1"
}
```

**关键：** 官方 `relations` 的主客体值在某些处理路径中可能是 `segments_info` 列表下标而不是 panoptic PNG 的原始 segment id。必须检查官方数据加载器的映射定义。Adapter 应实现两个独立字段：`local_relation_index` 与 `panoptic_segment_id`，在解码时按官方语义映射，不能未经检查把两者当作同一个 ID。

首批必须运行 `assert`：

- 主/客体索引合法；
- 每个 index 与 `segments_info` 记录一致，PNG 原始 segment id 可查询；
- binary mask 非空，与 bbox 匹配；
- 图片与掩码 H/W 一致；
- 关系有方向（`subject != object` 的默认检查，允许官方明确的异常）；
- category/predicate 名称映射稳定；
- image-level split 没有交叉。

## 2.3 正样本、已知负样本、未知与歧义：四种状态

| 标签 | 含义 | 是否用于 BCE | 生成方式 |
|---|---|---:|---|
| `1` | 人工确认该关系成立 | 是 | PSG 原标注复核/新增人工确认 |
| `0` | 人工确认该关系不成立 | 是 | 在明确谓词定义下人工核验的反例 |
| `unknown` | 没有核验，不能判断是否成立 | 否 | 原始 PSG 未标注关系、未审过的候选 |
| `ambiguous` | 当前单幅图确实不足以确定真假，或标注者争议未解决 | 否（默认） | 人工按协议标为“证据不足”，单独评估 abstain |

**容易犯的错误**：`negative` 不是“与当前正谓词不同”；同一实体对可以同时满足 `beside`、`holding`、`looking at`，也可能存在同义词。不能只用类别逻辑推导出语义互斥。

### 2.3.1 困难负样本的四类设计

- **同一类别对、不同交互**：例如同一组 `person/cup`，拿着 vs 桌上放着；这是控制类别共现捷径的关键；
- **同一实体对、语义近邻谓词**：`holding` 与 `touching`、`standing on` 与 `sitting on`，关系成立与否必须由人工重新确认；
- **同一谓词、姿态/场景相似但关系不成立**：不让模型用对象数量或背景粗线索取巧；
- **高分辨率局部细节反例**：粗尺度疑似成立，局部细节证明并非如此。

训练时可以用 `PU` / `masked BCE` 为大规模未核验关系提供弱监督；但**主论文二元真值评价仅基于经过核验的数据**。

## 2.4 自建 PSG-Active：有约束的标注工程

名称 `PSG-Active` 为暂定，需先查是否撞名。先做小规模 pilot（**500–1,000 条查询**），之后才根据误差和标注预算决定扩张。Pilot 只能诊断问题是否成立，不能支撑对 56 个谓词逐个断言方法有效。

每条人工核验记录示例：

```json
{
  "query_id": "sha256(image_id|subject_id|predicate_id|object_id)",
  "image_id": "...",
  "subject_segment_id": 101,
  "object_segment_id": 204,
  "predicate_id": 17,
  "relation_label": "positive",
  "observability": "sufficient|insufficient|disputed",
  "evidence": [
    {"box_xyxy": [32, 45, 84, 99], "role": "supports", "scope": "local"}
  ],
  "evidence_global": false,
  "coarse_observability": "yes|no|uncertain",
  "hard_negative_type": null,
  "annotator_ids": ["annotator_1", "annotator_2"],
  "adjudication": "agreed|third_review",
  "split": "dev"
}
```

**标注流程与准则**：

1. 将谓词写成可判定的自然语言规则：`holding` 需要什么证据、何种状态视作不成立、何时只能判为不可观察；
2. 标注者先独立看完整图像和对应 masks，回答真假或不可确定；不要先看模型输出、候选 ROI 或策略轨迹，避免暗示；
3. 对可以判断的关系，独立标记最小支持/反驳区域；允许多个非连续区域或 `global`，且不强求所有谓词都有唯一局部证据；
4. 对细节依赖再单独考察固定 coarse 图像，确定是否存在需要额外分辨率的情形；这一标注和主关系真值分开呈现；
5. 双人标注 + 争议第三方复核，记录 Cohen's κ（真假、可观察性分别）和证据区域一致性；
6. 保留 negative、positive、ambiguous 各层比例与谓词簇分布；预先固定 blind test，标注后不允许因模型表现重挑测试集；
7. 开发/正式测试按 image_id 不相交，图像相近度去重；记录标注成本与丢弃原因。

**推荐版本化**：`psg_active_v0_pilot`（仅研究可行性）→ `v1_train_dev`（方法开发）→ `v1_blind_test`（仅最终评价）。

## 2.5 两种 test 分布同时报告

- **Natural**：尽量贴近原始 PSG 谓词/图像/实体对分布，用于估计实际使用表现。统计失衡要如实呈现；
- **Balanced-Hard**：按谓词簇、positive/negative、尺度、类对共现分层重采样，专门放大反捷径和细节感知能力差异。`Balanced-Hard` 不能冒充真实世界分布。

所有主要模型在这两种测试分布上报告分数，并附 bootstrap 95% CI。凡同一张图像生成多个查询，bootstrap **以 image_id 分组重采样**，不应把每条查询假装为独立样本。

## 2.6 训练/校准/测试划分与防泄漏

图像级数据划分：`train`, `dev`, `calib`, `blind_test`，有必要另保留 `stress_test`。先遵守下载版本的官方分割，不得将带真值的官方 test 用于任何 Teacher 轨迹与阈值优化；从官方训练区内再分出独立 `dev/calib`。

对多查询联合策略：**来自相同图片的所有查询必须在同一个 split**。任何训练时共享 cache、近重复照片和跨数据集数据合并都要遵守这个约束。

## 2.7 VG150 / GQA 扩展

- VG150 只作框区域协议的附加域外基线，谓词与实体类别重新映射；不能直接把掩码主实验 mIoU 或 binary F1 当同协议对比；
- GQA 适合多关系组成与长尾组合，但其中的程序表示不等于真实的采证动作序列；
- PSG / Visual Genome 存在已知图像来源重叠，必须用 id 映射+感知哈希做去重，不可据来源不同就视为“跨域”。

## 2.8 审核补充：少量人工负例不足以从零训练关系模型

这是 v1.0 很容易忽略、也最可能导致工程做不下去的**实际瓶颈**。500–1,000 条 pilot 人工查询同时覆盖 56 个谓词、正负和歧义，很难单独训练一个具备稳定泛化的 binary relation scorer，更不足以直接给所有动作的效用网络提供高质量监督。**Pilot 的定位只能是验证标签定义、Oracle Gap 和是否值得追加标注。**

建议将关系评分器训练与“真假核验数据”区分为三层：

- **表征预训练层**：在 PSG 官方训练数据的已标注正三元组上做 subject/object/predicate 关系表示学习。可学习正关系的打分或正例对比，但**无标注不直接置零**。也可研究在非测试重叠数据上预训练的公开关系权重，明示预训练来源与泄漏核验；
- **可靠判别微调层**：使用单独的已核验正负 `verified_train` 样本训练二元 head（含每谓词频率/不均衡处理）；如果正负样本规模太小，保留原评分器并只在 `calib` 数据上拟合小型温度/阈值，而不是让大量参数过拟合几百条标签；
- **Teacher 效用层**：只能在足够多且**独立于当前 scorer 拟合样本**的 verified training queries 上构建明确的 `loss before-after` 标签。若独立标注不够，使用 `K` 折交叉拟合获取每条样本的 out-of-fold 预测；绝不从 blind test 生成 teacher utility。

建议资源分配采用滚动策略：

| 数据层 | 规模建议 | 用途 | 是否可用二元损失 |
|---|---:|---|---|
| PSG 官方已标注关系 | 由官方版本决定 | 关系表征学习、正例辅助任务 | 已有正例可以；未标注负例不行 |
| Pilot 已核验查询 | 500–1,000 条 | 标注 QA、任务有效性、初始阈值分布 | 可在 dev 上分析，不作为最终盲测 |
| `verified_train` | 扩展至至少数千条，依预算调整 | 二元 head、MVE/策略监督 | 正负明确才可使用 |
| `verified_dev` | 单独数百至约千条，按簇配置 | 方法选择与损失权重调试 | 可评价，不回流训练 |
| `verified_calib` | 独立数百至更多，视风险保证需求 | post-policy 校准 | 用于校准；不训练 policy |
| `verified_blind_test` | 预先冻结、尽量更多难例 | 最终精度/证据/预算综合评价 | **仅最终评价** |

上述数量仅是项目建议，不构成统计功效保证；**每谓词簇的实际样本数和正负基率**要先统计，再决定是否可报告该组 AUROC/F1。若无法获得足够人工核验训练样本，不应继续推进“所有 56 谓词可靠二元验证”主张。可将研究收窄到 8–15 个事先选定且证据结构清晰的谓词，做到真实可信的主实验，再把其余谓词作为未精标的辅助分析。

## 2.9 关系是否可从单张图片判断：区分“细节不够”与“客观不可知”

- `coarse_insufficient_fine_sufficient`：真正符合 BARV 研究动机；
- `coarse_sufficient`：应当快速停止，属于节省预算的主要来源；
- `fine_still_insufficient`：应拒绝或保留不确定，不应硬让模型通过不断 crop“猜出”真实运动/意图；
- `annotation_disputed`：排除主二元质量指标，并作为标注稳定性研究案例。

这四组必须在**方法开发前**定义和核查，不能只挑第一组做主测试然后声称自然场景下大幅提高性能。特别是 `driving on / parked on` 等可能需要时间信息的关系，单幅图像中的真实状态不一定可判定，不能把缺乏时序信息当成需要更高分辨率才能解决的问题。

---

# 第 3 部分：网络架构与可复用代码设计

## 3.1 总体架构（G-RV 版本）

```text
High-resolution RGB I ──────────────────────┐
                                           │ only selected ROIs read high-res
              resize to 256/320            ▼
       ┌───────────────────────┐    ┌─────────────────────┐
       │ Low-res Coarse Encoder│    │ Lazy ROI Fine Encoder│◄── selected crop a_t
       └──────────┬────────────┘    └─────────┬───────────┘
                  │                           │
        F_c, coarse masks                  f(a_t)
                  │                           │
       ┌──────────▼────────────┐              │
       │ Per-relation state    │◄─────────────┤ shared evidence cache
       │ q_j, predicate e_pj  │              │
       └──────────┬────────────┘              │
                  │                           │
       ┌──────────▼───────────────────────────▼──┐
       │  Global evidence scheduler / MVE network│
       │  per-query effect + joint gain + cost   │
       └─────────┬───────────────────────────────┘
                 │
      Action: select one ROI / STOP / ABSTAIN
                 │
          update all relevant queries
                 │
    Relation probabilities + evidence traces + real cost
```

**最重要的工程约束**：如果 fine encoder 在 `t=0` 已经处理整张高分辨率图、或用测试集提前计算了所有候选 crop 特征，**那么推理时不能声称取得了相应的惰性计算节约**。离线缓存可以用于 teacher 构建和训练效率；其成本需与线上推理分开列出。

## 3.2 Coarse encoder：冻结优先

首版可以从已有 YOLOVO 的 backbone/neck P4/P3 特征抽取，但应经过独立封装：不走 NMS、不依赖检测结果、不给检测任务计 mAP。为避免权重依赖和检测先验干扰，必须同时设置一个**同规模公开冻结视觉 backbone**作为清晰对照。

- RGB $I_c$：固定缩放与 pad 规则，记录 affine mapping；
- P4 等 feature：明确 `B,C,H_f,W_f`，禁止依赖隐式 hook 导致 batch 错配；
- Mask：low-res 下 nearest-neighbor resize，禁止 bilinear 后隐式阈值改变 tiny objects；
- 实体类别名是否作为输入：为了避免类别共现捷径，建议**主实验不输入类别标签文本**；辅助实验可提供类别名，并报告变化；
- 谓词文本 $e_p$：使用现有 `_VocabHead` 的文本向量接口或独立冻结 text encoder；固定模板和名称同义词映射。

## 3.3 Mask-aware query encoder

对某个实体低分辨率掩码 $M$，一个可重复的 MaskPool 基线是：

$$v(M)=\frac{\sum_{x,y}M(x,y)F_c(x,y)}{\sum_{x,y}M(x,y)+\epsilon}.$$

再计算 subject/object/union、边缘与接触候选区域、相对几何 $g$，构建

$$h_j=\mathrm{LN}(\mathrm{MLP}([v_s,v_o,v_{s\cup o},g_j,e_{p_j}])).$$

这种 masked average 是**简单但诚实的基线**，不是论文创新。随后再比较已有 `SoftSpatialPool` 的 coverage 参数与 MaskPool 的效果。

### 3.3.1 代码陷阱：原 `SoftSpatialPool` 的 `cov_lambda`

本地 `ultralytics/models/yolovo/relate/ra/pooling.py` 定义 `forward(F_map, boxes, cov=None)`；coverage 通过 `cov_lambda * log(cov+eps)` 加入注意力权重，而 `cov_lambda` 初始化为**0**。原主 `head.py::_compute()` 只传 `boxes`。因此，**简单打开 `cov=` 不代表已有 checkpoint 真正会利用掩码**。需单独训练 `cov_lambda`、测梯度与 loss 改善，或采用可控的显式 MaskPool。不能把接口的存在当作实际功能。

## 3.4 冻结关系评分器与 fine evidence update

最小实现：

- $z_{j,0}=f_{\theta}(h_j)$：coarse logit；
- Fine Encoder $E_f$ 对所选原图 crop 编码：$u_{t}=E_f(\operatorname{Crop}(I,a_t))$；
- 对关系 $j$ 的相关性权重 $r_{j,t}=\sigma(\mathrm{MLP}([h_j,u_t,e_{p_j},g_{j,a_t}]))$；
- 更新：$h_{j,t+1}=\mathrm{LN}(h_{j,t}+r_{j,t}W_u u_t)$，不相关的关系允许 $r\approx0$；
- 新 logit $z_{j,t+1}=f_{\theta}(h_{j,t+1})$；
- 如果没有新 crop，绝不能执行 $E_f$；对已缓存 crop，不执行第二遍 $E_f$。

为了可归因，第一版 $E_f$ 冻结，只训练更新网络和评分头；后续才可选择部分解冻或 LoRA/Adapter。训练时必须拥有 ROI query 的**实际 forward counter** 和输入大小的自动断言。

## 3.5 候选证据动作生成器

由 Mask 几何、图像边界、固定尺度与谓词条件构造有限动作池；第一版不要做连续坐标 RL。动作推荐：

| 动作名 | 几何定义 | 常见适用谓词 | 注意 |
|---|---|---|---|
| `CONTACT_NEAR` | 低分辨率两个 mask 的最近边界点附近方形 ROI | holding、touching、sitting on | 最近点未必等于真实接触；远距时自动禁用或扩大窗口 |
| `SUBJECT_DETAIL` | 主体紧致 bbox 按比例扩张 ROI | wearing、looking at | 占屏幕过大的主体容易产生昂贵 crop |
| `OBJECT_DETAIL` | 客体扩张 ROI | opening、eating、drinking | tiny object 需要足够源像素；不能由上采样凭空生成纹理 |
| `PAIR_UNION` | 两 mask 的 bbox union | riding、carrying | 容易过大；是否保留由 profile 决定 |
| `CONTEXT` | union 周围固定扩张区域 | in、on、in front of | 可能是低分辨率已足够的关系，策略应该跳过 |
| `SUBJECT_PART` | 基于固定比例/廉价图像显著性预定义候选，不引入昂贵预训练部件检测器 | looking at、holding | 若引入额外模型，必须把它的计算计入成本 |
| `REVISIT_SCALE` | 相同几何区域改用更高 source-res scale | 极小物体局部 | 多尺度采样可能重叠复算 |

最重要的规则：

1. 候选坐标允许使用**给定 masks 的几何边界信息**，但候选 ROI 的语义分数不得通过事先 fine 编码得到；
2. 原图裁剪应先在原像素坐标中提取，再 resize 到 fine encoder 输入；
3. 非法裁剪面积、无效 masks、超过预算动作标记 `legal=0`；
4. 不能用关系 GT 选择“看手部”作为 policy 输入，那只允许作为训练 teacher/标注分析；
5. 有多个 candidate 指向**完全相同的 crop key**时先合并，以避免虚高动作数和重复算力。

## 3.6 多查询共享 Cache：B 主线关键机制

缓存键至少包括：

```text
(image_uid, original_xyxy, crop_resize, crop_scale,
  normalization_version, fine_encoder_weights_hash, device_dtype)
```

值为 `fine_embedding`, `latency_cost`, `produced_at_step`；对**同一图像**的多个关系可以共享。禁止不同图像缓存错误复用；禁止把 GT label 或测试评估信息放进 cache。

对两个相同输入的 Crop，命中缓存时 $\Delta C_{\rm encoder}=0$，但索引检索、特征分配、额外融合/关系评分的实际成本仍须计入。

`E_cache` 的共享是工程基础设施，**不是独立的学术创新**；研究主张必须来自利用共享性学习更好的全局动作选择策略，并证明超过 naïve caching / greedy coverage 基线。

---

# 第 4 部分：核心方法——谓词条件化边际证据价值及联合调度

## 4.1 为什么普通预测熵不够？

熵 $H(p_j)$ 只表示当前不确定性，不告诉我们“哪个 ROI 能解决这个疑问”。某张图对 `holding` 的熵很高，但放大道路背景不会改善判断；另一条关系虽然低熵却是有偏先验导致的高置信错误，查看手部可能产生很大纠错价值。

因此，模型要估计在当前可见状态 $S_t$ 下对某个动作 $a$ 的**条件边际损失减少**，而不是依靠热力图最亮位置。

## 4.2 单查询期望价值

定义受校准的概率 $p_{j,t}$、任务损失 $\ell(y_j,p_{j,t})$、执行候选 $a$ 后的 $p_{j,t+1}^{(a)}$，则

$$v_j(a\mid S_t)=\mathbb E\big[\ell(y_j,p_{j,t})-\ell(y_j,p_{j,t+1}^{(a)})\mid S_t,a\big].$$

$\mathbb E$ 是对潜在标签/视觉变化的训练分布取条件期望；在**推理**时 $y_j$ 和未执行动作后的 $p_{j,t+1}^{(a)}$ 都不可见。参数模型 $\widehat v_{\phi}(q_j,a,S_t)$ 必须仅从允许的信息推断。

**不要将观察到的 teacher 后验 loss 降低称为“真实因果效应”**：它是同一图像、同一评分器下的**配对预测损失差**，一般不足以建立物理因果性。术语用 `conditional marginal utility` / `hindsight gain` 更准确。

## 4.3 多查询边际收益：含共享与公平性

定义当前同图核验集合 $\mathcal Q$，对一个候选 ROI $a$：

$$G(a\mid S_t)=\sum_{j\in\mathcal Q_{\rm active}} w_j\,\widehat v_{\phi}(q_j,a,S_t)
-\lambda\widehat{\Delta C}(a\mid E_{\rm cache})
-\mu\Omega(a,S_t).$$

这里：

- $w_j$ 初期可统一设为 $1/m$，之后按难度或业务代价变化，避免策略只服务最容易的关系；
- $\widehat{\Delta C}$ 是**给定缓存状态的新增计算**，已经算过的同 crop encoder 不应重复计；
- $\Omega$ 用于限制重复观察/近重叠浪费，不应惩罚真正互补的两尺度区域；
- $\mathcal Q_{\rm active}$ 不包括已经最终决定的查询，除非协议允许回撤已发布答案；首版建议最终决定不可回撤，便于严谨计费。

如果 $G(a)\le 0$，策略应考虑停止；但仅凭估计值小于零无法证明最优停止，应该和固定阈值及基于 calibrated confidence 的停止强基线比较。

## 4.4 新方法具体参数化：Pair–Predicate–Action 三元交互

对第 $j$ 条查询及动作 $a$：

$$k_{j,a}=\mathrm{MLP}([h_{j,t},e_{p_j},\psi(M_{s_j},M_{o_j},a),b_t,\mathrm{seen}(a)]).$$

再计算：

$$\widehat v_{j,a}=\mathrm{MLP}_{v}(k_{j,a}),\quad
\widehat \sigma_{j,a}=\mathrm{softplus}(\mathrm{MLP}_{u}(k_{j,a})).$$

$\psi$ 包含动作与主体/客体相交比例、距 mask 边界的归一化距离、crop 面积、位置/尺度、是否覆盖接触候选带等**不需要读取 fine 像素的廉价几何特征**。

预测值可改为**风险调整价值**：

$$\widehat v^{\rm conservative}_{j,a}=\widehat v_{j,a}-\kappa\widehat\sigma_{j,a},$$

用于降低明显过估动作的频率。但此项是**可选消融**；不能预设不确定性估计一定有效。

**与现有普通 attention 的区别必须经实验成立**：若用一个简单的谓词 attention score 排序就达到相同效果，那么三元交互价值网络不构成重要新贡献。

## 4.5 数据驱动的离线 Oracle 与 teacher

在**训练图像**上，对合法动作 `a` 枚举一次实际 fine crop 编码：

$$\widetilde{\Delta}_{j,a}=
\ell(y_j,p_{j,t})-\ell(y_j,p_{j,t+1}^{(a)}).$$

- 只有 `label∈{0,1}` 的经核验关系可使用 BCE 作为监督；
- 完整 PSG 未核验 query 不能因为未标注就当 $y=0$ 来计算 `teacher gain`；
- 如果要利用大量 PSG 关系正标注，可仅用于`positive-only auxiliary`、可靠 PU 目标或表示学习，不能作为双向“正确/错误” gain 的无偏真值；
- 若 teacher 用的是同一训练集拟合过的 scorer，容易给出过拟合的收益。建议使用**冻结的 held-out/交叉拟合 scorer** 生成训练轨迹，再独立复现；
- 每当 fine encoder/关系 head 有大变化，旧轨迹收益可能失效：首版保持打分器冻结到策略训练结束，必要时版本化并重采样 teacher；
- GT 只用于离线构建 $\widetilde{\Delta}$，**线上 state 绝不能包含** label、反事实更新后的 logit、所有 crop 的高分辨率特征。

Teacher 的动作选择可按 `best single-step` 和有上限的 2-step beam 近似，不声称全局最优；输出为清晰标记的 `hindsight oracle upper bound`，绝不可将它作为无需标签的可部署模型与 BARV 公平比较。

## 4.6 价值监督的损失与可学习性

$$\mathcal L_{value}=\frac1{N_A}\sum_{(j,a)}\operatorname{Huber}(\widehat v_{j,a},\widetilde\Delta_{j,a})$$

$$\mathcal L_{rank}=\frac1{|\mathcal P|}\sum_{(a^+,a^-)}\log\big(1+\exp[-(\widehat v_{j,a^+}-\widehat v_{j,a^-})]\big),$$

其中动作正负对由同一个 Query 下 hindsight gain 的排序产生，不跨不同成本量纲直接比较；需要设置 margin、采样比例并在 `dev` 上验证。

多查询 `joint gain` 可以先**用每个关系的 gain 求和**形成，不必提前发明难训练的 RL global reward。最后加一个可学习的共享增益校正 $\delta_\omega(S_t,a)$ 作为消融：

$$\widehat G=\sum_jw_j\widehat v_{j,a}+\delta_\omega(S_t,a)-\lambda\widehat{\Delta C}.$$

如果校正项并没有改善指标，就保留可解释的求和策略，不能为了提高模型复杂度硬加模块。

## 4.7 序贯采证的训练机制

**第一阶段**使用 imitation policy：teacher 在训练图片上给出贪心/beam 一步动作，学生模仿；  
**第二阶段**允许 policy rollouts：混入学生自己到达的状态，减少纯 teacher forcing 带来的分布偏移；  
**第三阶段**如确有必要再加入 RL：但须在同等训练算力下与 imitation-only 比较，RL 不作为先验必要创新。

对 G-RV，policy 的输出是**整个图像下一次要编码的 ROI**（而非每个关系独立一套动作），更新所有共享证据相关的关系状态。

### 4.7.1 停止策略和拒绝策略应分开

- **Stop acquiring**：不再花算力，进入最终决策阶段；
- **Abstain**：最终输出不确定，不给真/假结论；
- **Accept**：输出真/假决定。

不要把 `STOP` 当作 `ABSTAIN`：预算耗尽时可以继续选择接受负判断，也可以选择拒绝；两种动作有不同成本。训练时先学习可信概率输出，再单独校准接受阈值，最后探索联合训练，降低不可归因性。

## 4.8 动态策略后的风险与校准

已有 BCEA 明确指出：将采证策略粗暴插入已有 calibrated filter，会改变输出分布，原有覆盖/风险保证一般不再成立。

因此，本方案规定：

1. 先固定视觉特征、fine encoder、query policy 和停止逻辑；
2. 在**独立 `calib` split** 上对最终 post-policy scores 进行 calibration；
3. 简单路线：temperature / isotonic + per-budget 校准（样本不足时只做 pooled calibration 并报告限制）；
4. 如果宣称 conformal 风险保证，必须使用完整策略（包括 adaptivity）的非一致性分数在独立可交换样本上重新校准，并且只在实际满足其假设的群体中陈述保证；
5. 不得把单次 risk–coverage 曲线描述成分布无关的严格统计保证。

## 4.9 算法伪代码：多查询共享的懒执行策略

```python
@torch.no_grad()
def verify_image(image, masks, relation_queries, budget, coarse_model, fine_model,
                 scorer, policy, calibration, cost_model):
    coarse = coarse_model(resize_low(image))        # once per image
    states = scorer.init_states(coarse, downsample_masks(masks), relation_queries)
    cache = {}                                     # only observations acquired online
    trace = []
    spent = cost_model.coarse_actual_cost()

    while True:
        legal = build_geometry_actions(masks, states, cache, image.shape)
        legal = [a for a in legal if spent + cost_model.increment(a, cache) <= budget]
        if not legal:
            break
        action, utility = policy.select_next(states, legal, cache, budget - spent)
        if action == "STOP" or utility <= policy.stop_margin:
            break
        crop_key = make_crop_cache_key(image, action, fine_model)
        if crop_key in cache:
            obs, c_inc = cache[crop_key], cost_model.cache_lookup_cost()
        else:
            crop = crop_original_pixels(image, action)  # no upsample from coarse
            obs = fine_model(crop)                     # actually run only now
            cache[crop_key] = obs
            c_inc = cost_model.measured_increment(crop, fine_model, states)
        states = scorer.update_all_relevant(states, obs, action, relation_queries)
        spent += c_inc
        trace.append({"action": action, "increment_cost": c_inc})

    probs = calibration.apply(scorer.final_logits(states))
    decisions = calibrated_accept_or_abstain(probs, relation_queries)
    return decisions, probs, trace, spent
```

此伪代码是**接口契约**而不是可直接运行的实现：实际 latency 测量必须避免多次 GPU synchronize 导致测量偏差；成本 `spent` 的实际值与策略估计值同时记录，并在执行动作之前由 profiler/lookup 设置合法预算边界。

---

# 第 5 部分：损失、训练流水线与模型复现

## 5.1 训练需要的模型资产

- `YOLOVOmodal` 关系头：代码存在，但目前未在输入压缩包内确认匹配 `trainOVrel.py` 的特定关系 checkpoint；
- PSG 训练图像及原始正关系；
- 通过人工核验获得的真正负例与歧义标签；
- 已核验可用的基础视觉 backbone 权重（本地权重、许可、哈希）；
- 冻结文本编码器或已有 `_VocabHead` 谓词 bank；
- 精确 profiler（torch.profiler、CUDA events、`torch.cuda.max_memory_allocated`，必要时 Nsight Systems/Compute）。

**不存在的资产不能写进训练前置条件为“已下载/已训练”。**

## 5.2 建议训练阶段（各阶段可独立验收）

### S0：PSG Dataset + 二元查询构造

- parse 官方标注与 panoptic id，构造标准化 `image, masks, relations`；
- 从训练 split 中制作 `positive`, `unknown`；等人工审阅第一批负例后再建核验训练集；
- 200 张随机样本叠加可视化，标出 segment ID 和对应关系，人工核对至少 20 张；
- 检查小掩码、重叠区域、图像宽高不同等边界情况；
- 记录 `dataset_sha256`, `split_manifest_hash`, `schema_version`。

### S1：coarse 关系评分基线

- 只实现 `(image, masks, predicate) -> logit`；
- 先训练/冻结一个合理的二元关系评分器，positive/negative 必须可信；
- 训练表示时可以利用 PSG 正例做辅助多标签任务，但未核验缺失关系不按负例强制 BCE；
- 首轮用 simple MaskPool+MLP 以及借鉴 RA 的 RelationTransformer 两种评分器；
- 必做 `no-visual`, `geometry-only`, `predicate-only`, `object-category-only` 基线，以诊断 dataset shortcuts；
- 数据管线未通过 QA 前不进入 fine encoder 开发。

### S2：fine ROI 编码与真实延迟微基准

- 实现 CPU original-crop 与 GPU crop 两条实现（根据成本决定只保留更合适路线）；
- 支持 `T=0` 不编码 fine，`T>0` 才编码；
- 固定 ROI / 多尺度 ROI 做 oracle-gap 实验，确认哪些谓词确实受益于额外细节；
- 记录 `crop latency`, `fine encode latency`, `policy latency`, `update latency`, `cache lookup latency`；
- **同预算**与完整高分辨率及静态 ROI 对比。若 ROI 反而更慢，调整计算结构，而不是转而只报 FLOPs。

### S3：冻结评分器、离线 one-step teacher

- 使用 `train` 上已核验样本枚举动作，缓存动作后的预测与 loss；
- 为减低训练后见过标签带来的 optimistic gain，可做 K-fold/out-of-fold teacher 训练，或在 train 的子集上训练 scorer、在 train 的另一子集产生收益标签；
- teacher 应记录 scorer/fine encoder/checkpoint hash；模型权重更新后重新生成或明确冻结；
- 测 MVE 的 Spearman、动作 top-1 regret、pairwise ranking accuracy；
- 在随机难度样本中评估，不只看 oracle-gap 极大的筛选子集。

### S4：单查询动作策略

- 枚举 0、1、2 步，学习采证和停止；
- 每个预算点都比较同策略但没有 MVE、熵驱动、margin 驱动、固定 ROI、attention Top-K；
- 若单步已接近多步上界，停止扩展复杂 RL；
- 学习与静态策略的差异必须统计显著，不得只挑最优预算点。

### S5：多查询共享调度（论文更强候选）

- 同一图片构造 2、4、8、16 条关系查询（数量只是实验配置）；
- 一次 coarse 共享；候选动作汇总、去重，学习 joint marginal gain；
- 对比**同一个单查询模型逐条运行**、**逐条运行+naïve 缓存**、**图像级 greedy shared-cost**、**学习的 global policy**；
- 若只有缓存带来收益而学习策略不明显胜出，报告为工程优化，不声称新的学习方法；
- 同时检查服务最少受关注谓词的公平性，防止只提升容易谓词的宏平均前分布。

### S6：冻结策略后的概率校准与 blind evaluation

- 在 `calib` split 上完成温度与决策阈值；
- 固定策略、权重、数据划分、随机种子列表与所有评估代码；
- blind test 一次性运行，保留失败案例而非删除；
- 附性能置信区间、硬件 profile、训练总预算与 inference cost。

## 5.3 损失函数：从最小版本逐步增加

最小版：

$$\mathcal L=\mathcal L_{RV}+\alpha \mathcal L_{value}+\beta\mathcal L_{rank}.$$

其中 $\mathcal L_{RV}$ 只对有明确真值的关系训练。之后可添加：

$$\mathcal L_{policy}=\mathbb E\left[L_d(d,y)+\lambda C_{\rm image}\right]$$

可选公平性项 $\mathcal L_{fair}$ 与校准项 $\mathcal L_{cal}$，但只有在具体失败分析表明有必要时再纳入。不要一开始构造六七个损失同时训练，导致哪一项真正有效无法定位。

### 5.3.1 特别禁止的信息泄漏路径

1. teacher gain 只存在于训练 supervision 表中，不作为 policy forward 的任何一维输入；
2. `all_roi_embeddings` 可离线用于 teacher，但不得在 inference 里预计算再声称动态省算力；
3. 用于 policy 训练的价值真值不可以来自 `blind_test` 的 GT；
4. 读取 GT Mask 属于 oracle-region 协议，必须在表格里显式标注，不可称作“无监督发现实体”；
5. fine update 后的 logit 只在实际执行 ROI 后可见；
6. 判定哪些 Query 已结束不能通过测试 GT 的错误与否决策；
7. 已执行的 ROI 不可让后续未执行 Query 获得隐藏的未来 fine 特征，除非明确共享缓存且实际已有计算。

---

# 第 6 部分：代码审计——现有 YOLOVOmodal 到 BARV 的真实迁移路径

> 审计方式：当前 ZIP 的**静态源码审查**，不是全训练动态复现；行号与结论适用于此份压缩包版本，代码一旦变化需要重新确认。只修改与新研究直接相关的路径，不为“做代码创新”而改动无关 YOLO 功能。

## 6.1 原有关键模块与职责

| 真实文件 | 实际行为/可证实细节 | BARV 设计动作 |
|---|---|---|
| `ultralytics/models/yolovo/relate/head.py` | `YOLOVORelate._compute(base,boxes_b)`，首先 `self._fmap(base)`，再 `SoftSpatialPool` 和 `_Sampler`、`RelationTransformer` 等；`forward` 将特征 `.detach().float()` | 保留 RA-like 原关系头，设置可被主动策略调用的 **QueryScorerAdapter**；避免直接改成多模态端到端大工程 |
| `.../ra/pooling.py` | `cov` 可选，但 `cov_lambda` 初始化为零；注意力全图 patch，不是 mask 内 ROIAlign | 增加显式 MaskPool；若保留 `cov` 必须额外训练/测试非零影响 |
| `.../ra/deformable.py` | 在已有 scene map 上取特征，属于共享低分辨率特征读取 | 可复用 coarse context，**不能等同于 fine lazy computation** |
| `.../dataset.py` | `R sub obj rel_id` 依赖框行序；`relate_collate` 默认 jitter=0 | 新建独立 `PSGQueryDataset`，不要强行把 panoptic masks 存进原检测 txt；保持旧路径兼容 |
| `.../trainer.py` | 先跑整批 backbone，再关系头逐图训练，冻结一级模型；dataloader 使用无 jitter 的 `relate_collate` | BARV trainer 不使用该旧损失路径作为 binary verified baseline；复用已有 optimizer/AMP 思路即可 |
| `.../predict.py` | 调用 `DetectionPredictor` 后处理与 NMS，用 `r.boxes.xywhn` 调 `rel.score(boxes)` | BARV 推理必须绕开 NMS 和 DetectionPredictor，直接使用给定 masks |
| `.../results.py` | `RelateResults.triplets()` 的 `out.append` 前未见 `out=[]` 初始化 | 如果要复用旧 results，先修复并加测试；对新研究新增独立输出 schema 更稳妥 |
| `.../gen.py` | 按 VG prior 和框几何规则生成伪关系 | 仅做弱标签偏差与作弊基线，不做 PSG-Active 真值 |

## 6.2 三个细节可能直接影响真实推理正确性

**代码问题 A：批量索引**  
`head.py::score(boxes,topk=None,img_idx=0)` 默认 `img_idx=0`；`predict.py` 循环遍历 `results` 却使用 `rel.score(boxes)` 没传 index。batch>1 情况下存在重复使用第 0 张图特征的风险。新 `QueryScorerAdapter` 必须显式 `image_batch_index`；测试 B=2 输入不同图像、相同 masks，确认特征不会串图。

**代码问题 B：非空三元组结果容器**  
`results.py::triplets()` 中 `out.append` 之前未初始化 `out`，非空关系调用有潜在 `NameError`。只在需要继续支持旧可视化时修复，并加回归测试。

**代码问题 C：关系分数 shape 与注释混用**  
`head.py::score` 代码返回 `[N,5]`（包含 `exist`）；`results.py` 注释用 `[P,4]`，虽然读取前四列可能有效，但 BARV 应定义自己的张量 schema，不让隐式列号决定评估。

**工程推断而非已复现实验**：以上根据源码静态审查提出。没有真实数据/权重的动态测试，因此不能宣称当前部署已经出现了这些故障。

## 6.3 不应该改的部分

- 不动已有检测 head 逻辑，不“重写 YOLO Backbone”；
- 不必把 RA/YOLOVO 关系权重复制到新项目并修改模型内部所有类名；
- 不重新开发 COCO PNG 解码、混合精度、基础 DataLoader 与 profiler；
- 不为了“看起来像原创”去重写已有 `RelationTransformer`；
- 现有 `geo_budget/final_budget/eval_budget` 是关系候选预算，不是 lazy visual feature 的真实费用，保留原语义，不应直接借名。

## 6.4 新增文件的建议结构

```text
barv_project/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data_psg.yaml
│   ├── relation_baseline.yaml
│   ├── teacher_1step.yaml
│   ├── policy_single.yaml
│   ├── policy_joint.yaml
│   └── eval_hw.yaml
├── barv/
│   ├── data/psg_adapter.py
│   ├── data/query_builder.py
│   ├── data/annotation_protocol.py
│   ├── data/split_guard.py
│   ├── models/coarse_encoder.py
│   ├── models/mask_pool.py
│   ├── models/relation_query_encoder.py
│   ├── models/lazy_fine_encoder.py
│   ├── models/evidence_update.py
│   ├── policies/action_factory.py
│   ├── policies/marginal_value.py
│   ├── policies/joint_scheduler.py
│   ├── policies/stopping.py
│   ├── runtime/crop_cache.py
│   ├── runtime/cost_model.py
│   ├── runtime/engine.py
│   ├── train/train_relation.py
│   ├── train/make_teacher.py
│   ├── train/train_value.py
│   ├── train/train_joint_policy.py
│   └── evaluation/{metrics,protocols,profiling}.py
├── scripts/
│   ├── inspect_psg_schema.py
│   ├── build_split_manifest.py
│   ├── validate_annotations.py
│   ├── profile_roi_costs.py
│   └── run_ablation_matrix.py
├── tests/
│   ├── test_panoptic_index_mapping.py
│   ├── test_positive_unknown_negative.py
│   ├── test_feature_batch_alignment.py
│   ├── test_maskpool_zero_coverage.py
│   ├── test_lazy_forward_count.py
│   ├── test_teacher_future_leakage.py
│   ├── test_cache_key_versions.py
│   ├── test_shared_roi_cost.py
│   ├── test_post_policy_calibration.py
│   ├── test_metric_group_bootstrap.py
│   └── test_split_and_duplication.py
└── reports/
    ├── audit_code.md
    ├── audit_psg.md
    ├── teacher_quality.csv
    ├── gpu_profile.json
    ├── baseline_curve.csv
    ├── paper_ablation.csv
    └── failure_cases.md
```

## 6.5 关键 API contract（伪类型定义）

```python
# Shapes are illustrative; no hidden target during inference.
class RelationQuery:
    image_uid: str
    subject_mask: Tensor    # [H,W] bool; oracle-region P1
    object_mask: Tensor     # [H,W] bool
    predicate_id: int
    predicate_text: str

class EvidenceAction:
    xyxy_original: tuple[int, int, int, int]
    scale_id: int
    candidate_type: str
    cache_key: str

class QueryState:
    coarse_feature: Tensor
    relation_logits: Tensor
    selected_actions: list[EvidenceAction]
    verified_label: None  # online inference contract: must always be None

class BARVOutput:
    image_uid: str
    decisions: list[str]  # POS/NEG/ABSTAIN
    probabilities: Tensor
    trace: list[dict]
    measured_latency_ms: float
    estimated_flops: float
    fine_encoder_calls: int
```

不应把以上视作已经写入了压缩包。这里是新增模块的验收接口设计。

---

# 第 7 部分：公平的基线体系与实验矩阵

## 7.1 强制同架构基础设置

同一条比较曲线中的方法，必须尽可能使用**同一 backbone、同一 coarse/fine encoder、同一关系评分器、同一 Mask 真值、同一图像集与相同输入分辨率**。若某方法额外引入 VLM 或 SAM，需要独立报告其参数量、计算、预训练数据不同，不应将不等规格的模型直接比较得出策略优势。

## 7.2 单查询基线（Q-RV）

| ID | Baseline | 为什么必须有 |
|---|---|---|
| Q0 | category-pair frequency + geometry only | 排除无需视觉就能正确的捷径 |
| Q1 | coarse-only | 无新增证据的成本和质量下限 |
| Q2 | full-res fixed | 全图高质量高费用参考 |
| Q3 | fixed SUBJECT / OBJECT / UNION / CONTACT | 排除固定 ROI 就够好的可能 |
| Q4 | random legal ROI | 最基础采证比较 |
| Q5 | entropy/margin guided ROI | 排除单纯置信度可实现同收益 |
| Q6 | predicate-attention Top-K ROI | 排除注意力排序足够 |
| Q7 | oracle hindsight ROI | 上界，不是部署模型；标签已知，明确标注 |
| Q8 | MVE single-step | 检验真正增量价值预测 |
| Q9 | MVE multi-step without stop | 单步与多步的增益 |
| Q10 | MVE multi-step + calibrated stop | 决策与成本的效果 |
| Q11 | MVE + calibrated abstain | 风险覆盖，不可用“拒绝全部”赢 |

**建议把 BCEA/AdaptVision 的公开核心策略**在相同模型接口下重实现或复现可比变体（在论文中如实标注“adapted baseline”），不应因为预训练任务不同就直接把它们排除。若没有可公平移植的源码/实现，至少比较同概念的强启发式策略，并明确局限。

## 7.3 多查询基线（G-RV）

| ID | Baseline | 预算结构 |
|---|---|---|
| G0 | full-res once, all predicates | 每图完整编码并共享一次 |
| G1 | Q1 coarse only for all | 每图一份 coarse |
| G2 | Q-RV independently (no cache) | 不共享，展示上界开销 |
| G3 | Q-RV independently + exact crop cache | **必须有**，隔离纯缓存收益 |
| G4 | greedy ROI coverage, label-free | 利用 crop 对多个 masks 的几何覆盖程度 |
| G5 | entropy-weighted shared greedy | 当前 uncertainty × ROI 相关度 / 成本 |
| G6 | independent per-query MVE + naïve cache | 隔离价值预测 vs 全局调度 |
| G7 | joint MVE expected summed utility / cost | 主方法、可解释的共享策略 |
| G8 | learned joint correction / global policy | 复杂版本；只有比 G7 好才保留 |
| G9 | hindsight budgeted oracle | GT 已知的非部署上界 |

**核心判据：G7/G8 必须优于 G3/G5/G6 的同预算曲线，否则“联合调度”这个创新主张站不住。**

## 7.4 预算设置和报告方式

先 profiling 再定义预算点。建议固定几种相对于全图高分辨率 inference cost 的预算目标（例如 25/50/75/100%），但如果 coarse 本身已经超过低预算线，要报告**该预算不可实现**，不能硬塞 25% 标签。

对每种方法报告：

- end-to-end latency：包括 crop、policy、encoder、cache、fusion、postprocessing；warmup 与同步方式记录；
- GPU FLOPs/MACs、CPU 时间、峰值显存、batch throughput；
- 固定预算的质量与成本曲线；同一平均成本和**同一最大成本**下的比较；
- 模型平均查询次数、fine 编码调用次数、缓存命中率；
- 多 query 情况下每图 Query 数的分层性能；
- 真实时间预算下的超时/预算违反率。

**严禁**只统计 ROI 编码而漏掉共享 backbone 的费用；严禁计算 `fine_encoder` 为所有候选执行然后只把被选中动作 FLOPs 计入。

## 7.5 质量评估与证据质量

| 问题 | 指标 | 特别说明 |
|---|---|---|
| 核验真假 | balanced accuracy、macro-F1、AUROC、AP | 在 verified positive/negative 上，注明阳性比例 |
| 按谓词泛化 | per-predicate/per-cluster recall、macro-F1 | 56 类长尾，稀有类只报告 CI 不强宣称显著 |
| 概率可靠性 | ECE、Brier、可靠性图 | post-policy 校准；pre-policy 单独报告 |
| 拒绝机制 | selective risk vs coverage / AURC | 相同覆盖率比较，不能只报 accepted accuracy |
| 边际证据价值 | Spearman、top-k action ranking、regret | 只在 teacher held-out 可计算 gain 的开发样本上 |
| 证据定位 | annotated evidence IoU/pointing、证据类别命中 | 证据可以多区域/全局；区域 IoU 不作为唯一解释指标 |
| 证据作用 | 实测 loss 改善、margin 纠正率 | attention 热图不能自证“有效证据” |
| 真实效率 | p50/p90/p95 ms、FLOPs、峰显存、调用数 | 单图与多图、冷缓存与热缓存分开 |

如果需要官方 PSG `R@K/mR@K`：该指标评估场景图整体召回，需要**明确的所有候选关系生成流程**。Q-RV 给定特定实体对/谓词二分类与 PSG 全面 SGDet **不是相同任务**；不能把 PSG leaderboard 上的 R@50 数值直接与本模型二分类 accuracy 排列。只有另建统一穷举候选及排序流水线，才在完全相同 PredCls 协议下报告关系召回补充表。

## 7.6 六个不可省的消融

1. 去掉谓词输入（检验 predicate-conditioned 是否真实）；
2. MVE 换成 entropy/margin（检验“增量收益”是否优于不确定性）；
3. 各动作类型移除、统一随机、同样计算次数（检验哪种证据类型必需）；
4. 停止/拒绝单独移除（区分策略决策与高质量 posterior）；
5. 有无跨 Query 共享 Cache、有无全局调度（区分工程缓存与方法）；
6. `mask pool` vs `box pool`、有无类别文本、geometry-only（排除掩码改造/先验作弊）。

可选额外：不同采样 budget、tiny masks、长尾谓词、不同 coarse size、共享缓存容量限制、不同模型权重和推理后校准。

## 7.7 测试集冻结与统计学

- 至少 3 个随机种子独立训练；
- 对 blind test 按 `image_id` 聚类 bootstrap 至少 1,000 次，报告置信区间与配对差异；
- 固定预算点尽量前注册（project protocol），不按 test 最大提升选择展示点；
- 如果做多指标/多种 ablation 比较，使用 Holm 调整或至少明确探索性分析；
- failure cases 同样公开：小物体、无遮挡/遮挡、误标、过度查询、不同谓词受益冲突；
- 泛化数据集中的重复图像先严格去重。

---

# 第 8 部分：硬件预算、工程性能测量与训练资源

## 8.1 推荐起始配置（不是性能预测）

| 项目 | 起点 | 理由/必须测量 |
|---|---|---|
| coarse input | 长边/方形 256 或 320 | 与 full-res baseline 的真实差距要先验证 |
| fine ROI | 原图裁剪→224 或 256 | 不扩大无真实像素的 tiny ROI 假装增加细节 |
| fine encoder | 小型共享 CNN/Transformer（冻结） | 若太重 ROI 效率可能为负 |
| max query | 0/1/2/3 | 允许停止不查询，查询数不越预算 |
| hardware | 有 RTX 5090 可优先实验（需实际确认硬件） | 记录 CUDA、PyTorch、驱动、时钟、AMP 设置 |
| benchmark | batch=1, batch=8, multi-query 2/4/8/16 | 高吞吐与低延迟报告不同含义 |
| random seed | 3 个及以上 | 有预先注册范围，避免择优 |

## 8.2 端到端计时细则

1. 预热固定重复次数，用 CUDA events 测 GPU 部分、`time.perf_counter` 测完整端到端（包含 CPU preprocess 和 IO 若报告 end-to-end）；
2. 调整频率与功耗设置并记录；跨系统不要默认硬件条件等价；
3. 通过 `torch.cuda.synchronize()` 做准确观测，但不要把用于计时的同步插入在线生产推理路径；
4. 每种配置至少数百次推理（具体次数依方差调节），报告分位数与变异系数；
5. 对 ROI pipeline 同时测 `no crop`, `1 crop`, `2 crop`, `4 crop`, `same crop cache`, `different crop`；
6. 记录 profile 采样造成的开销，单独提供不插 profile 的真实 latency 基准；
7. 对 ROI 共享缓存计算真实命中率，不能假设不同关系都会查询完全相同区域；
8. 可以试 TensorRT/torch.compile，但必须**基线与方法同等优化**，且动态 crop shape/调度可能导致编译退化或频繁 recompile。

## 8.3 训练成本现实检查

Teacher 枚举每个查询若有 6 个动作、每步 1–3 次滚动，会快速放大离线计算。建议：

- 首版训练集合先用固定 4–6 种 ROI 进行 one-step 表；
- 相同图像/crop 在离线 teacher 可以临时缓存节省构表时间，并在日志单独统计构表 GPU-hours；
- 一图多 Query 时同源 crop 去重；
- 优先冻结 fine encoder，减少频繁 teacher 重构；
- 使用 beam width 2 等保守近似，只有单步无效才继续多步；
- 不要在未确认 oracle gap 前训练 1B/7B VLM 作为评分器。

---

# 第 9 部分：阶段验收、终止条件、论文可行性

## 9.1 Milestone / Gate 决策表

| Gate | 必须完成 | 继续条件（建议） | 失败时的处理 |
|---|---|---|---|
| **M0** 数据与接口 | PSG 对齐、schema、split、人工负例 pilot；200 图可视化 | 无系统性错位/索引泄漏 | 停止训练，修数据 |
| **M1** 关系基线 | coarse、full-res、MaskPool 与 geometry-only；已核验二元测试 | full-res 或细节 ROI 在一部分关系上确有可重现质量增益 | 若几乎无可提升空间，调整任务难例定义（不能按 test 事后筛选）或止损 |
| **M2** 真正的 lazy compute | fine.forward call counter、profile、相同预算固定裁剪基线 | 部分输入下 ROI 真正低于 full-res 推理成本 | 若高频查询后反而更慢，改 ROI batching/硬件执行器；没有收益就不宣称效率 |
| **M3** MVE 可预测性 | held-out action gain、rank、regret | 明显优于简单 entropy/attention 排序 | 若没有，取消 MVE 核心创新 |
| **M4** 单查询完整策略 | fixed budget / fixed coverage 曲线 | 对最强同预算启发式有稳定提升 | 若仅较弱启发式有效，不够投稿级 |
| **M5** 多查询联合调度 | shared-cost greedy、per-query MVE+cache、joint MVE 对比 | 优于最强带 cache 共享强基线；至少 2 个 query 密度有效 | 若只有缓存收益，改为系统优化工作或止损 |
| **M6** 投稿完整性 | blind test、独立校准、CI、跨集、代码开放与许可 | 提升不是单谓词/单预算/单种子驱动 | 重新界定结论而非包装成 SOTA |

上面的“明显优于”需要在实验前给出**量化效应阈值**（例如固定预算的 macro-F1 绝对值改善不低于一定百分点、或同风险下成本减少多少），并通过 pilot 估算不确定区间后固定。阈值是项目内部 go/no-go 判据，不能将其写成对最终成绩的承诺。

## 9.2 高水平论文最可能遭遇的审稿意见与回答要求

**Q1：“AdaptVision/BCEA 不就是这个思路吗？”**  
需要展示和这两者**方法及实验级**的对比，尤其是谓词动作增量收益建模、同图多关系共享，证明不仅是把 VQA 技术套到 SGG。

**Q2：“你给定 GT mask，这不是不现实吗？”**  
必须明确任务是 oracle-region 的 relation verification，不可假装做端到端开放世界发现。补充 predicted-mask/noisy-mask 实验并计入区域生成成本。

**Q3：“PSG 没标签的样本被当负例？”**  
展示 verified negative 生成说明，unknown mask，完整的数据审核报告；引用 ECCV 2024 针对 PSG 负标签的研究。

**Q4：“所谓主动性只是已经算完的全图 feature 里选 token？”**  
用 fine.forward counter=0（无查询）、源原图 crop 以及真实 profiler 证明 ROI 真正按需计算。对所有方法同口径计费。

**Q5：“模型看更多内容当然会提升。”**  
同预算、同视觉 encoder、同 crop 数目（控制实验），证明是策略选择而不是计算量更多。

**Q6：“所谓证据是否真正有效？”**  
使用独立人工证据标签、negative 对照和实际预测损失改善；注意遮挡/编辑会造成分布偏移，不能仅用 masked image score drop 建立因果解释。

**Q7：“多查询共享本来就是缓存？”**  
必须和 independent+exact-cache、shared greedy、entropy shared、geometry shared、full-res amortization 对照；若无超越，则不声称新学习方法。

**Q8：“动态策略之后置信度是否还可靠？”**  
固定策略后独立重校准；仅在满足严格条件时使用分布无关保证字样。

## 9.3 论文贡献写法（仅在取得对应证据后使用）

1. 提出关系**谓词条件化**的边际证据效用估计方法，预测追加细节对候选关系的期望纠错/风险降低价值；
2. 针对同一图像的关系集合，提出带真实新增计算代价的联合证据调度机制，支持证据跨关系复用与多步停止；
3. 基于 PSG 构建严格的 verified relation query / evidence / ambiguity 评价协议，避免标签缺失与 GT 泄漏，并在相同模型与成本上验证。

未通过对应 Gate 的贡献不得写进摘要。

## 9.4 适合投稿的场景与定位

- CVPR/ICCV/ECCV 风格偏好：提出独立新问题或通用方法 + 清楚的实验机制 + 强对照 + 显著的真实计算收益；
- 期刊进一步需要：系统性误差分析、跨数据集与不同计算设备的一般性证据、对失败场景的明确边界；
- **不能以换一个旧关系头为支点构建论文**；需要把新方法与简单贪心和已有预算式 VLM 策略严格区分。

---

# 第 10 部分：可执行的首轮工作包（非代码生成承诺）

建议从下面的任务顺序推进；每个工作包要求保存可检查输出：

**WP0：文献最近邻（开工前）**  
输出 `literature_nearest_neighbors.csv`，最低包括 AdaptVision、BCEA、RelateAnything、FleVRS、PSG 2022、PSG 2024 的问题设定、输入、动作空间、校准方式、真值与成本口径。

**WP1：数据 schema 与许可**  
输出 `psg_manifest.json`、SHA256、segment id 一致性审计、20–200 个样本可视化、训练/开发/测试无重复检验。

**WP2：baseline reproduction**  
输出 `relation_base_metrics.csv` 与 `no_visual_shortcut.csv`，在已核验 positive/negative 上有合理评价。

**WP3：oracle gap & ROI microbenchmark**  
输出 `coarse_full_roi_gap.csv`、`roi_latency.json`，若这两份无法体现“信息增量”和“成本可行”，暂不继续。

**WP4：teacher + MVE**  
输出 `teacher_actions_train.parquet`、`teacher_holdout_eval.csv`、`action_ranking_metrics.json`（不含盲测数据）；冻结教师后方可学习策略。

**WP5：single / joint policy**  
输出 Q-RV、G-RV Pareto 曲线和同预算的 ablation；显式跟踪 Query 数、缓存、ROI 真实编码调用。

**WP6：PSG-Active 标注与盲测**  
输出标注指南、QC 结果、分布统计、盲测分数、分组 bootstrap CI；遵守数据许可。

**WP7：写论文**  
优先完成 Problem、Dataset protocol、Method、Fair Budget、Ablations，再考虑是否加入开放词汇或 RGB-X 扩展。

---

# 第 11 部分：审核结论与 v1.0 → v2.0 变更日志

## 11.1 形式上的审查

- **已满足**：研究不以检测任务为目标；与给定 masks 的 relation verification 一致；已识别 PSG 不完整关系标签；已有按需 fine 编码的可操作结构；明确没有已有训练结果。
- **本次修正**：补入 BCEA 最近邻论文；剔除“答/拒/采证”组合首创的错误暗示；明确 G-RV 主线与 Q-RV 基础线；增加多关系共享和缓存强基线；明确官方竞赛 JSON 版本差异；追加训练后校准和 oracle leak 测试；确认已有 MaskPool 权重问题。
- **尚未验证**：PSG 数据是否已经下载/可用；PSG 内部索引实际字段；能否获得经核验负例的足够样本；真正 baseline performance；ROI latency 是否有收益；joint utility 是否优于 simple greedy；跨域性能。

## 11.2 三个最后必须回答的问题

1. **任务成立性**：在预注册的自然/困难查询分布上，低分辨率与局部细节之间是否存在稳定且有实际意义的预测差异？
2. **方法必要性**：预测条件边际价值、联合调度，是否严格优于已知正确的简易策略，而且提升来自选择机制而非缓存/增算力？
3. **系统真实性**：考虑图像共享 backbone、ROI crop、动态调度后，真实端到端 GPU 时间和显存是否值得？

只有这三个问题有数据支持，才建议把 BARV v2 作为主论文推进。

**最终建议**：先做 **PSG + verified negatives + frozen relation scorer + one-step lazy ROI oracle gap +真实 hardware profiler**。短期不引入新的目标检测、巨型 VLM、自由形式强化学习或 RGB-X。首篇如要突出原创方法，优先把**多查询共享的、谓词条件化边际证据调度**作为“待证”的核心贡献，而不是“视觉裁剪 + 不确定性停止”的简单组合。

---

# 附录 A：核验过的重要资源与论文入口

1. **OpenPSG 官方仓库与版本说明**：<https://github.com/Jingkang50/OpenPSG>（论文完整版和竞赛版分开，竞赛文件名与 GT 状态见 README）；统计：<https://psgdataset.org/data_stats.html>。
2. **Panoptic Scene Graph Generation**（ECCV 2022）：OpenPSG 项目 README 的 citation 与官网。
3. **A Fair Ranking and New Model for Panoptic Scene Graph Generation**（ECCV 2024）：<https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07817.pdf>（明确指出无法可靠区分 PSG 没标注的关系和真正负例）。
4. **FleVRS: Towards Flexible Visual Relationship Segmentation**（NeurIPS 2024）：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2e065133af98888ab11a549abed2cc3-Abstract-Conference.html>。
5. **AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition**（CVPR 2026）：<https://openaccess.thecvf.com/content/CVPR2026/html/Lin_AdaptVision_Efficient_Vision-Language_Models_via_Adaptive_Visual_Acquisition_CVPR_2026_paper.html>；arXiv <https://arxiv.org/abs/2512.03794>。
6. **Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Model**（arXiv 2606.16667，2026-06/07）：<https://arxiv.org/abs/2606.16667>；特别参考其选择后重校准与空间关系实验。
7. **RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs**（arXiv 2609.12552，2026-09）：<https://arxiv.org/abs/2609.12552>；代码 <https://github.com/Maelic/RelateAnything>。
8. **Visual Genome**：<https://visualgenome.org/>；**GQA**：<https://cs.stanford.edu/people/dorarad/gqa/>。
9. **CAGE-SGG**（arXiv 2604.22274）：<https://arxiv.org/abs/2604.22274>，**作者已撤稿且指出方法和评估存在影响结论的缺陷**；仅记录为检索覆盖项，不把其性能、结论当作有效证据。
10. **原代码本地审查**：`YOLOVOmodal_3871538535.zip` 内 `ultralytics/models/yolovo/relate/` 的 `head.py`, `ra/pooling.py`, `ra/deformable.py`, `dataset.py`, `trainer.py`, `predict.py`, `results.py`, `gen.py`。

## 附录 B：知识产权与开源发布检查

- `YOLOVOmodal` 使用 Ultralytics 相关代码，发布或商业部署应核对实际项目许可（例如 AGPL 要求）及修改/再分发义务；
- 本地移植的 `ra/` 注释中称来源代码按 MIT 移植，但上游仓库的当前 LICENSE/CITATION 描述可能不同。**源码头注释并不自动替代上游许可审查**，应查看所用 commit 和每个文件的真实许可证；
- PSG 官方仓库 MIT 不代表 COCO 图像/关系标注就必然与代码共享完全相同的授权条件，必须核对数据资产实际条款；
- 若将人工 PSG-Active 标注公开，区分**自建的 query/证据标注**和不能擅自重新分发的原始图像；
- 发布论文需引用相关上游模型、标注集与训练资源，记录预训练数据和模型来源。

## 附录 C：审查流程的真实性声明

本版是对两份 Markdown 内容、压缩包内实际源码与公开论文/项目页面的**文献和静态代码审查**；没有声称在该环境里下载 PSG 全量图像、获得真实 GPU 测试性能、运行训练或达到任何论文指标。文中所有 `M0–M6` 和结果上限是**实验计划**，不是已经完成的科研发现。


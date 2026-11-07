# 调研：「两阶段蒸馏（第一阶段 LLM/大模型 → 第二阶段蒸馏到小模型）」对本项目是否有用

- 时间：2026-11-07 15:10（只读调研，未改仓库任何文件，未起 GPU 进程）
- 范围：`/root/rivermind-data/YOLOVOmodal_3871538535`（GLP 关系头 + RA-4M）、官方仓库 `/root/rivermind-data/RelateAnything`、论文 `2609.12552v1.md`
- 事实与推测分开：凡标「估算」「未验证」「回忆」的均非实测

## 1. 结论

1. **论文的方法本身就是两阶段蒸馏，而且有三条不同的「大→小」链路**：(a) 26B VLM 标注语料 → 53M 关系模型；(b) 538M 的 dino.txt 文本塔 → 9M 文本 student；(c) 语料共标注统计 → 损失里的全部语义常数（同义词权重、P̂(q|p) 折扣、对称概率、反义扩展）。我们训练在 RA-4M 上，**只继承了 (a)**；(b)(c)(d) 全部没有：谓词仍用 MobileCLIP 编码、负例折扣仍按失真余弦推、正样本仍是单字符串（证据见 §2）。
2. **对两处实测差距的对应关系**：
   - 第二位差距（文本空间塌缩，反义对 cos 0.917 vs 官方 0.24）**不需要新跑任何大模型**：官方随附的 `text_student.pt` 就是蒸馏好的「第一阶段产物」，我们已验证它对已知谓词的输出与官方 W 的 cos=1.0；换 W 是零结构改动（text_dim 同为 512）。LLM 在文本侧的增量价值只在语料统计覆盖不到的地方：4,505 个只出现一次的谓词、反义对枚举（官方只有 4 个种子对，我们只有 8 个手写对）、对称谓词（我们只有 15 个手写）。
   - 第一位差距（冻结的 DIOR 遥感 20 类 YOLO11n 颈部特征不带物体/接触语义）**蒸馏解决不了**：蒸馏只能让学生逼近教师的输出，不能往学生的输入特征里注入它没有的信息。论文自己的消融说明关系 logit 的 87–93% 方差来自像素读出的 pair 上下文（Tab. 10），而我们的像素通道几乎是空的（动作类谓词 1.3%）。用官方模型做视觉 teacher 技术上可行（472k 图一次前向约 5–10 GPU 小时、约 12 GB 磁盘），但天花板要先用一个 <1 GPU 小时的线性探针实验测出来再决定。
   - VLM 直接对 472k 图逐对打软标签：单张 4080 上是数百 GPU 小时量级，不现实，而且是在重复 RA-4M 已经做过的事（RA-4M 就是 VLM 标签）。
3. **推荐顺序**（成本从低到高，§6 有表）：B 换官方 student 文本空间（1.7 h GPU）→ 线性探针测视觉天花板（<1 h GPU）→ C 共标注统计重建本体 + 同义词组正样本（1 天 CPU + 1.7 h GPU）→ LLM 补长尾本体（下载 4–14 h 后台 + 2–4 h GPU）→ 视觉 KD 仅在探针显示特征够用时做；否则直接走 A（可训练颈部副本）或换 DINOv3（权重已在 `model.pth` 里，不依赖被 gated 的 HF 下载）。
4. 与「换 DINOv3 骨干」的关系：文本侧三项（B/C/LLM 本体）与换骨干**互补**，换了骨干照样需要；视觉 KD 是换骨干的**低天花板替代品**，不是互补。

## 2. 论文里已有的「大模型 → 小模型」环节

| 环节 | 论文/代码证据 | 我们现状 | 缺口 |
|---|---|---|---|
| (a) VLM 标注语料再训小模型 | `2609.12552v1.md` L101：474,413 图 / 4.28M 关系 / 10,102 谓词，**104 GPU 小时**生成；L469（C.1）：Gemma 26B MoE（4B 激活）+ vLLM，三遍提议（语义/补全/空间）+ 确定性几何门，拒绝 11.3%（L107）；Tab. 19（L519）对比 GPT4SGG / MegaSG（Gemini 1.5 Flash）等无校验语料 | 训练集就是 RA-4M（`RA-4M/export_stats.json`：472,344 图、4,282,531 关系） | 已继承。论文 §10（L371）承认语料没有人工审核的精度估计 |
| (b) 文本塔蒸馏 teacher→student | L91：dino.txt 2048-d teacher 上 above/below cos 0.95、left/right 0.99；蒸馏 512-d student 后 syn/inv AUC 0.85→0.99、反义均值 0.92→0.09、同义 0.71、hubness 8.1→1.3。B.2（L445-453）六项目标：关系蒸馏、绝对锚、邻域保持 KL、反义 hinge、同义 hinge、软 hinge；teacher 目标先去掉前 3 个主成分。代码：`RelateAnything/training/distill/distill_text.py` L1-30、`relsgg/text/student.py` L3-4（6 层双向 transformer，约 9M 参数） | `wiring.py` L219-263 `encode_predicate_bank`：MobileCLIP-S0 模板系编码；`gap_analysis.json` text_space_ours：off-diag 均值 0.858、中位谓词 1,441 个 cos≥0.9 邻居、反义 0.917、同义 0.903；官方 student：0.142 / 1 / 0.24 / 0.704 | **未继承**。产物已在手：`assets/relsgg-vits16plus/text_student.pt`（45 MB） |
| (c) P̂(q\|p) 负例折扣从共标注统计拟合 | L87：706k 多标注框对上拟合，特征 = 共标注计数 + 文本余弦 + 频率，Elkan–Noto 校正，Eq. (3) 把 log(1−P̂) 加到负例 logit；两条豁免（标注谓词不降权、方向反义全权重）。代码：`training/estimate_false_negatives.py` 文档串：megasg train 706,454 框对（20.3%）多标注，35.2% 的关系在其上，34,270 个不同共标注谓词对；**去掉假阴性保护代价是稀有类召回的 33–68%**；`training/build_soft_supervision.py` L1-45：产物 `soft_supervision.npz`（pos_w / neg_lw / sym / inv_w / w_cooc）；`relsgg/training/losses.py` L31-67 `PredicateOntology.from_soft_supervision` | `ra/ontology.py` L54-73：`neg_lw` 由 MobileCLIP 余弦线性映射（close_thr 0.80、floor 0.25）；在均值 0.858 的空间里，cos 0.858 → neg_lw 0.78，cos 0.90 → 0.625，等于对几乎所有列做了一次差不多的降温，不携带语义；`inverse_mask` 只有 8 个手写对（L21-25） | **未继承**。数据在手（`RA-4M/annotations/ra4m_train_coco.json` 941 MB），纯 CPU 可算 |
| (d) 同义词组正样本 | L87：正样本 = 标注谓词的同义词组加权均值，权重 = 对 1,177 个词法同义对做的等渗回归 P(同义\|cos)；B.2.2（L455）：代价是精确匹配指标把同义答案算错（official `wearing` 71% 答 `wearing clothing`，README L70-72） | `head.py` L736-757：NCE 正样本是单字符串 `hot`；`pos_mask` 建了但 NCE 没用 | **未继承** |
| (e) 对称谓词概率 | `build_soft_supervision.py` L24-27：按反向标注率做 2 分量 beta-binomial EM（`beside` 2.5% vs 方向类 0.15%），swap hinge 权重 = 1−sym | `ontology.py` L34-37：15 个手写对称谓词，注释写明余弦推断会把 96.97% 的谓词误判为对称 | **未继承** |

用户说的「两阶段蒸馏、第一阶段用 LLM」在论文里对应 (a)（语料）+ (b)（文本塔）；(c)(d)(e) 不是 LLM，是语料统计，但它们才是让 (b) 的文本空间在损失里起作用的那一半。

## 3. 对我们的可行方案对比

### 3.1 文本侧

| 方案 | 第一阶段产出 | 第二阶段怎么接 | 成本 | 预期收益（估算） | 天花板/风险 |
|---|---|---|---|---|---|
| **B. W 换官方 student 空间** | 10,120 条直接取官方 bank，250 条用 `text_student.pt` 编码（`tools/run_ra_official_predcls.py` 已有这段逻辑） | `wiring.py` `encode_predicate_bank` 改为读 student；`relate_info.text_dim` 仍为 512（`logs/full_e4_resume.log` L3：`V=10370, text_dim=512`），头结构不动；**必须同步重建 `ontology.py` 的表**：student 空间 off-diag 0.14，现有 close_thr 0.80 下几乎没有列会被折扣，`inverse_mask` 的 8 对在 0.09 的反义余弦下也不再关键 | 0 下载；代码半天；4 轮 50k 代理 1.7 h GPU | 对准第二位差距：`holding` 不再需要在 1,441 个同向列里被挑出来；方向类不再靠 `q_spa` 兜底。rare 桶 mR@50（现 0.34–0.67 vs 官方 11.96）最可能先动 | 不解决像素通道为空的问题；只给框时的 action 类（1.3%）不会因此大涨。W_obj 仍用 MobileCLIP（物体名词），不冲突 |
| **C. 共标注统计重建本体 + 同义词组正样本** | 从 `ra4m_train_coco.json` 统计同一 (图, sub, obj) 的多谓词共现 → P̂(q\|p)（Elkan–Noto）；词法同义对（去脚手架后同一字符串）→ 等渗 P(同义\|cos)（cos 用 B 的空间）；反向标注率 → sym；4 个反义种子 × 同义核 → inv_w | 替换 `ontology.py` 四张表；`head.py` L736-757 NCE 的正样本从 `hot` 改为 `pos_w` 加权均值（对应 `relsgg/training/losses.py` `BatchLocalInfoNCE`），`lw` 直接用 log(1−P̂)；L759-780 swap hinge 权重改 (1−sym) | 1 天 CPU（941 MB JSON，88 核足够）+ 1.7 h GPU；官方脚本依赖其 pack 格式（`rels.npy`/`meta.json`）与 `pred_context_mc5.npz`，需移植不是照跑 | 官方实测：去掉假阴性保护损失 33–68% 稀有类召回（`estimate_false_negatives.py` 文档串，未在我们这边复现）。我们现在等于没有有效保护 | 同义词组正样本会让精确匹配 R@50 下降（论文 B.2.2），需要同时上一个同义感知指标，否则按 README 过关标准会误判 |
| **C-LLM. LLM 判本体补长尾** | 对每个谓词取其 top-50 文本邻居（B 的空间）让 LLM 逐条判：同义 / p 蕴含 q 的概率 / 反义 / 对称。10,370 × 50 = **518,500 条有向判断**，按「一个谓词一条 prompt、50 个候选一起判」= 10,370 条 prompt | 作为先验与 C 的统计合并（统计计数为 0 的对用 LLM 先验；4,505 个 hapax 谓词只能靠它）；反义对列表直接进 `inverse_mask`；对称表进 `sym` | token：每条 prompt 约 450 in + 400 out → **约 4.5M 输入 + 4M 输出 token**（估算）。7B bf16 用 HF transformers 批量生成在 4080 上约 300–800 tok/s（估算，未测）→ **1.5–4 h 独占 GPU**；vLLM 能快 3–5 倍但未安装。下载 15 GB 按实测 0.3–1.1 MB/s 需 **4–14 h**；AWQ 4-bit（约 5 GB）要装 autoawq。CPU 跑 7B 不现实（天级） | 只影响长尾与反义/对称表：val 436 个谓词里 411 个是 rare 桶，我们 rare mR@50 ≈ 0.3–0.7，这是最有空间的地方；但收益大小没有文献数字可引（未验证） | LLM 对「蕴含概率」的数值不可靠，只能当排序/先验用；需抽 200 对人工核对。GPU 窗口要和训练队列错开 |
| LLM 向量模型（如 Qwen3-Embedding-0.6B）替换 MobileCLIP | 重新编码 10,370 条 | 输出 1024-d，与 text_dim=512 不符，要改 `vocab_head.proj`/`relate_info`，属于结构改动 | 下载 1.2 GB（20–60 min），CPU 10 min 可测反义对余弦 | 未验证它是否分得开 above/below；论文 L91 的论点是对比式文本编码器普遍分不开，student 的反义 hinge 才是关键，而 student 已有 | 优先级低于 B；只值得做一次 10 分钟的余弦诊断 |

### 3.2 视觉侧

| 方案 | 第一阶段产出 | 第二阶段怎么接 | 成本 | 天花板 |
|---|---|---|---|---|
| **V. 官方 RA 模型做视觉 teacher，蒸馏 pair 表示 / logits** | 对 472k 训练图用 GT 框跑官方模型（`relsgg/model/relsgg.py` L359-362 有 `q`、`q_spa`、`logits [B,K,V]`、`pair_logits`），存：每个 GT 有序对的 `q`（512 fp16 = 1 KB）、`q_spa`（1 KB）、top-64 logits（idx int16 + val fp16 = 256 B）；另存每图 32 个教师采样负例对的 top-64 logits。GT 有序对约 3.5M（4.28M 关系，35.2% 落在 706k 多标注对上 → 约 2.78M 单标注 + 0.71M 多标注对）→ **约 8 GB + 3.9 GB ≈ 12 GB**（数据盘余 73 GB）。全部 128 对/图存 `q` 要 62 GB，不可行 | 接在 `head.py` `_losses`（L689）里加两项，模板照 `cand_ce`（L793-805）：`kd_logit` = 我们 `logits_raw[has_gt]`（L734 `cos_all` 或散列前 logits）与教师 top-64 列做温度 KL；`kd_feat` = 我们 `q`（L573）与教师 `q` 的 1−cos（**只有做了 B，两边 512-d 才在同一文本空间，这一项才有意义**）。权重走 `trainer.py` L376-377 `LAMBDA_RA` + L438-445 的环境变量注入方式（同 `YOLOVO_CAND_CE`）。教师目标按图 id 由 `dataset.py` 的 `build_relate_dataset`/`relate_collate` 带进 batch，槽位匹配照 L712-720 的 `gt_hot` 写法 | 教师前向：`scratch_audit/ra_full.log` 1,024 图 82 s（400 对/图，当时 GPU 与训练共用）= 12.5 img/s → 472k 图 **10.5 h**；空卡 + 128 对/图估 5–7 h（未验证）。训练侧每轮多读 12 GB，吞吐影响小。代码 1–2 天 | **蒸馏不能注入冻结 P4 里没有的信息。** 论文 Tab. 10（L313）：打乱像素 micro Acc −57.5%/−68.3%，pair 上下文占语义 logit 方差 87–93%；Appendix F（L739-745）：DINOv3 接触区特征的物体类别 1-NN 0.761（随机 0.068）。我们的像素通道实测是空的（README L51-66：action 1.3%，给 GT 类别后 44.4%），`WORKLOG.md` L513 写明一级权重是 DIOR 遥感 20 类、冻结后用在 COCO/Objects365 自然图上。KD 在这种学生上能转移的主要是教师的**类别先验**（PredCls 给类别时我们本来就有），不是像素证据 |
| **便宜的天花板实验（先做）** | 取 10k 训练图（约 56k 框、90k 关系）：(i) 我们的 `v_obj`/`v_union`/`v_contact`（`head.py` L438、L539-540，冻结 P4 池化，各 768-d）；(ii) 官方模型同一批 GT 框的池化特征 | sklearn 多类逻辑回归（CPU）：预测物体类别（497 类，`RA-4M/categories.json`）与谓词（按频次前 100）的 top-1 / macro-F1；再加一个 2 层 MLP 作上界 | GPU 20–30 min（YOLO11n 10k 图几分钟，官方 12 img/s 约 15 min）+ CPU 30 min | 判据：若物体类别探针我们比教师低 30 个点以上（参考值：论文 DINOv3 物体 1-NN 0.761，谓词线性 macro-F1 0.445–0.475，PSG），则瓶颈在特征，KD 天花板低，直接走 A / 换骨干；若探针差距小而端到端差距大，才是训练配方问题，KD 值得做 |
| **VLM 直接给软标签**（Qwen2.5-VL-7B 等对 472k 图逐对打分） | 每图一条 300-token 回答：4080 + HF transformers 约 2–4 s/图（估算）→ **260–520 GPU 小时**（11–22 天独占卡）；逐对打分（约 9 GT + 20 负例/图 ≈ 14M 条 prompt）更贵；50k 子集也要 28–56 h | 同 V 的 `kd_logit` 接法 | 论文自己的 104 GPU 小时是 26B-A4B MoE + vLLM 三遍（GPU 型号在正文未写，未验证） | **不现实**，且与 RA-4M 重复：RA-4M 本身就是 VLM 标签，再标一遍的增量只是软置信度 |

## 4. 离线可行性（实测，2026-11-07 15:00）

| 项 | 结果 |
|---|---|
| 显卡 | `nvidia-smi`：RTX 4080 SUPER，32,760 MiB，已用 15,573 MiB，利用率 100%；训练队列 PGID 1074560 存活（`ps` etime 20:47），未触碰 |
| CPU / 内存 / 磁盘 | 88 核，440 GB 内存；`/root/rivermind-data` 余 73 GB，`/` 余 4.8 GB（模型别下到 `/`） |
| 本机 LLM / VLM / 向量模型权重 | `find`（>200 MB 的 safetensors/gguf/bin/pt/pth/onnx）只有 `assets/relsgg-vits16plus/model.pth`（257 MB，含微调后的 DINOv3 ViT-S/16+）、两份 `mobileclip_s0.pt`；`~/.cache/huggingface` 不存在。**没有任何 LLM/VLM/文本向量模型** |
| 推理框架 | `ollama`/`vllm`/`llama-server` 均未安装；pip：transformers 4.49.0、torch 2.5.1+cu124、timm 1.0.30、onnxruntime 1.19.2；无 sentence-transformers / open_clip / bitsandbytes / accelerate / llama_cpp。transformers 4.49 支持 Qwen2.5 与 Qwen2.5-VL；Qwen3 系列需 ≥4.51（`ra_overlay` 里有更新的 transformers，`reports/baseline_cmp/README.md` L98 写 ≥4.56，可借用） |
| 网络 | `curl -sI`：hf-mirror.com 对所有 URL 返回 308，跟随后落到 huggingface.co 并 200（Qwen3-Embedding-0.6B、Qwen2.5-7B-Instruct、Qwen2.5-VL-7B-Instruct、Qwen3-VL-8B 的 config.json 都能取到）；**huggingface.co 与 github.com 本次测试直接 200**（「被墙」在 15:00 这一刻没有复现，可能随时间变化）；`facebook/dinov3-vits16plus` 401（gated，需 token） |
| 下载速度 | 1 MB range 经镜像 0.30 MB/s；20 MB range 直连 hf.co 1.15 MB/s；20 MB 经镜像 60 s 内未完成。按 0.3–1.1 MB/s：7B bf16（15 GB）4–14 h，Qwen3-Embedding-0.6B（1.2 GB）20–60 min，Qwen2.5-VL-7B（16 GB）4–15 h |
| 官方代码可复用 | `RelateAnything/training/build_soft_supervision.py`、`estimate_false_negatives.py`、`export_syn_kernel.py`、`distill/`（文本蒸馏全套）、`datagen/sgg_vllm_generate.py` + `datagen/prompts`（VLM 标注提示词）、`relsgg/training/losses.py`（组正样本 InfoNCE 参考实现）。依赖它们的 pack 格式与 `runs/packed/...` 产物，产物未随发布物提供，需用 RA-4M 自己重算 |

## 5. 文献（WebSearch 核实 + 论文引文；标注来源）

| 工作 | 第一阶段做了什么 | 第二阶段 | 报告收益 | 来源 |
|---|---|---|---|---|
| RelateAnything / RA-4M（本项目基线） | 26B MoE VLM 三遍提议 + 几何门标注 474k 图；dino.txt → 9M 文本 student；共标注统计 → 损失常数 | 53M 关系头，InfoNCE 组正样本 + P̂(q\|p) 折扣 | 文本空间 syn/inv AUC 0.85→0.99；去假阴性保护 −33~68% 稀有类召回（代码文档串） | 论文 L87/L91/L101，官方仓库（已核实） |
| LLM4SGG（Kim et al., CVPR 2024） | LLM（CoT + few-shot）把 caption 解析成三元组并对齐到 VG/GQA 类别，缓解谓词粗化与低密度 | 弱监督 SGG（VS³ 等）用对齐后的三元组训练 | COCO caption 上 VS³ R@50 6.60→8.91（+2.31），mR@50 2.88→7.11（+4.23）；GQA R@50 5.90→8.88 | WebSearch 核实（CVPR 开放获取 PDF 表格） |
| OvSGTR（Chen et al., ECCV 2024） | 用 Grounding DINO 从 caption 伪标注关系，预训练关系感知模型 | 微调时用冻结的预训练模型做 teacher，L1 对齐边特征（visual-concept retention distillation），防遗忘 novel 关系 | OvR-SGG VG：R@50 20.46（Swin-T）；蒸馏项是防遗忘而非提分主力 | ICCV 2025 RelGen 补充材料对其 L_distill 的描述 + NeurIPS 2025 ACC 表 1（WebSearch 核实） |
| ACC（NeurIPS 2025, Interaction-Centric Knowledge Infusion and Transfer） | 双向交互 prompt 生成伪监督（知识注入） | 交互一致 KD（点对点概念保持 + 相对交互保持）+ 交互引导 query 选择 | OvD+R-SGG 对 OvSGTR Swin-T +4.90% R@100；KD 项消融 +2.83% R@100 | WebSearch 核实（NeurIPS 论文 PDF） |
| RelGen（Min et al., ICCV 2025, Vision-Language Interactive Relation Mining） | 生成式关系模块 + 用自身模型给 base 集补伪标签（space-wise）+ 语义去偏 | 与 OvSGTR 叠加 | OvSGTR 20.5/23.9 → 21.1/25.0 R@50/100；novel 13.5→14.2 | WebSearch 核实 |
| GPT4SGG（2023）/ MegaSG（Gemini 1.5 Flash，1M 图，2025） | 多模态模型直接对图提示生成场景图，不校验 | 训小模型 | 论文 Tab. 19 只列产出方式；RA-4M 对 MegaSG 同图同框：密度 1.7×、词表 107× | 论文 L481、L519（已核实） |
| RECODE（Zero-shot VRD via composite visual cues from LLMs, NeurIPS 2023）、SDSGG（role-playing LLMs, NeurIPS 2024）、VLPrompt | LLM 生成每个谓词的视觉线索/场景描述，供 CLIP 类模型打分 | 零样本或开放词表 SGG | **回忆、未核实**：两者报告的是零样本/开放词表 mR 提升，数字未查 | 论文参考文献列表 L385-394 中出现（已核实存在，收益数字未核实） |

读法：文献里「LLM 第一阶段」的收益集中在**弱监督/伪标签**场景（标注稀缺时 LLM 把 caption 变成监督），收益 +2~4 R@50；而我们已经有 4.28M 条 VLM 标注，处境不同。视觉 KD 在这些工作里是**防遗忘**或 +2~3 点的辅助项，没有哪篇靠 KD 把一个弱骨干拉到强骨干水平。

## 6. 建议顺序（成本/收益排序）

| 序 | 方案 | 第一阶段产出 | 第二阶段接法 | 工时 / GPU | 过关标准（README「实验流程」口径：4 轮 50k 代理 vs `p50k_base_e4`） |
|---|---|---|---|---|---|
| 1 | **B 换官方 student 文本空间** | `W [10370,512]`（10,120 取 bank + 250 student 编码） | `wiring.py` 换编码源；`ontology.py` 阈值按新空间重定（或先只留显式表） | 0.5 天 + 1.7 h GPU | c2：只给框 pc_R@50 +0.5 或 pc_mR@50 +1.0；c3：pair_acc 不掉 >0.5（`progress.md` L46）；另看 README 差距表 action/other 与 rare 桶 |
| 2 | **线性探针测视觉天花板** | 10k 图两套池化特征 + LR 结果表 | 不进训练，只做决策 | 0.5 天 + <1 h GPU | 物体类别 top-1：我们 vs 教师差 >30 点 → 跳过 V，直接 A/换骨干 |
| 3 | **C 共标注本体 + 组正样本** | `soft_supervision.npz`（pos_w / neg_lw / sym / inv_w） | 替换 `ontology.py` 表；NCE 正样本改加权均值；swap 权重 (1−sym) | 1 天 CPU + 1.7 h GPU | 同上；同时报同义感知 R@50（官方 A3 口径）避免精确匹配误判 |
| 4 | **C-LLM 补长尾本体** | 10,370 条 prompt 的判定表（同义/蕴含/反义/对称） | 与 C 的统计合并为先验 | 下载 4–14 h（后台 `setsid nohup`）+ 2–4 h GPU + 0.5 天 | rare 桶 mR@50 相对 C 再 +1.0；抽 200 对人工核对一致率 ≥85% |
| 5 | **V 视觉 KD**（仅当第 2 步放行） | 12 GB 教师目标（`q`/`q_spa`/top-64 logits） | `_losses` 加 `kd_logit`/`kd_feat`，`LAMBDA_RA` 环境变量开关 | 2 天 + 5–10 h 教师前向 + 1.7 h GPU | 只给框 action/other 组 R@50 从 1–4% 明显抬升；否则结案 |
| 6 | **A 可训练颈部副本 → 换 DINOv3** | 无（结构改动） | README L90 的 A；DINOv3 ViT-S/16+ 微调权重已在 `model.pth` 内（257 MB，README L108），可直接做我们的骨干，不依赖 HF gated 下载 | A：1 天 + 1.7 h；换骨干：3–5 天 + 推理成本上升（论文 L365：ViT-S/16+ 检测器+头+解码 20 ms/帧 A40，需 torch.compile） | 同 README 三条 |

与「换 DINOv3 骨干」的关系：1/3/4 是文本侧，换不换骨干都要做，**互补**；5 是 6 的低天花板替代，二选一，不叠加；2 决定在 5 和 6 之间选哪个。

## 7. 风险与前提

- **单卡冲突**：A3 队列预计 18:30 结束，若过关会接 16 h 全量 4 轮；上面所有 GPU 项都要排在训练之间，LLM 推理与教师前向不能和训练并跑（显存 15.5 GB 已占，4080 全速时吞吐会互相拖慢）。
- **网络**：下载速度实测 0.3–1.1 MB/s，7B 级模型是半天到一天的后台任务；`huggingface.co`/`github.com` 本次可达但不保证持续。
- **许可**：RA-4M 为 CC-BY-NC-4.0（`RA-4M/README.md` 头部）；官方代码 `RelateAnything/LICENSE` 未核对条款，移植前看一眼（待确认）。
- **指标口径**：同义词组正样本会降低精确匹配 R@50（论文 B.2.2，官方 `wearing` 71% 答 `wearing clothing`）；不先上同义感知指标就按 c2 判，会把正确的改动判死。
- **教师也有缺陷**：官方只给框 R@50 26.61、pair_acc 34.26，mR@50 13.11；KD 的上限是教师，而教师的头部谓词本身被同义词稀释。
- **LLM 判断的可信度**：蕴含概率数值不可直接当权重，只能做排序先验并与统计合并；反义/对称表需人工抽检。
- **未验证项汇总**：7B 在 4080 上的 tok/s、教师空卡吞吐、VLM 逐图秒数、Qwen3-Embedding 的反义余弦、探针实验的实际差距、RECODE/SDSGG 的收益数字。

## 附：本次调研没做什么

- 没有加载 `weights/primary_dior20_best.pt` 核对类别名（缺 `mobileclip` 模块，加载失败；DIOR 遥感 20 类的结论引自 `WORKLOG.md` L513）。加载失败时 ultralytics 自动生成了 `logs/error_report_26.11.07_1.txt`，已删除，仓库恢复原状。
- 没有跑任何 GPU 进程、没有下载任何模型、没有改仓库文件。
- 没有核实 RECODE / SDSGG / VLPrompt 的具体数字（时间预算内只做了 2 次 WebSearch）。

# GLP 创新点与实现方案（详细版）

> 日期：2026-11-01
> 范围：三个创新点的**机制细节** + **代码级实现方案**（模块、张量、配置、训练/推理、RES 下游、代数、风险、测试、阶段）。
> 相关文档：`reports/PAPER_DRAFT_INTRO_METHOD_EXP.md`（论文口吻）、`reports/EVAL_PROTOCOL_C_AUC.md`（评测协议）、`reports/PLAN_LANGUAGE_SEG.md`（RES 计划）。
>
> **实现状态（2026-11-01 核验）**：本文原“已实现”表中 `field.py`、`score_all(return_field=True)`、场代数并不存在，已按本文补齐：
> `ultralytics/models/yolovo/relate/field.py`、`head.py` 的 field 分支/`score_all`、`wiring.py`/`namespace.py` 的 `field` 透传、
> `tests/test_glp_field.py`（23 项通过）、`relate/field_metrics.py`（公开+自定义指标）。
> **仍未实现**：RES/RefCOCO 数据与脚本、official RA adapter、SUNRGBD/SpatialSense 数据、学习风险头（V2 改为 `w_bg` 背景桶）。
> 本机没有 `reports/` 与 SUN/RefCOCO 数据，相关表不能声称已跑。

---

## 0. 一句话

**一个谓词场，三种用法**：
1. **读得准**：谓词 `e_p` → 各向异性高斯混合场 `Ω_p`，场条件化证据读取（主创新点）；
2. **用得上**：同一个场作为 RES 的空间先验、并在 K 个核上闭式组合；
3. **证得实**：场在决策路径上，deletion 是因果证据；场统计 → 风险弃权。

---

## 1. 代码现状与实现地图

### 1.1 已实现（可直接跑）

| 模块 | 文件 | 作用 |
|---|---|---|
| 场模块 | `ultralytics/models/yolovo/relate/field.py` | `FieldConfig`、`PredicateFieldRead`、`field_grid`、`field_stats` |
| 关系头接入 | `.../relate/head.py` | `score_candidates`、`_select_candidates`、field 分支、NCE scatter、`score_all`、`field_ablate` |
| 接线 | `.../relate/wiring.py` | `attach_relate_layer(..., field_cfg=)` |
| 训练入口 | `.../relate/namespace.py` | `train(..., field=)` |
| 日志 | `.../relate/trainer.py` | `field_diag`（gamma_l1 / delta_norm） |
| 训练脚本 | `scripts/train_relate_glp.py` | `--field off|glp|inject|mlp` |
| 诊断评测 | `scripts/eval_spatialsense_glp.py` | C-AUC/DSR-AUC；`--no_pair`、`--ablate` |
| SUN 评测 | `scripts/eval_sunrgbd_cells.py` | 内部 cell；缺 `--no_pair`/`--ablate` |
| 几何对照 | `scripts/geometry_prior_*.py` | no-pixel HGB |
| 数据 | `scripts/prepare_sunrgbd_relate.py`、`scripts/build_spatialsense_local.py` | 派生数据 / 本地 pack |
| 单测 | `tests/test_glp_field.py` | 9 项全过 |

### 1.2 未实现（本文档要补）

| 模块 | 计划文件 | 状态 |
|---|---|---|
| RES 数据/解析 | `scripts/refcoco_relation_subset.py` | 待写 |
| RES 零样本 rerank | `scripts/res_zero_shot_rerank.py` | 待写 |
| RES 场缓存 | `scripts/cache_field_heatmaps.py` | 待写 |
| RES 融合 | `scripts/res_field_fusion.py` | 待写 |
| RES 指标 | `scripts/res_metrics.py` | 待写 |
| `score_all` 返回场 | `head.py` | 待加 `return_field=True` |
| 场代数 | `field.py` + `tests/test_field_algebra.py` | 待写 |
| 风险头 | `field.py` + `scripts/risk_head.py` | 待写 |
| official RA adapter | `scripts/ra_adapter_eval.py` | 待写 |
| 机制指标 | `scripts/field_mechanism.py` | 待写 |

### 1.3 当前资产

- GLP checkpoint：`runs/glp/glp_sunrgbd/weights/best.pt`
- RA checkpoint：`runs/glp/ra_sunrgbd/weights/best.pt`
- 主检：`weights/yolovo11n_glp.pt`；检测器：`yolo11n.pt`
- SUN 派生数据：`data/glp/sunrgbd_relate/`
- SpatialSense 本地 pack：`/root/rivermind-data/datasets/ovsgg_local/`
- 当前 GPU：`torch.cuda.is_available() = False`（训练/评测需远程/Modal；CPU 可做数据、dump、bootstrap）

---

## 2. 创新点一（主）：谓词高斯场读取

### 2.1 问题

`DeformableRelRead` 的 `offset_mlp/weight_mlp` 只依赖 `qn = norm(queries)`；`e_p` 只在 `score_query_dual` 出现。19k 谓词共享同一份证据；空间谓词的判别信息在“去哪里看”。通道式条件改变特征通道，不改变采样分布。

### 2.2 形式化

```
w_{p,k} = α_p · softmax_k(logit_{p,k})          # k=(a,j), K=A·G=32
μ_{p,k} = c_a + h_a ⊙ 2tanh(δ_{p,k})
σ_{p,k} = (σ_local ⊙ h_a).clamp(0.02, 0.6)      # 各向异性 σ_x,σ_y
Ω_p(q)  = Σ_k w_{p,k} exp[−½ (q−μ_{p,k})^T Σ_{p,k}^{-1} (q−μ_{p,k})]
Ω̃_p(q) = exp(log Ω_p(q) − max log Ω_p(q))
```

- `α_p = sigmoid(gate_mlp(W))`：`_VocabHead` 已有的标量先验；场的总质量 = `α_p`。
- `A=4`：subject / object / union / contact；`G=8`；`π` 在 32 个核上全局归一化。
- 坐标：图像归一化 `[0,1]²`；`h_a=(w,h)/2`，`clamp_min 0.025`。
- `σ_local ∈ [0.1,2.0]`（半宽单位），乘 `h_a` 后 clamp 到 `[0.02,0.6]`。

### 2.3 FieldHyperNet

| 项 | 值 |
|---|---|
| 输入 | `[e_p(512); geo19(19); anchor_emb(16)] = 547` |
| MLP | `547 → 64 → 64 → G·5 = 40`（每个 anchor） |
| 输出 5 | `δ(2) + σ_local(2) + logit(1)` |
| 初始化 | 最后一层 weight 置零；δ bias 环形 `0.35(j+1)[cosθ_j,sinθ_j]`；σ/logit bias 置零 |
| `e_proj` | `512 → 32 → 512`，低秩 |
| `agg` | `512 → 32`，weight/bias 零初始化 |
| `out_norm` | LayerNorm(512) |
| `gamma` | `[512]`，零初始化 |
| `inject_off/inject_w` | inject/mlp 用；零权重 + ring bias |
| `q_res_proj`（head） | `512 → 64 → 512`，xavier |
| 合计 | **0.208M / 26.56M = 0.78%** |

### 2.4 前向数据流（精确张量）

| 张量 | 形状 | 说明 |
|---|---|---|
| `img` | `[B,3,640,640]` | 输入 |
| `P4` | `[B,C4,40,40]` | stride 16 |
| `F_map` | `[B,40,40,768]` | `fmap_proj` |
| `scene_d` | `[B,512,40,40]` | `scene_proj` |
| `boxes` | `[1,N,4]` | 归一化 cxcywh |
| `sub/obj` | `[1,K]` | sampler |
| `anchors` | `[1,K,4,4]` | sub/obj/union/contact |
| `geo19` | `[1,K,19]` | `RelGeomEncoder.features` |
| `r` | `[1,K,512]` | pair query（base read 后） |
| `q/q_spa` | `[1,K,512]` | dual query |
| `logits` | `[1,K,V]` | draft |
| `cand` | `[1,K,M]` | top-M |
| `e_cand` | `[1,K,M,512]` | `W[cand]` |
| `hin` | `[1,K,M,4,547]` | hyper 输入 |
| `raw` | `[1,K,M,4,8,5]` | hyper 输出 |
| `mu/sigma` | `[1,K,M,32,2]` | 核中心/各向异性 σ |
| `logw` | `[1,K,M,32]` | 核权重 |
| `q0` | `[1,K,M,32,2]` | 采样点 |
| `w_agg` | `[1,K,M,32]` | 场调制聚合 |
| `vals` | `[1,512,K,M,32]` | `grid_sample` |
| `z` | `[1,K,M,512]` | 读出 |
| `r_field` | `[1,K,M,512]` | 残差后 |
| `delta` | `[1,K,M,512]` | `r_field - r` |
| `q_f/q_spa_f` | `[1,K,M,512]` | 一阶修正 |
| `cos_m/logit_m` | `[1,K,M]` | 候选打分 |
| `field` | `mu/sigma/logw/q/anchors/log_omega` | detach |
| `field_grid` | `[64,64]` | 可视化/deletion |

**前向步骤**（`field.py::_chunk`）：

1. `hin` 拼接；hyper → `raw`；`δ=2tanh`；`σ_local`；`logit`；
2. `μ=c+h⊙δ`；`σ=(σ_local⊙h).clamp`；`logw=log_softmax(logit over 32)`；
3. `q0=μ`；`π=softmax(log_components(q0))`；`grad=Σπ(μ−q0)/σ²`；
4. `step=η h_a grad`；`cap=0.25h_a`；`q0=clamp(q0+step)`；
5. `log_omega=logsumexp(log_components(q0))`；减 max；
6. `w_agg=softmax(agg(LN(r)+E_proj(e_p)) + λ log_omega)`；
7. `grid=qpos*2−1`；`grid_sample(scene_d)`；`z=Σ w_agg vals`；
8. `r_field=r+γ⊙LN(z)`；返回 `r_field` 和 field dict。

### 2.5 关系头接入（`head.py::_compute`）

```python
V = self.vocab_head.W.shape[0]
m = min(self.field_cfg.n_candidates, V)
gtp = pred_labels[0] if pred_labels is not None else None
cand = self._select_candidates(logits, gtp, valid_mask[0], m, training)
e_cand = self.vocab_head.W[cand]
if field_ablate == "agnostic":
    e_cand = e_cand.mean(dim=2, keepdim=True).expand_as(e_cand)
elif field_ablate == "shuffled":
    e_cand = e_cand.roll(1, dims=2)
r_field, field = self.field_read(r, scene_d, anchors, geo19, e_cand, return_field=True)
delta = r_field - r.unsqueeze(2)
q_f = q.unsqueeze(2) + self.q_res_proj(delta)
q_spa_f = q_spa.unsqueeze(2) + self.q_res_proj(delta)
cos_m, logit_m = self.vocab_head.score_candidates(q_f, q_spa_f, cand)
logits = logits.scatter(-1, cand, logit_m.to(logits.dtype))
extra = {"cand_idx": cand[0], "field_cos": cos_m[0], "field": field,
         "field_diag": {"gamma_l1": ..., "delta_norm": ...}}
```

- `query_mode="full"`：对 `r_field` 重跑 `compose_norm(proj(...))` / `spa_proj`，跳过 `q_res_proj`；约 +30% FLOPs，只作消融。
- `_select_candidates`：eval 用 `topk`；训练把 GT 列置 `+inf` 后 topk，索引唯一。
- `score_candidates`：`Wc=W[idx]`，`a=alpha[idx]`；`cos_sem=(normalize(q_sem)*Wc).sum(-1)`；`cos=(1-a)cos_sem+a cos_spa`；`logit=cos*scale+bias`。

### 2.6 训练接入（`head.py::_losses`）

NCE 的候选列用场分数：

```python
cos = cos.scatter(1, out["cand_idx"][has_gt], out["field_cos"][has_gt])
```

- 分母仍跑全词表；
- `sigmoid`/`bg` 用 `out["logits"]`（已 scatter）；
- `swap` 保持 draft（已知限制）；
- 七项损失权重：NCE 1.0 / obj 0.1 / swap 0.5 / sigmoid 0.25 / bg 0.05 / geo 1.0 / rel 1.0。

### 2.7 零初始化与等价性

- `γ=0` → `r_field≡r` → `q_f≡q`；候选分数与 draft 只差 gather 舍入；
- 实测 `max|Δlogit|=8.9e-8`，argmax 一致；
- `W_q` xavier（不能零：两个零门串联会饿死分支）；
- 单测 `test_identity_init`、`test_gamma_opens_and_branch_trains` 覆盖。

### 2.8 消融模式

| mode | 位置来源 | 权重来源 | 场/梯度 |
|---|---|---|---|
| `glp` | hypernet δ/σ | `agg + λlogΩ̃` | 有 |
| `inject` | `inject_off(r+E_proj(e_p))` | `inject_w` | 无 |
| `mlp` | hypernet δ | `agg`（无 logΩ̃） | 无 |
| `agnostic` | 同 glp | e_p 换候选均值 | 有（身份消融） |
| `shuffled` | 同 glp | 候选 roll | 有（内容消融） |

### 2.9 实现状态

- 已完成：模块、接入、训练、评测、9 项单测、50 epoch RA/GLP；
- 待补：`score_all(return_field=True)`；`field_stats` 的 mass 统计；RES 用的 `field_grid` 接口（已有，需暴露）；
- 待跑：β=0 的 C-AUC、paired bootstrap、inject/MLP-field、official RA。

---

## 3. 创新点二：场作为可迁移先验 + 闭式代数

### 3.1 RES 先验：形式化

给定表达式 `x`，解析出 target noun、reference noun、relation `p`：

```
s_field(M) = Σ_{q∈M} Ω̃_p(q) / Σ_q Ω̃_p(q)
p'_M       = p_M · (1 + α · Ω̃_p|_M)
```

- `M`：候选 mask（Mask R-CNN / YOLO-seg / SAM / box mask）；
- `Ω̃_p`：target 候选框与 reference 框组成的 pair 的场；
- `α`：mask prior 的强度，初始 0，场冻结，只训 `α`/decoder；
- baseline：同一候选集不用场；geometry prior；random；oracle。

### 3.2 RES 数据与解析

**数据**

```
/root/rivermind-data/refcoco/
  annotations/refs(unc).p, refs(unc)+.p, refs(unc)gc.p,
              instances_train2014.json, instances_val2014.json
  images/train2014/, images/val2014/
  splits/relation_subset_{split}.json
  cache/field_heatmaps/{split}/{id}.npy
```

- RefCOCO annotations 从 `github.com/lichengunc/refer` 取；COCO 图像官方源当前不通，用 mirror/用户提供；先 val 子集。
- GT mask 从 COCO `instances_*.json` 的 `ann_id` 取。

**解析器** `scripts/refcoco_relation_subset.py`

- 9 谓词词表：`on / in / under / above / next to / behind / in front of / left of / right of`；
- 每条表达式抽 `target / reference / relation`；
- 子集：
  - `rel_all`：含任一谓词；
  - `rel_hard`：同图 ≥2 个同类 target（关系负责消歧）；
  - `full`：全 split；
- 人工抽检 100 条，报 parser accuracy；错误样本剔除并记录。

### 3.3 零样本 rerank

`scripts/res_zero_shot_rerank.py`：

1. detector（`yolo11n.pt`）出 target 候选框 + reference 框；
2. 候选 mask（Mask R-CNN / YOLO-seg / box mask）；
3. `rel.score_all(boxes, topk=500, return_field=True)`；取 field dict、`cand_idx`；
4. 对每个候选，用 `field_grid(field, k, m, hw=(64,64))` 得 `Ω̃_p`；
5. `s_field(M)`；选 mask；
6. baseline：detector top-1 / geometry / random / oracle；
7. mIoU/oIoU/P@0.5；`rel_all`/`rel_hard`/`full`；CI。

**需要的 API**：

```python
# head.py
@torch.no_grad()
def score_all(self, boxes, topk=None, img_idx=0, field_ablate=None, return_field=False):
    ...
    if return_field:
        return sub, obj, valid, logits, rel_logits, out.get("field"), out.get("cand_idx")
    return sub, obj, valid, logits, rel_logits
```

`out["field"]` 是 merged field dict；`out["cand_idx"]` 是 `[K,M]`；`field_grid` 已可用。

### 3.4 Fine-tune 融合

`scripts/cache_field_heatmaps.py`：

- 对 RefCOCO train/val 每条表达式：detector/GT boxes → field → `Ω̃` 64×64 → `.npy`；
- GT boxes 与 detector boxes 两版；场不参与反传。

`scripts/res_field_fusion.py`：

- baseline：CRIS/LAVT（选有 checkpoint、依赖可装的）；
- 融合：mask prior `p'=p(1+αΩ̃)`（主）；FiLM/attention bias（消融）；
- 场冻结，只训 `α`/decoder；`L_seg=BCE+Dice`；
- 对照：no field / field / agnostic / geometry / random / inject；
- 3 seeds；公开 mIoU/oIoU/P@0.5；per-relation；CI。

### 3.5 闭式代数

`field.py` 新增纯函数：

```python
def compose_and(field_a, field_b) -> dict: ...
def compose_or(field_a, field_b, w_a=0.5, w_b=0.5) -> dict: ...
def gate(field_p, field_r, tau=1.0) -> dict: ...
def transitive(field_ab, field_bc) -> dict: ...
def negate(field_p) -> dict: ...
```

公式：

```
AND:  Ω_{p∧q} ∝ Ω_p·Ω_q
      Λ = Σ_p^{-1}+Σ_q^{-1}
      μ = Λ^{-1}(Σ_p^{-1}μ_p + Σ_q^{-1}μ_q)
OR:   Ω_{p∨q} = w_pΩ_p + w_qΩ_q
gate: Ω_p·(Ω_r/maxΩ_r)^τ
传递: 递归组合，均值平移相加（近似，需校准）
否定: 1−Ω̃_p（校准近似）
```

- 混合×混合 → `K_pK_q` 核，按权重剪枝到 K；
- `field_grid` 直接读复合场；零额外前向；
- 测试：两个高斯的积的均值/精度；grid 形状；零前向。

### 3.6 状态

- RES 数据/解析/rerank/fusion：**未实现**；
- 代数：**未实现**；
- 依赖：`score_all(return_field=True)`；RefCOCO 数据；RES baseline；GPU。

---

## 4. 创新点三：证据定位 + 风险弃权

### 4.1 形式化

- **Deletion**：在 `scene_d` 特征层 mask 场 top-q% 框外质量；`DR(q)=Δ_field/Δ_random`；q∈{1,5,10,20}%；`DR≥2×`、paired bootstrap。
- **PSS**：固定候选/框，替换 `e_p`（循环 1/2/3 + 随机 5 seed）；`ΔC-AUC/ΔDSR-AUC`；GLP 应 > inject/MLP-field。
- **FMA**：`in` 的场质量落在容器 box 内的比例；同 pair 换 `e_on` 的差；GLP > inject。
- **Risk**：场统计 7-D → `P(correct)` → 三态 `POS/NEG/ABSTAIN`；AURC/ECE/risk-coverage；selective mIoU@80/90。

### 4.2 场统计

当前 `field_stats` 只返回 `kernel_entropy` 与 `anchor_boxes`；需补：

```python
def field_stats(field, anchors, boxes=None):
    return {
        "mass_in_sub": ..., "mass_in_obj": ..., "mass_in_contact": ...,
        "mass_outside": ..., "peak": ..., "kernel_entropy": ...,
        "alignment": ...,  # 采样点落在高质量区的比例
    }
```

- 用 `field_grid` 或解析 erf 闭式；
- risk 头：`Linear(7→64) → GELU → Linear(64→1)`；
- 训练：valid cells；小样本用 CV；
- 评测：AURC/ECE/risk-coverage；selective mIoU。

### 4.3 状态

- deletion/PSS/FMA/AURC 脚本：**未实现**；
- risk 头：**未实现**；
- 依赖：`field_grid`、`field_stats`、valid 数据、GPU/CPU。

---

## 5. 三个创新点的关系

| 用法 | 接口 | 创新点 | 主要实验 |
|---|---|---|---|
| 读得准 | field read | 1（主） | Table 1 RES；Table 2/3 C-AUC/DSR-AUC |
| 用得上 | mask prior + 闭式代数 | 2 | Table 1；Table 8 复合 |
| 证得实 | deletion + risk | 3 | Table 4；Table 6 全集 |

---

## 6. 端到端实现路线

### Phase 0（已完成）

- field 模块、head 接入、训练脚本、C-AUC/DSR-AUC 评测、9 项单测；
- RA/GLP 50 epoch；pilot：C-AUC RA 0.429 / GLP 0.533 / geometry 0.475。

### Phase 1：C-AUC 定稿（CPU/GPU）

1. `head.py::score_all(return_field=True)`；
2. `eval_spatialsense_glp.py --no_pair` 重跑 β=0；
3. `scripts/bootstrap_cells.py`：image-clustered paired bootstrap；
4. 扩 `in` cells：raw test 338（169/169）、valid 216、train 804；
5. domain-matched geometry（train 804）+ class prior + class+geometry；
6. inject / MLP-field 训练（50 epoch，batch 16，seed 42/43/44）；
7. official RA zero-shot + frozen adapter。

**交付**：C-AUC/DSR-AUC 最终表；Table 2/3。

### Phase 2：RES 零样本（CPU/GPU）

1. `refcoco_relation_subset.py`：RefCOCO annotations + 关系子集 + parser 抽检；
2. `res_zero_shot_rerank.py`：detector + 候选 mask + field rerank；
3. `res_metrics.py`：mIoU/oIoU/P@0.5 + per-relation + CI；
4. 先 box-mask 版本，再 Mask R-CNN/YOLO-seg。

**交付**：第一张公开 mIoU 表（baseline/geometry/random/field/oracle）。

### Phase 3：RES 融合（GPU）

1. `cache_field_heatmaps.py`：GT/detector boxes 两版；
2. `res_field_fusion.py`：CRIS/LAVT + mask prior/FiLM；field 冻结；
3. 对照 + 3 seeds；
4. Table 1/9。

### Phase 4：机制/代数/风险（CPU/GPU）

1. `field_mechanism.py`：deletion/PSS/FMA；
2. `field.py` 代数函数 + 测试；
3. risk 头 + `field_stats` 扩展；
4. Table 4/8。

---

## 7. 文件与命令清单

### 7.1 待写文件

| 文件 | 函数/内容 |
|---|---|
| `scripts/refcoco_relation_subset.py` | `load_refs`、`parse_relation`、`build_subset` |
| `scripts/res_zero_shot_rerank.py` | `detect_boxes`、`candidate_masks`、`field_score`、`evaluate` |
| `scripts/cache_field_heatmaps.py` | `cache_split`、`save_heatmap` |
| `scripts/res_field_fusion.py` | `MaskPrior`、`train`、`evaluate` |
| `scripts/res_metrics.py` | `miou`、`oiou`、`precision_at`、`bootstrap_ci` |
| `scripts/field_mechanism.py` | `deletion`、`pss`、`fma` |
| `scripts/risk_head.py` | `FieldRiskHead`、`train`、`aurc` |
| `tests/test_field_algebra.py` | AND/OR/gate/transitive/negate |
| `tests/test_res_pipeline.py` | parser / field score / metrics |

### 7.2 命令

```bash
# 单测
python -m pytest tests/test_glp_field.py -q

# 训练
python scripts/train_relate_glp.py --primary weights/yolovo11n_glp.pt \
  --data data/glp/sunrgbd_relate/data.yaml --field glp \
  --epochs 50 --batch 16 --workers 8 --seed 42 --name glp_sunrgbd

# C-AUC / DSR-AUC（β=0）
python scripts/eval_spatialsense_glp.py --ckpt runs/glp/glp_sunrgbd/weights/best.pt \
  --pack_valid $OVS/packed/spatialsense_valid --pack_test $OVS/packed/spatialsense_test \
  --cells_valid $OVS/datamix/spatialsense_valid_cells.json \
  --cells_test $OVS/datamix/spatialsense_test_cells.json \
  --no_pair --out reports/g1_glp_nopair.json

# 消融
python scripts/eval_spatialsense_glp.py ... --ablate agnostic
python scripts/eval_spatialsense_glp.py ... --ablate shuffled

# RES（待写）
python scripts/refcoco_relation_subset.py --refs data/refcoco/annotations --out data/refcoco/splits
python scripts/res_zero_shot_rerank.py --ckpt runs/glp/glp_sunrgbd/weights/best.pt \
  --detector yolo11n.pt --subset data/refcoco/splits/relation_subset_val.json --out reports/res_zero_shot.json
```

---

## 8. 测试与验收

### 8.1 已有单测（23 项全过）

`tests/test_glp_field.py`：identity init（V1/V2）、null 守恒/弃权、平移/尺度等变、V2 对象独立性 vs V1 对象依赖、
`score_candidates` vs 全词表、GT 强制候选、`score_all(return_field=True)`、field grid/stats、
AND/OR/传递闭式代数（单核与多核）、head 前向/反向（V1/V2）、指标累计器、FMA/EE/SFC。

### 8.2 待补/未跑

- RES parser 的 100 条抽检；RES metrics 与 `pycocotools`/`refer` 对齐（本机无 RefCOCO）；
- `field_grid` 在真实 SUNRGBD/SpatialSense `in`/left 单元格上的可视化 sanity（无数据）；
- official RA adapter（`assets/relsgg-vits16plus` 结构与 YOLOVO 不同）；
- risk 头：V2 已用 `w_bg` 背景桶替代，不再训练 sigmoid risk head。

### 8.3 评测指标（公开 + 自定义）

公开指标（`relate/val.py::RelateEvaluator`）：
- SGG：`R@20/50/100`、`mR@50/100`、`F1@50`（双端 IoU≥0.5 且谓词相同）、`predicate_acc`、
  `pair_existence_AP`、`DetPairRecall`、`PairRecall`、`GT_MRR`；
- 检测：P/R/mAP50/mAP50-95（一级检测器，与关系指标分开报）；
- RES（未跑）：`mIoU`、`oIoU`、`P@0.5`。

自定义指标（`relate/field_metrics.py`，直接对应两个创新点）：
- **FMA**（Field-Mass Agreement）：GT 谓词场落在 GT 目标框内的概率质量，
  `FMA = E[2·mass_in_obj − 1]`，越高越好；
- **WBG gap**（背景桶分离）：GT 谓词的 `w_bg` 应低于非 GT 候选，
  `gap = mean(w_bg^neg) − mean(w_bg^pos)`，越高越好（V2 才有）；
- **AURC**：以 `1 − w_bg` 为置信度、谓词 top-1 正确为标签的风险-覆盖面积，越低越好；
- **SFC**（Subject-Frame Consistency）：主体框平移/缩放后，局部核参数 `μ̃,Σ̃` 的不变误差，V2 应≈0；
- **EE**（Equivariance/round-trip Error）：`μ_abs` 经 `to_image_frame` 反变换回局部系的误差，应≈0。

报告口径：公开指标与 FMA/WBG/AURC 报均值；SFC/EE 报 fp32 参数版（网格版另报，受 64×64 插值误差限制）。
RA-4M 提供关系标注与图像；RefCOCO/SUNRGBD/SpatialSense 本机缺失，不能把 RA-4M smoke 结果冒充 RES/SUN 公开结果。

---

## 9. 风险与边界

| 风险 | 对策 |
|---|---|
| 场是 SUN 室内学的，COCO 域差 | RES 先 `rel_hard`；per-relation 报；备选 PhraseCut |
| 场 64×64 太粗 | mask prior 低分辨率做、上采样；FiLM 在 decoder 特征层 |
| detector/parser 错 | GT boxes 行隔离场；parser 抽检 100 条；错误剔除 |
| 强 baseline 太强，增益小 | 先 zero-shot 看信号；mask prior 从 α=0 学；报 full |
| 代数/风险未实现 | 先补函数+测试；实验表 4/8 留空，不假装 |
| official RA 集成成本 | 先 zero-shot 评；再 frozen adapter；port 只作代理 |
| GPU 不在 | Phase 1/2 CPU；Phase 3 远程；fine-tune 2–4 GPU 天 |
| 数据许可 | RefCOCO/COCO 研究用途；引用原数据；不重分发图像 |

---

## 10. 明确不做（本轮）

- 不重训场（冻结 checkpoint）；
- 不端到端重训分割 backbone（RES 是 plug-in）；
- 不做组合谓词分类（没有组合文本编码器）；
- 不刷 RES 总榜；只做 relation-heavy 子集 + full sanity；
- 不改 `YOLOVORelate` 结构；只加 eval 的 field 返回；
- 不把未实现的代数/风险写成已有结果。

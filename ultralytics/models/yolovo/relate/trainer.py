"""Relate trainer：二级关系训练器（mixin + 任务工厂）。

设计文档 §6：
- 注入保留（照抄 PrunedTrainerMixin 模式）：不做 YAML 重建。
- 冻结：backbone/neck/一级头/VCTE/MobileCLIP requires_grad=False 且 eval()；
  覆写父类 _setup_train 的强制解冻段（engine/trainer.py:315-320 会把冻结浮点参数解冻）。
- 排除 set_vcte_alpha 回调（每 epoch 改写 alpha buffer，污染冻结一级模型）。
- build_optimizer 只含关系层参数。
- loss_names 四项；save_model 重载注入 relate_info；EMA half 保护 W buffer。
- 关系模型 loss() 全路径接管（不触达 YOLOVOLoss / pose 默认 criterion）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from copy import copy

from ultralytics.utils import DEFAULT_CFG, LOGGER

if TYPE_CHECKING:  # mixin 类型垫片：让 super() 属性可解析（运行时为 object）
    from ultralytics.engine.trainer import BaseTrainer as _TypedBase
else:
    _TypedBase = object


def _find_relate_module(model):
    """识别关系头。"""
    from ultralytics.models.yolovo.relate.head import YOLOVORelate

    m = model.module if hasattr(model, "module") else model  # unwrap DDP
    for mod in m.model:
        if isinstance(mod, YOLOVORelate):
            return m, mod
    return m, None


class _RelateOnlyEMA:
    """ModelEMA 的关系层子集版：只对 ``prefix`` 开头的浮点张量做滑动平均。

    其余（冻结的一级模型/文本塔）在 deepcopy 时已与训练模型一致，且训练中不变，
    跳过它们对 EMA 权重无影响。接口与 ``ultralytics.utils.torch_utils.ModelEMA`` 一致
    （ema / updates / decay / enabled / update / update_attr），trainer 其余代码不感知。
    """

    def __init__(self, model, rel_idx: int, decay=0.9999, tau=2000, updates=0):
        import math
        from copy import deepcopy

        from ultralytics.utils.torch_utils import de_parallel

        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True
        self._idx = int(rel_idx)
        # 只取关系层的 state_dict（不含前缀）：整模型 state_dict() 每步重建一遍
        # 约 1,000 个张量的 OrderedDict，本身就占 3% 主线程时间。
        self._keys = [k for k, v in self.ema.model[self._idx].state_dict().items()
                      if v.dtype.is_floating_point]

    def update(self, model):
        if not self.enabled:
            return
        from ultralytics.utils.torch_utils import de_parallel

        self.updates += 1
        d = self.decay(self.updates)
        msd = de_parallel(model).model[self._idx].state_dict()
        esd = self.ema.model[self._idx].state_dict()
        for k in self._keys:
            v = esd[k]
            v *= d
            v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        if self.enabled:
            from ultralytics.utils.torch_utils import copy_attr

            copy_attr(self.ema, model, include, exclude)


class RelateTrainerMixin(_TypedBase):  # type: ignore[misc,valid-type]
    """Overlay for any YOLOVO task trainer: keep the relate graph, freeze primary."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        import os
        from pathlib import Path

        # 父类 __init__ 在 exist_ok=True 且 args.resume 为假时会删掉旧 results.csv；
        # 我们的续训走 YOLOVO_RESUME 环境变量（不走 args.resume），先把它备份下来。
        csv_backup = None
        if os.environ.get("YOLOVO_RESUME", "").strip() and overrides:
            p = Path(str(overrides.get("project") or "runs")) / str(overrides.get("name") or "") / "results.csv"
            if p.is_file():
                csv_backup = p.read_text(encoding="utf-8")
        super().__init__(cfg, overrides, _callbacks)
        if csv_backup and not self.csv.exists():
            self.csv.write_text(csv_backup, encoding="utf-8")
            LOGGER.info(f"[RelateTrain] 续训：已恢复 {self.csv}（{csv_backup.count(chr(10)) - 1} 行历史指标）")
        if getattr(self.args, "model_scale", None):
            raise ValueError("rel.train 不支持 model_scale/scale；关系层结构直接来自 checkpoint。")
        # 数据 yaml 必须带 rnames（谓词词表）
        self.relate_rnames = None
        self._field_metrics = None
        # YOLOVOTrainer.__init__ 注册了 set_vcte_alpha：它遍历所有模块做
        # `m.alpha.fill_(factor)`，会把关系头 vocab_head.alpha（逐谓词路由门）
        # 和冻结骨干的 VCTE alpha 一起覆写成常数。关系训练必须剔除该回调。
        cbs = self.callbacks.get("on_train_epoch_start", [])
        self.callbacks["on_train_epoch_start"] = [
            c for c in cbs if "vcte" not in getattr(c, "__qualname__", "").lower()
        ]
        if len(cbs) != len(self.callbacks["on_train_epoch_start"]):
            LOGGER.info("[RelateTrain] 已排除 set_vcte_alpha 回调（否则会覆写 alpha buffer）。")

    # ------------------------------------------------------------- model keep

    def get_model(self, cfg=None, weights=None, verbose=True):
        """使用注入的关系模型；不做 YAML 重建。"""
        if isinstance(self.model, torch.nn.Module):
            return self.model
        from ultralytics.nn.tasks import attempt_load_one_weight

        weights_obj, ckpt = attempt_load_one_weight(str(self.model))
        info = ckpt.get("relate_info") if isinstance(ckpt, dict) else None
        if not isinstance(info, dict) or not info:
            raise ValueError("rel.train 仅接受带 relate_info 的权重或已注入的关系模型。")
        weights_obj.relate_info = info
        self.model = weights_obj
        return self.model

    def compute_batch_loss(self, batch):
        """骨干整批一次前向（冻结）→ 关系头逐图重放（见 compute_loss）。"""
        preds = self.model.predict(batch["img"])  # head 训练前向暂存整批特征
        return self.compute_loss(batch, preds)

    def setup_model(self):
        """跳过 YAML 重建；丢弃 pickled criterion（关系 trainer 全路径接管 loss）。

        返回值交给父类 ``resume_training``：设置环境变量 ``YOLOVO_RESUME=<train-checkpoint.pt>``
        时加载关系头权重并返回该 ckpt dict（含 optimizer/epoch/best_fitness），父类据此恢复
        优化器状态与 start_epoch。不走 ``args.resume``：那条路会用 ``attempt_load_weights``
        重建模型，绕过两步构建。
        """
        if not isinstance(self.model, torch.nn.Module):
            self.model = self.get_model()
        if hasattr(self.model, "criterion"):
            delattr(self.model, "criterion")
        _, rel = _find_relate_module(self.model)
        if rel is None:
            raise ValueError("注入的模型没有 YOLOVORelate 层：请检查 model.rel.train 的构建路径。")
        return self._load_resume_ckpt(rel)

    def _load_resume_ckpt(self, rel):
        """从 train-checkpoint.pt 恢复关系头参数；返回 ckpt dict 或 None。

        兼容两种格式：旧版 ``model`` 是整模型 half 深拷贝（只取关系层参数，W/W_obj 缓冲
        不覆盖，避免 fp16 取整进 fp32 词表）；新版 ``rel_state`` 是关系层 fp32 state_dict。
        """
        import os

        path = os.environ.get("YOLOVO_RESUME", "").strip()
        if not path:
            return None
        if not os.path.isfile(path):
            raise FileNotFoundError(f"YOLOVO_RESUME 指向的文件不存在: {path}")
        # 旧格式 pickle 引用顶层 mobileclip.*：先注入 _vendor 路径
        import ultralytics.models.yolovo.text_tower.mobileclip.factory  # noqa: F401
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "rel_state" in ckpt:
            state = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["rel_state"].items()}
            missing, unexpected = rel.load_state_dict(state, strict=False)
            src = "rel_state(fp32)"
        else:
            m = ckpt.get("model")
            if m is None:
                raise ValueError(f"{path} 既没有 rel_state 也没有 model，不能续训。")
            _, rel_src = _find_relate_module(m)
            if rel_src is None:
                raise ValueError(f"{path} 的模型里没有 YOLOVORelate 层。")
            params = dict(rel_src.named_parameters())
            state = {k: v.detach().float() for k, v in params.items()}
            missing, unexpected = rel.load_state_dict(state, strict=False)
            # 只允许缓冲（W/W_obj/sigma_weights）缺失：它们由当前词表重新编码
            bufs = {k for k, _ in rel.named_buffers()}
            missing = [k for k in missing if k not in bufs]
            src = "model(half)->params"
        if missing or unexpected:
            raise RuntimeError(f"续训权重与当前关系头不匹配: missing={missing[:5]} unexpected={unexpected[:5]}")
        if os.environ.get("YOLOVO_RESUME_RESET_BEST", "0") == "1":
            # 旧 run 的 best_fitness 若来自口径不同的 val（例如换了 val 标签），应清零，
            # 否则新 run 的 best.pt 可能永远不会被写出。
            ckpt["best_fitness"] = None
        self.resume = True
        self._resume_meta = {"ema_updates": int(ckpt.get("ema_updates", 0) or 0), "path": path}
        LOGGER.info(f"[RelateTrain] 从 {path} 恢复关系头（{src}），epoch={ckpt.get('epoch')}，"
                    f"best_fitness={ckpt.get('best_fitness')}")
        return ckpt

    def build_dataset(self, img_path, mode="train", batch=None):
        """关系训练/验证数据集：R 行解析 + 行号寻址（标准校验会把 R 行当 corrupt）。"""
        from ultralytics.models.yolovo.relate.dataset import build_relate_dataset
        from ultralytics.models.yolovo.relate.wiring import resolve_rnames
        from ultralytics.utils.torch_utils import de_parallel

        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if self.relate_rnames is None:
            self.relate_rnames = resolve_rnames(data_yaml=self.data)
        return build_relate_dataset(
            self.args, img_path, batch, self.data, mode=mode, stride=gs, rnames=self.relate_rnames
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """关系数据加载器：relate_collate（训练侧 GT 框路径不用 jitter collate）。"""
        from torch.utils.data import DataLoader

        from ultralytics.data.build import InfiniteDataLoader
        from ultralytics.models.yolovo.relate.dataset import relate_collate

        dataset = self.build_dataset(dataset_path, mode, batch_size)
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        loader = (InfiniteDataLoader if mode == "train" else DataLoader)(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=workers,
            sampler=None,
            # 丢掉最后一个不满的 batch：否则这些图在逐图均值里权重偏大，
            # 还会和下一 epoch 的首个 batch 凑进同一个 nbs 累加步。
            drop_last=mode == "train",
            pin_memory=False,
            collate_fn=relate_collate,
            worker_init_fn=None,
            persistent_workers=workers > 0 and mode == "train",
        )
        return loader


    # --------------------------------------------------------------- freezing

    def _freeze_primary(self):
        """冻结一级全部参数（关系层之外），一级模块保持 eval。"""
        m, rel = _find_relate_module(self.model)
        rel_idx = rel.i
        frozen = trainable = 0
        for name, p in self.model.named_parameters():
            if name.startswith(f"model.{rel_idx}."):
                # logit_scale 在 vocab_head 内可学习（RA 原样）；顶层无此参数。
                # 例外：sigmoid/bg 都关闭时 logit_scale/bias 按 RA 规则冻结
                # （install_vocabulary 写入 _affine_frozen，重冻结时保持）。
                p.requires_grad = True
                trainable += p.numel() if p.requires_grad else 0
            else:
                p.requires_grad = False
                frozen += p.numel()
        if getattr(rel, "_affine_frozen", False):
            for p in (rel.vocab_head.logit_scale, rel.vocab_head.logit_bias):
                trainable -= p.numel() if p.requires_grad else 0
                p.requires_grad = False
        LOGGER.info(f"[RelateTrain] 冻结一级参数 {frozen / 1e6:.2f}M | 关系层可训练 {trainable / 1e6:.2f}M")
        # 一级模块 eval()（BN/dropout 不更新）
        for i, sub in enumerate(m.model):
            if i != rel_idx:
                sub.eval()
        # 文本塔不在 m.model 里，上面的循环管不到它：每个 epoch 的 _model_train
        # 会把整个模型翻回 train，BN 统计量随之漂移（实测 20 轮后
        # num_batches_tracked 涨了 20×156，running_mean 最大偏移 0.38）。冻结时补一次。
        te = getattr(m, "text_encoder", None)
        if te is not None and not any(p.requires_grad for p in te.parameters()):
            te.eval()

    def _setup_train(self, world_size=1):
        """覆写父类：先 dry-run 物化 pair_pool，再正常 setup，最后恢复冻结。

        pair_pool 在首次 forward 才创建；build_optimizer 在 super()._setup_train
        内部执行，晚于它则 pair_pool 不进 AdamW（视觉分支静默不更新），
        且 found_inf 不监控、clip_grad_norm_ 扫全模型会被其 Inf 梯度拖累。
        """
        self._materialize_pool()
        super()._setup_train(world_size)
        self._freeze_primary()  # 父类解冻后立即恢复
        # EMA 只跟踪关系层：父类 ModelEMA 每步遍历整模型 state_dict（一级模型 + 文本塔
        # 全部冻结，EMA 对它们恒等），实测占训练步 7.9% 的主线程时间。
        if getattr(self, "ema", None) is not None:
            _, rel = _find_relate_module(self.model)
            if rel is not None:
                updates = int(self.ema.updates)
                meta = getattr(self, "_resume_meta", None)
                if meta:
                    # 续训：EMA 从恢复的权重重新起步；updates 用保存值（旧格式没有则按
                    # 已完成 epoch 的优化器步数估算），避免 decay 从 0 再爬一次 tau=2000。
                    nb = len(self.train_loader)
                    est = (self.start_epoch * nb) // max(int(getattr(self, "accumulate", 1)), 1)
                    updates = int(meta.get("ema_updates") or est)
                self.ema = _RelateOnlyEMA(self.model, rel_idx=rel.i, updates=updates)

    def _materialize_pool(self):
        """物化延迟构建的子模块（进 AdamW 参数组）。

        v4: fmap_proj（P4→768 适配层）+ ontology 设备同步。
        v3: _PairPool/sub_proj/obj_proj。
        """
        _, rel = _find_relate_module(self.model)
        if rel is None:
            return
        device = next(self.model.parameters()).device
        if hasattr(rel, "_ensure_fmap_proj"):
            if not rel._feat_channels:
                LOGGER.warning(
                    "[RelateTrain] P4 通道数未知；fmap_proj 将在首次 forward 懒建，"
                    "不会进入 AdamW。请检查 attach_relate_layer 的 feat_channels。")
                return
            rel._ensure_fmap_proj(rel._feat_channels[rel.base_idx], device)

    def _model_train(self):
        """覆写父类：model.train() 会把一级模块设回 train 模式（BN 统计量会漂移），
        之后立即恢复冻结与 eval。父类在 on_train_epoch_start 回调之后才调用
        _model_train（engine/trainer.py:426→431），仅靠 epoch 回调无法覆盖。

        `_field_metrics` 也在这里重置：Mixin 上的 `on_train_epoch_start` 只是普通
        方法，不在 `self.callbacks` 里，训练循环永远不会调它。
        """
        super()._model_train()
        self._field_metrics = None
        self._freeze_primary()

    # --------------------------------------------------------------- optimizer

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """优化器只放关系层参数；按基础训练器惯例拆 bias/权重两组。

        warmup 期 j==0 组（bias）从 warmup_bias_lr 起步——若全部参数挤在
        第 0 组，26M 参数会以 0.1 的预热学习率更新（首轮 relatedness
        爆炸的根因）。分组后 bias 组极小，warmup_bias_lr 语义恢复正确。

        但关系头是随机初始化的，不是预训练 YOLO 头：0.1 的 bias warmup 会把
        out_norm.bias 从 0 推到 475（50k 第 1 轮），随后场残差吃掉整个 q。
        基类走 AdamW 时会强制 warmup_bias_lr=0（engine/trainer.py 的 auto 分支），
        这里覆写了 build_optimizer 绕过了该保护，所以显式对齐。
        """
        _, rel = _find_relate_module(model)
        if rel is None:
            raise ValueError("build_optimizer: 模型中没有 YOLOVORelate 层。")
        self.args.warmup_bias_lr = 0.0
        # 与基类 engine/trainer.py 的分组规则对齐：bias、Norm 权重、logit_scale 不衰减。
        # alpha 是路由门的系数，也不该被 AdamW 每步向 0 收缩。
        norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
        g0, g1 = [], []  # weights / no-decay
        for _mn, module in rel.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                no_decay = ("bias" in param_name or "alpha" in param_name
                            or "logit_scale" in param_name
                            or (isinstance(module, norm) and param_name == "weight"))
                (g1 if no_decay else g0).append(param)
        opt = torch.optim.AdamW(g1, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        opt.add_param_group({"params": g0, "weight_decay": decay})
        return opt

    # ------------------------------------------------------------------ loss

    def preprocess_batch(self, batch):
        """标准预处理。GT 框注入已移至 compute_batch_loss 逐图路径（单图流式废弃）。

        父类 `YOLOVOTrainer.preprocess_batch` 会把 `batch["cls"]` 重映射成
        batch 内文本的局部 id（0..K-1）。obj 对齐损失要索引按**全局类 id**
        排序的 W_obj，所以先把原始 cls 存一份。
        """
        if batch.get("cls") is not None:
            batch["cls_global"] = batch["cls"].clone()
        return super().preprocess_batch(batch)

    # 七项损失权重（RA config 原值）
    LAMBDA_RA = {"nce": 1.0, "obj": 0.1, "swap": 0.5, "sigmoid": 0.25,
                 "bg": 0.05, "geo": 1.0, "rel": 1.0}

    def compute_loss(self, batch, preds):
        """骨干一次前向 → 逐图 _compute，七项分开平均再加权（RA 权重）。"""
        from ultralytics.utils.torch_utils import de_parallel

        model = de_parallel(self.model)
        rel = None
        for mod in model.model:
            from ultralytics.models.yolovo.relate.head import YOLOVORelate as _R
            if isinstance(mod, _R):
                rel = mod
                break
        if rel is None:
            raise RuntimeError("compute_loss: 找不到 YOLOVORelate 层。")
        return self._compute_loss(batch, rel)

    def _compute_loss(self, batch, rel):
        """v4: 整批一次前向（图像按最大框数右填充）→ 逐图七项损失 → 分开平均。"""
        device = batch["img"].device
        idx = batch["batch_idx"].long().to(device)
        bboxes = batch["bboxes"].to(device)
        # obj 对齐损失用全局类 id（preprocess_batch 存的原文）；batch["cls"] 已被
        # YOLOVOTrainer 重映射成局部文本 id，用它索引 W_obj 会教错类别。
        cls_all = batch.get("cls_global", batch.get("cls"))
        if cls_all is not None:
            cls_all = cls_all.to(device)
        relations = batch.get("relations") or []
        n_imgs = batch["img"].shape[0]

        counts = torch.bincount(idx, minlength=n_imgs).long() if idx.numel() else \
            torch.zeros(n_imgs, dtype=torch.long, device=device)
        cnt = counts.tolist()  # 一次同步代替逐图 int(counts[i])（每图一次会掐断 GPU 流水）
        n_max = max(max(cnt), 2)
        boxes_pad = bboxes.new_zeros(n_imgs, n_max, 4)
        boxes_pad[..., 2:] = 0.01  # 填充槽用 1% 小框，避免 spatial_pool 退化成零尺寸
        cls_list, gt_list = [], []
        off = 0
        for i in range(n_imgs):
            n_i = cnt[i]
            boxes_pad[i, :n_i] = bboxes[off:off + n_i]
            cls_list.append(cls_all[off:off + n_i].reshape(-1).long()
                            if cls_all is not None else None)
            off += n_i
            rel_i = relations[i] if i < len(relations) else None
            gt_list.append([(int(r[0]), int(r[1]), int(r[2])) for r in rel_i]
                           if rel_i is not None and len(rel_i) else [])

        # 类别 id 是 PredCls 的定义输入，随框一起进关系头（padding 槽填 0，
        # 这些槽不会被任何 pair 选中）。
        cls_pad = None
        if cls_all is not None:
            cls_pad = torch.zeros(n_imgs, n_max, dtype=torch.long, device=device)
            for i in range(n_imgs):
                if cnt[i]:
                    cls_pad[i, :cnt[i]] = cls_list[i]

        # 关系头整批前向；损失仍逐图计算（七项损失内含逐图归一与 swap 对）
        o = rel._forward_pairs(rel._batch_feats[0], boxes_pad, counts, gt_list=gt_list,
                               box_cls=cls_pad)

        sums = {k: None for k in self.LAMBDA_RA}
        n_geo_imgs = 0
        nce_vals = []
        field_diag = {}
        # 训练期场指标（FieldMetricAccumulator）只用于 epoch 末的 [FieldMetrics] 日志行，
        # 不进损失。逐图全量累计占主线程约 7%（py-spy 实测），改为每 N 个 batch 采样一次；
        # YOLOVO_FIELD_METRICS_EVERY=1 恢复逐 batch 全量统计。
        self._batch_counter = getattr(self, "_batch_counter", -1) + 1
        fm_every = getattr(self, "_field_metrics_every", None)
        if fm_every is None:
            import os
            fm_every = max(int(os.environ.get("YOLOVO_FIELD_METRICS_EVERY", "8") or 1), 1)
            self._field_metrics_every = fm_every
        do_field_metrics = (self._batch_counter % fm_every) == 0
        for i in range(n_imgs):
            if cnt[i] < 2:
                continue
            out_i = {
                "logits": o["logits"][i],
                "sub_idx": o["sub_idx"][i], "obj_idx": o["obj_idx"][i],
                "valid_mask": o["valid_mask"][i], "rel_logits": o["rel_logits"][i],
                "boxes": boxes_pad[i:i + 1],
                "geo_scores": o["geo_scores"][i:i + 1],
                "zs": o["zs"][i:i + 1], "zo": o["zo"][i:i + 1],
                # 本图每个槽位「全部」GT 谓词的 (slot, pred) 稀疏索引；多谓词对不再 last-wins
                "slot_gt": (o["gt_slots"][i] if o.get("gt_slots") is not None else None),
            }
            if "logits_raw" in o:
                out_i["logits_raw"] = o["logits_raw"][i]
            # 未注入类别的 v_obj：obj 对齐损失用它，否则 cls_proj 自答
            if "v_obj_base" in o:
                out_i["v_obj_base"] = o["v_obj_base"][i:i + 1]
            if "field" in o:
                out_i["cand_idx"] = o["cand_idx"][i]
                out_i["field_cos"] = o["field_cos"][i]
                out_i["field"] = {k: v[i] for k, v in o["field"].items()}
            terms = rel._losses(
                out_i, [gt_list[i]], o["q"][i:i + 1], o["is_gt"][i:i + 1],
                o["sub_idx"][i:i + 1], o["obj_idx"][i:i + 1],
                o["valid_mask"][i:i + 1], o["v_obj"][i:i + 1], boxes_pad[i:i + 1],
                o["n_boxes"], cls_list[i], box_counts=counts[i:i + 1])
            n_geo_imgs += 1
            nce_vals.append(terms["nce"].detach())
            if out_i.get("field") is not None:
                from ultralytics.models.yolovo.relate.field_metrics import FieldMetricAccumulator
                if self._field_metrics is None:
                    self._field_metrics = FieldMetricAccumulator(
                        int(rel.vocab_head.W.shape[0]))
                if do_field_metrics:
                    self._field_metrics.update(out_i, [gt_list[i]])
                for k, v in o.get("field_diag", {}).items():
                    field_diag[k] = field_diag.get(k, 0.0) + v[i]  # 张量累加，末尾一次取值
            for k in self.LAMBDA_RA:
                v = terms[k] * self.LAMBDA_RA[k]
                sums[k] = v if sums[k] is None else sums[k] + v

        self._rel_log = {
            "n_geo_imgs": n_geo_imgs,
            "n_pred_pairs": int(torch.stack(nce_vals).ne(0).sum()) if nce_vals else 0,
        }
        if n_geo_imgs:
            self._rel_log.update({k: float(v) / n_geo_imgs for k, v in field_diag.items()})
        if n_geo_imgs == 0:
            p = next(rel.parameters())
            return 0.0 * p.sum(), torch.zeros(4, device=device)
        loss = sum(s / n_geo_imgs for s in sums.values())
        items = torch.stack([sums["geo"].detach(), sums["rel"].detach(),
                             sums["nce"].detach(), sums["swap"].detach()]) / n_geo_imgs
        return loss, items

    _rel_log = {}

    def label_loss_items(self, loss_items=None, prefix="train"):
        """train 侧四项 + 参与计数；val 侧返回空（验证不算关系损失，恒 0 不写 results.csv）。"""
        if prefix == "val":
            return []  # list：setup 侧 keys+[] 合法；验证器侧已做类型自适应（validator.py:253）
        keys = [f"{prefix}/{k}" for k in self.loss_names]
        if loss_items is not None:
            LOGGER.info(
                f"[RelateTrain] imgs(geo/exist)={self._rel_log.get('n_geo_imgs', 0)} "
                f"pairs(pred)={self._rel_log.get('n_pred_pairs', 0)} "
                f"field={ {k: round(v, 5) for k, v in self._rel_log.items() if k.startswith(('gamma', 'delta', 'w_bg'))} }"
            )
            if self._field_metrics is not None:
                LOGGER.info("[FieldMetrics] " + " ".join(
                    f"{k}={v:.4f}" for k, v in self._field_metrics.compute().items()))
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys

    # ------------------------------------------------------------- validator

    def get_validator(self):
        """返回关系验证器（照抄 val(coco=True) 编排）。"""
        from ultralytics.models.yolovo.relate.val import make_relate_validator

        task = self.args.task if self.args.task in ("detect", "segment", "obb", "pose") else "detect"
        v_cls = make_relate_validator(task)
        # 顺序必须与 _compute_loss 返回的 items 一致：[geo, rel, nce, swap]
        self.loss_names = ("geo", "rel", "nce", "swap")
        v = v_cls(dataloader=self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
        v.relate_info = getattr(self.model, "relate_info", None) or {}
        self.validator = v
        return v

    # ------------------------------------------------------------- save ckpt

    def save_model(self):
        """重载-注入-再保存：EMA half 前保护 W buffer，注入 relate_info。"""
        from ultralytics.utils.patches import torch_load
        from ultralytics.utils.torch_utils import de_parallel

        model = de_parallel(self.model)
        relate_info = getattr(model, "relate_info", None)
        # 先调父类保存（EMA/last/best 逻辑），再注入 relate_info
        super().save_model()
        # 训练态续训权重：strip_optimizer 会把 best/last 收成推理权重（epoch=-1、
        # EMA/优化器清空、fp16），"下轮从 best 续训"对它们不成立——
        # 每轮另存一份带优化器状态的 train-checkpoint.pt（epoch 末覆盖）。
        if self.last is not None and hasattr(self, "optimizer") and self.optimizer is not None:
            # 只存关系层 fp32 state_dict（约 110 MB）而不是整模型 half 深拷贝（1.1 GB）：
            # 一级模型冻结、可由 YOLOVO_PRIMARY 重建；half 会把 alpha/W 取整到 fp16，
            # 续训时精度已丢。读取侧 _load_resume_ckpt 兼容旧格式。
            _, rel_m = _find_relate_module(model)
            train_ckpt = {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "rel_state": {k: v.detach().cpu() for k, v in rel_m.state_dict().items()},
                "ema_updates": int(getattr(self.ema, "updates", 0)) if getattr(self, "ema", None) else 0,
                "optimizer": self.optimizer.state_dict(),
                "train_args": vars(self.args) if hasattr(self.args, "__dict__") else dict(self.args),
            }
            _t2_save = __import__("torch").save
            _t2_save(train_ckpt, self.last.parent / "train-checkpoint.pt")
        import torch as _t

        for path in (self.last, self.best):
            if path is None or not path.exists():
                continue
            ckpt = torch_load(path, map_location="cpu")
            m = ckpt.get("ema") or ckpt.get("model")
            if m is not None and hasattr(m, "named_buffers"):
                # W buffer 还原 FP32（EMA half 保存会压精度）
                for bn, bv in m.named_buffers():
                    if ("vocab_head.W" in bn or "W_obj" in bn) and bv.dtype != _t.float32:
                        state = m.state_dict()
                        state[bn] = bv.float()
                        m.load_state_dict(state)
                        _t.save(ckpt, path)
            ckpt["relate_info"] = relate_info
            _t.save(ckpt, path)


def make_relate_trainer_class(task_trainer_cls):
    """按任务拼接 RelateTrainerMixin（先例 make_pruned_trainer_class）。"""
    return type(f"Relate{task_trainer_cls.__name__}", (RelateTrainerMixin, task_trainer_cls), {})

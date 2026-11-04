"""RelateNamespace: model.rel.train / model.rel.val / model.rel.predict 入口。

设计文档: docs/plans/2026-10-05-yolovo-relate-design.md
- 不设 task 参数，任务恒取自 checkpoint。
- rel.train 直交 trainer（照抄 finetrain 先例），不经 Model.train() 的 YAML 重建。
- rel.val 只接受带 relate_info 的权重；SGDet/SGCls 双协议。
- rel.predict 走 YOLOVORelatePredictor，capture 机制取特征。
"""

from __future__ import annotations


class RelateNamespace:
    """YOLOVO 关系二级任务命名空间。"""

    def __init__(self, parent):
        self._parent = parent

    # ------------------------------------------------------------------ utils

    def _require_relate_ckpt(self, api: str):
        """rel.val / rel.predict 只接受带 relate_info 的权重。"""
        info = getattr(self._parent.model, "relate_info", None)
        if not info and isinstance(self._parent.ckpt, dict):
            info = self._parent.ckpt.get("relate_info")
        if not isinstance(info, dict) or not info:
            raise RuntimeError(
                f"model.rel.{api} 只接受带 relate_info 的二级关系权重。"
                "请先在一级权重上运行 model.rel.train 接入关系层。"
            )
        return info

    def _build_relate_model(self, inputs, rnames_override=None, data_yaml=None,
                            field_cfg=None):
        """两步构建：加载的一级模型 + 程序化追加 YOLOVORelate。"""
        from ultralytics.models.yolovo.relate.wiring import (
            attach_relate_layer, resolve_objnames, resolve_rnames)

        model = self._parent.model
        # 已带关系层（继续训练场景）：复用；若显式要求不同的 field 配置则 fail-fast，
        # 避免“传了 field 却被静默忽略”。
        info = getattr(model, "relate_info", None)
        if info:
            if field_cfg is not None:
                want = field_cfg.to_dict() if hasattr(field_cfg, "to_dict") else dict(field_cfg)
                got = info.get("field_cfg")
                if got is not None and got != want:
                    raise RuntimeError(
                        "该 checkpoint 已带关系层且 field_cfg 不同；不能原地换场配置。"
                        "请从一级权重重新 attach。")
            return model
        rnames = resolve_rnames(rnames_override, data_yaml)
        attach_relate_layer(model, inputs, rnames, field_cfg=field_cfg,
                            obj_names=resolve_objnames(data_yaml))
        return model

    # ------------------------------------------------------------------ train

    def train(self, inputs=None, data=None, rnames=None, trainer=None,
              field=None, **kwargs):
        """关系层二级训练：只更新 YOLOVORelate。

        Args:
            inputs: 接线 YAML 路径或列表（["P3","P4","P5"]）。已带关系层的权重忽略此参。
            data: 关系数据 yaml（含 rnames 扩展键）。
            rnames: 显式谓词词表覆盖（默认读 data yaml）。
            trainer: 自定义 trainer 类。
            field: FieldConfig / dict / "off"|"glp"|"inject"|"mlp"；不放进
                trainer args（default.yaml 没有该键）。
        """
        self._parent._check_is_pytorch_model()
        self._parent._raise_if_pruned("rel.train")

        from ultralytics.cfg import TASK2DATA
        from ultralytics.utils import DEFAULT_CFG_DICT, RANK, YAML, checks
        from ultralytics.nn.tasks import attempt_load_one_weight

        checks.check_pip_update_available()
        overrides = YAML.load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self._parent.overrides
        custom = {
            "data": data or overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA.get(self._parent.task, "coco.yaml"),
            "model": self._parent.overrides["model"],
            "task": self._parent.task,
        }
        args = {**overrides, **custom, **kwargs, "mode": "train"}
        if RANK != -1:
            raise RuntimeError("model.rel.train 第一版仅支持单卡训练（DDP 子进程无法访问注入的关系模型）。")

        # 两步构建：一级模型已加载，追加关系层（冻结与优化器约束在 trainer 内）
        from ultralytics.models.yolovo.relate.field import FieldConfig

        field_cfg = None if field is None else FieldConfig.from_any(field)
        model = self._build_relate_model(inputs, rnames_override=rnames, data_yaml=data,
                                         field_cfg=field_cfg)

        from ultralytics.models.yolovo.relate.trainer import make_relate_trainer_class

        trainer_cls = trainer or make_relate_trainer_class(self._parent.task_map[self._parent.task]["trainer"])
        rel_trainer = trainer_cls(overrides=args, _callbacks=self._parent.callbacks)
        rel_trainer.model = model
        rel_trainer.hub_session = self._parent.session
        self._parent.trainer = rel_trainer
        rel_trainer.train()

        if RANK in {-1, 0}:
            ckpt = rel_trainer.best if rel_trainer.best.exists() else rel_trainer.last
            self._parent.model, self._parent.ckpt = attempt_load_one_weight(ckpt)
            self._parent.overrides = self._parent.model.args
            self._parent.metrics = getattr(rel_trainer.validator, "metrics", None)
        return self._parent.metrics

    # -------------------------------------------------------------------- val

    def val(self, data=None, protocol: str = "sgdet", field_fuse: Optional[str] = None, **kwargs):
        """关系验证：SGDet（NMS 框，默认）或 PredCls（GT 框 oracle，别名 sgcls）。

        本实现只吃框、不预测物体类别，所以文献里的 SGCls 与 PredCls 在这里同义。
        field_fuse="max" 只允许场加分、"raw" 完全不用场（消融），缺省沿用模型内的设置。
        """
        info = self._require_relate_ckpt("val")
        self._parent._check_is_pytorch_model()
        # field_fuse 只对本次验证生效：写死在模型上会让之后的 train/predict 沿用，
        # 而训练期 sigmoid/bg 读的正是 out["logits"]。结束后按原值恢复。
        restore_fuse = []
        if field_fuse is not None:
            if field_fuse not in ("field", "max", "raw"):
                raise ValueError(f"field_fuse 只接受 'field' | 'max' | 'raw'，收到 {field_fuse!r}。")
        for m in self._parent.model.modules():
            if m.__class__.__name__ == "YOLOVORelate":
                restore_fuse.append((m, getattr(m, "field_fuse", "field")))
                if field_fuse is not None:
                    m.field_fuse = field_fuse

        from ultralytics.utils import DEFAULT_CFG_DICT

        if protocol == "predcls":
            protocol = "sgcls"
        if protocol not in ("sgdet", "sgcls"):
            raise ValueError(f"protocol 只接受 'sgdet' | 'sgcls' | 'predcls'，收到 {protocol!r}。")

        overrides = {k: v for k, v in {**self._parent.overrides, **kwargs}.items() if k != "protocol"}
        # Explicit data must override the checkpoint's train args (setdefault
        # silently kept the training yaml, so rel.val(data=subset) was ignored).
        overrides["data"] = data or overrides.get("data") or DEFAULT_CFG_DICT["data"]
        overrides["mode"] = "relval"
        if overrides.get("save_dir", None) is None and overrides.get("name", None) is None:
            overrides["name"] = f"val-rel-{protocol}"

        # 关系 dataloader（标准 loader 会丢 R 行 → n_gt=0）
        from types import SimpleNamespace
        from torch.utils.data import DataLoader
        from ultralytics.models.yolovo.relate.dataset import build_relate_dataset, relate_collate
        from ultralytics.data.utils import check_det_dataset

        data_cfg = check_det_dataset(overrides["data"])
        vargs = SimpleNamespace(
            # rect 必须与训练一致：训练是方形 640 letterbox，验证强制矩形会引入
            # 系统性的坐标偏移（1024×768 下图中心差 0.0586，59% 的框 >0.01）。
            # 取 ckpt 训练参数里的 rect，而不是写死 False。
            imgsz=overrides.get("imgsz", 640), rect=bool(overrides.get("rect", False)),
            cache=None, single_cls=False,
            task=self._parent.task, fraction=1.0, workers=overrides.get("workers", 2),
            mask_ratio=1, overlap_mask=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
            flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
            close_mosaic=10,
        )
        vds = build_relate_dataset(
            vargs, data_cfg.get("val") or data_cfg.get("test"), overrides.get("batch", 1),
            data_cfg, mode="val", stride=32, rnames=info.get("rnames"),
        )
        vloader = DataLoader(vds, batch_size=overrides.get("batch", 1), shuffle=False,
                             num_workers=overrides.get("workers", 2), collate_fn=relate_collate)

        from ultralytics.models.yolovo.relate.val import make_relate_validator

        validator = make_relate_validator(self._parent.task)(
            dataloader=vloader, save_dir=None, args=overrides, _callbacks=self._parent.callbacks
        )
        validator.protocol = protocol
        validator.relate_info = info
        try:
            stats = validator(model=self._parent.model)
        finally:
            for m, prev in restore_fuse:
                m.field_fuse = prev
        self._parent.metrics = getattr(validator, "metrics", None)
        return stats

    # ---------------------------------------------------------------- predict

    def predict(self, source, rnames=None, conf=None, **kwargs):
        """关系预测：检测 + NMS → top-60 → 关系打分，返回 RelateResults。"""
        info = self._require_relate_ckpt("predict")
        self._parent._check_is_pytorch_model()

        overrides = {**self._parent.overrides, **kwargs}
        if conf is not None:
            overrides["conf"] = conf

        from ultralytics.models.yolovo.relate.predict import YOLOVORelatePredictor

        predictor = YOLOVORelatePredictor(overrides=overrides)
        predictor.relate_info = info
        predictor.rnames_override = rnames
        # capture 标志由 predictor 在 stream 循环内开/关
        return predictor(source=source)

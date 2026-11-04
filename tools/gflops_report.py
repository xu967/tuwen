"""YOLOVO（检测主干）+ GLP 关系头 的 GFLOPs 统计。

只用本仓已有组件：YOLOVO 加载 checkpoint、rel 头 `_forward_pairs` 整条推理链，
FLOPs 由 torch 官方 `FlopCounterMode` 按 aten 算子累加（conv/mm/addmm/bmm 全覆盖，
不依赖 thop 只数 Conv/Linear 的估算）。

用法（仓库根目录）::

    export LD_LIBRARY_PATH=/root/rivermind-data/cudnn9/nvidia/cudnn/lib:$LD_LIBRARY_PATH
    python tools/gflops_report.py \
        runs/test/glp_proxy50k_v1_e100/weights/last.pt \
        --imgsz 640 --prompts 80 --boxes 60

默认在 CPU 上跑（GFLOPs 与设备无关），避免和正在训练的任务抢显存。
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flops(fn) -> float:
    """跑一次 fn()，返回 GFLOPs（FlopCounterMode 口径）。"""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    fc = FlopCounterMode(display=False)
    with torch.no_grad():
        with fc:
            fn()
    return fc.get_total_flops() / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=os.path.join(
        _ROOT, "runs/test/glp_proxy50k_v1_e100/weights/last.pt"), help="含 YOLOVORelate 头的 checkpoint")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--prompts", type=int, default=80, help="检测头的文本提示数（= max_texts）")
    ap.add_argument("--boxes", type=int, default=60, help="送进关系头的框数（NMS 后 top-N）")
    ap.add_argument("--device", default="cpu", help="cpu / 0；GFLOPs 与设备无关")
    args = ap.parse_args()

    sys.path.insert(0, _ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from ultralytics import YOLOVO
    from ultralytics.models.yolovo.relate.head import YOLOVORelate

    torch.manual_seed(0)
    model = YOLOVO(args.ckpt)
    net = model.model.float().eval()
    dev = torch.device(args.device)
    net.to(dev)

    names = [str(v) for v in net.names.values()]
    net.set_input_text(names[: args.prompts])
    x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=dev)

    rel = None
    for m in net.model:
        if isinstance(m, YOLOVORelate):
            rel = m
    if rel is None:
        print("该 checkpoint 没有 YOLOVORelate 层，只统计检测器。", file=sys.stderr)

    cap = {}
    if rel is not None:
        rel.register_forward_pre_hook(
            lambda mod, a: cap.__setitem__("feats", [t.detach() for t in a[0][1:]]))

    det = _flops(lambda: net(x))
    params = sum(p.numel() for p in net.parameters())
    print(f"checkpoint        : {args.ckpt}")
    print(f"params            : {params/1e6:.2f} M")
    print(f"输入              : 1x3x{args.imgsz}x{args.imgsz}, prompts={args.prompts}")
    print(f"检测器 (backbone+neck+head) : {det:.3f} GFLOPs")

    if rel is None:
        return 0
    # 关系头用的是 feats[base_idx]（head.py 的 base_idx），不是第 0 个接线层
    base = cap["feats"][getattr(rel, "base_idx", 0)]
    g = torch.Generator().manual_seed(0)
    boxes = torch.rand(1, args.boxes, 4, generator=g, device=dev) * 0.5 + 0.25
    boxes[..., 2:] = boxes[..., 2:] * 0.4 + 0.05

    print(f"关系头            : {sum(p.numel() for p in rel.parameters())/1e6:.2f} M params, "
          f"V={rel.n_predicates}, N={args.boxes}")
    rows = {}
    for budget, tag in ((rel.eval_budget, f"推理 budget={rel.eval_budget}"),
                        (None, f"训练 budget={rel.geo_budget}/{rel.final_budget}")):
        off = _flops(lambda: rel._forward_pairs(base, boxes, None, None, eval_budget=budget))
        rows[tag] = off
        print(f"  {tag:<24}: {off:.3f} GFLOPs")
        print(f"  {'':<24}  合计 {det + off:.3f} GFLOPs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

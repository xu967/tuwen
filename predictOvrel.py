"""YOLOVO 关系预测入口：model.rel.predict 消费 rel best.pt（检测 + 关系打分）。"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

from ultralytics import YOLOVO

model = YOLOVO("/root/ultralyticstxt/runs/test/yolovo-relate-1/weights/best.pt")

results = model.rel.predict(
    source=[
        "/root/datasets/M3FD_split_related/images/val/00063.png",
        "/root/datasets/M3FD_split_related/images/val/00091.png",
        "/root/datasets/M3FD_split_related/images/val/00099.png",
    ],
    imgsz=640,
    conf=0.25,
    iou=0.45,
    save=True,
    project="runs/test",
    name="yolovo-rel-predict",
)

# 打印每图关系三元组
for i, res in enumerate(results):
    rel = res.relations
    print(f"\n=== image {i}: {res.path} | boxes={len(res.boxes)} | relations={len(rel)} ===")
    if not len(rel):
        continue
    for e in rel.tolist():
        s, o, p = int(e[0]), int(e[1]), int(e[2])
        cs = res.names.get(int(res.boxes.cls[s]), "?") if s < len(res.boxes) else "?"
        co = res.names.get(int(res.boxes.cls[o]), "?") if o < len(res.boxes) else "?"
        pred = res.rnames.get(p, str(p))
        print(f"  {cs} --[{pred}]--> {co}  (fused={e[3]:.3f})")

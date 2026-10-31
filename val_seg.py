import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics import YOLOVO

# YOLOVO 实例分割验证脚本（与 detect 版 val.py 并列）
model = YOLOVO("runs/seg/yolovo-seg-train-30/weights/best.pt")

model.val(
    data="/root/datasets/tree_seg/tree_seg/data.yaml",
    imgsz=640,
    batch=8,
    workers=8,
    yolovo_debug=True,
    project="runs/seg",
    name="yolovo-seg-val-30",
)

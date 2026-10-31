import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics import YOLOVO

model = YOLOVO("runs/obb/yolovo-obb-train-5/weights/best.pt")

model.val(
    data="/root/datasets/obb/obb.yaml",
    imgsz=640,
    batch=16,
    workers=8,
    yolovo_debug=True,
    project='runs/obb',
    name='yolovo-obb-val-5'
)

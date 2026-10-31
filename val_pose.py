import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics import YOLOVO

model = YOLOVO("runs/pose/yolovo-pose-train-5/weights/best.pt")

model.val(
    data="/root/datasets/coco-pose/data.yaml",
    imgsz=640,
    batch=16,
    workers=8,
    yolovo_debug=True,
    project='runs/pose',
    name='yolovo-pose-val-5'
)

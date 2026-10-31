import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics import YOLOVO

model = YOLOVO("runs/pose/yolovo-pose-train-5/weights/best.pt")

# COCO-Pose数据集类别
pose_names = ["person"]

model.predict(
    source=[
        "/root/datasets/coco-pose/images/val/000000002473.jpg",
        "/root/datasets/coco-pose/images/val/000000002532.jpg",
        "/root/datasets/coco-pose/images/val/000000004134.jpg",
        "/root/datasets/coco-pose/images/val/000000006894.jpg",
    ],
    text=pose_names,
    imgsz=640,
    conf=0.25,
    iou=0.45,
    save=True,
    yolovo_debug=True,
    project="runs/pose",
    name="yolovo-pose-predict",
)

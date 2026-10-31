import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics import YOLOVO

model = YOLOVO("runs/obb/yolovo-obb-train-5/weights/best.pt")

# OBB数据集类别
obb_names = ["bridge", "harbor", "storage_tank"]

model.predict(
    source=[
        "/root/datasets/obb/images/test/1001.png",
        "/root/datasets/obb/images/test/1003.png",
        "/root/datasets/obb/images/test/1016.png",
        "/root/datasets/obb/images/test/1017.png",
    ],
    text=obb_names,
    imgsz=640,
    conf=0.25,
    iou=0.45,
    save=True,
    yolovo_debug=True,
    project="runs/obb",
    name="yolovo-obb-predict",
)

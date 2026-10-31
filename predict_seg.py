import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics import YOLOVO

# YOLOVO 实例分割预测脚本（与 detect 版 predict.py 并列）
model = YOLOVO("runs/seg/yolovo-seg-train-30/weights/best.pt")

# tree_seg 类别为单类 "tree"，开放词汇场景下通过 text prompt 注入
tree_seg_prompts = ["tree"]

test_dir = "/root/datasets/tree_seg/tree_seg/images/test"
test_images = [
    os.path.join(test_dir, "DJI_20240513164329_0003_D_1_part_1.png"),
    os.path.join(test_dir, "DJI_20240513164335_0006_D_1_part_1.png"),
    os.path.join(test_dir, "DJI_20240513164339_0008_D_1_part_0.png"),
    os.path.join(test_dir, "DJI_20240513164343_0010_D_1_part_2.png"),
]

model.predict(
    source=test_images,
    text=tree_seg_prompts,
    imgsz=640,
    conf=0.25,
    iou=0.45,
    save=True,
    yolovo_debug=True,
    project="runs/seg",
    name="yolovo-seg-predict",
)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ultralytics import YOLOVO
# 创建 YOLOVO 模型
if __name__ == "__main__" :
    model = YOLOVO("yolovo11n-obb.yaml")
    # model = YOLOVO('yolovo11n-obb.pt')
    # model.load("yolovo11n-obb.pt")

    # 训练配置
    model.train(
        data="/root/datasets/obb/obb.yaml",
        epochs=5,
        cache=True,
        imgsz=640,
        batch=16,
        workers=8,
        lr0=5e-4,
        # lrf=0.01,
        # device = '0,
        # optimize='AdamW',
        project="runs/obb",
        name="yolovo-obb-train-5",
        close_mosaic=1,
        # exist_ok=True,
        resume=False,
        # fraction=0.4,
        yolovo_debug=True,
    )

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ultralytics import YOLOVO
# 创建 YOLOVO 模型
if __name__ == "__main__" :
    # 关系二级训练（GLP）：设置 YOLOVO_RELATE_DATA 后启用；默认行为保持不变。
    rel_data = os.environ.get("YOLOVO_RELATE_DATA")
    if rel_data:
        model = YOLOVO(os.environ.get("YOLOVO_PRIMARY", "yolo11n.pt"))
        model.rel.train(
            inputs=os.environ.get("YOLOVO_RELATE_INPUTS", "P4").split(","),
            data=rel_data,
            field=os.environ.get("YOLOVO_FIELD", "glp"),
            epochs=int(os.environ.get("YOLOVO_EPOCHS", "1")),
            imgsz=640,
            batch=16,
            workers=8,
            lr0=1e-3,
            project="runs/test",
            name=os.environ.get("YOLOVO_NAME", "glp-relate"),
            exist_ok=True,
        )
    else:
        model = YOLOVO("yolovo11n.yaml")
        # model = YOLOVO('yolovo11n.pt')
        # model.load("yolovo11n.pt")

        # 训练配置
        model.train(
            data="/root/datasets/M3FD_split/data.yaml",
            epochs=100,
            cache=True,
            imgsz=640,
            batch=16,
            workers=8,
            lr0=5e-4,
            # lrf=0.01,
            # device = '0,
            # optimize='AdamW',
            project="runs/test",
            name="yolovo-train-1",
            close_mosaic=10,
            # exist_ok=True,
            resume=False,
            # fraction=0.4,
            yolovo_debug=True,
        )

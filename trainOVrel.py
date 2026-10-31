"""YOLOVO 关系二级训练入口（trainOVrel.py）。

设计文档: docs/plans/2026-10-05-yolovo-relate-design.md
用法: /home/zhizi/miniconda3/envs/mm/bin/python trainOVrel.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


from ultralytics import YOLOVO

if __name__ == "__main__":
    # 一级权重：M3FD detect 训练产物（mAP50 0.617），task 恒取自 checkpoint
    model = YOLOVO("/root/ultralyticstxt/runs/test/yolovo-train-2/weights/best.pt")

    # 关系二级训练：只更新 YOLOVORelate，一级全冻结
    model.rel.train(
        inputs=["P3", "P4", "P5"],                            # 接线；也可写 cfg yaml 路径
        data="/root/datasets/M3FD_split_related/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,                                              # 用户指定 16（v3 逐图重放下每图都被监督）
        workers=2,
        # amp=Flase,
        lr0=1e-3,
        project="runs/",
        name="yolovo-relate-1",
        exist_ok=True,
    )

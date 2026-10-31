import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics import YOLOVO

# YOLOVO 实例分割训练脚本（与 detect 版 train.py 并列）
if __name__ == "__main__":
    # 从 yolovo11-seg.yaml 构建实例分割模型（open-vocab segment 头）
    model = YOLOVO("yolovo11-seg.yaml")
    # model = YOLOVO("runs/seg/yolovo-seg-train-30/weights/best.pt")  # 续训

    # tree_seg: 多模态(RGB+IR)实例分割数据集，类别 names={0: tree}
    # YOLOVO v1 使用 RGB+Text，IR 模态由 mm/ 框架保留接入点。
    model.train(
        data="/root/datasets/tree_seg/tree_seg/data.yaml",
        epochs=30,          # 正式训练：30 个 epoch
        cache=False,
        imgsz=640,
        batch=8,
        workers=8,
        lr0=5e-4,
        # optimize="AdamW",
        project="runs/seg",
        name="yolovo-seg-train-30",
        close_mosaic=5,
        exist_ok=True,
        resume=False,
        # fraction=0.2,     # 需要更快的冒烟测试可开启子集
        yolovo_debug=True,
    )

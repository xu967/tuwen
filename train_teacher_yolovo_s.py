# YOLOVO-s 教师训练脚本
# 用于蒸馏实验：训练一个 s 尺寸的 YOLOVO 教师模型

from ultralytics import YOLOVO


if __name__ == "__main__":
    model = YOLOVO("ultralytics/cfg/models/yolovo/yolovo11.yaml")
    model.train(
        data="/home/zhizi/work/multimodel/ultralyticmm/data.yaml",
        epochs=100,
        batch=16,
        scale="s",
        device="0",
        cache=True,
        exist_ok=True,
        project="ResTest",
        name="DistillExp/teacher-yolovo-s",
    )

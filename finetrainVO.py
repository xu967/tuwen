# YOLOVO 剪枝后微调训练示例脚本
#
# 注意事项:
# - 只接受带 prune_info 标记的剪枝权重 (由 model.prune(save_dir=...) 生成)
# - 不允许把原始 best.pt 当成 finetrain() 输入
# - 不要传 scale / model_scale 参数，剪枝后结构必须直接来自 checkpoint
# - 参数风格与 train.py 一致，直接改参数、直接运行

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ultralytics import YOLOVO


if __name__ == "__main__":
    model = YOLOVO("./YOLOVO-pruned/pruned.pt")

    model.finetrain(
        data="/root/datasets/M3FD_split/data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        device="0",
        cache=True,
        exist_ok=True,
        project="runs/prune",
        name="yolovo-pruned-l1-r30-finetrain",
    )

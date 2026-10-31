"""YOLOVO 关系验证入口：model.rel.val 消费 rel best.pt（完整验证 + 关系双联图）。"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

from ultralytics import YOLOVO

model = YOLOVO("runs/test/yolovo-relate-1/weights/best.pt")

stats = model.rel.val(
    data="/mnt/SSD/work/projecttest/M3fd_related/data.yaml",
    plots=True,
    project="runs/test",
    name="yolovo-rel-val",
)

print("\n=== rel.val 独立验证结果 ===")
for k, v in stats.items():
    print(f"  {k}: {v}")

# YOLOVO 剪枝脚本（零注册引擎，视觉侧）
#
# 引擎说明：依赖发现全自动（算子级追踪）。YAML `text:` 整段与 MobileCLIP
# 文本塔不剪；其余模块要么被正确剪枝，要么被保守跳过并在报告中说明原因。
# 覆盖 YOLOVO 全任务：detect / segment / obb / pose。
from pathlib import Path

from ultralytics import YOLOVO

if __name__ == "__main__":
    # 正式剪枝时，建议直接加载已经训练完成的权重，而不是空白 YAML。
    model = YOLOVO("/root/ultralyticstxt/runs/test/yolovo-train-14/weights/best.pt")
    # model = YOLOVO("yolovo11n.yaml")

    # 剪枝结果会在该目录下生成：
    # 1. pruned.pt          -> 运行时权威结构（后续 finetrain/val/predict 用它）
    # 2. pruned.yaml        -> 完整可加载配置；配合 state_dict 可重建剪后结构
    # 3. prune_spec.json    -> 逐单元剪枝决策 + 保守跳过原因清单（审计）
    # 4. prune_report.json  -> 引擎完整报告（save_report=True 时）

    # ===== 结构化剪枝（物理移除通道，直接得到小模型）=====
    model.prune(
        method="l1",      # 可选: 'l1', 'l2', 'lamp', 'bn', 'random'
        ratio=0.30,
        save_dir="./YOLOVO-pruned/",
        round_to=8,
        min_ch=8,
        save_report=True,
        # scope=[0, 1, 2, ...],  # 可选：限制剪枝的视觉层号范围
    )

    # ===== 非结构化 / 2:4 半结构化稀疏（结构不变，权重置零；text 段保持稠密）=====
    # model.prune_sparse(sparsity=0.5, pattern="unstructured", save_dir="./YOLOVO-sparse/")
    # model.prune_sparse(pattern="semi24", save_dir="./YOLOVO-24/")

    # 剪枝后可直接微调（无需落盘重载；DDP 多卡时请从 pruned.pt 重载）：
    # model.finetrain(data="data.yaml", epochs=50, device="0")
    #
    # 或从落盘产物开始：
    # model = YOLOVO("./YOLOVO-pruned/pruned.pt")
    # model.finetrain(data="data.yaml", epochs=50, device="0")
    #
    # 注意：剪枝模型调用 train() 会直接报错（防止按 YAML 重建丢权重），必须用 finetrain()。

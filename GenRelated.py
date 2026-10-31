"""YOLOVO 关系数据集生成入口：RelGen（prior 后端）消费源检测数据集 data.yaml。"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

from ultralytics.models.yolovo.relate.gen import RelGen

gen = RelGen(
    tau=0.9,                # 词表累计质量截断（实测 0.56 会截灭语义谓词）
    # save_dir="",  # None 时默认 {源}_related
    seed=0,
)

stats = gen.run("/root/datasets/M3FD_split/data.yaml")

print("\n=== RelGen 生成结果 ===")
print(stats)
for sp, d in stats.per_split.items():
    print(f"  {sp}: imgs={d['imgs']} boxes={d['boxes']} rels={d['rels']} "
          f"no_rel(few/gen/tau)={d.get('no_rel_few_boxes', 0)}/{d.get('no_rel_gen_empty', 0)}/{d.get('no_rel_tau_dropped', 0)}")
print(f"  rnames({len(stats.rnames)}): {stats.rnames}")

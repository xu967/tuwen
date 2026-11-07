# BARV 难负例对照

cache `/root/rivermind-data/YOLOVOmodal_3871538535/data/barv_cache_contact/queries.npz`，seed 42，train_core 86108 / dev 9388 / test 7278；负例来源 {'contact_other': 30239, 'contact_unlabeled': 45428}

| 模型 | dev AUROC | test AUROC | test vs contact_other | test vs contact_unlabeled | test bacc |
|---|---:|---:|---:|---:|---:|
| geometry_lr | 0.6881 | 0.6803 | 0.7064 | 0.6621 | 0.6290 |
| no_visual_mlp | 0.7548 | 0.7500 | 0.7819 | 0.7276 | 0.6799 |
| visual | 0.7740 | 0.7767 | 0.8096 | 0.7536 | 0.6986 |

test 按谓词 AUROC（visual / geometry_lr）：

- holding: 0.7151 / 0.6192
- carrying: 0.8188 / 0.7189
- wearing: 0.7170 / 0.5102
- sitting on: 0.7642 / 0.6712
- standing on: 0.7415 / 0.6063
- riding: 0.8357 / 0.7557
- touching: 0.5717 / 0.6661
- leaning on: 0.7488 / 0.6212
- lying on: 0.8703 / 0.7297

pilot（远离负例，`runs/barv/pilot_seed42`）：visual test AUROC 0.9960，geometry_lr 0.9942。

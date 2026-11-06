# BARV Cached Baseline Pilot

## Run: coarse visual scorer vs geometry-only

- Date: 2026-11-06 (Asia/Shanghai)
- Goal: run the first cached-feature baseline and test whether image features add signal beyond the current geometry-derived label rule.
- Data: `data/barv_cache/queries.npz`; 85,373 queries from 28,545 images; 44,923 positives and 40,450 rule-generated negatives. Split counts: train_core 59,992; train_teacher 6,687; dev 6,657; calib 5,110; test 4,998; psgval 1,929. Image UID intersections across all six splits were zero.
- Environment: NVIDIA RTX 4080 SUPER; Python 3.11; PyTorch 2.5.1+cu124; OpenCV 4.13.0.92; scikit-learn 1.8.0. A temporary NumPy 2.2.6 path overlay was required because the base environment has NumPy 1.26.4, which cannot import the installed OpenCV build. No global packages were changed.

### Coarse visual scorer

- Config: `configs/barv_psg.yaml`; seed 42; 30 epochs; batch 4,096; learning rate 0.001; weight decay 0.01; early-stop patience 8; trained on train_core, selected on dev.
- Best dev result: epoch 29, AUROC 0.9974.
- Held-out test: accuracy 0.9818; balanced accuracy 0.9820; macro-F1 0.9817; Brier 0.01464; ECE 0.00486; AUROC 0.9960; AUPRC 0.99735.
- Artifact: `runs/barv/pilot_seed42/relation.pt`; history: `runs/barv/pilot_seed42/relation_history.json`.

### Geometry-only baseline

- Trained with `barv.train.train_prior_baseline` on the same train_core split; selected/inspected on dev and evaluated once on test.
- Dev AUROC: 0.99385.
- Held-out test: accuracy 0.9750; balanced accuracy 0.9756; macro-F1 0.9749; Brier 0.02056; ECE 0.01372; AUROC 0.99420; AUPRC 0.99574.
- Artifact: `runs/barv/pilot_seed42/geometry_only.pkl`; dev metrics: `runs/barv/pilot_seed42/geometry_only_dev.json`.

## Interpretation

The image-based model improves test AUROC by only 0.0018 over geometry-only, while geometry-only already reaches 0.9942. Positives are official PSG triples, but negatives are generated using `mask gap > threshold`; the scorer receives box coordinates, IoU, offsets and area features that can approximate this rule. This pilot therefore exposes a geometry shortcut risk. It does not establish semantic relation verification or ROI information gain.

Next experiment should use human-reviewed hard negatives matched by subject/object categories and box geometry, then rerun the geometry-only and visual baselines. Proceed to coarse/full/ROI oracle-gap and hardware-latency measurements only if visual evidence improves on that baseline. Do not train teacher/MVE from this cache result yet.

## Reproduction and limits

The training call used the existing `QueryCache`, `train_relation`, `train_prior_baseline`, `predict_logits` and `binary_metrics` functions directly. The README's `scripts/run_barv.py` and `tests/test_barv.py` do not exist, and the configured source PSG JSON and COCO image directory are absent, so this is a cache-level pilot only. The visual training and test evaluation completed and artifacts were saved; the one-off wrapper then exited with `KeyError: 'best_epoch'` while formatting its final status line because that key is not returned at the top level. The printed metrics above were produced before that reporting-only error.

This is one seed and one cached split. No bootstrap intervals, source-data rebuild, manually verified negative set, ROI microbenchmark, or full BARV policy evaluation was run.

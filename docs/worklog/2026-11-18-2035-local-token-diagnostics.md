# Phase 2 局部 Token 诊断实现

- 时间：2026-11-18 20:35
- 范围：关系识别局部 subject/object Token 提取与 CPU 单测
- 结论：完成

## 做了什么

- 新增 `ultralytics/models/yolovo/relate/local_tokens.py`，从运行时场景特征图按归一化 `cxcywh` 框采样 subject/object 网格 Token，并返回采样坐标和有效性 mask。
- 新增 `tests/test_local_tokens.py`，覆盖输出形状、坐标/特征对齐、Letterbox 画布、无效框、pair padding、空 pair、反向梯度和非法输入。
- 将改动放在基于 `origin/main` 的独立分支 `feat/local-token-diagnostics-20261118`，避免带入当前工作分支的其他历史和工作区变更。

## 没做什么

- 没有接入 `head.py`、`trainer.py`、`train.py` 或修改 E2/E4 的分数和训练配置。原因是 Phase 2 只验证 Token 提取与诊断接口。
- 没有实现 OT、Top-M 谓词候选、损失或可视化渲染工具。原因是这些属于后续阶段，当前输出的坐标与 mask 可作为可视化输入。
- 没有启动训练、推理或 GPU 任务。原因是本阶段仅需 CPU 单测。
- 初次本地提交时没有推送到 GitHub。原因是远端仓库 API 显示 `xu967/tuwen` 为公开仓库，等待用户确认是否允许公开发布。用户确认后已按下方更新推送。

## 待确认

- 公开推送权限已确认，分支推送完成。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python tests/run_tests.py tests/test_local_tokens.py` | 初次导入失败，当前 shell 的 NumPy ABI 与 OpenCV 不兼容 |
| `PYTHONPATH=/root/rivermind-data/envs/numpy2_overlay /opt/conda/bin/python tests/run_tests.py tests/test_local_tokens.py` | 通过，4 passed，0 failed，0 skipped |
| 独立分支 CPU 验证，设置 NumPy overlay 与 CUDNN 动态库路径后运行 4 个新测试 | 通过，4 passed，0 failed |
| `awk 'length($0) > 120 ...'` 检查新增 Python 文件 | 通过，无超长行输出 |
| `nvidia-smi -L` | 检出 RTX 2080 Ti，本阶段没有启动 GPU 工作 |

## 风险与遗留

- 当前提取器按每个对象框的规则网格采样，尚未通过验证其是否比全局池化或 Cross-Attention 更有效。
- 默认网格为 `4×4`。实际 Token 数及代价矩阵内存仍需结合后续 batch、pair 数和谓词候选数做预算。
- 尚未在训练路径验证提取器对关系识别指标的影响，后续需按实验方案进行消融。

## 更新 2026-11-18 20:39

- 用户确认将独立分支发布到公开仓库 `xu967/tuwen`。
- 已推送 `feat/local-token-diagnostics-20261118`，提交 `04ae8da203c467152d131afd75b0dcd68a8aaa66` 与远端分支一致。
- 推送后 `git status --short --branch` 显示工作区干净，分支跟踪 `origin/feat/local-token-diagnostics-20261118`。
- 本次更新仅记录公开推送状态，没有源代码变更。

# MCOA Eye-Level 运行说明

## 1. 链路定位

这条链路属于公开数据过渡阶段的 V2 实验线：

- 使用 MCOA 的 AS-OCT 数据
- 以 eye-level 多帧样本组织训练
- 用于验证“按眼组织样本 + 多图聚合”的最小工程闭环

注意：

- 它不是最终的 vault 主任务
- 当前任务是 eye-level 分类，不是最终的 vault 回归
- 它的作用是为后续真实临床数据接入和多模态主线铺路

## 2. 正式运行环境

当前项目正式环境：

- 环境名：`cv_mamba_project`
- 解释器路径：`D:\tools\anaconda\envs\cv_mamba_project\python.exe`

当前已确认该环境中可用的核心依赖：

- `Python 3.9.25`
- `torch 2.5.1`
- `torchvision 0.20.1`
- `pandas 2.3.3`
- `Pillow 11.1.0`

说明：

- 后续正式实验和正式验证优先以这个环境为准
- 不要把默认 `python` 缺少 `torch` 当作当前项目的正式阻塞结论

## 3. eye-level manifest 生成

使用脚本：

- `scripts/build_mcoa_eye_manifest.py`

输入：

- image-level manifest，例如：
  - `data/manifests/mcoa_manifest_small.csv`
  - `data/manifests/mcoa_manifest_medium.csv`

输出：

- eye-level manifest，例如：
  - `data/manifests/mcoa_eye_manifest_small.csv`
  - `data/manifests/mcoa_eye_manifest_medium.csv`

当前已确认的状态：

- `mcoa_manifest_small.csv`
  - 在 eye-level 下当前不可用
  - 原因是同一只眼的切片跨了 `train/val`
  - 按当前默认冲突策略会被全部丢弃
- `mcoa_manifest_medium.csv`
  - 是当前可用入口
  - 已完成最小运行验证

## 4. 最小运行命令

先生成 eye-level manifest：

```powershell
D:\tools\anaconda\envs\cv_mamba_project\python.exe scripts/build_mcoa_eye_manifest.py --input_manifest data/manifests/mcoa_manifest_medium.csv --output_manifest data/manifests/mcoa_eye_manifest_medium.csv
```

再运行最小 smoke test：

```powershell
D:\tools\anaconda\envs\cv_mamba_project\python.exe scripts/pretrain_mcoa_eye.py --manifest_path data/manifests/mcoa_eye_manifest_medium.csv --epochs 1 --batch_size 2 --num_workers 0 --max_slices 8
```

这是当前已经实跑通过的正式环境命令。

## 5. 运行产物位置

日志文件：

- `artifacts/logs/`

checkpoint：

- `artifacts/checkpoints/`

当前 eye-level 训练脚本会保存：

- `artifacts/checkpoints/mcoa_eye_latest.pth`
- `artifacts/checkpoints/mcoa_eye_best.pth`

## 6. 日志文件记录内容

当前 `scripts/pretrain_mcoa_eye.py` 会把最小实验信息写入日志文件，至少包括：

- 解释器路径
- 实际执行命令
- manifest 路径
- 关键超参数
  - `epochs`
  - `batch_size`
  - `num_workers`
  - `max_slices`
  - `lr`
  - 是否使用 ImageNet pretrain
- train / val 结果
  - `train_loss`
  - `val_loss`
  - `val_accuracy`
- checkpoint 保存路径

日志文件命名风格示例：

- `mcoa_eye_smoketest_mcoa_eye_manifest_medium_YYYYMMDD_HHMMSS.log`

## 7. 当前已知限制 / 注意事项

- 当前是 eye-level 分类，不是最终 vault 回归
- 当前主要基于 `mcoa_manifest_medium.csv`
- `mcoa_manifest_small.csv` 不适合作为当前 eye-level 训练入口
- 当前模型是简单的 `2D backbone + mean pooling`
- 当前链路已经完成最小运行验证，但仍然只是公开数据过渡线
- 当前 eye-level 分组依赖现有文件名规则和 manifest 构建脚本，不应把它误认为最终临床数据接口

## 8. 适合什么时候看这份文档

这份文档最适合用于：

- 重新跑 MCOA eye-level 实验前快速查命令
- 查看日志和 checkpoint 位置时
- 把这条链路交接给别人或另一个 AI 助手时

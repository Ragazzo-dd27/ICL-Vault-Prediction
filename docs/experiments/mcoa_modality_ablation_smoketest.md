# MCOA 单模态 / 双模态对照与缺失模态验证

## 1. 实验目的

这轮实验不是为了追求最好指标，而是做一轮真实性检查：

- 比较 `OCT-only`、`ASP-only`、`OCT+ASP`
- 验证当前 `has_oct / has_asp` 缺失模态路径是否真的能跑通

正式环境：

- `cv_mamba_project`
- `D:\tools\anaconda\envs\cv_mamba_project\python.exe`

统一设置：

- manifest: `data/manifests/mcoa_multimodal_manifest_medium.csv`
- epochs: `1`
- batch_size: `2`
- num_workers: `0`
- max_slices: `8`
- seed: `42`
- lr: `1e-4`

## 2. 三组正式对照结果

| 模式 | train_loss | val_loss | val_accuracy | 日志 |
| --- | ---: | ---: | ---: | --- |
| `oct_only` | `0.7727` | `0.7003` | `0.5000` | `artifacts/logs/mcoa_oct_only_smoketest_mcoa_multimodal_manifest_medium_20260413_171324.log` |
| `asp_only` | `0.6986` | `0.6946` | `0.5000` | `artifacts/logs/mcoa_asp_only_smoketest_mcoa_multimodal_manifest_medium_20260413_171434.log` |
| `oct_asp` | `0.7681` | `0.6998` | `0.5000` | `artifacts/logs/mcoa_oct_asp_smoketest_mcoa_multimodal_manifest_medium_20260413_171434.log` |

对应 checkpoint：

- `oct_only`
  - `artifacts/checkpoints/mcoa_oct_only_latest.pth`
  - `artifacts/checkpoints/mcoa_oct_only_best.pth`
- `asp_only`
  - `artifacts/checkpoints/mcoa_asp_only_latest.pth`
  - `artifacts/checkpoints/mcoa_asp_only_best.pth`
- `oct_asp`
  - `artifacts/checkpoints/mcoa_oct_asp_latest.pth`
  - `artifacts/checkpoints/mcoa_oct_asp_best.pth`

## 3. 人为缺失模态验证

### 强制缺失 ASP

命令设置：

- `mode=oct_asp`
- `force_missing_asp_ratio=1.0`

结果：

- train / val 跑通
- checkpoint 写入成功
- 日志写入成功
- 指标：
  - `train_loss=0.7762`
  - `val_loss=0.7009`
  - `val_accuracy=0.5000`

日志：

- `artifacts/logs/mcoa_smoketest_mcoa_multimodal_manifest_medium_mode-oct_asp_miss-asp-1p0_20260413_171806.log`

### 强制缺失 OCT

命令设置：

- `mode=oct_asp`
- `force_missing_oct_ratio=1.0`

结果：

- train / val 跑通
- checkpoint 写入成功
- 日志写入成功
- 指标：
  - `train_loss=0.7211`
  - `val_loss=0.6947`
  - `val_accuracy=0.5000`

日志：

- `artifacts/logs/mcoa_smoketest_mcoa_multimodal_manifest_medium_mode-oct_asp_miss-oct-1p0_20260413_171922.log`

说明：

- 这两次缺失模态验证都完成了最小 smoke test
- 当前实现里的缺失模态兜底不是“只在代码结构上存在”，而是已经通过了真实训练路径验证

## 4. 务实结论

- 在这轮极小规模、1 epoch 的 smoke test 下，三组设置的 `val_accuracy` 都是 `0.5000`
- 当前还看不出 `OCT+ASP` 相比单模态有明确优势
- 但 `OCT+ASP` 至少没有明显更差，说明双分支融合在工程上是成立的
- 从这轮结果看，当前双模态线更像是“工程闭环已站住，方法优势尚未显现”
- 缺失模态路径已经完成真实 smoke test，说明：
  - ASP 缺失时链路可跑
  - OCT 缺失时链路可跑
  - `has_oct / has_asp` + 占位张量的最小方案是成立的

## 5. 当前阶段的意义

这轮实验带来的主要价值不是结论性指标，而是工程层面的确认：

- 公开数据代理线已经能支持单模态 / 双模态 / 缺失模态三类实验
- 未来如果接入真实临床多模态数据，可以继续沿用：
  - eye-level sample 组织
  - modality flag
  - 占位张量兜底
  - 统一日志与 checkpoint 记录方式

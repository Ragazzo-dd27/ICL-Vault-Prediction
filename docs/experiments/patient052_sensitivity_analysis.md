# patient_052 exclusion sensitivity analysis

## 背景

AS-OCT seed ensemble top-error analysis 显示，`patient_052` 双眼是当前 combined batch_01 + batch_02 test set 中最大的误差来源：

- `batch_02__patient_052_OS_20250501`
- `batch_02__patient_052_OD_20250501`

这两个样本均位于 low-vault 区间，模型预测明显高于人工核对的 POD1 vault label：

- patient_052 OS: label about 432 um, prediction about 790 um, absolute error about 358 um;
- patient_052 OD: label about 434 um, prediction about 744 um, absolute error about 310 um.

由于这两个样本对整体指标影响较大，我们进行了一个只用于理解结果稳定性的 sensitivity analysis：比较保留全部 test samples 与排除 `patient_052` 双眼后的 AS-OCT seed ensemble 指标变化。

需要强调：模型预测误差大本身不能作为删除样本的理由。

## 医生复查结论

医生已确认 `patient_052` 双眼的以下信息均无问题：

- POD1 vault 标签；
- AS-OCT 输入图像；
- 左右眼眼别；
- 检查日期；
- visit 对齐关系。

因此，`patient_052` 双眼不能从主结果中删除，应保留为真实有效的 top-error / model failure cases。

## 原始主结果

原始 test set 使用全部 25 只眼，并保留 `patient_052` 双眼：

- n = 25
- MAE = 96.33 um
- RMSE = 130.94 um
- R2 = 0.221

这是当前论文和实验记录中的主结果。

## 排除 patient_052 后的 sensitivity analysis

排除 `patient_052` 双眼后，test set 剩余 23 只眼：

- n = 23
- MAE = 75.65 um
- RMSE = 94.22 um
- R2 = 0.572

该结果说明 `patient_052` 双眼对整体 test MAE 和 RMSE 有明显影响，但该分析仅用于敏感性评估，不作为主结果。

## 被排除样本

| sample | abs error |
|---|---:|
| `batch_02__patient_052_OS_20250501` | 357.91 um |
| `batch_02__patient_052_OD_20250501` | 310.45 um |

两个样本均属于 batch_02，并且均为 low-vault 区间中的明显 overestimation case。

## Batch-Level 影响

排除 `patient_052` 双眼前后，batch-level MAE 变化如下：

| batch | original MAE | after exclusion MAE |
|---|---:|---:|
| batch_01 | 80.49 um | 80.49 um |
| batch_02 | 110.95 um | 70.36 um |

因此，`patient_052` 双眼主要影响 batch_02 的误差统计。batch_01 指标不受影响。

## 解释原则

本分析是 sensitivity analysis，不是主结果。医生已确认 `patient_052` 双眼的标签、图像、眼别、日期和 visit 对齐均无问题，因此主分析必须保留这两个样本。

该结果应解释为：当前 AS-OCT seed ensemble 在部分 low-vault cases 上存在明显 overestimation，而不是数据应被删除。

## 当前状态

`patient_052` 双眼已经完成医生复查，结论为数据无误。

当前处理决定：

- 保留 `patient_052` 双眼；
- 将其作为真实有效的模型失败病例；
- 在实验解释中将其描述为 low-vault overestimation case；
- 主结果继续使用完整 test set，即 n = 25。

## 后续方向

如果医生确认数据无误，后续应按以下方向处理：

- 保留 `patient_052` 作为模型失败病例；
- 分析 low-vault overestimation 的误差模式；
- 后续扩大 low-vault 样本量；
- 考虑 vault-range-aware evaluation；
- 探索 low-vault-sensitive training strategy，例如对 low-vault 区间做分层评估、重加权训练或专门的 calibration 分析。

如果未来发现其他样本存在真实标签或对齐问题，应按数据质控规则修正或排除，并重新生成 corrected analysis。但该规则不适用于当前已确认无误的 `patient_052` 双眼。

## 相关输出

Sensitivity analysis 输出目录：

`artifacts/reports/combined_batch_01_02/as_oct_error_analysis/patient052_sensitivity/`

主要文件：

- `patient052_exclusion_sensitivity_summary.csv`
- `excluded_patient052_samples.csv`
- `patient052_exclusion_batch_summary.csv`
- `patient052_exclusion_sensitivity_summary.md`
- `figures/mae_before_after_patient052_exclusion.png`
- `figures/pred_vs_gt_before_after_patient052_exclusion.png`

# Low-vault error analysis

## 1. 分析目的

本分析围绕当前 combined batch_01 + batch_02 AS-OCT seed ensemble 在 low-vault test cases 上的 overestimation 问题进行专项整理。分析目标不是重新训练模型，而是解释当前最强 AS-OCT baseline 的主要误差模式，并为论文 Discussion / Limitation 和后续实验设计提供依据。

所有结果均基于已有 manifest、prediction 和 error analysis outputs。未修改任何数据、manifest、prediction、checkpoint 或训练结果。

## 2. Vault Range 分布

按真实 POD1 vault label 将样本分为：

- low: `<500 um`
- medium: `500-800 um`
- high: `>800 um`

combined AS-OCT strict manifest 中各 split 的分布如下：

| split | low | medium | high |
|---|---:|---:|---:|
| train | 41 | 58 | 15 |
| val | 9 | 9 | 5 |
| test | 8 | 14 | 3 |

Low-vault samples 在 train、val、test 中均存在，因此当前 low-vault error pattern 不能简单解释为训练集中完全缺失该类样本。不过，test set 的 low-vault 样本仍然只有 8 只眼，结论需要保持 pilot-level 的谨慎解释。

## 3. AS-OCT Ensemble 按 Vault Range 的误差

AS-OCT seed ensemble 在 test set 中按 vault range 的误差如下：

| vault range | n | MAE | mean signed error | overestimation |
|---|---:|---:|---:|---:|
| low | 8 | 137.41 um | +132.65 um | 7/8 |
| medium | 14 | 75.48 um | -31.24 um | 5/14 |
| high | 3 | 84.08 um | -84.08 um | 0/3 |

Signed error 定义为 `prediction - label`。因此：

- positive signed error 表示 overestimation；
- negative signed error 表示 underestimation。

结果显示，当前 AS-OCT seed ensemble 在 low-vault eyes 上存在明显 overestimation，而在 high-vault eyes 上存在 underestimation。整体上，这提示模型可能存在一定 regression-to-the-mean tendency：低值被预测得偏高，高值被预测得偏低。

## 4. patient_052 影响

`patient_052` 双眼是 low-vault test samples 中最大的误差来源：

| sample | label | prediction | abs error |
|---|---:|---:|---:|
| `batch_02__patient_052_OS_20250501` | 432.0 um | 789.9 um | 357.91 um |
| `batch_02__patient_052_OD_20250501` | 434.0 um | 744.5 um | 310.45 um |

医生已确认 `patient_052` 双眼的 POD1 vault 标签、AS-OCT 图像、眼别、日期和 visit 对齐均无问题。因此，`patient_052` 双眼必须保留为真实有效的模型失败病例，不能因为误差大而从主结果中删除。

Sensitivity analysis 显示：

- 包含 `patient_052` 时，low-vault MAE = 137.41 um；
- 排除 `patient_052` 后，low-vault MAE = 71.82 um；
- overall MAE 从 96.33 um 变为 75.65 um。

但排除 `patient_052` 的结果仅作为 sensitivity analysis，不作为主结果。主分析仍保留完整 test set。

## 5. Measurement / Fusion 对 Low-Vault 的影响

进一步比较了 prediction-level measurement / fusion 方法在 low-vault test samples 上的表现：

| method | low-vault MAE |
|---|---:|
| AS-OCT seed ensemble | 137.41 um |
| concat fusion ensemble | 151.05 um |
| measurement ensemble | 204.28 um |
| weighted late fusion | 204.28 um |
| three-way weighted fusion | 153.57 um |
| residual correction | 218.32 um |

当前 measurement / fusion 方法均未解决 low-vault overestimation。相反，measurement-only、late fusion 和 residual correction 在 low-vault subset 上表现更差，提示现有术前结构化参数和简单融合策略还不能稳定校正低 vault 区间的系统性高估。

## 6. 阶段性结论

当前最强 AS-OCT 模型的主要不足不是整体预测失败，而是在 low-vault 子集上存在系统性高估趋势。`patient_052` 双眼是最典型的模型失败病例，但经医生复查确认数据无误，因此应作为真实有效的 high-error cases 保留在主分析中。

这一结果提示后续改进方向应优先围绕 vault range 展开，而不是直接堆叠更复杂的 fusion architecture：

- 做 vault-range-aware evaluation；
- 增加 low-vault samples；
- 设计 low-vault-sensitive training strategy；
- 评估 calibration 或 range-stratified loss weighting；
- 在样本量更大后再考虑复杂多模态融合。

## 7. 对论文的建议表述

当前论文可在 Discussion / Limitation 中谨慎加入以下表述：

> Additional range-stratified analysis suggested that low-vault eyes tended to be overestimated, including two clinically verified high-error eyes from the same patient. These samples were retained in the primary analysis and indicate that future work should pay particular attention to low-vault prediction.

该表述强调了模型限制，同时避免将真实有效的失败病例误解释为应删除的数据异常。

## 8. 相关输出

Low-vault error analysis 输出目录：

`artifacts/reports/combined_batch_01_02/low_vault_error_analysis/`

主要文件：

- `vault_range_distribution_by_split.csv`
- `as_oct_error_by_vault_range.csv`
- `low_vault_test_samples.csv`
- `low_vault_method_comparison.csv`
- `patient052_low_vault_case_summary.csv`
- `low_vault_error_analysis_summary.md`
- `figures/vault_range_distribution_by_split.png`
- `figures/signed_error_by_vault_range.png`
- `figures/low_vault_predictions_bar.png`
- `figures/low_vault_method_comparison.png`

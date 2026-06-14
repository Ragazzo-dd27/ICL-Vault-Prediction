# Current status after paper v2 revision

## 1. 当前论文主线

当前 ICL POD1 vault prediction 项目的论文主线已经稳定为：

**Preoperative AS-OCT and 2DAnalysis Measurements for Early Vault Prediction after ICL Implantation**

论文 v2 的核心思路是：基于真实临床导出数据，评估术前 AS-OCT 图像和术前 CASIA2 2DAnalysis measurements 对 POD1 vault prediction 的价值，并以 pilot baseline 的形式报告当前模型性能、误差模式和局限性。

当前阶段不再继续扩展复杂模型结构，而是优先稳固结果解释、误差分析和 repeated split 稳定性评估。

## 2. Combined Cohort 数据规模

当前分析基于 batch_01 + batch_02 combined cohort。

主要 manifest 规模如下：

| manifest | samples | train | val | test |
|---|---:|---:|---:|---:|
| combined AS-OCT strict | 162 | 114 | 23 | 25 |
| combined measurement ready | 160 | 111 | 24 | 25 |
| combined fusion ready | 159 | 111 | 23 | 25 |

所有 split 均为基于 `global_patient_uid` 的 patient-level split，避免同一患者双眼跨 split 泄漏。

当前不使用 UBM。结构化 measurement 输入仅使用真正术前 CASIA2 2DAnalysis measurements。POD1 postoperative 2DAnalysis measurements 只作为 vault label 来源，不作为输入特征。

## 3. 三条 Baseline 结果

### AS-OCT-only baseline

AS-OCT-only baseline 使用术前 AS-OCT raw image 作为输入，模型为 ResNet18 ImageNet fine-tune。

Combined strict AS-OCT-only 3 seeds 的平均 test MAE 约为：

- mean test MAE = 107.85 um

进一步对 3 个 AS-OCT seeds 做 seed ensemble 后：

- AS-OCT seed ensemble test MAE = 96.33 um

这是当前项目的主结果和最强 baseline。

### Preop measurement-only baseline

Measurement-only baseline 使用真正术前 2DAnalysis features：

- CCT
- ACD Epi
- ACD Endo
- CLR
- ATA

Linear / Ridge / Random Forest 的 test MAE 约为 136-138 um，MLP test MAE 约为 147.74 um。

结论：术前结构化 measurement features 具有预测价值，但整体弱于 AS-OCT-only。它们更适合作为辅助信息或误差解释变量，而不是当前阶段的主输入。

### AS-OCT + measurement concat fusion baseline

Concat fusion 使用 ResNet18 image encoder、measurement MLP 和 concat regression head。

3 seeds 平均 test MAE 约为：

- mean test MAE = 111.49 um

Fusion 接近 AS-OCT-only run-level 表现，但没有稳定超过 AS-OCT-only，更没有超过 AS-OCT seed ensemble。

## 4. 当前主结果

当前论文 v2 中建议保留的主结果为：

- **AS-OCT seed ensemble test MAE = 96.33 um**

该结果基于 combined batch_01 + batch_02 test set，n = 25 eyes。

该结果仍属于 pilot baseline，不应解释为最终临床性能。当前 test set 较小，后续需要 repeated patient-level split 或更多数据验证其稳定性。

## 5. Low-Vault Error Analysis

Range-stratified error analysis 显示，AS-OCT seed ensemble 在 low-vault test cases 上存在明显 overestimation：

- low-vault test samples: n = 8
- low-vault MAE = 137.41 um
- low-vault mean signed error = +132.65 um
- overestimation = 7/8

相比之下：

- medium-vault MAE = 75.48 um，mean signed error = -31.24 um
- high-vault MAE = 84.08 um，mean signed error = -84.08 um

这提示当前模型存在一定 regression-to-the-mean tendency：低 vault 倾向被高估，高 vault 倾向被低估。

该分析已经整理在：

`docs/experiments/low_vault_error_analysis.md`

## 6. patient_052 Sensitivity Analysis

`patient_052` 双眼是当前 test set 中最大的 top-error 来源：

- OS: label about 432 um, prediction about 790 um, abs error about 358 um
- OD: label about 434 um, prediction about 744 um, abs error about 310 um

Sensitivity analysis 显示，如果排除 `patient_052` 双眼：

- overall MAE 从 96.33 um 降至 75.65 um
- low-vault MAE 从 137.41 um 降至 71.82 um

但是，医生已确认 `patient_052` 双眼的 POD1 vault label、AS-OCT 图像、眼别、日期和 visit 对齐均无问题。

因此，`patient_052` 必须保留在主结果中，作为真实有效的 top-error / model failure cases。不能因为模型误差大而删除样本。

该分析已经整理在：

`docs/experiments/patient052_sensitivity_analysis.md`

## 7. Low-Vault Weighted Loss Pilot

为测试 low-vault overestimation 是否能通过训练策略缓解，进行了 seed42 的 low-vault weighted loss pilot。

主要结果：

| setting | overall MAE | low MAE | medium MAE | high MAE | low signed error |
|---|---:|---:|---:|---:|---:|
| original seed42 | 111.55 | 158.38 | 86.60 | 103.08 | +154.56 |
| low_weight=1.5 seed42 | 114.53 | 104.32 | 107.00 | 176.91 | +84.39 |
| low_weight=2.0 seed42 | 119.40 | 123.04 | 105.88 | 172.81 | +103.62 |
| extreme_weight=1.5 seed42 | 108.99 | 148.25 | 82.71 | 126.93 | +134.68 |
| AS-OCT seed ensemble | 96.33 | 137.41 | 75.48 | 84.08 | +132.65 |

结论：

- `low_weight=1.5` 最明显改善 low-vault MAE 和 low-vault signed error；
- 但它牺牲 overall、medium 和 high-vault 表现；
- `extreme_weight=1.5` 是 weighted single-seed 中 overall 最平衡的设置，但没有解决 low-vault overestimation；
- 所有 weighted-loss single-seed 设置仍弱于 AS-OCT seed ensemble。

因此，weighted loss 暂不作为主结果，也不建议当前直接扩展 seed2026 / seed3407。它应作为 low-vault overestimation 的探索性训练策略记录。

该分析已经整理在：

`docs/experiments/low_vault_weighted_loss_pilot.md`

## 8. 当前不继续推进的方向

基于当前结果，暂不继续推进以下方向：

- 复杂 fusion；
- cross-attention；
- Transformer-based multimodal model；
- 更大容量 fusion head；
- 基于当前小 test set 直接选择复杂模型。

原因：

- concat fusion 没有稳定超过 AS-OCT-only；
- prediction-level late fusion 和 residual correction 没有超过 AS-OCT ensemble；
- measurement features 具有局部互补性，但当前尚未形成稳定整体收益；
- test set 只有 25 只眼，复杂模型选择风险较高。

当前更重要的是稳定性评估和误差解释，而不是继续堆模型复杂度。

## 9. 下一步计划

下一步建议推进：

**Repeated patient-level split stability evaluation**

目标：

- 检查 AS-OCT-only baseline 在不同 patient-level split 下是否稳定；
- 评估 AS-OCT seed ensemble 的 MAE / RMSE / R2 波动；
- 观察 low-vault overestimation 是否在不同 split 中持续存在；
- 判断 patient_052 类似 top-error cases 对单次 split 的影响；
- 为论文 v2 的 robustness / limitation 提供更稳健依据。

在 repeated split 结果出来前，当前论文主线应保持：

- AS-OCT seed ensemble 作为主 baseline；
- measurement-only 和 concat fusion 作为对照；
- low-vault overestimation 作为主要 error pattern；
- patient_052 作为医生确认的数据无误但模型失败的案例；
- weighted loss 作为探索性补充，不作为主结果。

## 10. 当前状态总结

当前 ICL POD1 vault prediction 项目已经完成：

- batch_01 + batch_02 数据整理；
- POD1 vault label 人工核对；
- true preoperative measurement-only manifest；
- combined patient-level split；
- AS-OCT-only baseline；
- measurement-only baseline；
- concat fusion baseline；
- final baseline summary；
- top-error review；
- patient_052 sensitivity analysis；
- low-vault error analysis；
- low-vault weighted loss pilot。

论文 v2 结果部分已经足够稳定。后续工作重点应从“增加模型复杂度”转向“验证结果稳定性和解释误差模式”。

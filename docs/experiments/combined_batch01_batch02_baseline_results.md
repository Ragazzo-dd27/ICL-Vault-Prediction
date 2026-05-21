# Batch 01 + Batch 02 combined POD1 vault prediction pilot baseline results

## 1. 实验背景

Batch 01 和 Batch 02 的真实临床导出数据已经完成独立整理、POD1 vault 标签核对、true preoperative CASIA2 2DAnalysis measurement 核对，以及 combined patient-level split 构建。本阶段的目标是比较三类输入对 POD1 vault prediction 的贡献：

- 术前 AS-OCT raw image；
- true preoperative 2DAnalysis measurement features；
- AS-OCT image + preoperative measurement fusion。

本实验仍属于 pilot baseline。所有模型均预测人工核对后的 POD1 vault mean。UBM 暂未作为模型输入。Postoperative 2DAnalysis measurements are not used as input features；POD1 postoperative 2DAnalysis 仅作为标签来源。

## 2. 数据规模

Combined 数据使用基于 `global_patient_uid` 的 patient-level split，避免同一患者的双眼跨 train / val / test 泄漏。Batch 01 和 Batch 02 的 patient/sample 均保留 batch 前缀，避免匿名编号冲突。

| manifest | samples | train | val | test | 说明 |
|---|---:|---:|---:|---:|---|
| combined AS-OCT strict | 162 | 114 | 23 | 25 | AS-OCT-only baseline |
| combined measurement ready | 160 | 111 | 24 | 25 | true preoperative measurement-only baseline |
| combined fusion ready | 159 | 111 | 23 | 25 | AS-OCT + preoperative measurement concat fusion |

当前所有 baseline 均不使用 UBM。所有结构化 measurement 输入均来自真正术前 CASIA2 2DAnalysis；术后 2DAnalysis measurement 不作为输入特征。

## 3. AS-OCT-only baseline

AS-OCT-only baseline 使用术前 AS-OCT raw image 作为输入，模型为 ResNet18 regression，并使用 ImageNet pretrained weights 进行 full fine-tuning。实验包含 3 个 random seeds：

- `combined_as_oct_strict_imagenet_seed42_e30`
- `combined_as_oct_strict_imagenet_seed2026_e30`
- `combined_as_oct_strict_imagenet_seed3407_e30`

Run-level 平均 test MAE 约为 **107.85 um**。在 error complementarity analysis 中，对 3 个 AS-OCT seed 的 test predictions 做 seed ensemble 后，test MAE 约为 **96.33 um**。在当前 combined pilot split 上，AS-OCT-only ImageNet fine-tune 是整体最强 baseline。

这一结果说明，术前 AS-OCT raw image 对 POD1 vault prediction 已经包含较强的可学习信息；即使没有加入 UBM 和结构化参数，图像模型仍可达到当前最佳平均表现。

## 4. Preop measurement-only baseline

Preop measurement-only baseline 仅使用真正术前 CASIA2 2DAnalysis measurement features：

- `cct_mean_um`
- `acd_epi_mean_mm`
- `acd_endo_mean_mm`
- `clr_mean_um`
- `ata_mean_mm`

评估模型包括 Linear Regression、Ridge Regression、Random Forest Regression 和 MLP Regressor。当前 combined measurement ready manifest 为 160 samples，train / val / test = 111 / 24 / 25。

主要结果为：

- Linear / Ridge / Random Forest 的 test MAE 约为 **136-138 um**；
- MLP 的 test MAE 约为 **147.74 um**；
- measurement-only 整体弱于 AS-OCT-only，但明显优于随机猜测，说明术前结构化参数具有预测价值。

从解释角度看，CCT、ACD、CLR、ATA 等参数与术后 vault 存在可利用关系，但 5 个结构化特征无法完全替代 AS-OCT 图像中的局部形态信息。因此 measurement features 更适合作为辅助信息，而不是当前阶段的主输入。

## 5. Concat fusion baseline

Concat fusion baseline 使用 ResNet18 image encoder 提取 AS-OCT image feature，并用一个小型 MLP 编码 5 个 true preoperative measurement features，随后将两个分支 concat 后送入 regression head。该模型不使用 UBM，也不使用术后 2DAnalysis measurement 作为输入。

Fusion runs 包括：

- `combined_fusion_ready_concat_seed42_e30`
- `combined_fusion_ready_concat_seed2026_e30`
- `combined_fusion_ready_concat_seed3407_e30`

3 seeds 的 mean test MAE 约为 **111.49 um**。该结果接近 AS-OCT-only run-level 平均，但没有稳定超过 AS-OCT-only。Concat fusion seed ensemble 的 test MAE 约为 **110.96 um**，仍弱于 AS-OCT-only seed ensemble 的 **96.33 um**。

可能原因包括：

- fusion head 增加了额外参数，小样本条件下更容易过拟合；
- measurement features 与 AS-OCT image information 存在部分冗余；
- 简单 concat 未必能学到稳定的模态互补关系；
- measurement features 中存在 confirmed outlier 或单记录样本时，结构化分支可能引入额外噪声；
- 当前 test set 只有 25 只眼，少数样本会明显影响均值。

## 6. Late fusion / residual correction

为判断 measurement features 是否能在 prediction level 提供稳定增益，进一步进行了 late fusion 和 residual correction 分析。该分析只基于已有 val/test predictions，不训练新的深度图像模型。

主要结果：

| 方法 | test MAE |
|---|---:|
| AS-OCT seed ensemble | 96.33 um |
| measurement model/seed ensemble | 135.26 um |
| concat fusion seed ensemble | 110.96 um |
| weighted late fusion | 135.26 um |
| three-way weighted fusion | 112.35 um |
| residual correction | 131.75 um |

Weighted late fusion 在 validation set 上选择了 `w_as_oct=0.00, w_measurement=1.00`，即完全偏向 measurement prediction；但在 test set 上 MAE 为 135.26 um，明显差于 AS-OCT ensemble。这提示当前 validation set 太小，基于 val MAE 选择融合权重不稳定。

Residual correction 在 validation set 上表现较好，但 test MAE 为 131.75 um，说明基于 23 个 validation samples 拟合 residual pattern 容易过拟合。Three-way weighted fusion 也没有超过 AS-OCT ensemble。

Error complementarity analysis 显示 measurement-only 在 **7/25** 个 test samples 上是最优方法，说明 measurement features 的确存在局部互补信息；但 concat fusion 只在 **4/25** 个 test samples 上最优，且相比 AS-OCT-only 变差的样本多于改善的样本。因此，当前 measurement 信息尚未通过简单融合方式转化为稳定整体收益。

## 7. 主要结论

当前 combined batch_01 + batch_02 pilot baseline 的最强路线是 **AS-OCT-only ImageNet fine-tune / seed ensemble**。术前 AS-OCT raw image 是目前最稳定、最有效的输入模态。

True preoperative measurement features 具有明确预测价值，并且在部分样本上优于 AS-OCT-only，提示其可能提供局部互补信息。但 measurement-only 整体性能弱于 AS-OCT-only，更适合作为辅助输入或校正信息。

简单 concat fusion、weighted late fusion 和 residual correction 暂未形成稳定增益。当前不建议直接进入复杂 cross-attention 或高容量多模态模型。更合理的方向是先扩大数据量、做更稳定的 repeated split / K-fold 评估，并探索更保守的融合方式。

所有结论都应保持克制：当前 test set 只有 25 只眼，结果属于 pilot baseline，不是最终临床结论。

## 8. 当前限制

- 样本量仍有限，combined test set 只有 25 只眼；
- Batch 01 和 Batch 02 来自同一数据来源，外部泛化能力尚未验证；
- UBM 尚未完成 eye/date/visit 级别对齐，因此未纳入模型；
- 当前 fusion 为简单 concat，没有显式建模更复杂的模态关系；
- validation set 较小，late fusion / residual correction 的权重或残差模式选择不稳定；
- measurement features 来自人工核对的 true preoperative 2DAnalysis，后续仍需持续 QC；
- 当前结果仍属于 pilot baseline，不应解读为最终临床性能。

## 9. 下一步计划

后续建议优先固化 AS-OCT-only ImageNet fine-tune / seed ensemble 作为当前主 baseline，并围绕该 baseline 做更深入误差分析：

- 复查 top-error samples，确认术前 AS-OCT 输入、POD1 vault label 和图像质量；
- 分析 high-vault / low-vault 样本的系统性偏差；
- 在新增数据纳入后重新训练，并考虑 K-fold 或 repeated split；
- 继续探索更保守的融合方法，例如 uncertainty-based gating、late fusion with stronger validation、或 residual fusion with stronger regularization；
- 暂缓复杂 cross-attention，等待样本量更大、fusion 需求更明确后再推进。

## 10. 输出文件索引

Final baseline summary：

- `artifacts/reports/combined_batch_01_02/final_baseline_summary/final_baseline_summary.csv`
- `artifacts/reports/combined_batch_01_02/final_baseline_summary/final_run_level_summary.csv`
- `artifacts/reports/combined_batch_01_02/final_baseline_summary/final_baseline_summary.md`
- `artifacts/reports/combined_batch_01_02/final_baseline_summary/figures/`

Error complementarity analysis：

- `artifacts/reports/combined_batch_01_02/error_complementarity/test_error_complementarity_by_sample.csv`
- `artifacts/reports/combined_batch_01_02/error_complementarity/method_error_summary.csv`
- `artifacts/reports/combined_batch_01_02/error_complementarity/improvement_summary.csv`
- `artifacts/reports/combined_batch_01_02/error_complementarity/vault_range_error_summary.csv`
- `artifacts/reports/combined_batch_01_02/error_complementarity/error_complementarity_summary.md`
- `artifacts/reports/combined_batch_01_02/error_complementarity/figures/`

Late fusion / residual correction：

- `artifacts/reports/combined_batch_01_02/late_fusion_analysis/late_fusion_summary.csv`
- `artifacts/reports/combined_batch_01_02/late_fusion_analysis/late_fusion_test_predictions.csv`
- `artifacts/reports/combined_batch_01_02/late_fusion_analysis/late_fusion_weight_search.csv`
- `artifacts/reports/combined_batch_01_02/late_fusion_analysis/three_way_weight_search.csv`
- `artifacts/reports/combined_batch_01_02/late_fusion_analysis/late_fusion_analysis.md`
- `artifacts/reports/combined_batch_01_02/late_fusion_analysis/figures/`

# Repeated patient-level split stability evaluation 阶段性报告

## 一、分析目的

本轮 repeated patient-level split stability evaluation 的目的，是检查 combined batch_01 + batch_02 cohort 上当前 POD1 vault prediction baseline 对 patient-level split 的敏感性。由于当前样本量仍然有限，单次 train/val/test split 可能会受到少数 low-vault、high-vault 或 top-error 病例分布的影响，因此需要用多个 patient-level split 观察模型表现是否稳定。

本分析不替代原始主结果。原始主结果仍然基于已经完成的 combined cohort 主 split 和 AS-OCT seed ensemble。

本分析不包含 fusion、weighted loss、cross-attention 或 Transformer，也不作为继续堆复杂模型结构的依据。本轮重点是稳定性、误差分布和 split sensitivity。

## 二、数据与 split 设置

输入 AS-OCT strict manifest:

- `data/manifests/vault_as_oct_only_pod1_manifest_combined_strict.csv`

数据规模：

- 总样本数: 162
- 总 patient 数: 90

已生成 split：

- 5 个 standard repeated patient-level splits
- 2 个 patient_052 forced-test stress splits

本轮 AS-OCT standard repeated split evaluation 只使用 5 个 standard repeated splits，不包含 patient_052 forced-test splits。

每个 standard split 的样本数：

- train / val / test = 114 / 24 / 24

所有 split 均为 patient-level split，同一个 patient 不跨 train / val / test。

## 三、Measurement-only repeated split 结果

Measurement-only baseline 使用真正术前 2DAnalysis measurements：

- CCT
- ACD Epi
- ACD Endo
- CLR
- ATA

5 个 standard repeated splits 上的 test MAE mean ± std：

| Model | Test MAE mean ± std |
| --- | --- |
| Ridge | 148.41 ± 37.08 um |
| Linear | 148.72 ± 37.61 um |
| Random Forest | 160.43 ± 44.11 um |

patient_052 forced-test stress splits 下，measurement-only 模型也出现低 vault 高估，说明 patient_052 双眼并不是 AS-OCT 图像模型独有的异常点，而是当前输入信号对 low-vault 真实结果建模不足的一个代表性病例。

解释：measurement-only 是结构化术前参数 baseline。它整体不优于原始 AS-OCT 主结果，但在 repeated split 下可以作为稳定性参照，帮助判断 AS-OCT-only 模型在不同 split 中的波动是否过大。

## 四、AS-OCT standard repeated split single-seed 结果

AS-OCT-only repeated split evaluation 设置：

- model seed = 42
- ResNet18 ImageNet fine-tune
- 使用 standard repeated patient-level splits

5 个 split 的 test MAE：

| Split seed | Test MAE |
| --- | ---: |
| 42 | 136.06 um |
| 1001 | 171.86 um |
| 2002 | 141.11 um |
| 2026 | 137.87 um |
| 3407 | 195.09 um |

总体：

- mean ± std = 156.40 ± 26.11 um

与 measurement-only Ridge 结果对比：

- AS-OCT single-seed repeated split: 156.40 ± 26.11 um
- Measurement-only Ridge repeated split: 148.41 ± 37.08 um

阶段性结论：single-seed AS-OCT 在 repeated patient-level split 下没有稳定优于 measurement-only。与原始主结果相比，repeated split single-seed 结果明显更保守，提示当前小样本 cohort 对 split 分布较敏感。

## 五、Original split sanity reproduction

为了确认 repeated split wrapper 没有引入训练配置错误，我们使用原始 combined AS-OCT strict manifest 中已有 train/val/test split 做了 sanity reproduction。

原始记录：

- seed42 test MAE 约 111.55 um

sanity reproduction：

- test MAE = 116.83 um
- RMSE = 146.66 um
- R2 = 0.023
- 与原始记录差异 = 5.28 um

解释：sanity reproduction 与原始 seed42 结果接近，说明当前 repeated split wrapper 和训练配置基本可信。repeated split 结果变差，更可能来自 split 本身的样本组成差异，而不是代码配置错误。

## 六、Prediction distribution diagnostic

对 AS-OCT-only standard repeated split predictions 做 prediction distribution diagnostic 后，发现：

- prediction range / label range 平均比例 = 0.39
- 5 个 split 的 prediction range 都明显窄于 label range
- low-vault signed error = +140.86 um
- high-vault signed error = -437.17 um

解释：模型存在明显 regression-to-the-mean，即低 vault 被高估，高 vault 被低估。这个现象比单纯 overall MAE 更重要，因为它提示模型没有充分覆盖真实 vault 分布的两端。

seed3407 test MAE 最高，主要与该 split 的 high-vault test 样本较多有关。该 split 的 high-vault underestimation 明显，导致整体 MAE 被拉高。

## 七、3-seed ensemble pilot

为了判断 seed ensemble 是否可以缓解 repeated split 波动，我们只对两个代表性 standard split 做了 3-seed ensemble pilot：

- split_seed2026: 中等表现 split
- split_seed3407: 最差表现 split，且 high-vault test 样本较多

结果如下：

| Split seed | Seed42 MAE | 3-seed ensemble MAE | Delta |
| --- | ---: | ---: | ---: |
| 2026 | 137.87 um | 136.40 um | -1.48 um |
| 3407 | 195.09 um | 190.21 um | -4.88 um |

Range-specific error 仍然存在：

- split2026 low signed error = +162.84 um
- split3407 low signed error = +128.22 um
- split2026 high signed error = -437.85 um
- split3407 high signed error = -383.84 um

结论：3-seed ensemble 可以带来小幅改善，但不能解决 low-vault overestimation 和 high-vault underestimation。它更像是降低随机波动的轻微修正，而不是解决当前模型核心误差模式的方法。

## 八、阶段性结论

当前不建议扩展到全部 5 个 standard splits 的 3-seed ensemble。两个 pilot split 的改善幅度较小，训练成本较高，而且没有解决 range-specific error。

当前也不建议继续跑 patient_052 forced-test AS-OCT，除非后续导师明确要求 stress analysis。patient_052 双眼已经由医生确认标签、图像、眼别、日期和 visit 对齐无误，应作为真实有效模型失败病例保留。

当前不建议继续做复杂 fusion、cross-attention、Transformer 或 weighted loss。已有结果显示，简单 concat fusion、late fusion、residual correction、weighted loss 都没有稳定超过当前主结果，也没有解决 low-vault / high-vault 两端误差。

原始 AS-OCT seed ensemble MAE = 96.33 um，仍然作为当前主结果。Repeated split 结果应作为 robustness / limitation 内部分析，而不是替代主结果。

当前模型主要限制是：

- small-cohort split sensitivity
- regression-to-the-mean
- low-vault overestimation
- high-vault underestimation

后续真正有价值的方向是增加样本量，尤其是 low-vault 和 high-vault 样本，而不是继续堆模型结构。

## 九、对论文的处理建议

暂时不建议大改论文 v2。当前 repeated split 结果更适合作为内部 robustness / limitation 记录，而不是主结果表。

如果需要在 Discussion / Limitation 中加入，可考虑一句克制表述：

> Preliminary repeated-split analysis suggested performance variability and a regression-to-the-mean tendency, indicating the need for larger cohorts and range-aware validation.

不建议把完整 repeated split 表格放入主结果。当前主结果仍应保持 AS-OCT seed ensemble 与主要 baseline comparison。

## 十、下一步建议

- 准备给导师的阶段性汇报。
- 保留当前 repeated split 结果作为实验记录。
- 暂停复杂模型开发。
- 暂停 full repeated-split 3-seed ensemble 扩展，除非导师明确要求。
- 后续优先推进数据质量闭环和更多病例收集。
- 增加 low-vault 与 high-vault 样本后，再重新评估 range-aware validation 或更稳健的训练策略。

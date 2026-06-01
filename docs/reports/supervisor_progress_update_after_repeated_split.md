# Repeated split 后阶段性进展汇报

## 一、当前论文主线

目前论文题目方向已经收敛为：

**Preoperative AS-OCT and 2DAnalysis Measurements for Early Vault Prediction after ICL Implantation**

当前核心任务是基于术前 AS-OCT 图像和术前 CASIA2 2DAnalysis measurements，预测 ICL 术后第 1 天 POD1 vault。

本阶段重点不是继续扩展复杂模型，而是确认当前真实数据 baseline 的可靠性、误差模式和 split 稳定性。

## 二、当前数据规模

目前已经完成 batch_01 + batch_02 的 combined cohort 整理。

主要 manifest 规模如下：

- AS-OCT-only strict manifest: 162 samples
- measurement-only ready manifest: 160 samples
- AS-OCT + measurement fusion ready manifest: 159 samples

所有主实验均使用 patient-level split，避免同一 patient 的双眼跨 train / val / test 泄漏。

## 三、主实验结果

当前 combined cohort 上的主要结果如下：

- AS-OCT-only ImageNet fine-tune 3-seed ensemble test MAE = **96.33 um**
- Measurement-only baseline test MAE 约 **136-138 um**
- AS-OCT + measurement concat fusion test MAE 约 **111.49 um**

目前最强主结果仍然是 **AS-OCT seed ensemble**。结构化术前 measurement features 有预测价值，但在当前数据规模下，简单 concat fusion 没有稳定超过 AS-OCT-only。

## 四、误差分析

误差分析显示，当前 AS-OCT seed ensemble 存在较明显的 vault range 相关误差：

- low-vault cases 存在 overestimation
- high-vault cases 存在 underestimation

patient_052 双眼是当前 test set 中最大的误差来源。医生已经复查确认 patient_052 双眼的 POD1 vault 标签、AS-OCT 图像、眼别、日期和 visit 对齐均无问题。因此 patient_052 不能因为误差大而删除，应作为真实有效的模型失败病例保留。

这一点对后续论文表述很重要：patient_052 更适合作为模型局限性和 low-vault overestimation 的代表病例，而不是质控排除样本。

## 五、Repeated split stability evaluation

为了检查当前结论对 patient-level split 的敏感性，我们生成并评估了 5 个 standard repeated patient-level splits。

主要结果：

- AS-OCT single-seed repeated split MAE = **156.40 ± 26.11 um**
- Measurement-only Ridge repeated split MAE = **148.41 ± 37.08 um**

此外，我们做了 original split sanity reproduction：

- 原始记录 seed42 test MAE 约 111.55 um
- sanity reproduction test MAE = 116.83 um
- 差异约 5.28 um

这说明 repeated split wrapper 和训练配置基本可信，repeated split 下 AS-OCT 结果变差大概率不是代码错误，而是 split 本身带来的样本组成差异。

Prediction distribution diagnostic 进一步显示：

- prediction range 明显窄于 label range
- low-vault 被高估
- high-vault 被低估

也就是说，当前模型存在 regression-to-the-mean tendency。

我们还对 split2026 和 split3407 做了 3-seed ensemble pilot：

- split2026: seed42 MAE 137.87 um，3-seed ensemble MAE 136.40 um
- split3407: seed42 MAE 195.09 um，3-seed ensemble MAE 190.21 um

3-seed ensemble 只有小幅改善，不能解决 low-vault overestimation 和 high-vault underestimation。

## 六、阶段性判断

基于目前结果，我的阶段性判断是：

- 当前不建议继续盲目做复杂 fusion、cross-attention 或 Transformer。
- 当前也不建议继续扩大 weighted loss 或 repeated ensemble 实验。
- 主要限制不是模型结构不够复杂，而是小样本、vault range 两端样本不足，以及 patient-level split sensitivity。
- 原始 AS-OCT seed ensemble 仍然可以作为当前主结果。
- Repeated split 结果更适合作为 robustness / limitation 记录，而不是替代主结果。

换句话说，目前继续堆模型结构的性价比不高。更关键的问题是数据量和 vault range 分布，尤其是 low-vault 和 high-vault 样本不足。

## 七、下一步建议

建议下一步先请导师判断是否需要把 repeated split stability analysis 写进论文。

如果要写，建议只放在 Discussion / Limitation 中，用一句简洁表述，例如：

> Preliminary repeated-split analysis suggested performance variability and a regression-to-the-mean tendency, indicating the need for larger cohorts and range-aware validation.

不建议把完整 repeated split 表格放入主结果。

后续优先方向：

- 扩大样本量，尤其是 low-vault 和 high-vault 病例。
- 继续完善数据质量闭环。
- 保留 patient_052 等真实模型失败病例，用于误差分析和后续模型改进。
- 暂停复杂模型开发，除非导师明确要求补充 stress analysis 或 robustness experiment。

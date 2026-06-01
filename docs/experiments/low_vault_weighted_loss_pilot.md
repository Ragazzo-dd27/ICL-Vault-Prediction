# Low-vault weighted loss pilot

## 1. 实验背景

前序 low-vault error analysis 显示，当前 AS-OCT seed ensemble 在 low-vault test cases 上存在明显 overestimation：

- low-vault test samples: n = 8;
- AS-OCT seed ensemble low-vault MAE = 137.41 um;
- low-vault mean signed error = +132.65 um;
- overestimation = 7/8。

其中 `patient_052` 双眼是最明显的 high-error cases。医生已确认其 POD1 vault 标签、AS-OCT 图像、眼别、日期和 visit 对齐均无问题，因此这些样本必须保留为真实有效的模型失败病例。

基于这一误差模式，本实验尝试在 AS-OCT-only baseline 中加入 vault-range weighted loss，测试 low-vault overestimation 是否能通过训练策略部分缓解。

## 2. 实验设置

本实验是最小可控 pilot，不作为最终主结果：

- 只在 AS-OCT-only baseline 上探索训练策略；
- 不修改模型结构；
- 不使用 fusion；
- 不使用 UBM；
- 不删除 `patient_052`；
- 只做 seed42 pilot；
- 不修改任何 manifest、prediction、checkpoint 或既有训练结果。

训练目标仍为 POD1 vault regression。Weighted loss 根据原始 vault label 所属区间对 normalized squared error 加权。

## 3. 对比方法

本次比较以下方法：

- `original_seed42`
- `low_weight_1p5_seed42`
- `low_weight_2p0_seed42`
- `extreme_weight_1p5_seed42`
- `as_oct_seed_ensemble`

其中 `as_oct_seed_ensemble` 不是单次 weighted-loss 训练，而是现有 AS-OCT ImageNet fine-tune 三个 seed 的 ensemble，用作当前最强参考。

## 4. 核心结果

| method | overall MAE | low MAE | medium MAE | high MAE | low signed error |
|---|---:|---:|---:|---:|---:|
| original_seed42 | 111.55 | 158.38 | 86.60 | 103.08 | +154.56 |
| low_weight_1p5_seed42 | 114.53 | 104.32 | 107.00 | 176.91 | +84.39 |
| low_weight_2p0_seed42 | 119.40 | 123.04 | 105.88 | 172.81 | +103.62 |
| extreme_weight_1p5_seed42 | 108.99 | 148.25 | 82.71 | 126.93 | +134.68 |
| as_oct_seed_ensemble | 96.33 | 137.41 | 75.48 | 84.08 | +132.65 |

## 5. 主要发现

`low_weight=1.5` 对 low-vault 的改善最明显：

- low-vault MAE 从 original seed42 的 158.38 um 降至 104.32 um；
- low-vault mean signed error 从 +154.56 um 降至 +84.39 um。

这说明 low-vault overestimation 确实可以被训练策略部分缓解。

但是，该改善伴随明显代价：

- overall MAE 从 111.55 um 增至 114.53 um；
- medium-vault MAE 从 86.60 um 增至 107.00 um；
- high-vault MAE 从 103.08 um 增至 176.91 um。

`low_weight=2.0` 也改善 low-vault MAE，但 overall 更差：

- overall MAE = 119.40 um；
- low-vault MAE = 123.04 um；
- high-vault MAE = 172.81 um。

`extreme_weight=1.5` 是 weighted single-seed 中 overall 最好的设置：

- overall MAE = 108.99 um；
- medium-vault MAE = 82.71 um；
- high-vault MAE = 126.93 um。

但它对 low-vault 的改善有限：

- low-vault MAE = 148.25 um；
- low-vault mean signed error = +134.68 um。

所有 weighted-loss single-seed 设置仍弱于 AS-OCT seed ensemble：

- AS-OCT seed ensemble overall MAE = 96.33 um；
- AS-OCT seed ensemble 在 overall、medium 和 high vault 上均表现最好。

因此，weighted loss 的主要价值是解释性和探索性：它证明 low-vault overestimation 可以被训练策略影响，但当前设置尚未形成更好的整体模型。

## 6. 结论

Weighted loss pilot 表明，low-vault overestimation 并非完全不可调；`low_weight=1.5` 可以明显降低 low-vault MAE 和 positive signed error。

然而，目前 weighted loss 不是更好的整体模型。Low-vault 改善通常伴随 medium/high 或 overall 性能损失。当前不建议将 weighted-loss run 作为论文主结果，也不建议立即扩展 seed2026 / seed3407。

主线结果仍应保持：

- AS-OCT seed ensemble 作为当前最强 baseline；
- low-vault overestimation 作为 error analysis / limitation；
- patient_052 作为真实有效模型失败病例保留。

## 7. 后续方向

建议暂停 weighted-loss 大规模训练，将本实验作为探索性结果记录。

如果后续继续推进 low-vault-sensitive strategy，可考虑：

- balanced sampler；
- range-aware calibration；
- 更系统的小网格，例如 `low_weight=1.2 / 1.5 / 1.8`；
- 分层 validation 或 repeated split 后再判断是否扩展多 seed；
- 保持 AS-OCT seed ensemble + error analysis 作为主线。

在样本量有限的当前阶段，不建议直接把 weighted loss 或复杂 fusion 推为主结果。

## 8. 相关输出文件

输出目录：

`artifacts/reports/combined_batch_01_02/low_vault_weighted_loss_comparison/`

主要文件：

- `weighted_loss_pilot_summary.csv`
- `weighted_loss_pilot_by_sample.csv`
- `weighted_loss_pilot_summary.md`

Figures：

- `figures/weighted_loss_overall_range_mae_comparison.png`
- `figures/weighted_loss_low_signed_error_comparison.png`
- `figures/weighted_loss_delta_vs_original.png`
- `figures/weighted_loss_patient052_errors.png`

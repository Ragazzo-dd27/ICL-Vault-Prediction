# Batch 01 真实数据 AS-OCT-only POD1 vault regression baseline experiment

## 1. 实验背景

本轮实验基于新增的 batch_01 真实临床导出数据。该批次共包含 46 个匿名 patient，已完成 CASIA2 2DAnalysis 中 POD1 vault 值的人工核对，并构建为 eye-level POD1 verified label table。

在当前阶段，我们从 91 只眼的 POD1 verified labels 中，排除了 8 个 `large_between_scan_difference` 样本，得到 83 只眼 clean samples。实验目标是先建立一个可复现的 AS-OCT-only POD1 vault regression baseline，用于真实数据 pipeline smoke test 和后续方法比较；这不是最终多模态模型，也不是最终临床结论。

## 2. 数据整理流程

数据整理采用逐步保守推进的方式，尽量保持原始导出数据和中间人工核对结果不被覆盖。

1. **Raw export inspection**  
   对 batch_01 raw export 进行扫描，建立初始 manifest 和 patient-level summary，统计 AS-OCT、2DAnalysis、UBM 等图像类型，并保持匿名 `patient_uid`。

2. **CASIA2 2DAnalysis measurement crop**  
   从 2DAnalysis 图像中生成 measurement table crop，用于人工核对 vault、CCT、ACD、ATA、CLR 等测量值。该步骤不启用 OCR，避免将未核对文本当成可靠标签。

3. **POD1 manual review**  
   从 POD1 records 中筛选术后第 1 天记录，形成人工核对表。人工填写后的 vault 值作为后续 verified label 来源。

4. **Eye-level POD1 verified label construction**  
   将同一 `patient_uid + eye` 下的 POD1 verified records 聚合为 eye-level label。通常每只眼有 2 条 2DAnalysis scan，使用 POD1 vault mean 作为 regression target，同时保留 scan-level 值、range 和 QC flag。

5. **Formal POD1 manifest draft**  
   将 POD1 verified label 与术前 AS-OCT raw、术前 2DAnalysis candidate 以及 patient-level UBM availability 对齐，生成 formal manifest draft。该 draft 仍不是最终训练 manifest。

6. **AS-OCT-only strict manifest**  
   为避免 AS-OCT-only baseline 意外加载 UBM，将 strict manifest 中 `has_ubm=False`、`ubm_path` 置空，并标记 `ubm_alignment_status=not_used_in_as_oct_only_baseline`。

7. **DataLoader smoke test**  
   使用 clean strict manifest 进行 Dataset/DataLoader smoke test。AS-OCT 图像 resize 到 `224 x 224`，batch 中 `oct_images` shape 为 `(B, 3, 224, 224)`，`ubm_images=None`，确认 AS-OCT-only 数据链路可运行。

## 3. 当前关键数据规模

- batch_01 patients: 46
- eye-level POD1 verified labels: 91
- clean samples: 83
- clean manifest train/val/test split: 58 / 12 / 13
- full manifest train/val/test split: 63 / 14 / 14
- excluded samples: 8 个 `large_between_scan_difference` samples

## 4. 训练设置

- Input: preoperative AS-OCT raw image
- Target: POD1 vault mean，单位为 micrometers
- Model: ResNet18 regression
- Image size: `224 x 224`
- Label normalization: 使用 train split 的 vault label mean/std
- Loss: normalized label 上的 MSE
- Metrics: 反归一化到 micrometers 后计算 MAE、RMSE、R2
- Split: patient-level split，避免同一 patient 的双眼跨 split 泄漏
- UBM: 在 AS-OCT-only baseline 中有意禁用，不作为模型输入

## 5. Baseline runs

当前已完成以下 AS-OCT-only POD1 baseline runs：

- random init, 3 seeds
- ImageNet fine-tune, 3 seeds
- ImageNet freeze, 1 seed
- full manifest ImageNet seed42 sensitivity experiment

Run-level 结果见：

- `artifacts/reports/as_oct_pod1_baseline_batch_01/summary.csv`
- `artifacts/reports/as_oct_pod1_baseline_batch_01/summary.md`

Family-level 汇总见：

- `artifacts/reports/as_oct_pod1_baseline_batch_01/group_summary.csv`
- `artifacts/reports/as_oct_pod1_baseline_batch_01/group_summary.md`

截至当前 summary：

| experiment_family | n_runs | best_val_mae_mean | best_val_mae_std | test_mae_mean | test_mae_std | test_rmse_mean | test_rmse_std | test_r2_mean | test_r2_std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random_init | 3 | 182.54 | 7.06 | 138.33 | 10.92 | 167.68 | 16.57 | 0.002 | 0.191 |
| imagenet_freeze | 1 | 174.33 |  | 151.19 |  | 178.35 |  | -0.122 |  |
| imagenet_finetune | 4 | 165.89 | 16.78 | 153.48 | 2.13 | 179.42 | 5.28 | -0.120 | 0.086 |

需要注意，上表中的 `imagenet_finetune` family 包含一个 full manifest sensitivity run。若只看 clean manifest 的 3 个 ImageNet fine-tune seeds，其 validation MAE 大致在 149.76 到 164.30 um 之间，整体优于 random init 和 freeze backbone 在 validation set 上的表现。

## 6. 主要发现

1. **ImageNet fine-tune 在验证集上优于 random init 和 freeze backbone。**  
   clean manifest 的 ImageNet fine-tune runs 在 validation MAE 上整体更低，说明在当前小样本真实数据上，ImageNet 初始化并全量 fine-tuning 对 AS-OCT vault regression 有一定帮助。

2. **random init 在当前小 test set 上平均 test MAE 更低，但需要谨慎解释。**  
   random init family 的 test MAE mean 当前较低，但 test set 只有 13 只眼，单次 patient-level split 下结果容易受样本构成影响。这更适合作为 pilot observation，而不是稳定结论。

3. **freeze backbone 没有明显优势。**  
   ImageNet pretrained frozen backbone 的 validation 和 test 表现均未显示出优势，提示完全冻结自然图像特征可能不足以适配 AS-OCT vault regression。

4. **clean manifest 相比 full manifest 在验证集上更稳定。**  
   full manifest sensitivity run 的 best validation MAE 高于 clean ImageNet seed42 run，支持对 `large_between_scan_difference` 样本进行 label QC 的必要性。

5. **当前结果是 pilot baseline，不是最终临床结论。**  
   该实验主要证明真实数据 pipeline、label construction、strict AS-OCT-only manifest、DataLoader 和 baseline training 可以闭环运行。

## 7. 当前限制

- 样本量仍小，clean manifest 只有 83 只眼。
- test set 只有 13 只眼，test MAE 和 R2 容易受个别样本影响。
- 当前模型只使用 AS-OCT raw image，不包含 UBM、术前 2DAnalysis measurements 或 clinical features。
- POD1 vault 标签来自人工核对的 2DAnalysis，而非独立金标准。
- `large_between_scan_difference` 样本需要进一步临床确认，当前 clean manifest 暂时排除这些样本。
- 当前仍是单套 patient-level split，不同 split seed 下的稳定性尚未系统评估。

## 8. 下一步计划

- 将当前误差分析图整理进组会材料。
- 分析 prediction error 最大的样本，检查是否与图像质量、术前/术后日期、scan 差异或标签 QC 有关。
- 后续加入术前 2DAnalysis measurements 和 clinical features，评估 image + structured features 的增益。
- 后续再考虑 UBM alignment 和多模态融合，避免在 patient-level UBM 尚未精确对齐时过早做 multimodal 结论。
- 新数据继续纳入相同 pipeline，保持 raw inspection、manual review、strict manifest 和 baseline reporting 的一致性。

## 9. 输出文件索引

主要 summary 和报告文件：

- `artifacts/reports/as_oct_pod1_baseline_batch_01/summary.csv`
- `artifacts/reports/as_oct_pod1_baseline_batch_01/summary.md`
- `artifacts/reports/as_oct_pod1_baseline_batch_01/group_summary.csv`
- `artifacts/reports/as_oct_pod1_baseline_batch_01/group_summary.md`

结果图目录：

- `artifacts/reports/as_oct_pod1_baseline_batch_01/figures/`

Manifest 文件：

- `data/manifests/vault_as_oct_only_pod1_manifest_batch_01_full_strict.csv`
- `data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean_strict.csv`
- `data/manifests/formal_vault_manifest_batch_01_pod1_draft.csv`

训练日志目录：

- `artifacts/logs/as_oct_pod1_baseline_batch_01/`

预测结果目录：

- `artifacts/predictions/as_oct_pod1_baseline_batch_01/`

Checkpoint 目录：

- `artifacts/checkpoints/as_oct_pod1_baseline_batch_01/`

相关脚本：

- `scripts/build_as_oct_only_pod1_training_manifest.py`
- `scripts/build_strict_as_oct_only_manifest.py`
- `scripts/smoke_test_as_oct_pod1_dataset.py`
- `scripts/train_as_oct_pod1_baseline.py`
- `scripts/summarize_as_oct_pod1_baseline_runs.py`
- `scripts/plot_as_oct_pod1_baseline_results.py`

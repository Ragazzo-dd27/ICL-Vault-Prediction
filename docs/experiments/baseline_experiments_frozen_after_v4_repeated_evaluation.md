# Baseline Experiments Frozen After V4 Repeated Evaluation

- Freeze date: generated from completed combined v4 outputs.
- Data version: combined batch_01 + batch_02 + batch_03 + batch_04 v4 manifests and repeated patient-level splits.
- Label correction: patient_100 OS AS-OCT label corrected from 7901 um to 701 um after manual source verification.
- Frozen baselines: measurement-only RF, corrected AS-OCT-only, and AS-OCT + measurement concat fusion.
- Primary protocol: seed42 patient-level split; AS-OCT and fusion primary ensembles use three model seeds.
- Repeated protocol: split seeds 42, 1001, 2002, 2026, 3407; AS-OCT and fusion use fixed model_seed42.

## Final Result Paths
- Final comparison package: `artifacts/reports/combined_batch_01_02_03_04/final_baseline_comparison`
- Measurement-only repeated: `artifacts/reports/combined_batch_01_02_03_04/measurement_only_repeated_splits`
- Corrected AS-OCT repeated: `artifacts/reports/combined_batch_01_02_03_04/as_oct_only_repeated_splits_label_corrected_patient100_os`
- Fusion repeated: `artifacts/reports/combined_batch_01_02_03_04/fusion_repeated_splits_fixed_model_seed42`

## Frozen Primary Results
| model | MAE | RMSE | R2 | prediction_range_ratio |
| --- | --- | --- | --- | --- |
| Measurement RF | 169.44 | 227.5 | 0.24 | 0.34 |
| Corrected AS-OCT | 178.46 | 242.82 | 0.12 | 0.31 |
| Fusion | 182.8 | 241.87 | 0.14 | 0.32 |

## Frozen Repeated Results
| model | MAE_mean_std | MAE_min | MAE_max | prediction_range_ratio_mean |
| --- | --- | --- | --- | --- |
| Measurement RF | 163.15 +/- 14.09 | 144.517 | 181.831 | 0.37 |
| Corrected AS-OCT | 168.31 +/- 13.41 | 150.091 | 184.504 | 0.329 |
| Fusion | 171.35 +/- 12.15 | 158.04 | 184.65 | 0.352 |

## Decision
- No further tuning will be performed around the current primary/repeated baseline results.
- No additional calibration, weighted loss, lower learning rate, complex fusion, or repeated split generation is part of the frozen primary analysis.
- Secondary experiments may be added only if requested by a supervisor or reviewer.

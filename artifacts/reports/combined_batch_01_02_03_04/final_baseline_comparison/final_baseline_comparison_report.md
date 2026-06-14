# Final Baseline Comparison Report

## 1. Cohort and Evaluation Protocol
- Three completed baselines are frozen: measurement-only Random Forest, corrected AS-OCT-only, and AS-OCT + measurement concat fusion.
- Primary results are reported separately from repeated split stability results.
- Repeated evaluation uses patient-level split seeds 42, 1001, 2002, 2026, and 3407. AS-OCT and fusion repeated results use fixed model_seed42.
- Primary AS-OCT and fusion ensemble results are not averaged together with repeated fixed-seed results.

## 2. Primary Split Results
| model | evaluation_variant | n_test | MAE | RMSE | R2 | mean_signed_error | prediction_range_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Measurement RF | primary seed42 Random Forest | 49 | 169.444 | 227.504 | 0.237 | -4.703 | 0.342 |
| Corrected AS-OCT | primary corrected 3-seed ensemble | 51 | 178.457 | 242.82 | 0.116 | -56.436 | 0.31 |
| Fusion | primary 3-seed ensemble | 49 | 182.797 | 241.868 | 0.138 | -51.665 | 0.323 |

## 3. Repeated Split Stability
| model | MAE_mean_std | MAE_median | MAE_IQR | MAE_min | MAE_max | RMSE_mean | RMSE_std | prediction_range_ratio_mean | prediction_range_ratio_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Measurement RF | 163.15 +/- 14.09 | 164.338 | 13.839 | 144.517 | 181.831 | 219.971 | 19.242 | 0.37 | 0.043 |
| Corrected AS-OCT | 168.31 +/- 13.41 | 170.514 | 15.489 | 150.091 | 184.504 | 224.831 | 20.093 | 0.329 | 0.09 |
| Fusion | 171.35 +/- 12.15 | 174.201 | 21.189 | 158.04 | 184.65 | 232.98 | 17.054 | 0.352 | 0.074 |

- Best average repeated performance: Measurement RF.
- Most stable repeated MAE: Fusion.

## 4. Paired Model Comparisons
| comparison | delta_mean | delta_std | model_a | model_b | model_a_wins | model_b_wins |
| --- | --- | --- | --- | --- | --- | --- |
| AS-OCT vs RF | 5.165 | 7.429 | Corrected AS-OCT | Measurement RF | 1 | 4 |
| Fusion vs AS-OCT | 3.035 | 8.125 | Fusion | Corrected AS-OCT | 3 | 2 |
| Fusion vs RF | 8.2 | 10.983 | Fusion | Measurement RF | 1 | 4 |
- These are descriptive paired comparisons across five splits; no significance claim is made.

## 5. Range-Specific Error
| model | low_MAE_mean_std | medium_MAE_mean_std | high_MAE_mean_std | low_overestimation_proportion | high_underestimation_proportion |
| --- | --- | --- | --- | --- | --- |
| Measurement RF | 172.57 +/- 32.83 | 90.07 +/- 15.49 | 364.70 +/- 48.50 | 0.945 | 1.0 |
| Corrected AS-OCT | 174.50 +/- 27.69 | 98.30 +/- 15.29 | 381.19 +/- 105.98 | 0.957 | 0.975 |
| Fusion | 162.50 +/- 31.86 | 103.90 +/- 9.20 | 393.24 +/- 86.57 | 0.942 | 0.978 |
- High-vault cases remain the largest range-specific error source for all three model families.

## 6. Regression-to-the-Mean / Range Compression
- All three models show prediction range compression, with prediction range / label range ratios well below 1.0.
- Low-vault overestimation and high-vault underestimation are consistent across model families.

## 7. patient_100 OS Label Correction
- The original AS-OCT label 7901 um for `batch_03__patient_100_OS_20240517` was manually verified as a transcription error and corrected to 701 um.
- This correction affects AS-OCT-only results because the sample belongs to the AS-OCT cohort.
- Measurement-only and fusion cohorts/results are unaffected because this OS sample is not in the fusion/measurement-ready cohorts.
- Superseded 7901-label AS-OCT outputs are excluded from all formal tables in this package.

## 8. Main Scientific Findings
- Measurement-only RF currently provides the best average repeated-split performance.
- Corrected AS-OCT-only is competitive but does not consistently outperform RF.
- Simple concatenation fusion does not provide stable incremental benefit over measurement-only RF.
- Fusion is not consistently better than corrected AS-OCT-only across the five repeated splits.
- Image information is not completely invalid: AS-OCT and fusion are competitive on some splits, but the image models do not provide robust average superiority under this protocol.

## 9. Limitations
- The repeated split analysis has only five patient-level splits and should be interpreted descriptively.
- High-vault sample counts remain small, so high-range metrics are informative but unstable.
- The fusion architecture is intentionally simple and may not exhaust all possible multimodal approaches.

## 10. Recommended Manuscript Narrative
Measurement-only RF achieved the strongest average repeated-split performance. AS-OCT-only and simple concat fusion were competitive but did not consistently improve over measurement-only modeling. Across all models, low-vault overestimation, high-vault underestimation, and prediction range compression persisted, indicating that the dominant limitation is not merely modality choice but also range-dependent regression behavior.

## 11. Recommended Supervisor Update Narrative
The v4 baselines are now frozen after correcting the patient_100 OS label error and rerunning the affected AS-OCT analyses. Measurement-only RF remains the strongest and most stable baseline. Corrected AS-OCT is close but wins only one of five paired splits against RF. Fusion wins one of five against RF and three of five against corrected AS-OCT, so simple fusion does not justify a stronger claim yet.

## 12. Whether Further Experiments Are Necessary
- No further primary baseline experiments are necessary before reporting these findings.
- Complex fusion, weighted loss, lower learning rate, or calibration should be reserved as secondary experiments only if requested by a supervisor or reviewer.
- The current results are comparative evidence, not a clinical-readiness claim.

## Direct Answers
- Which model has best average performance? Measurement RF.
- Which model is most stable? Fusion by repeated MAE sample SD.
- Does AS-OCT stably outperform measurement-only? No; AS-OCT wins 1/5 against RF.
- Does fusion bring stable gain? No; fusion wins 1/5 against RF and 3/5 against corrected AS-OCT.
- Is image information completely ineffective? No, but it is not robustly superior under this baseline protocol.
- Is primary seed42 representative? Yes for the broad repeated-split range, but primary and repeated protocols are reported separately.
- Are low/high vault biases stable? Yes; low-vault overestimation and high-vault underestimation persist across models.

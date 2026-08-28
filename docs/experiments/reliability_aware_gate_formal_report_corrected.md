# Reliability-Aware Soft Gate Formal Evaluation Corrected

## 1. Correction Notice

This document supersedes the fixed-additive comparator statements in `docs/experiments/reliability_aware_gate_formal_report.md` without overwriting that historical report.

The frozen additive alpha is `0.35` and its resolved semantics are Measurement weight. The corrected fixed additive formula is:

`fixed_additive_pred = 0.35 * Measurement + 0.65 * AS-OCT`

G2 predictions themselves are unchanged. Measurement predictions are unchanged. AS-OCT predictions are unchanged. Oracle results are unchanged. Only the fixed additive comparator and downstream comparisons that depended on it are corrected. The source of the error was alpha semantic inversion in `scripts/run_reliability_aware_gate_experiment.py`.

## 2. Formal Repeated-Split Results

| split_seed | measurement_rf_test_mae_um | as_oct_v0_test_mae_um | matched_best_unimodal_mae_um | fixed_additive_fusion_test_mae_um | g2_test_mae_um | oracle_best_of_two_mae_um | oracle_fraction_captured | g2_beats_matched_best_unimodal | g2_beats_fixed_additive | corrected_fixed_additive_beats_matched_best_unimodal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42.000 | 202.686 | 216.838 | 202.686 | 200.770 | 209.142 | 166.282 | -0.177 | False | False | True |
| 1001.000 | 165.782 | 161.705 | 161.705 | 154.177 | 166.153 | 128.092 | -0.132 | False | False | True |
| 2002.000 | 176.343 | 170.115 | 170.115 | 159.003 | 172.307 | 124.144 | -0.048 | False | False | True |
| 2026.000 | 177.520 | 195.480 | 177.520 | 185.294 | 181.693 | 147.063 | -0.137 | False | True | False |
| 3407.000 | 138.062 | 155.747 | 138.062 | 140.290 | 129.890 | 96.665 | 0.197 | True | True | False |

## 3. Comparison With Baselines

| Method | Repeated MAE mean | Repeated MAE SD | Wins vs matched best unimodal | High-Vault MAE | Oracle fraction captured |
| --- | --- | --- | --- | --- | --- |
| Measurement RF | 172.08 | 23.35 | NA | 309.27 | NA |
| AS-OCT V0 | 179.98 | 25.58 | NA | 369.05 | NA |
| Matched best unimodal | 170.02 | 23.53 | Reference | NA | NA |
| Corrected fixed additive fusion | 167.91 | 24.56 | 3.00 | 346.71 | NA |
| G2 Reliability-Aware Soft Gate | 171.84 | 28.64 | 1.00 | 351.98 | -0.06 |
| Oracle best-of-two | 132.45 | 26.11 | NA | 277.72 | NA |

The G2 repeated mean MAE remains 171.84 +/- 28.64 um. G2 beats the matched best unimodal model in 1/5 splits and beats corrected fixed additive fusion in 2/5 splits.

## 4. High-Vault Analysis

| method | high_vault_mae_mean_um |
| --- | --- |
| Measurement RF | 309.27 |
| AS-OCT V0 | 369.05 |
| Corrected fixed additive fusion | 346.71 |
| G2 Reliability-Aware Soft Gate | 351.98 |
| Oracle best-of-two | 277.72 |

G2 does not alleviate the High-Vault failure mode. The corrected fixed additive High-Vault mean MAE is 346.71 um, while G2 High-Vault mean MAE is 351.98 um.

## 5. Gate Weight Behavior

Gate behavior and coefficient diagnostics are unchanged from the historical G2 formal archive because G2 predictions and fitted gate coefficients were not recomputed.

## 6. Multi-View Subgroup Diagnostic

| split_seed | multiview_test_n | multiview_measurement_rf_mae_um | multiview_as_oct_mae_um | multiview_fixed_additive_mae_um | multiview_g2_mae_um | multiview_oracle_mae_um |
| --- | --- | --- | --- | --- | --- | --- |
| 42.00 | 24.00 | 230.70 | 237.63 | 218.91 | 230.03 | 199.19 |
| 1001.00 | 30.00 | 178.88 | 158.53 | 154.91 | 170.36 | 135.60 |
| 2002.00 | 36.00 | 176.07 | 189.32 | 174.97 | 197.53 | 129.71 |
| 2026.00 | 40.00 | 176.70 | 201.40 | 190.92 | 184.46 | 155.54 |
| 3407.00 | 25.00 | 150.01 | 127.20 | 123.22 | 131.11 | 94.39 |

This subgroup is a secondary diagnostic only and was not used for model selection.

## 7. GO / NO-GO Decision

NO-GO.

Reliability-aware eye-specific gating did not provide stable improvement over the matched best unimodal model or the corrected validation-tuned fixed additive fusion across repeated patient-level splits.

## 8. Interpretation

Theoretical oracle complementarity still exists. However, the current reliability proxies did not reliably convert that headroom into learnable complementarity. The corrected comparator strengthens the conclusion that a more complex gate is not justified under the current information setting.

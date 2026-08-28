# Leakage-Safe Validation-Tuned Additive Fusion

## 1. Why protocol correction was required

The previous additive fusion result used one pooled global alpha selected from matched validation predictions across all repeated outer split seeds. A cross-split role audit found that every reported outer test split contained patients and eyes that had contributed labels to the pooled validation objective in another split. The previous global-alpha result is therefore retained as a historical diagnostic, not as the final unbiased repeated-test formal result.

## 2. Previous pooled-alpha contamination

Historical pooled-global alpha diagnostic:

- alpha = 0.35, Measurement weight
- MAE = 167.91 +/- 24.56 um
- wins vs matched best unimodal = 3/5
- label: NOT VALID AS UNBIASED REPEATED-TEST FORMAL RESULT

## 3. Corrected per-outer-split protocol

For each formal outer split, alpha was selected using only that split's validation predictions and labels. The original grid, formula, selection metric, and tie rule were preserved.

Formula:

`pred_additive = alpha * Measurement + (1 - alpha) * AS-OCT`

Grid: `0.00, 0.05, ..., 1.00`

Tie rule: among alpha values whose validation MAE is within 0.5 um of the best validation MAE, choose the largest alpha.

## 4. Alpha selection

| split_seed | n_val | n_test | selected_alpha_measurement | selected_alpha_as_oct | best_validation_mae | selected_validation_mae | tie_rule_affected_selected_alpha |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42.0000 | 56.0000 | 56.0000 | 0.9000 | 0.1000 | 170.3931 | 170.7851 | True |
| 1001.0000 | 56.0000 | 56.0000 | 0.5500 | 0.4500 | 120.6875 | 121.0710 | True |
| 2002.0000 | 56.0000 | 56.0000 | 0.0500 | 0.9500 | 172.4356 | 172.8272 | True |
| 2026.0000 | 56.0000 | 56.0000 | 0.4000 | 0.6000 | 164.6762 | 165.1140 | True |
| 3407.0000 | 56.0000 | 56.0000 | 0.0500 | 0.9500 | 167.3108 | 167.6893 | True |

Alpha summary:

- mean = 0.390
- SD = 0.360
- median = 0.400
- min/max = 0.050/0.900

## 5. Leakage prevention

Same-split validation/test patient overlap was zero for all five splits. Same-split validation/test eye overlap was zero for all five splits. Test labels were not used for alpha selection. No pooled cross-split validation selection remains in the final protocol.

## 6. Repeated held-out test performance

| split_seed | n_test | measurement_rf_test_mae_um | as_oct_v0_test_mae_um | matched_best_unimodal_mae_um | leakage_safe_additive_test_mae_um | concat_test_mae_um | g2_test_mae_um | oracle_best_of_two_mae_um |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42.000 | 56.000 | 202.686 | 216.838 | 202.686 | 200.544 | 203.293 | 209.142 | 166.282 |
| 1001.000 | 56.000 | 165.782 | 161.705 | 161.705 | 155.071 | 163.409 | 166.153 | 128.092 |
| 2002.000 | 56.000 | 176.343 | 170.115 | 170.115 | 167.600 | 174.793 | 172.307 | 124.144 |
| 2026.000 | 56.000 | 177.520 | 195.480 | 177.520 | 183.839 | 192.518 | 181.693 | 147.063 |
| 3407.000 | 56.000 | 138.062 | 155.747 | 138.062 | 152.656 | 144.921 | 129.890 | 96.665 |

Final leakage-safe additive repeated MAE:

`171.94 +/- 20.22 um`

## 7. Comparison with unimodal and concat

| split_seed | additive_delta_vs_measurement_um | additive_delta_vs_as_oct_um | additive_delta_vs_matched_best_unimodal_um | additive_delta_vs_concat_um | additive_delta_vs_g2_um |
| --- | --- | --- | --- | --- | --- |
| 42.000 | -2.142 | -16.294 | -2.142 | -2.749 | -8.598 |
| 1001.000 | -10.711 | -6.634 | -6.634 | -8.338 | -11.082 |
| 2002.000 | -8.744 | -2.516 | -2.516 | -7.193 | -4.708 |
| 2026.000 | 6.318 | -11.641 | 6.318 | -8.679 | 2.146 |
| 3407.000 | 14.594 | -3.091 | 14.594 | 7.735 | 22.766 |

Wins:

- vs Measurement RF: 3/5
- vs AS-OCT V0: 5/5
- vs matched best unimodal: 3/5
- vs concat: 4/5
- vs G2: 3/5

## 8. High-Vault performance

| split_seed | n_high | measurement_high_vault_mae_um | as_oct_high_vault_mae_um | matched_best_unimodal_high_vault_mae_um | leakage_safe_additive_high_vault_mae_um | concat_high_vault_mae_um | g2_high_vault_mae_um | oracle_high_vault_mae_um |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42.000 | 18.000 | 318.847 | 415.735 | 318.847 | 325.975 | 392.944 | 386.625 | 308.935 |
| 1001.000 | 9.000 | 403.927 | 397.860 | 397.860 | 401.197 | 383.895 | 391.848 | 374.885 |
| 2002.000 | 7.000 | 300.860 | 386.706 | 300.860 | 382.413 | 306.358 | 415.179 | 254.079 |
| 2026.000 | 16.000 | 290.765 | 353.084 | 290.765 | 328.157 | 314.735 | 338.835 | 271.970 |
| 3407.000 | 8.000 | 231.945 | 291.890 | 231.945 | 288.648 | 267.888 | 227.413 | 178.741 |
| mean | 58.000 | 309.269 | 369.055 | 308.055 | 345.278 | 333.164 | 351.980 | 277.722 |

Leakage-safe additive High-Vault MAE was `345.28 um`.

## 9. Range compression

| method | target_range_um | prediction_range_um | prediction_range_ratio | prediction_sd_to_target_sd | low_signed_error_um | medium_signed_error_um | high_signed_error_um |
| --- | --- | --- | --- | --- | --- | --- | --- |
| as_oct_v0 | 1114.300 | 385.123 | 0.367 | 0.360 | 297.192 | 49.240 | -362.575 |
| concat | 1114.300 | 433.662 | 0.402 | 0.424 | 255.616 | 46.946 | -329.544 |
| g2 | 1114.300 | 428.295 | 0.402 | 0.382 | 295.610 | 61.765 | -342.416 |
| leakage_safe_additive | 1114.300 | 412.806 | 0.387 | 0.364 | 315.466 | 72.131 | -336.888 |
| measurement_rf | 1114.300 | 549.688 | 0.505 | 0.477 | 341.206 | 96.310 | -305.820 |

Leakage-safe additive prediction range ratio was `0.387`. Range compression persists.

## 10. Oracle headroom

| split_seed | matched_best_unimodal_mae_um | leakage_safe_additive_test_mae_um | oracle_best_of_two_mae_um | oracle_fraction_captured | oracle_fraction_median | oracle_fraction_min | oracle_fraction_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42.000 | 202.686 | 200.544 | 166.282 | 0.059 | NA | NA | NA |
| 1001.000 | 161.705 | 155.071 | 128.092 | 0.197 | NA | NA | NA |
| 2002.000 | 170.115 | 167.600 | 124.144 | 0.055 | NA | NA | NA |
| 2026.000 | 177.520 | 183.839 | 147.063 | -0.207 | NA | NA | NA |
| 3407.000 | 138.062 | 152.656 | 96.665 | -0.353 | NA | NA | NA |
| summary | NA | NA | NA | -0.050 | 0.055 | -0.353 | 0.197 |

Mean oracle fraction captured was `-0.050`.

## 11. Per-eye diagnostic

| pooled_repeated_test_n | unique_eyes | mean_delta_um | median_delta_um | fraction_improved | fraction_worsened | fraction_abs_delta_lt_10um | fraction_improvement_gt_25um | fraction_worsening_gt_25um |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 280.000 | 207.000 | 39.492 | 19.987 | 0.121 | 0.879 | 0.311 | 0.018 | 0.446 |

The pooled repeated-test diagnostic may include the same physical eye in different outer test splits. This is a repeated-split robustness diagnostic, not an independent-eye cohort.

## 12. Comparison with historical global-alpha diagnostic

| Protocol | MAE mean | MAE SD | Wins vs matched best unimodal | Formal status |
| --- | ---: | ---: | ---: | --- |
| Historical pooled-global alpha=0.35 | 167.91 | 24.56 | 3/5 | NOT VALID AS UNBIASED REPEATED-TEST FORMAL RESULT |
| Leakage-safe per-split validation alpha | 171.94 | 20.22 | 3/5 | Final corrected formal result |

Differences between these rows reflect protocol correction, not model retraining or inference changes.

## 13. Implication for manuscript

The previous paper-readiness conclusion must be updated using the leakage-safe formal result. The corrected result should replace the pooled-global alpha diagnostic wherever an unbiased repeated-test additive comparator is required.

G2-only metrics remain unchanged, and G2 remains NO-GO unless its predefined decision criteria are changed for independent reasons.

## 14. Final formal additive result

Final leakage-safe additive fusion result:

`171.94 +/- 20.22 um`

Paper-readiness decision:

B. PAPER-READY SUPPORTING METHOD, NOT STRONG ENOUGH AS CORE NOVELTY

Recommended manuscript role:

primary simple fusion baseline / supporting fusion result

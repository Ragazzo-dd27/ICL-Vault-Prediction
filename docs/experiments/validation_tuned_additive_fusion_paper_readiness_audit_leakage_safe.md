# Validation-Tuned Additive Fusion Paper-Readiness Audit Leakage-Safe

## 1. Basis for re-evaluation

This paper-readiness conclusion supersedes the previous pooled-global alpha additive conclusion for formal repeated-test reporting. The previous `167.91 +/- 24.56 um` result is retained as a historical diagnostic only because cross-split role contamination was detected in the pooled global alpha selection.

## 2. Leakage-safe formal result

The corrected protocol selects alpha independently within each outer split validation set, using the original grid, formula, validation MAE metric, and tie rule. It then evaluates that alpha once on the corresponding held-out test split.

Final leakage-safe additive MAE:

`171.94 +/- 20.22 um`

Wins:

- vs matched best unimodal: 3/5
- vs concat: 4/5
- vs G2: 3/5

## 3. High-Vault and range behavior

High-Vault MAE:

`345.28 um`

Prediction range ratio:

`0.387`

The leakage-safe additive result should be interpreted with its High-Vault and range-compression limitations intact.

## 4. Oracle and per-eye diagnostics

Mean oracle fraction captured:

`-0.050`

Per-eye diagnostic:

| pooled_repeated_test_n | unique_eyes | mean_delta_um | median_delta_um | fraction_improved | fraction_worsened | fraction_abs_delta_lt_10um | fraction_improvement_gt_25um | fraction_worsening_gt_25um |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 280.000 | 207.000 | 39.492 | 19.987 | 0.121 | 0.879 | 0.311 | 0.018 | 0.446 |

## 5. Decision

B. PAPER-READY SUPPORTING METHOD, NOT STRONG ENOUGH AS CORE NOVELTY

Recommended manuscript role:

primary simple fusion baseline / supporting fusion result

This decision is derived from the leakage-safe outer-test result only. It does not reuse the previous pooled-global alpha paper-readiness decision.

## 6. G2 comparison

G2 was not rerun. Existing corrected G2 predictions/results were used. G2-only metrics and the G2 NO-GO decision remain unchanged. The additive-vs-G2 comparison is updated only through the corrected additive protocol.

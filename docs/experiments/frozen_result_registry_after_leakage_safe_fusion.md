# Frozen Result Registry After Leakage-Safe Fusion

This registry is the manuscript-facing evidence freeze after the leakage-safe validation-tuned additive fusion correction. Manuscript revisions should cite formal numbers from this registry and its listed source artifacts, not from superseded pooled-alpha or alpha-inverted outputs.

No manuscript or LaTeX source was modified while creating this registry. No model was trained, no CNN inference was run, and no RF/G2 experiment was rerun.

## Authoritative Source Artifacts

- Full-cohort v5 baselines: `artifacts/v5_overall_comparison/v5_model_ranking_summary.csv`
- Full-cohort v5 formal summary: `artifacts/v5_1_final_summary/v5_1_formal_experiment_summary.csv`
- Leakage-safe additive archive: `artifacts/v5_2_matched_fusion_audit/additive_fusion_leakage_safe_formal/`
- Corrected G2 archive: `artifacts/reports/v5_2_matched_fusion_audit/reliability_aware_gate_multiview_v1_formal_corrected/`
- Gate learnability summary: `artifacts/v5_2_final_summary/v5_2_gate_summary.csv`
- Alpha semantics resolution: `artifacts/v5_2_matched_fusion_audit/alpha_semantics_resolution.json`
- Cross-split leakage audit: `artifacts/v5_2_matched_fusion_audit/additive_fusion_cross_split_leakage_audit.json`

## A. Full-Cohort / Primary Baseline Results

These are full-cohort v5 baseline results. They are not the same cohort/protocol as the matched v5.2 repeated fusion audit and must not be blended without explanation.

| Method | Cohort / protocol | Deployable | Role | Formal status | Repeated MAE mean +/- SD (um) | Primary split MAE (um) | Range ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Measurement-only RF | Full v5 measurement cohort, repeated patient-level splits | Yes | Primary baseline | Formal | 172.79 +/- 22.38 | 200.79 | 0.510 |
| AS-OCT-only V0 | Full v5 AS-OCT cohort, repeated patient-level splits | Yes | Primary baseline | Formal | 173.78 +/- 15.77 | 175.00 | 0.457 |
| Simple concat fusion | Full v5 fusion cohort, repeated patient-level splits | Yes | Naive multimodal baseline | Formal | 175.79 +/- 23.15 | 193.50 | 0.402 |

Interpretation: the full-cohort concat fusion baseline does not show stable improvement over the unimodal baselines.

## B. Matched Repeated-Split Results

Matched v5.2 repeated-split results use the same held-out test eyes for Measurement RF, AS-OCT V0, concat, leakage-safe additive fusion, G2, matched best unimodal, and Oracle Best-of-Two. There are 5 outer splits with 56 test eyes per split.

| Method | Deployable | Primary model or diagnostic | Formal / historical | Cohort | Split protocol | Repeated MAE mean +/- SD (um) | Wins vs matched best unimodal |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| Measurement RF | Yes | Primary unimodal baseline | Formal | Matched v5.2 test eyes | Repeated patient-level outer splits | 172.08 +/- 23.35 | 0/5 |
| AS-OCT V0 | Yes | Primary unimodal baseline | Formal | Matched v5.2 test eyes | Repeated patient-level outer splits | 179.98 +/- 25.58 | 0/5 |
| Concat fusion | Yes | Naive fusion baseline | Formal | Matched v5.2 test eyes | Repeated patient-level outer splits | 175.79 +/- 23.15 | 0/5 |
| Leakage-safe validation-tuned additive fusion | Yes | Simple conventional fusion baseline / supporting method | Formal | Matched v5.2 test eyes | Per-split validation-only alpha selection, then held-out test evaluation | 171.94 +/- 20.22 | 3/5 |
| G2 reliability-aware soft gate | Yes, but not recommended as final predictor | Adaptive fusion learnability diagnostic | Formal corrected, NO-GO | Matched v5.2 test eyes | Gate fit on same-split validation only; held-out test evaluation | 171.84 +/- 28.64 | 1/5 |
| Matched best unimodal | No | Diagnostic benchmark | Formal diagnostic | Matched v5.2 test eyes | Post-hoc split-level selection of better unimodal test MAE | 170.02 +/- 23.53 | Reference |
| Oracle Best-of-Two | No | Retrospective theoretical diagnostic | Formal diagnostic | Matched v5.2 test eyes | Post-hoc per-eye selection of lower absolute unimodal error | 132.45 +/- 26.11 | Diagnostic only |

### Leakage-Safe Additive Definition

Formal split-specific Measurement alphas:

| Split seed | Measurement alpha | AS-OCT weight | Selected validation MAE (um) | Held-out test MAE (um) |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.90 | 0.10 | 170.7851 | 200.5439 |
| 1001 | 0.55 | 0.45 | 121.0710 | 155.0710 |
| 2002 | 0.05 | 0.95 | 172.8272 | 167.5996 |
| 2026 | 0.40 | 0.60 | 165.1140 | 183.8386 |
| 3407 | 0.05 | 0.95 | 167.6893 | 152.6560 |

Formula:

`pred_additive = alpha * Measurement + (1 - alpha) * AS-OCT`

Alpha semantics: Measurement weight.

## C. Vault-Range Results

Matched leakage-safe archive supports Low / Medium / High range summaries. Mean matched repeated-split range MAEs:

| Method | Low MAE (um) | Medium MAE (um) | High MAE (um) |
| --- | ---: | ---: | ---: |
| Measurement RF | 341.21 | 129.56 | 309.27 |
| AS-OCT V0 | 297.19 | 122.76 | 369.05 |
| Concat fusion | 255.62 | 127.08 | 333.16 |
| Leakage-safe additive fusion | 315.47 | 122.07 | 345.28 |
| G2 reliability-aware gate | 295.61 | 117.72 | 351.98 |

Interpretation: High-Vault prediction remains difficult. Leakage-safe additive fusion does not solve High-Vault error.

## D. High-Vault

Formal matched High-Vault MAE means:

| Method | High-Vault MAE (um) |
| --- | ---: |
| Measurement RF | 309.27 |
| AS-OCT V0 | 369.05 |
| Matched best unimodal | 308.06 |
| Concat fusion | 333.16 |
| Leakage-safe additive fusion | 345.28 |
| G2 reliability-aware gate | 351.98 |
| Oracle Best-of-Two | 277.72 |

The High-Vault result should be treated as a persistent failure mode, not a solved endpoint.

## E. Range Compression

Matched prediction range ratios:

| Method | Prediction range ratio | Prediction SD / target SD |
| --- | ---: | ---: |
| Measurement RF | 0.505 | 0.477 |
| AS-OCT V0 | 0.367 | 0.360 |
| Concat fusion | 0.402 | 0.424 |
| Leakage-safe additive fusion | 0.387 | 0.364 |
| G2 reliability-aware gate | 0.402 | 0.382 |

Interpretation: prediction range compression persists. Additive averaging does not correct dynamic-range compression.

## F. Oracle Complementarity

Oracle Best-of-Two is a retrospective theoretical diagnostic. It is not a deployable model and not a fusion algorithm.

| Quantity | Value |
| --- | ---: |
| Oracle repeated MAE mean +/- SD (um) | 132.45 +/- 26.11 |
| Oracle High-Vault MAE (um) | 277.72 |
| Leakage-safe additive mean oracle fraction captured | -0.0498 |
| Leakage-safe additive median oracle fraction captured | 0.0547 |
| Oracle fraction captured range | -0.3526 to 0.1974 |

Interpretation: the modalities contain retrospective per-eye complementarity, but leakage-safe additive fusion does not reliably capture that headroom.

## G. Gate Diagnostics

G0/G1 gate learnability diagnostics:

| Diagnostic | Winner AUC mean +/- SD | OOF MAE mean +/- SD (um) | Wins vs nested fixed alpha | Oracle fraction captured | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| G0 prediction-only | 0.521 +/- 0.069 | 163.88 +/- 22.46 | 2/5 | -0.0928 | NO-GO |
| G1 prediction + measurements | 0.456 +/- 0.107 | 165.77 +/- 21.20 | 1/5 | -0.1599 | NO-GO |

G2 reliability-aware soft gate:

| Quantity | Value |
| --- | ---: |
| G2 repeated MAE mean +/- SD (um) | 171.84 +/- 28.64 |
| Wins vs matched best unimodal | 1/5 |
| Wins vs corrected fixed additive comparator in corrected G2 archive | 2/5 |
| Wins vs leakage-safe additive correction | Additive beats G2 in 3/5 splits |
| G2 High-Vault MAE (um) | 351.98 |
| Mean extreme alpha fraction | 0.825 |
| Coefficient stability | Mixed signs across splits for all gate features |
| Final decision | NO-GO |

Interpretation: adaptive eye-specific flexibility did not translate into stable benefit. It is also not correct to claim that fixed additive definitively outperforms G2; the better statement is that neither simple additive nor adaptive reliability-aware fusion consistently exploited the available oracle complementarity.

## H. Invalidated / Superseded Results

Do not use the following as manuscript evidence:

| Result / artifact family | Reason | Manuscript status |
| --- | --- | --- |
| Pooled global alpha result, 167.91 +/- 24.56 um | Alpha selected from pooled cross-split validation; cross-split role contamination detected | DO NOT USE IN MANUSCRIPT as formal repeated-test evidence |
| `0.35 * AS-OCT + 0.65 * Measurement` fixed-additive comparator | Alpha semantics reversed; frozen alpha is Measurement weight | DO NOT USE IN MANUSCRIPT |
| Provisional additive paper-readiness audits before alpha and leakage resolution | Superseded by formal semantic and leakage-safe correction | DO NOT USE IN MANUSCRIPT |
| Old uncorrected G2 fixed-additive comparator outputs | Propagated alpha semantic inversion in fixed-additive comparator columns | DO NOT USE IN MANUSCRIPT |
| Any additive result that did not pass final QC | Not part of authoritative formal archive | DO NOT USE IN MANUSCRIPT |

The historical pooled-alpha result may be retained only as a correction/provenance note outside main claims. It must not enter the Abstract, main Results, main tables, Conclusion, or contribution claims.

## Terminology Freeze

Matched best unimodal: for each outer split, the better-performing unimodal model is selected post-hoc according to that split's two unimodal test MAEs. This is a diagnostic benchmark, not a deployable prospectively selected baseline.

Oracle Best-of-Two: for each test eye, the unimodal prediction with smaller absolute error is selected post-hoc. This is a retrospective theoretical diagnostic, not a deployable model and not a fusion algorithm.

Leakage-safe additive fusion: conventional late fusion with split-specific validation-only alpha selection. It is deployable in form, but its evidence supports only a simple supporting fusion baseline, not a strong novel method.

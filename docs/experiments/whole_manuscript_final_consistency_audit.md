# Whole-Manuscript Final Consistency Audit

This is a read-only consistency audit of the current main manuscript and current Supplementary after the leakage-safe evidence freeze and all manuscript rewrites. No manuscript source, supplementary source, title, experiment artifact, prediction CSV, model, or figure file was modified. No experiment, training, inference, commit, or push was performed.

## Executive verdict

Status: NOT READY FOR TITLE FINALIZATION

No CRITICAL factual or protocol error was found in the current manuscript/supplementary sources. The manuscript no longer uses the invalid pooled-alpha additive result as formal evidence, no reversed additive alpha formula was found, and the main numerical claims map to the frozen registry.

However, three issues should be addressed before title finalization and final Overleaf compile:

- HIGH: Figure 1 visually places the POD1 target-label box under the left-column heading "Preoperative Inputs", which could imply that POD1 vault is an input despite the box text and Methods stating otherwise.
- HIGH: Related Work states that additive fusion is "used only as matched diagnostics", while the final protocol treats leakage-safe additive as a deployable conventional late-fusion baseline/supporting method evaluated in the strict matched cohort.
- MEDIUM: Clinical risk/planning claims in the opening Introduction paragraph are plausible and likely supported by existing cited literature, but the risk sentence itself has no immediate citation marker.

## Authoritative result map

### Full-cohort repeated evaluation

| Method | Cohort/protocol | Role | Repeated MAE mean +/- SD (um) | Primary split MAE (um) | Range ratio |
| --- | --- | --- | ---: | ---: | ---: |
| Measurement RF | Full v5 measurement cohort | Formal deployable unimodal baseline | 172.79 +/- 22.38 | 200.79 | 0.510 |
| AS-OCT V0 | Full v5 AS-OCT cohort | Formal deployable unimodal baseline | 173.78 +/- 15.77 | 175.00 | 0.457 |
| Feature concatenation | Full v5 fusion cohort | Formal deployable conventional baseline | 175.79 +/- 23.15 | 193.50 | 0.402 |

### Strict matched repeated evaluation

| Method | Role | Repeated MAE mean +/- SD (um) | Wins vs matched best |
| --- | --- | ---: | ---: |
| Measurement RF | Deployable matched unimodal baseline | 172.08 +/- 23.35 | 0/5 |
| AS-OCT V0 | Deployable matched unimodal baseline | 179.98 +/- 25.58 | 0/5 |
| Concat fusion | Deployable conventional fusion baseline | 175.79 +/- 23.15 | 0/5 |
| Leakage-safe additive fusion | Deployable conventional late-fusion baseline/supporting method | 171.94 +/- 20.22 | 3/5 |
| Matched best unimodal | Retrospective split-level diagnostic | 170.02 +/- 23.53 | Reference |
| Oracle Best-of-Two | Retrospective per-eye theoretical diagnostic | 132.45 +/- 26.11 | Diagnostic |
| G2 reliability-aware soft gate | Formal corrected adaptive learnability diagnostic, NO-GO | 171.84 +/- 28.64 | 1/5 |

### Failure-mode results

| Method | High-Vault MAE (um) | Prediction range ratio |
| --- | ---: | ---: |
| Measurement RF | 309.27 | 0.505 |
| AS-OCT V0 | 369.05 | 0.367 |
| Concat fusion | 333.16 | 0.402 |
| Leakage-safe additive fusion | 345.28 | 0.387 |
| G2 reliability-aware gate | 351.98 | 0.402 |
| Oracle Best-of-Two | 277.72 | NA |

### Gate diagnostics

| Diagnostic | Winner AUC mean +/- SD | OOF/test MAE mean +/- SD (um) | Wins | Oracle fraction captured | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| G0 prediction-only | 0.521 +/- 0.069 | 163.88 +/- 22.46 OOF | 2/5 vs nested fixed alpha | -0.0928 | NO-GO |
| G1 prediction + measurements | 0.456 +/- 0.107 | 165.77 +/- 21.20 OOF | 1/5 vs nested fixed alpha | -0.1599 | NO-GO |
| G2 reliability-aware soft gate | NA | 171.84 +/- 28.64 test | 1/5 vs matched best | about -0.06 | NO-GO |

## Critical errors

None found.

Specific CRITICAL checks passed:

- No formal use of `167.91 +/- 24.56 um` in `main.tex` or current Supplementary.
- No formal use of `166.23 +/- 23.34 um` in `main.tex` or current Supplementary.
- No `2.11 um` additive gain in `main.tex` or current Supplementary.
- No reversed additive formula `0.35 * AS-OCT + 0.65 * Measurement`.
- Oracle is not presented as deployable.
- Matched best unimodal is not presented as a deployable model.
- G2 is not presented as the proposed final predictor.
- Full-cohort and strict matched cohorts are explicitly distinguished in Abstract, Methods, Results, and Supplementary.

## High-priority warnings

### HIGH: Figure 1 target-label grouping

File: `manuscript/overleaf_current/figures/figure1_multimodal_framework_final.png`

The figure is captioned and referenced correctly, and the figure itself labels the target box as "POD1 postoperative vault". However, the target-label box appears in the left panel under the heading "A. Preoperative Inputs". This may visually imply that POD1 vault is one of the preoperative inputs. The main Methods text correctly states that postoperative POD1 CASIA2 reports were used only to define the label and were never used as predictive inputs, so this is a figure/readability risk rather than a manuscript-text protocol error.

Recommended correction before final layout: visually separate the target label from the "Preoperative Inputs" panel or relabel the left group so that the target cannot be read as a preoperative input.

### HIGH: Related Work additive role wording

File: `manuscript/overleaf_current/main.tex`, Related Work, line 58.

Current wording says: "Additive fusion, oracle best-of-two, and low-complexity gating are used only as matched diagnostics."

This conflicts with the final evidence hierarchy, where leakage-safe additive fusion is a deployable conventional late-fusion baseline/supporting method in the strict matched cohort, while Oracle and gate diagnostics are diagnostics. This wording does not change any numerical result, but it blurs the additive role.

Recommended correction before title finalization: distinguish additive as a conventional matched late-fusion baseline and reserve "diagnostic" wording for Oracle/matched best/gate diagnostics.

## Medium-priority wording issues

### MEDIUM: Clinical citation proximity

File: `manuscript/overleaf_current/main.tex`, Introduction opening paragraph.

The paragraph states that low vault may increase lens-contact/cataract risk, high vault may narrow the angle/elevate IOP, and preoperative estimates could support lens selection/planning/risk assessment. These claims are standard and likely supported by the existing ICL/vault citations that follow in the next paragraph, but the risk sentence itself has no immediate citation marker.

Recommended correction before final submission: either attach an existing citation to the risk/planning sentence or confirm that the following citation cluster sufficiently supports the claim under target journal style.

### MEDIUM: "Can be exploited" phrasing in Abstract/objective

The Abstract and Introduction ask whether complementarity "can be exploited" by conventional or adaptive fusion. This is framed as a research question and is answered negatively/with caution, so it is acceptable. If further tightening is desired, "can be reliably exploited" could be preserved consistently wherever this question appears.

## Low-priority issues

### LOW: Long Abstract and dense Results paragraphs

The Abstract is within the documented word count, but it is dense. The Results contains several long paragraphs with multiple numeric clauses. This is a layout/readability issue only.

### LOW: Long table/figure captions

Figure 1 and Table `tab:v52_matched` captions are accurate but long. They may need final Overleaf layout inspection.

## Cohort consistency

PASS with one wording watch.

The Abstract explicitly separates "full-cohort repeated evaluation" from "strict matched cohort". The Results separates full-cohort v5 baseline characterization from strict matched v5.2 comparisons. The Methods states that modality-specific full cohorts and stricter matched cohorts are distinct. Supplementary retains full-cohort v5 reference tables and separately labels strict matched v5.2 interaction analysis.

The only wording watch is the Abstract's compact combination of full-cohort and strict-matched results in one paragraph. It is clear enough because each number group is introduced by cohort, but final copyediting should preserve those cohort labels.

## Alpha semantics

PASS.

Additive alpha semantics:

`pred_additive = alpha_s * Measurement + (1 - alpha_s) * AS-OCT`

`alpha_s` is the Measurement weight.

The Methods equation, Methods text, Results text, Discussion text, and Supplementary v5.2 equation all match this definition. The formal split-specific Measurement weights are reported correctly where values appear:

- seed42: 0.90
- seed1001: 0.55
- seed2002: 0.05
- seed2026: 0.40
- seed3407: 0.05

No stale global `alpha=0.35` formula appears in the manuscript or current Supplementary.

## G2 alpha semantics

PASS.

G2 is defined as:

`pred_G2 = alpha_i * AS-OCT + (1 - alpha_i) * Measurement`

`alpha_i` is explicitly stated to be the AS-OCT weight, distinct from additive `alpha_s`, which is the Measurement weight. The G2 reliability features are described as prediction-dispersion reliability proxies, not calibrated uncertainty estimates.

## Oracle / matched-best roles

PASS.

Oracle Best-of-Two is consistently described as retrospective, per-eye, post-hoc/test-label dependent, theoretical, non-deployable, and not a fusion algorithm/model. Matched best unimodal is consistently described as a retrospective split-level diagnostic benchmark, not a deployable prospectively selected model.

## G0/G1/G2 roles

PASS.

G0/G1/G2 are described as fusion learnability diagnostics. G2 is not called the proposed method, final predictor, or successful adaptive fusion architecture. The NO-GO conclusion is limited to evaluated strategies, current inputs, and current cohort.

## High-Vault / range compression

PASS.

High-Vault is consistently presented as persistent/unresolved. The registry-supported High-Vault values are correctly used in the main Results/Discussion and Supplementary where relevant:

- Measurement RF: 309.27 um
- AS-OCT V0: 369.05 um
- Concat: 333.16 um
- Leakage-safe additive: 345.28 um
- G2: 351.98 um
- Oracle: 277.72 um

Prediction-range compression is described as persistent and compatible with regression-to-the-mean behavior, not as a proven architectural cause. The leakage-safe additive range ratio is correctly reported as 0.387.

## Abstract consistency

PASS with cohort-label watch.

Every Abstract number maps to the frozen registry:

- Measurement RF full-cohort repeated MAE: 172.79 +/- 22.38 um
- AS-OCT V0 full-cohort repeated MAE: 173.78 +/- 15.77 um
- Feature concatenation full-cohort repeated MAE: 175.79 +/- 23.15 um
- Leakage-safe additive strict matched repeated MAE: 171.94 +/- 20.22 um
- Oracle strict matched repeated MAE: 132.45 +/- 26.11 um
- G2 strict matched repeated MAE: 171.84 +/- 28.64 um

The Abstract correctly states that additive and G2 did not provide robust/stable improvement and that High-Vault/range compression remained persistent failure modes.

## Main vs Supplementary consistency

PASS.

Supplementary now uses leakage-safe additive, corrected G2, retrospective Oracle wording, and matched-best split-level wording. It no longer retains the contaminated pooled additive row as a formal result. Supplementary v5.1 exploratory material is labeled as validation-only screening or exploratory, and no v5.1 exploratory result is promoted over frozen v5 baselines.

## Figure/Table consistency

PASS for source and captions, with one HIGH Figure 1 visual warning.

Figure 1:

- First figure in main source: yes.
- Uses `figure*`: yes.
- Label: `fig:multimodal_framework`.
- File path exists: `manuscript/overleaf_current/figures/figure1_multimodal_framework_final.png`.
- Caption has no performance numbers.
- Caption does not call G2 proposed or Oracle deployable.
- Visual formula for additive appears to use Measurement-weight semantics.
- G2/Oracle appear as diagnostic branches.
- Warning: POD1 target label is visually under the "Preoperative Inputs" heading.

Other figures/tables:

- `fig_v5_overall_comparison.pdf` remains as a later Results figure and its caption is consistent with frozen v5 comparison.
- Table captions distinguish deployable predictors and retrospective diagnostics.
- Table column counts passed source-level check.

## Method-Results alignment

PASS.

Method components and Results coverage:

- AS-OCT model: Method defined; Results report full-cohort and matched AS-OCT performance.
- Measurement model: Method defined; Results report full-cohort and matched Measurement performance.
- Feature concat: Method defined; Results report full-cohort and matched concat performance.
- Leakage-safe additive: Method defined; Results report strict matched leakage-safe additive performance.
- Matched complementarity: Method defines matched best and Oracle; Results report both.
- G0/G1/G2: Method defines diagnostics; Results report G0/G1/G2 outcomes.
- Evaluation/range/failure analysis: Method defines MAE/RMSE/R2, strata, and range compression; Results report corresponding values.

No result without method and no major method without result was found.

## Introduction-Discussion-Conclusion alignment

PASS.

The Introduction asks whether preoperative AS-OCT and CASIA2 measurements contain complementary information and whether this complementarity can be learned by fusion. The Results answer with competitive unimodal baselines, weak conventional fusion, large retrospective Oracle headroom, and NO-GO gate diagnostics. The Discussion explains complementarity versus learnability without overclaiming. The Conclusion restates the same limited answer without adding new claims.

## Related Work audit

MEDIUM/HIGH wording issue found.

The Related Work correctly states that Tensor Fusion, low-rank multimodal fusion, co-attention, cross-modal attention, and Transformer fusion were not implemented or presented as study methods. It does not imply that the current paper implemented those complex fusion architectures.

The problematic sentence is the additive role statement at line 58, where additive fusion is grouped with Oracle and low-complexity gating as "used only as matched diagnostics." This should be revised in a later editing pass.

## Citation/clinical claim checks

CITATION/CLAIM CHECK NEEDED.

Existing bibliography supports the general clinical and modeling background. All citation keys in `main.tex` are defined in `references.bib`. No new citation key is needed from the audit alone.

The Introduction opening paragraph contains clinical risk/planning claims without an immediate citation marker. This should be checked against journal expectations. Do not add references automatically without user approval.

## LaTeX/source checks

PASS at source level. Local LaTeX compile was not performed.

Checks performed:

- `abstract`, `table`, `table*`, `figure`, `figure*`, `equation`, and `enumerate` begin/end counts matched in main and supplementary source.
- No duplicate labels found within main or supplementary source.
- No undefined references found within main or supplementary source.
- All `\cite{...}` keys in `main.tex` are present in `references.bib`.
- All `\includegraphics{...}` paths in main and supplementary source exist.
- Tabular column counts passed for main and Supplementary.
- Supplementary compile structure uses `\input{supplementary_v5_experiments}` and the input file exists.

## Title readiness

Verdict: MINOR REVISION.

Current title:

`Preoperative AS-OCT and CASIA2 2DAnalysis Measurements for Early ICL Vault Prediction: A Matched-Cohort Multimodal Analysis`

Assessment:

- Accurate: yes.
- Not overclaiming: yes.
- Does not imply a new architecture: yes.
- Matches current contribution broadly: yes.
- Highlights multimodal analysis: yes.
- Does not strongly foreground complementarity versus learnability: mild limitation.
- Long but acceptable for a technical biomedical manuscript.

Do not finalize title until the HIGH/MEDIUM wording issues above are resolved or explicitly accepted.

## Layout TODOs

- LAYOUT TODO: Figure 1 is wide and detailed; verify at Overleaf compile that text remains readable in two-column `figure*` placement.
- LAYOUT TODO: Abstract is dense; check journal word count and readability.
- LAYOUT TODO: Table `tab:v52_matched` and range table should be checked in final PDF for spacing.
- LAYOUT TODO: Long paragraphs in Results and Discussion may need final page-flow trimming after compile.
- LAYOUT TODO: Long figure/table captions may affect float placement.

## Final readiness decision

NOT READY FOR TITLE FINALIZATION.

Minimum issue set to resolve first:

1. Fix or explicitly accept the Figure 1 target-label grouping risk.
2. Correct Related Work wording so leakage-safe additive is not described as "only" a diagnostic.
3. Decide whether the Introduction clinical risk/planning sentence needs an immediate existing citation.

Once these are resolved, the manuscript appears ready to proceed to title finalization, Overleaf final compile, page/layout optimization, and then a git evidence-freeze commit.

## Source files inspected

- `docs/experiments/frozen_result_registry_after_leakage_safe_fusion.md`
- `docs/experiments/manuscript_restructuring_blueprint_after_leakage_safe_fusion.md`
- `docs/experiments/additive_fusion_leakage_safe_formal_report.md`
- `docs/experiments/reliability_aware_gate_formal_report_corrected.md`
- `docs/experiments/additive_fusion_alpha_semantics_resolution.md`
- `docs/experiments/additive_fusion_cross_split_leakage_audit.md`
- `docs/experiments/method_rewrite_after_leakage_safe_fusion.md`
- `docs/experiments/results_rewrite_after_leakage_safe_fusion.md`
- `docs/experiments/discussion_rewrite_after_leakage_safe_fusion.md`
- `docs/experiments/introduction_rewrite_after_leakage_safe_fusion.md`
- `docs/experiments/conclusion_rewrite_after_leakage_safe_fusion.md`
- `docs/experiments/abstract_rewrite_after_leakage_safe_fusion.md`
- `docs/experiments/supplementary_sync_after_leakage_safe_fusion.md`
- `manuscript/overleaf_current/main.tex`
- `manuscript/overleaf_current/supplementary_main.tex`
- `manuscript/overleaf_current/supplementary_v5_experiments.tex`
- `manuscript/overleaf_current/references.bib`

# Figure 1 Framework Redesign After Leakage-Safe Fusion

This audit documents the redesign of the manuscript-level overall framework figure after the leakage-safe Method and Results rewrites. No experiment was run, no model was trained, no inference was performed, no result artifact was modified, and `main.tex` was not updated to reference the new figure.

## 1. Old Figure 1 Path

The current `manuscript/overleaf_current/main.tex` does not contain a dedicated framework Figure 1. The first figure currently referenced in the main manuscript is:

- path: `manuscript/overleaf_current/figures/fig_v5_overall_comparison.pdf`
- LaTeX source: `\includegraphics[width=0.95\textwidth]{figures/fig_v5_overall_comparison.pdf}`
- caption: `Frozen v5 comparison. (a) Test MAE across five repeated patient-level splits. (b) Range-specific MAE for low-, medium-, and high-vault eyes. (c) Prediction-range ratio, where lower values indicate stronger compression.`
- label: `fig:v5_overall`
- manuscript reference: no explicit `Figure~\ref{fig:v5_overall}` text reference was found in the current main manuscript.

Legacy unreferenced framework/data-curation figures remain in `manuscript/overleaf_current/figures/`, including `fig1_data_curation_pipeline.png`, `fig1_v4_data_curation_pipeline.png`, and `fig1_v4_data_curation_pipeline_wide.png`.

## 2. Why the Old Figure No Longer Matches the Manuscript

The current first manuscript figure is a results summary, not an overall study framework. It does not communicate the post-correction manuscript positioning:

- multimodal prediction framework plus diagnostic framework;
- full separation of prediction paths and diagnostic paths;
- conventional fusion rather than a proposed novel adaptive architecture;
- Oracle Best-of-Two as retrospective and non-deployable;
- G2 as a fusion learnability diagnostic rather than a final predictor;
- validation-only tuning and held-out repeated patient-level evaluation.

## 3. New Figure Structure

The new figure is a landscape academic framework schematic with four main zones:

- A. Preoperative Inputs
- B. Unimodal Experts
- C. Conventional Multimodal Fusion
- D. Matched Diagnostics

A bottom evaluation band spans the full figure and shows strict patient-level repeated evaluation.

## 4. Solid Prediction Paths

Solid navy arrows denote the primary predictive modeling and conventional fusion path:

- Preoperative AS-OCT Images to AS-OCT Prediction Model
- Preoperative CASIA2 2DAnalysis Measurements to Measurement-Based Model
- unimodal predictors to Modality-Specific Vault Predictions
- unimodal outputs to Feature Concatenation and Validation-Tuned Additive Late Fusion
- conventional fusion strategies to POD1 Vault Prediction

## 5. Dashed Diagnostic Paths

Dashed gray arrows denote secondary diagnostic analysis:

- modality-specific predictions to Matched Predictions
- matched predictions to Split-Level Best Unimodal Diagnostic
- matched predictions to Oracle Best-of-Two
- complementarity diagnostics to Fusion Learnability Diagnostics
- prediction and diagnostic outputs to Error / Failure-Mode Analysis

## 6. Additive Role

Validation-Tuned Additive Late Fusion is shown as one conventional multimodal fusion strategy, visually parallel to Feature Concatenation. It is not presented as a novel method or the central contribution.

The figure uses the generic leakage-safe formula:

`y_add = alpha_s y_M + (1-alpha_s) y_O`

and states:

`alpha_s: same-split validation only`

No global `alpha = 0.35` value is shown.

## 7. Oracle Role

Oracle Best-of-Two is shown inside the matched diagnostic branch and explicitly marked:

- Retrospective
- Non-deployable

It is visually separated from the POD1 Vault Prediction output and is not presented as a fusion algorithm.

## 8. G2 Role

G2 is visually downgraded to a sub-item inside Fusion Learnability Diagnostics:

- `G0 / G1 gate diagnostics`
- `Reliability-aware G2`
- `Prediction + reliability proxies; diagnostic only`

G2 is not shown as a proposed final model or as the main prediction output. The figure intentionally does not show G2 performance, NO-GO status, or all six G2 input features.

## 9. Evaluation Protocol Representation

The bottom evaluation band states:

- Strict Patient-Level Repeated Evaluation
- Train to Validation to Held-Out Test
- Repeated outer splits
- Matched test cohort for multimodal comparison
- Validation-only fusion/gate tuning; no test-driven tuning

No performance numbers or leakage-correction history are included in the figure.

## 10. Generated Files

Generated files:

- `manuscript/overleaf_current/figures/figure1_multimodal_framework.svg`
- `manuscript/overleaf_current/figures/figure1_multimodal_framework.pdf`
- `manuscript/overleaf_current/figures/figure1_multimodal_framework.png`

The SVG was generated with editable text preserved. The PDF was exported locally from the same vector drawing backend. The PNG is a high-resolution preview.

## 11. Recommended Caption

Recommended caption:

`Overview of the multimodal prediction and diagnostic framework. Preoperative AS-OCT images and CASIA2 2DAnalysis measurements were modeled using modality-specific predictors. Conventional fusion was evaluated using feature concatenation and leakage-safe validation-tuned additive late fusion on strictly matched cohorts. Retrospective complementarity and the learnability of eye-specific modality weighting were further assessed using Oracle Best-of-Two and gate-based diagnostic analyses under repeated patient-level evaluation.`

No performance conclusion should be added to the caption.

## 12. Remaining TODOs

- `main.tex` has not yet been updated to reference the new figure, per task instruction.
- The current Results figure `fig_v5_overall_comparison.pdf` remains referenced in `main.tex`; a later manuscript integration pass should decide whether it becomes Figure 2 or moves elsewhere.
- Verify final visual placement after LaTeX compilation in Overleaf or a machine with a LaTeX distribution.

## Final Check

- G2 is not visually presented as proposed final method: pass
- additive is not visually presented as novel method: pass
- Oracle marked retrospective/non-deployable: pass
- prediction vs diagnostic pathways visually distinguishable: pass
- patient-level repeated evaluation visible: pass
- validation-only fusion tuning visible: pass
- no stale alpha=0.35 global protocol: pass
- no result numbers: pass
- no unsupported Transformer/Cross-Attention: pass
- High-Vault/range compression only shown as analysis: pass
- editable vector source generated: pass

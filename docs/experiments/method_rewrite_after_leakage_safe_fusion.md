# Method Rewrite After Leakage-Safe Fusion

## 1. Modified Manuscript File(s)

- `manuscript/overleaf_current/main.tex`

Only Method-related content was modified. Title, Abstract, Introduction, Related Work, Results, Discussion, Conclusion, tables, figures, bibliography, and result values were intentionally not rewritten in this task.

## 2. Old Method Structure

The previous manuscript separated:

- `Study Design and Problem Formulation`
  - Cohort Construction and Leakage Control
  - Prediction Task
  - Patient-Level Evaluation
- `Method`
  - Overall Multimodal Prediction Framework
  - AS-OCT Representation Learning
  - Structured Measurement Representation
  - Multimodal Fusion and Interaction Analysis
  - Training and Small-Sample Controls

This structure mixed formal prediction methods, matched diagnostics, additive fusion, and gate diagnostics in one broad subsection. It also contained result-level wording about the earlier gate diagnostic decision.

## 3. New Method Structure

The rewritten manuscript now uses:

- `Methods`
  - Study Overview
  - AS-OCT Prediction Model
  - Measurement-Based Prediction Model
  - Conventional Multimodal Fusion
    - Feature Concatenation
    - Leakage-Safe Validation-Tuned Additive Fusion
  - Matched Modality Complementarity Analysis
  - Fusion Learnability Diagnostics
  - Evaluation Protocol

The new structure separates full-cohort baseline characterization from matched-cohort fusion and complementarity diagnostics.

## 4. Major Scientific Corrections

- Replaced the old pooled-alpha additive wording with the leakage-safe per-outer-split validation-only protocol.
- Explicitly defined additive alpha as Measurement weight.
- Explicitly distinguished additive alpha semantics from G2 alpha semantics.
- Repositioned G0/G1/G2 as fusion learnability diagnostics, not as a proposed successful adaptive fusion method.
- Defined matched best unimodal as a retrospective split-level diagnostic benchmark.
- Defined Oracle Best-of-Two as a retrospective theoretical diagnostic, not a deployable model.
- Added full-cohort versus matched-cohort separation in Method.
- Removed result-level NO-GO wording from Method and left results for the Results section.

## 5. Additive Protocol Wording

The Method now defines:

`pred_additive = alpha_s * Measurement + (1 - alpha_s) * AS-OCT`

Key protocol elements:

- `alpha_s` is the Measurement weight.
- `alpha_s` is outer-split specific.
- Grid: `{0, 0.05, ..., 1}`.
- Selection metric: same-split validation MAE.
- Tie rule: select the largest alpha within 0.5 um of the best validation MAE.
- The selected alpha is frozen before applying to the corresponding held-out test set.
- No pooled cross-split validation objective is used.
- Test labels are not used for alpha selection.

The Method does not report the five selected alpha values or additive performance numbers.

## 6. Oracle Wording

Oracle Best-of-Two is defined as:

`e_oracle = min(|Measurement - label|, |AS-OCT - label|)`

The Method states that Oracle:

- uses test labels post hoc,
- quantifies theoretical modality complementarity,
- is never used for model fitting or model selection,
- is not deployable,
- is not a fusion algorithm.

## 7. G2 Diagnostic Wording

G2 is described as a linear soft gate:

`alpha_i = sigmoid(w^T z_i + b)`

`pred_G2 = alpha_i * AS-OCT + (1 - alpha_i) * Measurement`

G2 alpha semantics were verified from `scripts/run_reliability_aware_gate_experiment.py`: G2 `alpha_i` is the AS-OCT weight. This is separate from additive-fusion `alpha_s`, which is the Measurement weight.

G2 inputs are documented as:

- AS-OCT prediction,
- Measurement prediction,
- absolute prediction disagreement,
- AS-OCT view-wise prediction dispersion after imputation,
- AS-OCT single-view indicator,
- RF tree prediction dispersion.

Gate training is documented as validation-only scaling and fitting with Smooth-L1 loss, beta 25 um, learning rate 0.03, L2 penalty `1e-4`, and 2500 full-batch gradient-descent iterations.

## 8. Full vs Matched Cohort Clarification

The Method now states that:

- modality-specific full cohorts are used for frozen baseline characterization according to real input availability;
- the strict matched cohort is used for fair same-eye multimodal comparison, additive fusion, Oracle, and gate diagnostics;
- matched analyses align predictions by split seed, sample identifier, patient identifier, eye, and ground-truth vault.

## 9. Unsupported Old Claims Removed or Repositioned

The rewritten Method avoids presenting:

- additive fusion as a core algorithmic innovation,
- G2 as a successful proposed final model,
- Oracle as a deployable model,
- cross-attention, Transformer, Mamba, ROI, or V2/V3 variants as main formal methods,
- calibrated uncertainty or predictive uncertainty estimates.

Transformer, Mamba, ROI, and range-aware AS-OCT variants are mentioned only to state that they were not part of the main Method.

## 10. Compilation Status

Compilation was not executed because no local LaTeX compiler was available in the environment.

Checked commands:

- `latexmk`: not found
- `pdflatex`: not found
- `xelatex`: not found
- `lualatex`: not found
- `tectonic`: not found
- `bibtex`: not found

Source-level checks performed:

- equation begin/end counts matched;
- table and figure begin/end counts matched;
- Method-only citation set remains limited to existing `resnet,imagenet`;
- no Method result leakage for additive MAE, G2 MAE, Oracle MAE, High-Vault values, alpha saturation, or oracle fraction;
- no reversed additive alpha formula found in Method.

## 11. Remaining TODOs

- Compile the manuscript in Overleaf or an environment with a LaTeX distribution.
- Results still contain superseded additive values from the pre-leakage-safe manuscript. This was not modified because the task explicitly limited edits to Method-related content.
- Abstract, Results, Discussion, and Conclusion still need later synchronized updates to the leakage-safe result registry.
- Figure 1/framework artwork was intentionally not modified.

## 12. Sections Intentionally Not Modified

- Title
- Abstract
- Introduction
- Related Work
- Experiments and Results
- Discussion
- Conclusion
- Figure files
- Tables and result values
- Bibliography
- Supplementary files

## 13. Self-Check

- no `167.91` formal additive result in Method: pass
- no reversed alpha formula in Method: pass
- no pooled-validation additive protocol in Method: pass
- additive alpha semantics = Measurement weight: pass
- per-split validation-only alpha selection: pass
- Oracle explicitly non-deployable retrospective diagnostic: pass
- matched best unimodal distinct from Oracle: pass
- G2 explicitly diagnostic: pass
- G2 alpha semantics verified from source code: pass
- reliability proxy not called calibrated uncertainty: pass
- single-view imputation leakage-safe: pass
- patient-level repeated evaluation described: pass
- full cohort and matched cohort distinguished: pass
- no unsupported Cross-Attention/Transformer claim as current method: pass
- no Method result leakage: pass
- manuscript compiles: not checked locally, compiler unavailable

# Introduction Rewrite After Leakage-Safe Fusion

This audit documents the manuscript Introduction rewrite after the leakage-safe evidence freeze, Method rewrite, Results rewrite, Figure 1 integration, and Discussion rewrite. No experiment was run, no model was trained, no inference was performed, no figure was modified, no formal artifact was modified, and no commit or push was performed.

## 1. Old Introduction Narrative

The previous Introduction already contained the major ingredients of the final manuscript, but its narrative still leaned toward a broad unified framework and included a contribution claim that the study showed greater fusion-network complexity was not justified. That placed a negative result too early and risked framing the paper around fusion escalation rather than around the explicit distinction between retrospective complementarity and learnable fusion.

## 2. New Introduction Narrative

The rewritten Introduction now follows:

- clinical motivation for preoperative POD1 vault prediction
- two preoperative input families: structured biometry / CASIA2 2DAnalysis measurements and AS-OCT imaging
- multimodal opportunity and the unresolved problem of whether complementarity can be learned
- study question focused on complementarity versus learnability under strict patient-level evaluation
- defensible contribution list

## 3. Rewritten Research Gap

The gap is now framed as:

`the presence of distinct modality-specific information does not guarantee that a deployable fusion rule can improve prediction for unseen eyes.`

The Introduction explicitly distinguishes:

- retrospective differences between unimodal errors
- learnable complementarity available to a deployable fusion model at inference time

## 4. Rewritten Study Objective

The study objective is:

`whether preoperative AS-OCT images and CASIA2 2DAnalysis measurements contain complementary information for POD1 vault prediction, and whether this complementarity can be reliably exploited by conventional or adaptive multimodal fusion under strict patient-level evaluation.`

The objective uses evaluate / quantify / assess language and does not claim a superior fusion method.

## 5. New Contribution Claims

The revised contribution list states that the study:

1. establishes a patient-level multimodal evaluation separating modality-specific and strictly matched cohorts;
2. systematically evaluates conventional fusion using feature concatenation and leakage-safe validation-tuned additive late fusion;
3. quantifies retrospective modality complementarity and distinguishes oracle headroom from deployable fusion performance;
4. assesses whether complementarity can be translated into stable eye-specific weighting with gate-based learnability diagnostics and failure-mode analysis.

These contribution claims are supported by the current Method, Results, and Discussion.

## 6. Old Unsupported Novelty Claims Removed

The Introduction no longer claims or implies:

- a novel multimodal fusion architecture
- a reliability-aware adaptive fusion method as the proposed model
- robust additive-fusion gain
- fusion superiority over unimodal models
- state-of-the-art fusion
- cross-attention / Transformer contribution
- that greater fusion-network complexity was proven unnecessary as a front-loaded contribution claim

## 7. Citation Handling

No new citation keys were invented. Existing Introduction citations were retained where they support clinical background, measurement-based prediction, AS-OCT/image-based prediction, and pretrained convolutional representation claims:

- `icl_sizing_rule`
- `icl_vault_biometry`
- `nakamura_icl_sizing`
- `yang_vault_formula`
- `wu_vault_formula`
- `ophthalmic_deep_learning`
- `as_oct_deep_learning`
- `resnet`
- `imagenet`

No bibliography file was modified.

## 8. Stale Claims Outside Introduction

The current task allowed only Introduction edits. Remaining stale or not-yet-synchronized areas outside Introduction include:

- Abstract still states that validation-tuned additive fusion gained only `2.11 um` on average.
- Conclusion still refers broadly to `fixed weighting` and should later be synchronized with leakage-safe additive and G2 NO-GO wording.

These were recorded but not modified.

## 9. Source-Level Check

Source-level checks were performed because local LaTeX compilation is not available. Checks covered:

- LaTeX environment begin/end counts
- citation keys against `references.bib`
- label/reference consistency
- stale invalid evidence in the Introduction section
- unsupported novelty or proposed-model phrasing in the Introduction section

The source-level checks passed.

## 10. Remaining TODOs

- Rewrite Abstract using the leakage-safe evidence hierarchy.
- Rewrite Conclusion using the updated Method, Results, Discussion, and Introduction.
- Compile in Overleaf or a machine with a LaTeX distribution.

## 11. Self-Check

- research gap = complementarity versus learnability: pass
- no claim that fusion robustly improves performance: pass
- no G2 proposed-model framing: pass
- no additive core-method framing: pass
- no Oracle deployable framing: pass
- no old `167.91` / `166.23` / `2.11` evidence in Introduction: pass
- contribution claims supported by current manuscript: pass
- no unsupported `novel` / `first` / `state-of-the-art` claim in Introduction: pass
- Related Work not duplicated: pass
- no section outside Introduction intentionally modified in this task: pass
- citation keys valid: pass

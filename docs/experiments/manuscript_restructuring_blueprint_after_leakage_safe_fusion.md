# Manuscript Restructuring Blueprint After Leakage-Safe Fusion

This blueprint supersedes `docs/experiments/manuscript_restructuring_blueprint_after_fusion_audit.md` for manuscript planning after the leakage-safe additive fusion correction. It does not modify manuscript or LaTeX source.

The authoritative evidence registry is:

- `docs/experiments/frozen_result_registry_after_leakage_safe_fusion.md`
- `artifacts/v5_2_matched_fusion_audit/frozen_result_registry_after_leakage_safe_fusion.json`

## 1. Evidence Freeze

The authoritative leakage-safe validation-tuned additive fusion result is:

- Formula: `pred_additive = alpha * Measurement + (1 - alpha) * AS-OCT`
- Alpha semantics: Measurement weight
- Split-specific Measurement alphas: seed42 `0.90`, seed1001 `0.55`, seed2002 `0.05`, seed2026 `0.40`, seed3407 `0.05`
- Repeated MAE: `171.94 +/- 20.22 um`
- Wins vs matched best unimodal: `3/5`
- Wins vs concat: `4/5`
- Wins vs G2: `3/5`
- High-Vault MAE: `345.28 um`
- Prediction range ratio: `0.387`
- Mean oracle fraction captured: `-0.0498`

The previous pooled-global alpha result `167.91 +/- 24.56 um` is no longer valid as unbiased repeated-test evidence because cross-split role contamination was detected. It may appear only as historical/provenance material, not in the Abstract, main Results, main tables, Conclusion, or contribution claims.

## 2. Updated Scientific Question

Recommended core question:

**Do preoperative AS-OCT images and CASIA2 2DAnalysis measurements contain complementary information for POD1 vault prediction, and can this complementarity be reliably exploited by conventional or adaptive multimodal fusion under strict patient-level evaluation?**

Chinese working version:

**术前 AS-OCT 与 CASIA2 measurements 是否包含 POD1 vault prediction 的互补信息，以及在严格 patient-level evaluation 下，这种互补性能否被 conventional 或 adaptive multimodal fusion 稳定利用？**

This is deliberately broader and more rigorous than a simple "does fusion improve MAE" question.

## 3. Updated Narrative

Recommended narrative:

1. Two unimodal experts capture different information.
2. Naive concat does not provide stable benefit.
3. Leakage-safe validation-tuned additive fusion shows split-dependent but not robust overall improvement.
4. Matched audit reveals substantial Oracle Best-of-Two headroom.
5. Therefore, retrospective complementarity exists.
6. Simple gate and reliability-aware G2 test whether eye-specific trust can be learned.
7. Adaptive fusion also fails to produce stable benefit.
8. High-Vault errors and prediction range compression persist.
9. Conclusion: the current modalities contain complementary predictive information, but this complementarity is not readily learnable as stable fusion under the currently available inputs.

## 4. Recommended Paper Positioning

Recommended title-level positioning remains suitable:

**Matched multimodal fusion and complementarity analysis for early ICL vault prediction**

More precise framing:

**Multimodal fusion evaluation and complementarity analysis for early ICL vault prediction**

Avoid framing the paper as:

- a novel fusion method paper
- a successful adaptive fusion paper
- a state-of-the-art fusion architecture paper

## 5. Method Blueprint

Recommended Method structure:

### 3.1 Study Overview

Define the study as a leakage-controlled, patient-level, repeated-split evaluation of AS-OCT images and CASIA2 2DAnalysis measurements for POD1 vault prediction. Explain that fusion experiments are evaluated as conventional baselines and learnability diagnostics, not as post-hoc algorithm escalation.

### 3.2 AS-OCT Prediction Model

Describe the frozen AS-OCT V0 model, single selected preoperative AS-OCT view protocol for formal V0 predictions, and its role as the image unimodal expert.

### 3.3 Measurement-Based Prediction Model

Describe the frozen Measurement RF using preoperative CASIA2 2DAnalysis measurements. Keep it clearly separate from any raw measurement feature additions that were not part of the frozen formal protocols.

### 3.4 Conventional Multimodal Fusion

#### 3.4.1 Feature Concatenation

Define concat as the implemented naive feature-level fusion baseline. It is deployable but did not show stable improvement.

#### 3.4.2 Leakage-Safe Validation-Tuned Additive Fusion

Define additive fusion as a conventional late-fusion baseline:

`pred_additive = alpha * Measurement + (1 - alpha) * AS-OCT`

For each outer split, alpha is selected using only that split's validation predictions and labels, then frozen before held-out test evaluation. There is no pooled cross-split alpha selection and no test-driven tuning.

### 3.5 Matched Modality Complementarity Analysis

Define the matched v5.2 cohort and aligned repeated test predictions. Include:

- matched coverage
- split-level matched best unimodal
- Oracle Best-of-Two
- disagreement / residual diagnostics where needed

Matched best unimodal must be defined as a post-hoc split-level diagnostic benchmark, not a deployable model.

Oracle Best-of-Two must be defined as a post-hoc per-eye theoretical diagnostic, not a model or fusion algorithm.

### 3.6 Fusion Learnability Diagnostics

Place G0/G1 and reliability-aware G2 here. G2 should be described as a diagnostic adaptive fusion experiment using predefined reliability signals, not the proposed final predictor.

Core conclusion to support in Results:

Adaptive eye-specific flexibility did not translate into stable benefit.

### 3.7 Evaluation Protocol

Emphasize:

- patient-level splits
- repeated outer splits
- same-split validation-only fusion tuning
- no pooled cross-split alpha selection
- no test-driven tuning
- MAE as primary metric
- High-Vault and prediction range compression as key failure-mode analyses

## 6. Results Blueprint

### 4.1 Cohort and Evaluation Cohorts

Separate full-cohort v5 baseline cohorts from matched v5.2 fusion audit cohorts. Do not blend these numbers without explicitly naming the cohort and protocol.

### 4.2 Unimodal Prediction Performance

Report Measurement RF and AS-OCT V0 as complementary unimodal experts. Avoid saying either modality is uniformly superior.

### 4.3 Conventional Multimodal Fusion Performance

Report concat and leakage-safe additive fusion together:

- Full-cohort concat: `175.79 +/- 23.15 um`, no stable gain over unimodal baselines.
- Matched leakage-safe additive: `171.94 +/- 20.22 um`, wins `3/5` vs matched best unimodal and `4/5` vs concat.

Required interpretation:

Additive does **not** demonstrate a robust overall improvement over the best unimodal model. It is a simple conventional fusion baseline with split-dependent supporting gains.

### 4.4 Vault-Range and Range-Compression Analysis

Report Low / Medium / High behavior if table space allows. High-Vault and range compression should be main Results topics because they remain central failure modes.

Key numbers:

- Leakage-safe additive High-Vault MAE: `345.28 um`
- Measurement High-Vault MAE: `309.27 um`
- AS-OCT High-Vault MAE: `369.05 um`
- Concat High-Vault MAE: `333.16 um`
- G2 High-Vault MAE: `351.98 um`
- Oracle High-Vault MAE: `277.72 um`
- Leakage-safe additive prediction range ratio: `0.387`

### 4.5 Matched Oracle Complementarity

Report Oracle Best-of-Two as a formal diagnostic:

- Oracle repeated MAE: `132.45 +/- 26.11 um`
- Leakage-safe additive mean oracle fraction captured: `-0.0498`

Interpretation:

Substantial retrospective eye-level complementarity exists, but the tested fusion approaches do not reliably exploit it.

### 4.6 Fusion Learnability Diagnostics

Report G0/G1 and G2 concisely:

- G0 winner AUC: `0.521 +/- 0.069`, NO-GO
- G1 winner AUC: `0.456 +/- 0.107`, NO-GO
- G2 repeated MAE: `171.84 +/- 28.64 um`
- G2 wins vs matched best unimodal: `1/5`
- G2 High-Vault MAE: `351.98 um`
- G2 final decision: NO-GO

Do not state that fixed additive definitively outperforms G2. Instead:

**Neither simple additive nor adaptive reliability-aware fusion consistently exploited the available oracle complementarity.**

### 4.7 Summary of Main Findings

Summarize the evidence hierarchy:

- Unimodal models are competitive and complementary.
- Naive concat does not stably improve prediction.
- Leakage-safe additive fusion is paper-ready only as a supporting conventional baseline.
- Oracle complementarity is substantial but retrospective.
- G0/G1/G2 show that current signals do not support stable eye-specific modality trust.
- High-Vault error and range compression persist.

## 7. Updated Contribution Claims

Defensible contributions:

1. Rigorous patient-level multimodal evaluation of preoperative AS-OCT and CASIA2 2DAnalysis measurements for early vault prediction.
2. Matched analysis quantifying substantial retrospective eye-level complementarity between the two modalities.
3. Systematic evaluation of conventional and adaptive fusion showing that observed oracle complementarity is not readily translated into stable predictive gain.
4. Characterization of persistent High-Vault error and prediction range compression.

Claims to abandon:

- additive improves prediction robustly
- additive provides a stable overall improvement over the best unimodal model
- novel fusion architecture
- successful adaptive fusion
- state-of-the-art fusion
- G2 is the proposed final predictor
- Oracle performance is achievable/deployable
- High-Vault difficulty is solved
- prediction range compression is corrected

## 8. Main Text and Supplementary Allocation

Main text should keep:

- unimodal baselines
- concat baseline
- leakage-safe additive fusion
- Oracle complementarity
- concise G2 NO-GO
- High-Vault and range behavior

Supplementary should hold:

- full per-split alpha grid
- split-specific alphas
- G0/G1 details
- G2 coefficients
- alpha saturation diagnostics
- full repeated split tables
- leakage/correction provenance
- sensitivity analyses

## 9. Novelty Risk

The current paper no longer has credible "fusion performance improvement" as its main novelty.

Stronger novelty axes:

- rigorous matched design
- complementarity quantification
- fusion learnability diagnosis
- methodological evaluation rigor
- failure-mode characterization

Venue implications:

- Strong algorithm venue: high risk unless additional validated algorithmic contribution is added later.
- Biomedical AI / medical informatics: more suitable.
- Clinical AI application: potentially suitable if cohort/reporting and clinical interpretation are strong.

No external venue search was performed.

## 10. Required Manuscript Guardrails

Do not use as formal manuscript evidence:

- `167.91 +/- 24.56 um` pooled-global alpha result
- `0.35 * AS-OCT + 0.65 * Measurement`
- old G2 uncorrected fixed-additive comparator outputs
- provisional additive audits before alpha/leakage resolution
- any additive result that did not pass final QC

The manuscript must explicitly distinguish:

- full-cohort baselines vs matched repeated-split fusion audit
- matched best unimodal vs Oracle Best-of-Two
- deployable models vs diagnostic benchmarks
- retrospective complementarity vs learnable fusion

## 11. Recommended Rewrite Order

1. Freeze registry and tables.
2. Update figure/table plan.
3. Rewrite Method first, especially Sections 3.4 to 3.7.
4. Rebuild Results around the RQ sequence.
5. Rewrite Discussion around complementarity vs learnability.
6. Revise Introduction and contributions after Results are stable.
7. Rewrite Abstract last.
8. Finalize title after target positioning is approved.

## 12. Current Protocol Status

No unresolved protocol issue is apparent from the final leakage-safe archive and registry creation step.

Remaining manuscript risk is interpretive, not computational: future drafting must avoid overclaiming additive or adaptive fusion and must keep invalidated outputs out of main evidence.

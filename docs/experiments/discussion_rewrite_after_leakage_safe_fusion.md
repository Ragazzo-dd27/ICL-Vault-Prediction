# Discussion Rewrite After Leakage-Safe Fusion

This audit documents the manuscript Discussion rewrite after the leakage-safe evidence freeze, Method rewrite, Results rewrite, and Figure 1 integration. No experiment was run, no model was trained, no inference was performed, no figure was modified, no formal artifact was modified, and no commit or push was performed.

## 1. Old Discussion Structure

The previous Discussion used the following structure:

- Why AS-OCT and Measurements Are Potentially Complementary
- Why Simple Concatenation Did Not Improve Overall Accuracy
- Range-Specific Behavior and Prediction Compression
- Theoretical Versus Learnable Complementarity
- Small-Sample Adaptation and Model Complexity
- Clinical Variables Required for Further Improvement
- Limitations

This structure already contained useful components, but it was not aligned with the final leakage-safe Results hierarchy and did not fully incorporate the corrected additive and G2 framing.

## 2. New Discussion Structure

The rewritten Discussion uses:

- Principal Findings
- Retrospective Multimodal Complementarity
- Why Fusion Did Not Reliably Exploit Complementarity
- Fusion Learnability Diagnostics
- Persistent High-Vault Error and Prediction Range Compression
- Methodological Implications and Limitations

This follows the post-correction scientific logic: predictive signal, retrospective complementarity, failed stable fusion exploitation, adaptive learnability diagnostics, persistent tail/range failures, and methodological implications.

## 3. Stale Fusion Claims Removed

The rewritten Discussion avoids:

- additive fusion as a robust overall improvement
- fixed fusion as consistently beneficial
- successful multimodal fusion
- successful reliability-aware fusion
- G2 as a proposed final method
- Oracle as deployable
- High-Vault error as solved
- range compression as alleviated
- proven overfitting claims for G2

No `167.91 +/- 24.56 um` or `166.23 +/- 23.34 um` result is used in the rewritten Discussion.

## 4. Oracle Interpretation

Oracle Best-of-Two is interpreted as retrospective, post-hoc, and non-deployable. The Discussion states that Oracle indicates partially non-overlapping unimodal errors and retrospective predictive complementarity, but not biological complementarity or prospectively actionable complementarity by itself.

The central sentence is:

`retrospective modality complementarity does not imply learnable complementarity under the current inputs, cohort size, and reliability proxies.`

## 5. Additive Interpretation

Leakage-safe additive fusion is interpreted as a simple conventional late-fusion baseline with split-dependent gains. The Discussion reports:

- repeated MAE: `171.94 +/- 20.22 um`
- wins vs concat: `4/5`
- wins vs matched best unimodal: `3/5`

The text explicitly states that this does not establish robust overall superiority over matched unimodal references.

## 6. Split-Specific Alpha Interpretation

The Discussion reports formal split-level Measurement alphas:

- seed42: `0.90`
- seed1001: `0.55`
- seed2002: `0.05`
- seed2026: `0.40`
- seed3407: `0.05`

These are interpreted as evidence that a stable population-level modality balance was difficult to identify in the current cohort. The text explicitly avoids interpreting split-level alphas as patient-level modality requirements.

## 7. G2 Interpretation

G0/G1/G2 are described as learnability diagnostics rather than proposed final models. The Discussion reports:

- G2 repeated MAE: `171.84 +/- 28.64 um`
- G2 wins vs matched best unimodal: `1/5`
- G2 decision: `NO-GO`

Near-extreme gate weights and cross-split coefficient instability are described as being consistent with difficult robust estimation, not as proof of a specific overfitting mechanism.

## 8. High-Vault Interpretation

High-Vault remains an unresolved failure mode. The Discussion reports:

- leakage-safe additive High-Vault MAE: `345.28 um`
- G2 High-Vault MAE: `351.98 um`
- Oracle High-Vault MAE: `277.72 um`

The text states that the gap again separates retrospective headroom from learnable improvement.

## 9. Range-Compression Interpretation

Prediction-range compression is described as a persistent pattern. The Discussion reports:

- leakage-safe additive prediction range ratio: `0.387`

The text frames compression as compatible with regression-to-the-mean behavior in a modest and imbalanced cohort, without claiming a proven single architectural cause.

## 10. Limitations

The rewritten Discussion includes:

- retrospective single-project design
- modest cohort size
- lack of external validation
- repeated-split variability
- repeated-test pooled diagnostics may contain the same physical eye in multiple outer test splits
- low/high vault sparsity
- only two preoperative modality families
- missing device/procedure variables
- reliability proxies are prediction dispersion, not calibrated uncertainty
- NO-GO scope limited to evaluated strategies, current inputs, and current cohort

## 11. Novelty Framing

The Discussion states that the main contribution is not a new fusion architecture. It frames the contribution as:

- leakage-controlled patient-level matched evaluation
- quantification of retrospective modality complementarity
- systematic testing of whether complementarity can be converted into stable fusion gain
- distinction between retrospective complementarity and deployable learnability
- failure-mode characterization

No unsupported `novel`, `first`, or `state-of-the-art` claim was added.

## 12. Stale Claims Outside Discussion

The current task allowed only Discussion edits. Remaining stale or not-yet-synchronized areas outside Discussion include:

- Abstract still states that validation-tuned additive fusion gained only `2.11 um` on average.
- Conclusion still refers broadly to `fixed weighting` and should later be synchronized with leakage-safe additive and G2 NO-GO wording.
- Introduction may later need contribution-claim tightening after the final Abstract/Conclusion pass, although no specific invalid additive number was found there.

These were recorded but not modified.

## 13. Unresolved Issues

No unsupported Discussion sentence was identified after the rewrite. All numeric claims in the Discussion trace to the frozen registry or listed authoritative reports.

Remaining manuscript work:

- Abstract rewrite
- Conclusion rewrite
- possible Introduction/contribution synchronization
- final LaTeX compilation in Overleaf or an environment with a TeX distribution

## 14. Source-Level Check Status

Source-level checks were performed because no local LaTeX compiler is available. Checks covered:

- LaTeX environment begin/end counts
- label uniqueness
- reference targets
- citation keys against `references.bib`
- stale invalid additive/G2 numbers in the Discussion section
- accidental presence of unsupported novelty claims in the Discussion section

The source-level checks passed.

## 15. Self-Check

- no old `167.91` positive additive claim in Discussion: pass
- no `166.23` result in Discussion: pass
- additive not described as robust improvement: pass
- Oracle explicitly retrospective: pass
- complementarity is distinct from learnability: pass
- G2 remains diagnostic / NO-GO: pass
- G2 failure not overclaimed as proven overfitting: pass
- High-Vault remains unresolved: pass
- range compression accurately discussed: pass
- split-level alpha variability not misinterpreted as patient-level: pass
- missing clinical/device variables framed as future work: pass
- no unsupported novelty claim: pass
- no section outside Discussion intentionally modified in this task: pass

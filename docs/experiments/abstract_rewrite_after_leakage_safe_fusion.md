# Abstract Rewrite After Leakage-Safe Fusion

This audit documents the manuscript Abstract rewrite after the leakage-safe evidence freeze and the completed Method, Results, Figure 1, Discussion, Introduction, and Conclusion rewrites. No experiment was run, no model was trained, no inference was performed, no figure was modified, no formal artifact was modified, and no commit or push was performed.

## 1. Old Abstract Narrative

The previous Abstract followed the older baseline-and-fusion narrative. It included many cohort and model details and retained the stale statement that validation-tuned additive fusion gained only `2.11 um` on average. That statement no longer matches the final leakage-safe additive-fusion evidence.

## 2. New Abstract Narrative

The rewritten Abstract follows:

- clinical prediction problem;
- objective: complementarity and fusion learnability under strict patient-level evaluation;
- methods: preoperative AS-OCT, CASIA2 2DAnalysis measurements, modality-specific predictors, conventional fusion, matched diagnostics, gate-based diagnostics, and vault-range analysis;
- results: full-cohort unimodal/concat references, leakage-safe additive result, retrospective Oracle result, reliability-aware gate result;
- conclusion: retrospective multimodal complementarity is distinct from learnable/deployable fusion.

## 3. Stale Evidence Removed

The rewritten Abstract removes or avoids:

- `2.11 um` stale additive-gain statement
- `167.91 +/- 24.56 um`
- `166.23 +/- 23.34 um`
- stable additive improvement
- successful multimodal fusion
- proposed reliability-aware fusion
- G2 as final method
- Oracle as deployable model
- High-Vault solved
- range compression alleviated

## 4. Formal Results Included

The Abstract includes only registry-supported formal results:

- Measurement RF repeated MAE: `172.79 +/- 22.38 um`
- AS-OCT V0 repeated MAE: `173.78 +/- 15.77 um`
- Feature concatenation repeated MAE: `175.79 +/- 23.15 um`
- Leakage-safe additive fusion repeated MAE: `171.94 +/- 20.22 um`
- Oracle Best-of-Two repeated MAE: `132.45 +/- 26.11 um`
- Reliability-aware eye-specific gating repeated MAE: `171.84 +/- 28.64 um`

## 5. Oracle Wording

Oracle Best-of-Two is described as a `retrospective Oracle Best-of-Two diagnostic` that indicates theoretical complementarity. It is not described as deployable or as an achievable clinical model.

## 6. Conventional Fusion Wording

Conventional fusion is described as feature concatenation and leakage-safe validation-tuned additive late fusion. The Abstract states that leakage-safe additive fusion did not show robust overall improvement over matched unimodal references.

## 7. Adaptive Fusion Wording

Adaptive fusion is described as reliability-aware eye-specific gating and gate-based learnability diagnostics. It is not described as a proposed final predictor. The Abstract states that it likewise did not provide stable held-out improvement.

## 8. Failure-Mode Wording

The Abstract includes one concise failure-mode sentence:

`High-Vault prediction and prediction-range compression remained persistent failure modes.`

No High-Vault MAE or range-ratio values are included in the Abstract.

## 9. Final Conclusion Wording

The final sentence states that the findings distinguish retrospective multimodal complementarity from its learnability and suggest that richer clinical, device, or surgical variables may be needed for reliable extreme-vault prediction and modality weighting.

## 10. Word Count

The rewritten Abstract is 230 English words.

## 11. Stale Claims Elsewhere

A full-manuscript stale search after the Abstract rewrite found no remaining `2.11`, `167.91`, or `166.23` in `manuscript/overleaf_current/main.tex`.

The title still uses the current matched-cohort multimodal analysis framing and was not modified. No keywords block is present in the current `main.tex`. The supplementary file `manuscript/overleaf_current/supplementary_v5_experiments.tex` still contains the historical `167.91` / `2.11` additive row and requires a later supplementary synchronization pass. Supplementary files were not edited in this task.

## 12. Source-Level Check

Source-level checks were performed because local LaTeX compilation is not available. Checks covered:

- Abstract begin/end syntax
- LaTeX environment begin/end balance
- label/reference consistency
- citation keys against `references.bib`
- stale invalid evidence in the Abstract
- unsupported novelty or proposed-method phrasing in the Abstract

The source-level checks passed.

## 13. Self-Check

- no `2.11` old additive gain in Abstract: pass
- no `167.91` formal result in Abstract: pass
- no `166.23` reversed-alpha result in Abstract: pass
- additive not described as robust improvement: pass
- Oracle retrospective/non-deployable: pass
- complementarity is distinct from learnability: pass
- G2 not proposed final model: pass
- High-Vault/range compression not claimed solved: pass
- all included numbers map to frozen registry: pass
- no unsupported novel/first/state-of-the-art claim: pass
- no section outside Abstract intentionally modified in this task: pass
- Abstract word count acceptable: pass

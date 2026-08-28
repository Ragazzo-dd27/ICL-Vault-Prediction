# Conclusion Rewrite After Leakage-Safe Fusion

This audit documents the manuscript Conclusion rewrite after the leakage-safe evidence freeze, Method rewrite, Results rewrite, Figure 1 integration, Discussion rewrite, and Introduction rewrite. No experiment was run, no model was trained, no inference was performed, no figure was modified, no formal artifact was modified, and no commit or push was performed.

## 1. Old Conclusion Framing

The previous Conclusion summarized formal results in detail and retained broad wording that theoretical complementarity could not be converted into stable gains by `fixed weighting` or low-complexity gating. Although directionally close to the final narrative, this wording was not precise enough after the leakage-safe additive correction and could imply an outdated fixed-fusion framing.

## 2. New Conclusion Framing

The rewritten Conclusion is a single concise paragraph. It states that the study systematically evaluated POD1 ICL vault prediction from:

- preoperative AS-OCT images
- CASIA2 2DAnalysis measurements
- conventional fusion
- matched complementarity analysis
- adaptive fusion diagnostics

The final takeaway is that retrospective complementary predictive information was present, but the evaluated conventional and adaptive fusion strategies did not consistently translate it into improved held-out prediction under the current cohort and input setting.

## 3. Stale Claims Removed

The rewritten Conclusion removes or avoids:

- `2.11 um` gain
- `167.91 +/- 24.56 um`
- `166.23 +/- 23.34 um`
- stable additive improvement
- fixed weighting as an outdated broad comparator phrase
- successful multimodal fusion
- adaptive fusion improvement
- proposed G2 framework
- superior fusion
- robust fusion gain
- High-Vault solved
- range compression alleviated

## 4. Oracle Wording

The Conclusion refers to `retrospective complementary predictive information`. It does not describe Oracle as deployable, achievable clinical performance, or a model. The detailed Oracle definition remains in Methods/Results/Discussion.

## 5. Conventional Fusion Wording

Conventional fusion is summarized as `feature-concatenation` and `leakage-safe additive late-fusion`. The Conclusion states that these evaluated strategies did not consistently translate complementarity into improved held-out prediction. It does not report additive MAE or wins and does not claim robust additive superiority.

## 6. Adaptive Fusion Wording

Adaptive fusion is summarized as `gate-based adaptive strategies`. The Conclusion does not name G0/G1/G2 individually and does not claim overfitting or general ineffectiveness of adaptive fusion. The statement is limited to the evaluated strategies, current cohort, and input setting.

## 7. High-Vault / Range-Compression Wording

The Conclusion states that High-Vault error and prediction-range compression remained persistent limitations. It does not introduce new causal explanations or numeric values.

## 8. Future-Work Wording

Future work is limited to one sentence:

`Future work should assess whether richer clinical, device, and surgical variables can improve extreme-vault prediction and provide more informative signals for multimodal fusion.`

This does not promise improvement.

## 9. Stale Abstract Claims

The current task allowed only Conclusion edits. The Abstract still contains a stale statement that validation-tuned additive fusion gained only `2.11 um` on average. This should be corrected in the Abstract rewrite.

## 10. Source-Level Check

Source-level checks were performed because local LaTeX compilation is not available. Checks covered:

- LaTeX environment begin/end balance
- label/reference consistency
- citation key consistency
- stale invalid evidence in the Conclusion section
- unsupported novelty or proposed-method phrasing in the Conclusion section

The source-level checks passed.

## 11. Self-Check

- no `2.11` in Conclusion: pass
- no `167.91` in Conclusion: pass
- no `166.23` in Conclusion: pass
- no stable additive improvement claim: pass
- no G2 proposed-method framing: pass
- Oracle remains retrospective: pass
- complementarity is distinct from learnability: pass
- High-Vault remains unresolved: pass
- range compression remains unresolved: pass
- future work does not promise improvement: pass
- no unsupported novelty claim: pass
- Conclusion concise: pass
- no section outside Conclusion intentionally modified in this task: pass

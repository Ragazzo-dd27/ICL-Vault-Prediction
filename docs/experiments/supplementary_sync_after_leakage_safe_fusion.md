# Supplementary Sync After Leakage-Safe Fusion

This audit documents the supplementary evidence synchronization after the main manuscript was rewritten around the leakage-safe frozen evidence. No experiment was run, no model was trained, no inference was performed, no formal artifact was modified, and `main.tex` was not modified in this task.

## 1. Modified Supplementary File

- `manuscript/overleaf_current/supplementary_v5_experiments.tex`

## 2. Stale Numbers Found

The supplementary v5.2 matched interaction table previously contained:

- `167.91 +/- 24.56 um`
- `2.11` average gain

These values came from the superseded pooled-validation global-alpha diagnostic and are not valid as current formal held-out repeated-test evidence.

## 3. Stale Numbers Replaced/Removed

The stale formal row:

- `Fixed additive fusion & 167.91 +/- 24.56 & 3/5 & 2.11 & Limited, unstable`

was removed from the current supplementary table. No historical contaminated performance row was retained in the Supplementary to avoid confusion with formal evidence.

## 4. Additive Formal Values Used

The supplementary v5.2 table now uses the leakage-safe additive result:

- repeated MAE: `171.94 +/- 20.22 um`
- wins vs matched best unimodal: `3/5`
- role: validation-only late fusion, not robustly superior

The text now records the formal split-specific Measurement alphas:

- seed42: `0.90`
- seed1001: `0.55`
- seed2002: `0.05`
- seed2026: `0.40`
- seed3407: `0.05`

The formula is stated as:

`pred_additive = alpha_s * Measurement + (1 - alpha_s) * AS-OCT`

where `alpha_s` is the Measurement weight.

## 5. G2 Corrected Values Used

The matched interaction table now includes corrected G2 evidence:

- repeated MAE: `171.84 +/- 28.64 um`
- wins vs matched best unimodal: `1/5`
- role: corrected adaptive diagnostic; `NO-GO`

G2 is not described as a proposed final method.

## 6. Oracle Wording

Oracle Best-of-Two is described as:

- retrospective
- non-deployable
- per-eye diagnostic

It is not presented as a deployable model.

## 7. Full vs Matched Cohort Clarification

The supplementary file retains the full-cohort v5 reference tables for modality-specific baseline characterization and separately labels the v5.2 table as a strict matched repeated-split interaction analysis. The v5.2 section explicitly states that predictions were aligned on the same fusion cohort and fusion patient-level splits.

## 8. Unresolved Supplementary Values

No unresolved supplementary value was identified in the modified v5.2 table. Other v5.1 exploratory values were not changed because they are retained as exploratory screening/formal exploratory results and were outside the stale additive correction target.

## 9. Remaining Historical / Provenance Entries

The historical pooled-validation contaminated additive result was not retained as a row in the Supplementary. The text notes only that the previous pooled-validation global-alpha diagnostic is superseded and not used as a formal held-out repeated-test result.

## 10. Source-Level Check

Source-level checks were performed after synchronization:

- LaTeX environment begin/end balance
- label uniqueness
- reference targets
- citation keys
- table column consistency
- stale invalid additive numbers in current supplementary files

The source-level checks passed.

## 11. Self-Check

- no `167.91` as current formal result: pass
- no `2.11` gain as current formal evidence: pass
- no `166.23` reversed-alpha result: pass
- leakage-safe additive is authoritative: pass
- no reversed alpha semantics: pass
- G2 corrected evidence only: pass
- Oracle retrospective: pass
- matched best unimodal is distinct from Oracle: pass
- full vs matched cohorts distinguished: pass
- no unsupported positive fusion claim: pass
- `main.tex` not modified in this task: pass

# Results Rewrite After Leakage-Safe Fusion

This audit documents the manuscript Results rewrite performed after the leakage-safe additive fusion correction. No experiment was run, no model was trained, no CNN inference was performed, and no formal artifact was modified.

## 1. Modified File

- `manuscript/overleaf_current/main.tex`

Only the `\section{Experiments and Results}` block was intentionally rewritten. Title, Abstract, Introduction, Related Work, Methods, Discussion, Conclusion, figure image files, bibliography, and formal artifacts were not intentionally edited during this Results rewrite.

## 2. Old Results Structure

The previous Results section used the following structure:

- Cohort and Data Quality
- Implementation Details
- Main Baseline Results
- Repeated Patient-Level Stability
- Range-Specific Errors and Prediction Compression
- Exploratory Improvements
- Matched Multimodal Interaction Analysis
- Gate Learnability Diagnostic

This structure mixed full-cohort baseline results, exploratory V5.1 material, matched audit findings, and gate diagnostics in a way that no longer reflected the final leakage-safe evidence hierarchy.

## 3. New Results Structure

The Results section now uses the following structure:

- Cohort and Evaluation Cohorts
- Unimodal Prediction Performance
- Conventional Multimodal Fusion Performance
- Vault-Range and Prediction-Range Behavior
- Matched Modality Complementarity
- Fusion Learnability Diagnostics
- Summary of Main Findings

This sequence follows the frozen post-correction narrative: unimodal performance, conventional fusion, leakage-safe additive fusion, range failure, matched oracle complementarity, fusion learnability diagnostics, and summary.

## 4. Replaced Stale Additive Results

The old Results table row:

- `Fixed additive fusion & 167.91 +/- 24.56 & 3/5`

was removed from the Results section as formal evidence. The Results section now reports the final leakage-safe validation-tuned additive fusion result:

- `171.94 +/- 20.22 um`
- wins vs matched best unimodal: `3/5`
- wins vs concat: `4/5`
- split-specific Measurement weights varied across splits: `0.05--0.90`

The Results section does not use:

- `167.91 +/- 24.56 um` as formal additive evidence
- `166.23 +/- 23.34 um`
- `0.35 * AS-OCT + 0.65 * Measurement`
- old uncorrected G2 fixed-additive comparator outputs

## 5. Full-Cohort vs Matched-Cohort Corrections

The Results section now explicitly separates:

- full-cohort v5 baseline characterization: AS-OCT-only, measurement-only, concat-fusion cohorts
- strict matched v5.2 repeated-split comparisons: Measurement RF, AS-OCT V0, concat fusion, leakage-safe additive fusion, G2, matched best unimodal, and Oracle Best-of-Two

The matched cohort is described as 280 repeated held-out test predictions across five outer splits using common fusion-cohort split assignments.

## 6. Oracle Positioning

Oracle Best-of-Two is now described as:

- retrospective
- per-eye
- non-deployable
- not a fusion algorithm

The Results report:

- Oracle repeated MAE: `132.45 +/- 26.11 um`
- matched best unimodal repeated MAE: `170.02 +/- 23.53 um`
- retrospective headroom: `37.57 um`
- Oracle High-Vault MAE: `277.72 um`

The text states that oracle complementarity does not imply prospectively learnable fusion.

## 7. G2 Positioning

G2 is now positioned as a diagnostic adaptive gate, not as the proposed final predictor. The Results report:

- G2 repeated MAE: `171.84 +/- 28.64 um`
- wins vs matched best unimodal: `1/5`
- G2 High-Vault MAE: `351.98 um`
- mean near-extreme alpha fraction: `0.825`
- coefficient signs mixed across splits
- G2 decision: `NO-GO`

The Results avoid claiming fixed additive is definitively superior to G2. The stated conclusion is that neither leakage-safe additive fusion nor reliability-aware eye-specific gating consistently translated retrospective oracle complementarity into improved held-out prediction.

## 8. High-Vault and Range-Compression Positioning

The Results now use the matched leakage-safe range values:

| Method | Low MAE | Medium MAE | High MAE | Range ratio |
| --- | ---: | ---: | ---: | ---: |
| Measurement RF | 341.21 | 129.56 | 309.27 | 0.505 |
| AS-OCT V0 | 297.19 | 122.76 | 369.05 | 0.367 |
| Concat fusion | 255.62 | 127.08 | 333.16 | 0.402 |
| Leakage-safe additive | 315.47 | 122.07 | 345.28 | 0.387 |
| G2 reliability-aware gate | 295.61 | 117.72 | 351.98 | 0.402 |

The Results state that High-Vault prediction remains difficult and that additive averaging did not correct prediction-range compression.

## 9. Tables Modified

Within the Results section:

- `tab:v5_cohort`: retained, with framing updated as full-cohort baseline cohort composition.
- `tab:v5_main_results`: retained, with interpretation updated as full-cohort baseline characterization.
- `tab:v52_matched`: replaced with strict matched repeated-split multimodal comparison including deployable predictors and retrospective diagnostics.
- `tab:v5_range`: replaced with matched repeated-split range MAE and prediction-range ratio table.

No figure image file was modified.

## 10. Stale Claims Elsewhere

The current task prohibited changes outside Results. Source search identified stale or pre-correction claims outside the Results section that should be updated in later manuscript passes:

- Abstract still states that validation-tuned additive fusion gained only `2.11 um` on average.
- Discussion still refers to near-random G0/G1 and fixed-alpha gating without the final leakage-safe additive/G2 framing.
- Conclusion still uses broad "fixed weighting" wording that should be synchronized with leakage-safe additive and G2 NO-GO positioning.

These were recorded but not modified.

## 11. Unresolved Values

No unresolved numeric conflict was found within the rewritten Results section. The Results numbers were taken from:

- `docs/experiments/frozen_result_registry_after_leakage_safe_fusion.md`
- `docs/experiments/additive_fusion_leakage_safe_formal_report.md`
- `docs/experiments/reliability_aware_gate_formal_report_corrected.md`
- `docs/experiments/manuscript_restructuring_blueprint_after_leakage_safe_fusion.md`

The corrected G2 report still contains a historical corrected global fixed-additive comparator. The rewritten Results intentionally use the final leakage-safe additive archive and frozen registry for formal additive claims.

## 12. Source-Level Consistency

LaTeX source-level checks were performed after the rewrite because no local LaTeX compiler was available. Checks covered:

- `\begin{table}` / `\end{table}` balance
- `\begin{table*}` / `\end{table*}` balance
- `\begin{figure*}` / `\end{figure*}` balance
- `\begin{equation}` / `\end{equation}` balance
- label/reference consistency for labels introduced or retained in Results
- stale invalid additive numbers and reversed-alpha formula search

No source-level structural issue was detected in the rewritten Results section.

## 13. Figure 1 TODO

No figure image files were modified. If the final paper adopts the new "retrospective complementarity vs learnable fusion" framing visually, the main schematic/first figure should be redesigned in a later figure-specific task. This Results rewrite did not alter figure assets.

## 14. Compilation Status

Local LaTeX compilation was not performed because common LaTeX executables were not available in the environment. Source-level checks were used instead.

## 15. Final Consistency Statement

The rewritten Results section:

- does not use the pooled contaminated `167.91 +/- 24.56 um` additive result as formal evidence
- does not use the alpha-inverted `166.23 +/- 23.34 um` result
- uses leakage-safe additive fusion as the formal additive result
- distinguishes full-cohort baselines from matched repeated-split comparisons
- distinguishes matched best unimodal from Oracle Best-of-Two
- treats Oracle as retrospective and non-deployable
- treats G2 as diagnostic and NO-GO
- keeps High-Vault error and prediction-range compression as unresolved failure modes

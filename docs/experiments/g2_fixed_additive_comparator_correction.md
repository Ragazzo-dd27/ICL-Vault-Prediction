# G2 Fixed Additive Comparator Correction

## Correction Scope

Alpha semantics have been formally resolved for the frozen validation-tuned additive fusion comparator.

- Frozen alpha: `0.35`
- Alpha semantics: Measurement weight
- Correct formula: `0.35 * Measurement + 0.65 * AS-OCT`

This correction uses existing formal per-eye predictions only. No model was retrained. No CNN inference was run. No G2 fitting was rerun. Alpha was not searched again.

## What Changed

Only the fixed additive comparator and downstream comparisons that depended on it were corrected.

Unchanged:

- G2 predictions themselves are unchanged.
- Measurement predictions are unchanged.
- AS-OCT predictions are unchanged.
- Oracle results are unchanged.
- Gate coefficients are unchanged.
- Reliability feature summaries are unchanged.

Corrected:

- `fixed_additive_pred_um`
- `fixed_additive_abs_error_um`
- fixed additive per-split MAE
- fixed additive High-Vault MAE
- fixed additive multi-view subgroup MAE
- G2 delta and wins versus fixed additive
- aggregate fixed-additive-dependent summaries

## Source Of Error

The source of the error was alpha semantic inversion in `scripts/run_reliability_aware_gate_experiment.py`.

The historical G2 archive interpreted `alpha=0.35` as AS-OCT weight:

`0.35 * AS-OCT + 0.65 * Measurement`

The original v5.2 matched additive fusion audit generated and froze alpha as Measurement weight:

`0.35 * Measurement + 0.65 * AS-OCT`

## Corrected Archive

The historical archive was preserved:

`artifacts/reports/v5_2_matched_fusion_audit/reliability_aware_gate_multiview_v1_formal/`

The corrected archive was created:

`artifacts/reports/v5_2_matched_fusion_audit/reliability_aware_gate_multiview_v1_formal_corrected/`

## Correction QC

Correction QC passed.

The corrected fixed additive per-split MAEs reproduce the original v5.2 matched audit additive MAEs within numerical tolerance.

Corrected fixed additive repeated MAE:

`167.906904 +/- 24.559626 um`

Corrected fixed additive wins versus matched best unimodal:

`3/5`

## Corrected Formal Result

G2 repeated MAE remains:

`171.836996 +/- 28.639685 um`

G2 wins versus corrected fixed additive:

`2/5`

Corrected fixed additive High-Vault MAE:

`346.706880 um`

G2 High-Vault MAE:

`351.979976 um`

## Final Decision

NO-GO.

After correction, G2 still does not provide stable improvement over the matched best unimodal model or the corrected validation-tuned fixed additive fusion comparator. The corrected result does not justify increasing gate complexity under the current information setting.

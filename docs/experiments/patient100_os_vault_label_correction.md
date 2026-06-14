# patient_100 OS Vault Label Correction

- Correction date: 2026-06-13
- Target: `batch_03__patient_100_OS_20240517`
- Human source verification: completed against original POD1 CASIA2 report/image.
- Correction: `manual_vault_mean_um` 7901.0 um -> 701.0 um.
- Scan values retained: scan1 = 684 um, scan2 = 718 um.
- Reason: manual mean transcription error.
- Provenance audit: `artifacts/reports/combined_batch_01_02_03_04/as_oct_only_repeated_splits/patient100_os_7901_label_provenance_audit.md`
- Upstream backup: `data/manifests/vault_label_candidates_batch_03_eyelevel_manual_review_before_patient100_label_correction_20260613_183737.csv`
- Structured correction record: `artifacts/reports/combined_batch_01_02_03_04/label_corrections/patient100_os_vault_label_correction.csv`

Only the upstream eye-level manual review file was manually corrected. Downstream AS-OCT label/manifests were rebuilt programmatically from that corrected source and existing split assignments. No model training was started.


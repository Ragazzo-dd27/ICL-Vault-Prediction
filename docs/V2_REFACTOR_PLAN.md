# V2 Refactor Plan

## 1. Current Project Status

The repository currently contains three prototype lines for the ICL postoperative Vault prediction project:

1. A U-Net segmentation rehearsal based on the public Keratitis dataset.
2. A ResNet18 backbone pretraining prototype based on the public MCOA dataset.
3. A multimodal Vault prediction prototype that is still primarily driven by simulated data.

The current repository is functional as a research prototype, but its engineering structure is still flat and script-centric:

- training and inference entrypoints live in the repository root
- model code and dataset code are mixed into `models/` and `utils/`
- checkpoints, logs, figures, and code coexist in the same top-level space
- the multimodal task does not yet have a stable real-data schema

## 2. V2 Refactor Goals

The V2 phase focuses on engineering scaffolding rather than new model ideas.

Primary goals:

1. Establish a clearer and more maintainable project directory layout.
2. Introduce a unified dataset schema for future real clinical data integration.
3. Standardize training and inference entrypoints under `scripts/`.
4. Separate reusable library code under `src/icl_vault/`.
5. Keep all existing prototype code runnable during the transition.

Non-goals for this stage:

- deleting old prototype files
- rewriting existing model implementations
- changing current business logic
- moving legacy files before migration paths are validated

## 3. New Directory Structure

```text
ICL_Vault_Project/
├─ configs/
│  ├─ data/
│  ├─ model/
│  └─ experiment/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  ├─ manifests/
│  └─ splits/
├─ docs/
├─ scripts/
├─ src/
│  └─ icl_vault/
│     ├─ data/
│     │  ├─ datasets/
│     │  ├─ transforms/
│     │  ├─ collate.py
│     │  └─ schema.py
│     ├─ models/
│     │  ├─ segmentation/
│     │  ├─ backbones/
│     │  ├─ multimodal/
│     │  └─ losses/
│     ├─ engine/
│     │  ├─ trainer.py
│     │  ├─ evaluator.py
│     │  ├─ checkpoint.py
│     │  └─ logger.py
│     └─ utils/
├─ artifacts/
│  ├─ checkpoints/
│  ├─ logs/
│  ├─ figures/
│  └─ predictions/
├─ tests/
├─ README.md
└─ requirements.txt
```

Notes:

- `src/icl_vault/` is intended to host reusable V2 library code.
- `scripts/` is intended to host thin execution entrypoints only.
- `data/manifests/` and `data/splits/` will become the canonical control layer for dataset membership and splits.
- `artifacts/` is intended to receive future V2 outputs without affecting legacy `checkpoints/` and `logs/`.

## 4. Legacy File Classification

### 4.1 Can Be Kept

- `models/unet.py`
- `models/multimodal_net.py`
- `utils/plot_metrics.py`
- `README.md`
- files under `docs/proposal/`
- legacy checkpoints and logs for reproducibility

These files are currently useful as prototype references and should remain untouched during early V2 scaffolding.

### 4.2 Need Migration

- `train_unet.py`
- `pretrain_backbone.py`
- `train_multimodal.py`
- `inference.py`
- `demo.py`
- `main.py`
- `utils/dataset.py`
- `utils/mcoa_dataset.py`

These files contain logic that should eventually be wrapped or migrated into:

- `scripts/` for entrypoints
- `src/icl_vault/data/` for dataset/schema logic
- `src/icl_vault/engine/` for train/eval/checkpoint/logging flow

### 4.3 Need Rewrite

- `utils/multimodal_dataset.py`
- future real-data manifest builder for multimodal clinical samples
- future split-generation utilities for patient-level train/val/test separation

Reason:

- the current multimodal dataset is simulation-only and cannot serve as the long-term V2 data interface
- real clinical integration requires an explicit schema, sample identifier policy, manifest format, and missing-modality handling

## 5. Phase 1 Refactor Priorities

1. Create the V2 directory skeleton without changing legacy behavior.
2. Define a minimal canonical sample schema in `src/icl_vault/data/schema.py`.
3. Prepare thin V2 entrypoints under `scripts/`.
4. Add minimal engine placeholders so later migration has a stable landing zone.
5. Create a conservative `requirements.txt` based on currently observed imports.

Suggested next implementation order after scaffolding:

1. Rewrite the multimodal dataset interface around manifests and schema.
2. Rebuild the main vault training entrypoint around config + engine abstractions.
3. Migrate the MCOA pretraining dataset to a more explicit label source.

## 6. Known Risks and TODO

### Known Risks

- The final real clinical data table format is not yet defined in this repository.
- The multimodal sample alignment strategy is not yet fixed.
- Missing modality policy is not yet defined.
- Patient-level split rules are not yet defined.
- Legacy and V2 checkpoint naming may diverge unless normalized later.

### TODO

- TODO: Confirm the authoritative clinical feature list and units.
- TODO: Confirm whether OCT and UBM are paired one-to-one at sample level or visit level.
- TODO: Confirm target label source and post-processing for Vault regression.
- TODO: Define a manifest format, likely CSV or JSONL, for multimodal samples.
- TODO: Define patient-level split generation rules to avoid leakage.
- TODO: Decide whether V2 will remain plain PyTorch or adopt a training framework.
- TODO: Add tests after the first real V2 dataset and trainer abstractions exist.

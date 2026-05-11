"""Build AS-OCT-only POD1 vault regression smoke-test manifests.

This creates the first real-data AS-OCT-only POD1 smoke-test / baseline
manifest from the POD1 formal draft. It is not the final multimodal training
manifest: UBM is retained as patient-level metadata, but AS-OCT raw is the
intended model input for this version.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
LABEL_SOURCE = "POD1_CASIA2_2DAnalysis_manual_verified"
INPUT_STRATEGY = "as_oct_primary_image_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AS-OCT-only POD1 vault regression manifests for batch_01."
    )
    parser.add_argument(
        "--formal_manifest_in",
        type=str,
        default="data/manifests/formal_vault_manifest_batch_01_pod1_draft.csv",
        help="Input POD1 formal vault manifest draft.",
    )
    parser.add_argument(
        "--full_out",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_full.csv",
        help="Output full AS-OCT-only POD1 manifest.",
    )
    parser.add_argument(
        "--clean_out",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean.csv",
        help="Output clean AS-OCT-only POD1 manifest excluding large between-scan differences.",
    )
    parser.add_argument(
        "--split_out",
        type=str,
        default="data/splits/pod1_batch_01_patient_level_split.csv",
        help="Output patient-level train/val/test split CSV.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Fixed random seed for patient-level split.")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def split_paths(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return sorted(dict.fromkeys(item.strip().replace("\\", "/") for item in text.split(";") if item.strip()))


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def eye_to_side(eye: str) -> str:
    value = str(eye).strip().upper()
    if value == "OD":
        return "R"
    if value == "OS":
        return "L"
    return value


def path_exists(path_value: object) -> bool:
    paths = split_paths(path_value)
    if not paths:
        return False
    path = Path(paths[0])
    if path.is_absolute():
        return path.exists()
    return (PROJECT_ROOT / path).exists()


def resolve_existing_path_text(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path.as_posix()
    if not path.is_absolute() and (PROJECT_ROOT / path).exists():
        return path_value

    alternates = []
    normalized = path_value.replace("\\", "/")
    if "/real_export_batch_01/patients/" in normalized:
        alternates.append(normalized.replace("/real_export_batch_01/patients/", "/real_export_batch_01/patient/"))
    if "/real_export_batch_01/patient/" in normalized:
        alternates.append(normalized.replace("/real_export_batch_01/patient/", "/real_export_batch_01/patients/"))

    for alternate in alternates:
        alternate_path = Path(alternate)
        if alternate_path.is_absolute() and alternate_path.exists():
            return alternate_path.as_posix()
        if not alternate_path.is_absolute() and (PROJECT_ROOT / alternate_path).exists():
            return alternate
    return path_value


def resolve_existing_paths_text(path_value: object) -> str:
    return ";".join(resolve_existing_path_text(path) for path in split_paths(path_value))


def first_path(value: object) -> str:
    paths = split_paths(value)
    return paths[0] if paths else ""


def first_ubm_path(row: pd.Series) -> str:
    paths = split_paths(row.get("ubm_horizontal_paths", "")) + split_paths(row.get("ubm_vertical_paths", ""))
    return sorted(dict.fromkeys(paths))[0] if paths else ""


def build_patient_split(patients: Iterable[str], seed: int) -> pd.DataFrame:
    patient_list = sorted(str(patient) for patient in patients)
    rng = random.Random(seed)
    rng.shuffle(patient_list)

    total = len(patient_list)
    train_count = round(total * SPLIT_RATIOS["train"])
    val_count = round(total * SPLIT_RATIOS["val"])
    # Keep the split deterministic and make the remainder test.
    train_patients = set(patient_list[:train_count])
    val_patients = set(patient_list[train_count : train_count + val_count])

    rows = []
    for patient_uid in sorted(patient_list):
        if patient_uid in train_patients:
            split = "train"
        elif patient_uid in val_patients:
            split = "val"
        else:
            split = "test"
        rows.append({"patient_uid": patient_uid, "patient_id": patient_uid, "split": split, "seed": seed})
    return pd.DataFrame(rows, columns=["patient_uid", "patient_id", "split", "seed"])


def append_note(existing: object) -> str:
    base = "" if pd.isna(existing) else str(existing).strip()
    addition = "AS-OCT-only POD1 training manifest; UBM not used as model input in this version."
    if base:
        return f"{base} | {addition}"
    return addition


def build_training_manifest(formal_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    split_map: Dict[str, str] = dict(zip(split_df["patient_uid"], split_df["split"]))
    rows: List[Dict[str, object]] = []

    for row in formal_df.sort_values(by=["patient_uid", "eye", "sample_id"], kind="stable").to_dict(orient="records"):
        patient_uid = str(row["patient_uid"])
        oct_paths = resolve_existing_paths_text(row.get("preop_as_oct_raw_paths", ""))
        oct_path = first_path(oct_paths)
        vault_label = pd.to_numeric(pd.Series([row.get("pod1_vault_mean_um", "")]), errors="coerce").iloc[0]
        has_ubm = bool(normalize_bool_series(pd.Series([row.get("has_ubm", False)])).iloc[0])
        ubm_path = resolve_existing_path_text(first_ubm_path(pd.Series(row))) if has_ubm else ""

        rows.append(
            {
                "sample_id": row["sample_id"],
                "patient_id": patient_uid,
                "patient_uid": patient_uid,
                "eye_side": eye_to_side(row["eye"]),
                "eye": row["eye"],
                "split": split_map.get(patient_uid, ""),
                "oct_path": oct_path,
                "oct_paths": oct_paths,
                "num_preop_as_oct_raw": len(split_paths(oct_paths)),
                "ubm_path": ubm_path,
                "topography_path": "",
                "has_oct": bool(oct_path),
                "has_ubm": has_ubm,
                "has_topography": False,
                "device_oct": "CASIA2",
                "device_ubm": "",
                "ubm_alignment_status": row.get("ubm_alignment_status", ""),
                "vault_label": float(vault_label) if pd.notna(vault_label) else "",
                "pod1_vault_mean_um": row.get("pod1_vault_mean_um", ""),
                "pod1_vault_range_um": row.get("pod1_vault_range_um", ""),
                "label_qc_flag": row.get("label_qc_flag", ""),
                "label_status": row.get("label_status", ""),
                "verify_status": row.get("verify_status", ""),
                "training_ready_status": row.get("training_ready_status", ""),
                "input_strategy": INPUT_STRATEGY,
                "label_source": LABEL_SOURCE,
                "notes": append_note(row.get("notes", "")),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "patient_id",
            "patient_uid",
            "eye_side",
            "eye",
            "split",
            "oct_path",
            "oct_paths",
            "num_preop_as_oct_raw",
            "ubm_path",
            "topography_path",
            "has_oct",
            "has_ubm",
            "has_topography",
            "device_oct",
            "device_ubm",
            "ubm_alignment_status",
            "vault_label",
            "pod1_vault_mean_um",
            "pod1_vault_range_um",
            "label_qc_flag",
            "label_status",
            "verify_status",
            "training_ready_status",
            "input_strategy",
            "label_source",
            "notes",
        ],
    )


def validate_manifest(df: pd.DataFrame, clean: bool = False) -> Dict[str, object]:
    vault_labels = pd.to_numeric(df["vault_label"], errors="coerce")
    missing_oct_path = int(df["oct_path"].fillna("").astype(str).str.strip().eq("").sum())
    nonexistent_oct_path = int((~df["oct_path"].map(path_exists)).sum())
    split_missing = int(df["split"].fillna("").astype(str).str.strip().eq("").sum())
    duplicate_sample_ids = int(df["sample_id"].duplicated().sum())
    invalid_labels = int((vault_labels.isna() | (vault_labels <= 0)).sum())
    split_counts_per_patient = df.groupby("patient_uid")["split"].nunique()
    leaked_patients = split_counts_per_patient[split_counts_per_patient > 1].index.tolist()
    clean_large_diff = 0
    if clean:
        clean_large_diff = int((df["label_qc_flag"] == "large_between_scan_difference").sum())

    return {
        "missing_oct_path": missing_oct_path,
        "nonexistent_oct_path": nonexistent_oct_path,
        "split_missing": split_missing,
        "duplicate_sample_ids": duplicate_sample_ids,
        "invalid_labels": invalid_labels,
        "leaked_patients": leaked_patients,
        "clean_large_diff": clean_large_diff,
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def print_distribution(title: str, series: pd.Series) -> None:
    print(f"{title}:")
    if series.empty:
        print("  none")
        return
    for value, count in series.value_counts().sort_index().items():
        print(f"  {value}: {count}")


def print_summary(
    full_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    split_df: pd.DataFrame,
    full_validation: Dict[str, object],
    clean_validation: Dict[str, object],
    full_out: Path,
    clean_out: Path,
    split_out: Path,
) -> None:
    labels = pd.to_numeric(full_df["vault_label"], errors="coerce").dropna()

    print(f"Full manifest rows: {len(full_df)}")
    print(f"Clean manifest rows: {len(clean_df)}")
    print(f"Patients: {split_df['patient_uid'].nunique()}")
    print("Train/val/test patient counts:")
    for split, count in split_df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0).items():
        print(f"  {split}: {count}")
    print("Train/val/test sample counts (full):")
    for split, count in full_df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0).items():
        print(f"  {split}: {count}")
    print("Train/val/test sample counts (clean):")
    for split, count in clean_df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0).items():
        print(f"  {split}: {count}")
    print_distribution("Full label_qc_flag distribution", full_df["label_qc_flag"])
    print_distribution("Clean label_qc_flag distribution", clean_df["label_qc_flag"])
    print(f"Samples missing UBM: {int((~normalize_bool_series(full_df['has_ubm'])).sum())}")
    print(
        "vault_label stats: "
        f"mean={labels.mean():.2f}, std={labels.std():.2f}, min={labels.min():.2f}, max={labels.max():.2f}"
    )
    print(f"Missing oct_path samples: {full_validation['missing_oct_path']}")
    print(f"Nonexistent oct_path samples: {full_validation['nonexistent_oct_path']}")
    print(f"Missing split samples: {full_validation['split_missing']}")
    print(f"Duplicate sample_id count: {full_validation['duplicate_sample_ids']}")
    print(f"Invalid vault_label samples: {full_validation['invalid_labels']}")
    print(
        "Patients crossing splits: "
        f"{', '.join(full_validation['leaked_patients']) if full_validation['leaked_patients'] else 'none'}"
    )
    print(f"Clean manifest large_between_scan_difference count: {clean_validation['clean_large_diff']}")
    print(f"Full output: {relative_path(full_out)}")
    print(f"Clean output: {relative_path(clean_out)}")
    print(f"Split output: {relative_path(split_out)}")


def main() -> None:
    args = parse_args()
    formal_manifest_in = resolve_project_path(args.formal_manifest_in)
    full_out = resolve_project_path(args.full_out)
    clean_out = resolve_project_path(args.clean_out)
    split_out = resolve_project_path(args.split_out)

    formal_df = pd.read_csv(formal_manifest_in)
    split_df = build_patient_split(formal_df["patient_uid"].unique(), seed=args.seed)
    full_df = build_training_manifest(formal_df=formal_df, split_df=split_df)
    clean_df = full_df[full_df["label_qc_flag"] != "large_between_scan_difference"].copy()

    full_validation = validate_manifest(full_df)
    clean_validation = validate_manifest(clean_df, clean=True)

    if full_validation["split_missing"]:
        raise ValueError("Some samples have empty split assignments.")
    if full_validation["duplicate_sample_ids"]:
        raise ValueError("sample_id must be unique in the full manifest.")
    if full_validation["leaked_patients"]:
        raise ValueError(f"Patient-level split leakage found: {full_validation['leaked_patients']}")
    if clean_validation["clean_large_diff"]:
        raise ValueError("Clean manifest still contains large_between_scan_difference samples.")

    write_csv(full_df, full_out)
    write_csv(clean_df, clean_out)
    write_csv(split_df, split_out)
    print_summary(
        full_df=full_df,
        clean_df=clean_df,
        split_df=split_df,
        full_validation=full_validation,
        clean_validation=clean_validation,
        full_out=full_out,
        clean_out=clean_out,
        split_out=split_out,
    )


if __name__ == "__main__":
    main()

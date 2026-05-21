"""Validate preoperative CASIA2 measurement records for baseline use.

Postoperative 2DAnalysis measurements must not be used as preoperative input
features. This validator checks the clean batch_01 samples, filters out
excluded or non-verified records, detects leakage risk, and reports whether
each sample is ready for a future preop measurement-only baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMNS = ["cct_um", "acd_epi_mm", "acd_endo_mm", "clr_um", "ata_mm"]
RANGES = {
    "cct_um": (300, 800),
    "acd_epi_mm": (1.0, 7.0),
    "acd_endo_mm": (1.0, 7.0),
    "ata_mm": (6.0, 15.0),
    "clr_um": (-6000, 3000),
}
EXCLUDE_NOTE = "exclude_from_preop_measurement_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate batch_01 preop measurement records.")
    parser.add_argument(
        "--measurements",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_preop_manual_review_priority_checked_patient042_fixed.csv",
    )
    parser.add_argument(
        "--clean_manifest",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean_strict.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/reports/as_oct_pod1_baseline_batch_01/preop_measurements",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def normalize_measurements(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "verify_status" not in normalized.columns and "verified" in normalized.columns:
        normalized["verify_status"] = normalized["verified"]
    if "verify_status" not in normalized.columns:
        normalized["verify_status"] = ""
    if "is_preop" not in normalized.columns:
        normalized["is_preop"] = True
    normalized["is_preop"] = normalize_bool_series(normalized["is_preop"])
    normalized["has_vault"] = normalize_bool_series(normalized["has_vault"]) if "has_vault" in normalized.columns else False
    normalized["verify_status"] = normalized["verify_status"].fillna("").astype(str).str.strip().str.lower()
    normalized["extraction_method"] = normalized["extraction_method"].fillna("").astype(str).str.strip().str.lower()
    normalized["notes"] = normalized["notes"].fillna("").astype(str)
    for column in KEY_COLUMNS + ["vault_um"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def path_candidates(path_text: object) -> List[Path]:
    if pd.isna(path_text):
        return []
    text = str(path_text).strip()
    if not text or text.lower() == "nan":
        return []
    raw = Path(text)
    candidates = [raw if raw.is_absolute() else PROJECT_ROOT / raw]
    normalized = text.replace("\\", "/")
    if "/real_export_batch_01/patients/" in normalized:
        alt = normalized.replace("/real_export_batch_01/patients/", "/real_export_batch_01/patient/")
        candidates.append(Path(alt) if Path(alt).is_absolute() else PROJECT_ROOT / alt)
    if "/real_export_batch_01/patient/" in normalized:
        alt = normalized.replace("/real_export_batch_01/patient/", "/real_export_batch_01/patients/")
        candidates.append(Path(alt) if Path(alt).is_absolute() else PROJECT_ROOT / alt)
    return candidates


def path_exists(path_text: object) -> bool:
    return any(path.exists() for path in path_candidates(path_text))


def is_excluded(df: pd.DataFrame) -> pd.Series:
    return df["verify_status"].eq("excluded") | df["notes"].str.contains(EXCLUDE_NOTE, case=False, regex=False)


def is_candidate(df: pd.DataFrame) -> pd.Series:
    return (
        df["is_preop"]
        & df["verify_status"].eq("verified")
        & df["extraction_method"].eq("manual_verified")
        & ~is_excluded(df)
    )


def leakage_risk(df: pd.DataFrame) -> pd.Series:
    vault_present = df["vault_um"].notna() if "vault_um" in df.columns else pd.Series(False, index=df.index)
    has_vault = df["has_vault"] if "has_vault" in df.columns else pd.Series(False, index=df.index)
    raw_text = df["vault_raw_text"].fillna("").astype(str).str.strip() if "vault_raw_text" in df.columns else pd.Series("", index=df.index)
    raw_text_present = raw_text.ne("") & raw_text.ne("---") & raw_text.str.lower().ne("nan")
    return vault_present | has_vault | raw_text_present


def missing_key_measurement(df: pd.DataFrame) -> pd.Series:
    missing = pd.Series(False, index=df.index)
    for column in KEY_COLUMNS:
        missing = missing | df[column].isna()
    return missing


def outlier_status(row: pd.Series) -> str:
    statuses: List[str] = []
    confirmed = "outlier_confirmed_from_crop" in str(row.get("notes", ""))
    for column, (lower, upper) in RANGES.items():
        value = row.get(column)
        if pd.isna(value):
            continue
        if value < lower or value > upper:
            statuses.append("outlier_confirmed" if confirmed else "outlier_recheck_needed")
    if not statuses:
        return ""
    if "outlier_recheck_needed" in statuses:
        return "outlier_recheck_needed"
    return "outlier_confirmed"


def collect_problem_records(df: pd.DataFrame) -> pd.DataFrame:
    records = df.copy()
    records["is_candidate"] = is_candidate(records)
    records["is_excluded_record"] = is_excluded(records)
    records["leakage_risk_preop_vault_present"] = leakage_risk(records)
    records["missing_key_measurement"] = missing_key_measurement(records)
    records["image_path_exists"] = records["image_path"].map(path_exists) if "image_path" in records.columns else False
    records["measurement_crop_path_exists"] = (
        records["measurement_crop_path"].map(path_exists) if "measurement_crop_path" in records.columns else False
    )
    records["outlier_status"] = records.apply(outlier_status, axis=1)
    records["problem_flags"] = records.apply(record_flags, axis=1)
    return records[records["problem_flags"].astype(str).str.strip().ne("")].copy()


def record_flags(row: pd.Series) -> str:
    flags: List[str] = []
    if not bool(row.get("is_preop", False)):
        flags.append("not_preop")
    if str(row.get("verify_status", "")) in {"pending", "uncertain"}:
        flags.append(str(row.get("verify_status")))
    if bool(row.get("is_excluded_record", False)):
        flags.append("excluded")
    if not bool(row.get("image_path_exists", True)):
        flags.append("missing_image_path")
    if not bool(row.get("measurement_crop_path_exists", True)):
        flags.append("missing_measurement_crop_path")
    if bool(row.get("is_candidate", False)) and bool(row.get("leakage_risk_preop_vault_present", False)):
        flags.append("leakage_risk_preop_vault_present")
    if bool(row.get("is_candidate", False)) and bool(row.get("missing_key_measurement", False)):
        flags.append("missing_key_measurement")
    outlier = str(row.get("outlier_status", ""))
    if outlier:
        flags.append(outlier)
    return ";".join(dict.fromkeys(flags))


def sample_status(clean_row: Dict[str, object], sample_records: pd.DataFrame) -> Dict[str, object]:
    sample_id = str(clean_row["sample_id"])
    candidate_records = sample_records[is_candidate(sample_records)].copy() if not sample_records.empty else sample_records.copy()
    verified_records = sample_records[sample_records["verify_status"].eq("verified")] if not sample_records.empty else sample_records.copy()
    has_values = {column: (candidate_records[column].notna().any() if not candidate_records.empty else False) for column in KEY_COLUMNS}
    has_leakage = bool(leakage_risk(candidate_records).any()) if not candidate_records.empty else False
    has_missing_key = bool(missing_key_measurement(candidate_records).any()) if not candidate_records.empty else True
    outlier_statuses = candidate_records.apply(outlier_status, axis=1).tolist() if not candidate_records.empty else []
    has_unconfirmed_outlier = "outlier_recheck_needed" in outlier_statuses
    has_confirmed_outlier = "outlier_confirmed" in outlier_statuses
    has_pending_uncertain = bool(sample_records["verify_status"].isin(["pending", "uncertain"]).any()) if not sample_records.empty else False

    if candidate_records.empty:
        status = "missing_preop_measurement"
    elif has_leakage:
        status = "leakage_risk"
    elif all(has_values.values()) and not has_unconfirmed_outlier:
        status = "measurement_ready_with_confirmed_outlier" if has_confirmed_outlier else "measurement_ready"
    else:
        status = "manual_review_needed"
    if has_pending_uncertain and status == "measurement_ready":
        status = "manual_review_needed"

    return {
        "sample_id": sample_id,
        "patient_uid": clean_row["patient_uid"],
        "eye": clean_row["eye"],
        "split": clean_row["split"],
        "num_candidate_preop_records": int(len(candidate_records)),
        "num_verified_preop_records": int(len(verified_records[~is_excluded(verified_records)])) if not verified_records.empty else 0,
        "has_cct": has_values["cct_um"],
        "has_acd_epi": has_values["acd_epi_mm"],
        "has_acd_endo": has_values["acd_endo_mm"],
        "has_clr": has_values["clr_um"],
        "has_ata": has_values["ata_mm"],
        "has_leakage_risk": has_leakage,
        "has_missing_key_measurement": has_missing_key,
        "has_unconfirmed_outlier": has_unconfirmed_outlier,
        "measurement_ready_status": status,
    }


def validate_samples(clean_df: pd.DataFrame, measurements_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for clean_row in clean_df.sort_values(["patient_uid", "eye", "sample_id"], kind="stable").to_dict(orient="records"):
        sample_records = measurements_df[measurements_df["sample_id"].astype(str).eq(str(clean_row["sample_id"]))]
        rows.append(sample_status(clean_row, sample_records))
    return pd.DataFrame(rows)


def patient042_checks(measurements_df: pd.DataFrame) -> Dict[str, object]:
    p42 = measurements_df[measurements_df["patient_uid"].astype(str).eq("patient_042")].copy()
    true_preop = p42[p42["notes"].str.contains("newly_added_patient042_true_preop_2danalysis", regex=False)]
    old_excluded = p42[
        p42["notes"].str.contains("possible_postop_record_originally_misclassified", regex=False)
        | p42["notes"].str.contains(EXCLUDE_NOTE, regex=False)
    ]
    true_preop_verified = bool(true_preop["verify_status"].eq("verified").all()) if not true_preop.empty else False
    true_preop_no_vault = bool((~leakage_risk(true_preop)).all()) if not true_preop.empty else False
    old_records_excluded = bool(is_excluded(old_excluded).all()) if not old_excluded.empty else False
    old_candidate_count = int(is_candidate(old_excluded).sum()) if not old_excluded.empty else 0
    return {
        "new_true_preop_records": int(len(true_preop)),
        "new_true_preop_verified": true_preop_verified,
        "new_true_preop_has_vault_false": true_preop_no_vault,
        "old_possible_postop_records": int(len(old_excluded)),
        "old_records_excluded": old_records_excluded,
        "old_records_entering_candidate_count": old_candidate_count,
        "passed": bool(len(true_preop) >= 4 and true_preop_verified and true_preop_no_vault and old_records_excluded and old_candidate_count == 0),
    }


def write_summary_md(
    path: Path,
    clean_count: int,
    record_count: int,
    candidate_count: int,
    excluded_count: int,
    pending_uncertain_count: int,
    status_counts: Dict[str, int],
    p42: Dict[str, object],
) -> None:
    lines = [
        "# Preoperative measurement validation summary",
        "",
        "本报告验证 clean 83 samples 对应的术前 CASIA2 2DAnalysis measurement 表是否可用于后续 preop measurement-only baseline。Postoperative 2DAnalysis measurements must not be used as preoperative input features.",
        "",
        "## Overall counts",
        "",
        f"- clean manifest samples: {clean_count}",
        f"- preop measurement records: {record_count}",
        f"- verified non-excluded baseline candidate records: {candidate_count}",
        f"- excluded records: {excluded_count}",
        f"- pending / uncertain records: {pending_uncertain_count}",
        "",
        "## Sample readiness",
        "",
    ]
    for key in [
        "measurement_ready",
        "measurement_ready_with_confirmed_outlier",
        "manual_review_needed",
        "missing_preop_measurement",
        "leakage_risk",
    ]:
        lines.append(f"- {key}: {status_counts.get(key, 0)}")
    lines.extend(
        [
            "",
            "## patient_042 special check",
            "",
            f"- new true preop records: {p42['new_true_preop_records']}",
            f"- new true preop all verified: {p42['new_true_preop_verified']}",
            f"- new true preop has no vault leakage: {p42['new_true_preop_has_vault_false']}",
            f"- old possible postop records: {p42['old_possible_postop_records']}",
            f"- old records excluded: {p42['old_records_excluded']}",
            f"- old records entering candidate count: {p42['old_records_entering_candidate_count']}",
            f"- patient_042 check passed: {p42['passed']}",
            "",
            "## Interpretation",
            "",
            "只有 `measurement_ready` 和 `measurement_ready_with_confirmed_outlier` 样本适合作为 preop measurement-only baseline 的候选输入。`manual_review_needed`、`missing_preop_measurement` 和 `leakage_risk` 样本需要人工处理后再进入建模。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    args = parse_args()
    measurements_path = resolve_project_path(args.measurements)
    clean_manifest_path = resolve_project_path(args.clean_manifest)
    output_dir = resolve_project_path(args.output_dir)
    report_path = output_dir / "preop_measurement_validation_report.csv"
    problem_path = output_dir / "preop_measurement_problem_records.csv"
    summary_path = output_dir / "preop_measurement_validation_summary.md"

    measurements_df = normalize_measurements(pd.read_csv(measurements_path))
    clean_df = pd.read_csv(clean_manifest_path)
    validation_df = validate_samples(clean_df, measurements_df)
    problem_df = collect_problem_records(measurements_df)
    p42 = patient042_checks(measurements_df)

    candidate_count = int(is_candidate(measurements_df).sum())
    excluded_count = int(is_excluded(measurements_df).sum())
    pending_uncertain_count = int(measurements_df["verify_status"].isin(["pending", "uncertain"]).sum())
    status_counts = validation_df["measurement_ready_status"].value_counts().to_dict()

    write_csv(validation_df, report_path)
    write_csv(problem_df, problem_path)
    write_summary_md(
        summary_path,
        clean_count=len(clean_df),
        record_count=len(measurements_df),
        candidate_count=candidate_count,
        excluded_count=excluded_count,
        pending_uncertain_count=pending_uncertain_count,
        status_counts=status_counts,
        p42=p42,
    )

    print(f"Clean manifest samples: {len(clean_df)}")
    print(f"Preop measurement records: {len(measurements_df)}")
    print(f"Verified non-excluded records: {candidate_count}")
    print(f"Excluded records: {excluded_count}")
    print(f"Pending / uncertain records: {pending_uncertain_count}")
    print(f"measurement_ready samples: {status_counts.get('measurement_ready', 0)}")
    print(
        "measurement_ready_with_confirmed_outlier samples: "
        f"{status_counts.get('measurement_ready_with_confirmed_outlier', 0)}"
    )
    print(f"manual_review_needed samples: {status_counts.get('manual_review_needed', 0)}")
    print(f"missing_preop_measurement samples: {status_counts.get('missing_preop_measurement', 0)}")
    print(f"leakage_risk samples: {status_counts.get('leakage_risk', 0)}")
    print(f"patient_042 check: {p42}")
    print(f"Validation report: {relative_path(report_path)}")
    print(f"Problem records: {relative_path(problem_path)}")
    print(f"Summary markdown: {relative_path(summary_path)}")


if __name__ == "__main__":
    main()

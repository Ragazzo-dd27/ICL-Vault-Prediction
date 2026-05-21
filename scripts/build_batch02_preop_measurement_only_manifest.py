"""Build batch_02 preoperative measurement-only POD1 manifest.

This is a true preoperative measurement-only manifest. Postoperative
2DAnalysis measurements must not be used as input features.

The script does not modify inputs, does not train models, and keeps batch_02
independent from batch_01.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["cct_um", "acd_epi_mm", "acd_endo_mm", "clr_um", "ata_mm"]
SORT_COLUMNS = ["patient_uid", "eye", "sample_id", "exam_date", "exam_time", "analysis_index", "image_path"]
READY_STATUSES = {"measurement_ready", "measurement_ready_with_confirmed_outlier"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build batch_02 preop measurement-only manifests.")
    parser.add_argument(
        "--checked_in",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_02_preop_manual_review_priority_ready_checked_v2.csv",
        help="Input batch_02 preop checked_v2 CSV.",
    )
    parser.add_argument(
        "--validation_report_in",
        type=str,
        default="artifacts/reports/batch_02_data_curation/preop_measurements/batch_02_preop_measurement_validation_report.csv",
        help="Input batch_02 preop measurement validation report.",
    )
    parser.add_argument(
        "--ready_as_oct_manifest_in",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_02_ready.csv",
        help="Input batch_02 AS-OCT-only ready manifest.",
    )
    parser.add_argument(
        "--strict_as_oct_manifest_in",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_02_strict.csv",
        help="Input batch_02 AS-OCT-only strict manifest.",
    )
    parser.add_argument(
        "--labels_in",
        type=str,
        default="data/manifests/vault_label_candidates_batch_02_pod1_verified.csv",
        help="Input batch_02 POD1 verified label table.",
    )
    parser.add_argument(
        "--ready_out",
        type=str,
        default="data/manifests/vault_preop_measurement_only_pod1_manifest_batch_02_ready.csv",
        help="Output ready preop measurement-only manifest.",
    )
    parser.add_argument(
        "--strict_out",
        type=str,
        default="data/manifests/vault_preop_measurement_only_pod1_manifest_batch_02_strict.csv",
        help="Output strict preop measurement-only manifest.",
    )
    parser.add_argument(
        "--summary_md_out",
        type=str,
        default="artifacts/reports/batch_02_data_curation/preop_measurements/batch_02_preop_measurement_manifest_summary.md",
        help="Output Markdown summary.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def sort_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_analysis_index_sort"] = pd.to_numeric(out.get("analysis_index"), errors="coerce")
    by: list[str] = []
    for column in SORT_COLUMNS:
        if column in out.columns:
            by.append("_analysis_index_sort" if column == "analysis_index" else column)
    return out.sort_values(by=by, kind="stable", na_position="last").drop(
        columns=["_analysis_index_sort"], errors="ignore"
    )


def join_unique(values: Iterable[object]) -> str:
    items = sorted({str(value).strip() for value in values if str(value).strip() and str(value).lower() != "nan"})
    return ";".join(items)


def eye_to_side(eye: object) -> str:
    token = str(eye).strip().upper()
    if token == "OD":
        return "R"
    if token == "OS":
        return "L"
    return ""


def prepare_checked(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["verify_status"] = normalize_text(out["verify_status"]).str.lower()
    out["extraction_method"] = normalize_text(out["extraction_method"]).str.lower()
    out["has_vault"] = normalize_bool_series(out["has_vault"])
    out["notes"] = normalize_text(out.get("notes", pd.Series("", index=out.index)))
    for column in FEATURES + ["vault_um"]:
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["excluded_from_preop_baseline"] = out["notes"].str.contains(
        "exclude_from_preop_measurement_baseline", case=False, na=False
    )
    out["candidate_record"] = (
        out["verify_status"].eq("verified")
        & out["extraction_method"].eq("manual_verified")
        & (~out["has_vault"])
        & out["vault_um"].isna()
        & (~out["excluded_from_preop_baseline"])
    )
    out["complete_record"] = out["candidate_record"] & out[FEATURES].notna().all(axis=1)
    return out


def scan_value(values: List[float], index: int) -> object:
    return values[index] if len(values) > index else ""


def aggregate_measurements(checked_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    complete = sort_records(checked_df[checked_df["complete_record"]].copy())
    all_candidates = checked_df[checked_df["candidate_record"]].copy()

    for sample_id, group in complete.groupby("sample_id", dropna=False):
        sample_id = str(sample_id)
        source_group = all_candidates[all_candidates["sample_id"].astype(str).eq(sample_id)]
        first = group.iloc[0]
        row: dict[str, object] = {
            "sample_id": sample_id,
            "patient_id": first["patient_uid"],
            "patient_uid": first["patient_uid"],
            "eye_side": eye_to_side(first["eye"]),
            "eye": first["eye"],
            "split": first.get("split", ""),
            "num_preop_measurement_records": int(len(source_group)),
            "num_complete_preop_measurement_records": int(len(group)),
            "measurement_source_images": join_unique(group["image_path"]),
            "measurement_crop_paths": join_unique(group["measurement_crop_path"]),
        }
        for feature in FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce").dropna().astype(float).tolist()
            prefix = {
                "cct_um": "cct",
                "acd_epi_mm": "acd_epi",
                "acd_endo_mm": "acd_endo",
                "clr_um": "clr",
                "ata_mm": "ata",
            }[feature]
            suffix = "um" if feature in {"cct_um", "clr_um"} else "mm"
            row[f"{prefix}_mean_{suffix}"] = float(pd.Series(values).mean()) if values else ""
            row[f"{prefix}_scan1_{suffix}"] = scan_value(values, 0)
            row[f"{prefix}_scan2_{suffix}"] = scan_value(values, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_manifest(
    aggregated_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    as_oct_manifest: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    validation = validation_df[
        [
            "sample_id",
            "measurement_ready_status",
            "has_single_valid_preop_record",
            "has_leakage_risk",
            "notes",
        ]
    ].rename(columns={"notes": "measurement_validation_notes"})
    label_cols = [
        "sample_id",
        "pod1_vault_mean_um",
        "pod1_vault_range_um",
        "label_qc_flag",
        "label_status",
        "verify_status",
    ]
    label_info = labels_df[label_cols].rename(columns={"verify_status": "label_verify_status"})
    base_cols = ["sample_id", "patient_id", "patient_uid", "eye_side", "eye", "split"]
    as_oct_base = as_oct_manifest[base_cols + ["label_qc_flag", "label_status", "verify_status", "vault_label"]].rename(
        columns={
            "label_qc_flag": "as_oct_label_qc_flag",
            "label_status": "as_oct_label_status",
            "verify_status": "as_oct_verify_status",
            "vault_label": "as_oct_vault_label",
        }
    )

    duplicate_base_cols = [col for col in base_cols if col != "sample_id" and col in aggregated_df.columns]
    df = as_oct_base.merge(aggregated_df.drop(columns=duplicate_base_cols), on="sample_id", how="left")
    df = df.merge(validation, on="sample_id", how="left")
    df = df.merge(label_info, on="sample_id", how="left")

    df["label_qc_flag"] = df["pod1_vault_range_um"].notna().map(lambda _: "")
    df["label_qc_flag"] = df["as_oct_label_qc_flag"].fillna(df["label_qc_flag"])
    df["label_status"] = df["as_oct_label_status"].fillna(df["label_status"] if "label_status" in df else "")
    df["verify_status"] = df["as_oct_verify_status"].fillna(df["label_verify_status"])
    df["vault_label"] = pd.to_numeric(df["pod1_vault_mean_um"], errors="coerce")
    df["measurement_input_status"] = df["measurement_ready_status"].map(
        {
            "measurement_ready": "ready",
            "measurement_ready_with_confirmed_outlier": "ready_with_confirmed_outlier",
        }
    ).fillna("")

    notes: list[str] = []
    for _, row in df.iterrows():
        row_notes = [
            "batch_02 true preoperative measurement-only manifest",
            "postoperative 2DAnalysis measurements excluded from input features",
        ]
        if bool(row.get("has_single_valid_preop_record", False)):
            row_notes.append("single_valid_preop_measurement_record")
        if row.get("measurement_ready_status") == "measurement_ready_with_confirmed_outlier":
            row_notes.append("confirmed_outlier_or_acd_cct_difference")
        validation_notes = str(row.get("measurement_validation_notes", "")).strip()
        if validation_notes and validation_notes.lower() != "nan":
            row_notes.append("validation_notes: " + validation_notes)
        notes.append(" | ".join(row_notes))
    df["notes"] = notes

    output_columns = [
        "sample_id",
        "patient_id",
        "patient_uid",
        "eye_side",
        "eye",
        "split",
        "cct_mean_um",
        "cct_scan1_um",
        "cct_scan2_um",
        "acd_epi_mean_mm",
        "acd_epi_scan1_mm",
        "acd_epi_scan2_mm",
        "acd_endo_mean_mm",
        "acd_endo_scan1_mm",
        "acd_endo_scan2_mm",
        "clr_mean_um",
        "clr_scan1_um",
        "clr_scan2_um",
        "ata_mean_mm",
        "ata_scan1_mm",
        "ata_scan2_mm",
        "num_preop_measurement_records",
        "num_complete_preop_measurement_records",
        "measurement_ready_status",
        "measurement_input_status",
        "vault_label",
        "pod1_vault_mean_um",
        "pod1_vault_range_um",
        "label_qc_flag",
        "label_status",
        "verify_status",
        "measurement_source_images",
        "measurement_crop_paths",
        "notes",
    ]
    for column in output_columns:
        if column not in df.columns:
            df[column] = ""
    return df[output_columns]


def validate_output(df: pd.DataFrame, checked_df: pd.DataFrame) -> dict[str, object]:
    labels = pd.to_numeric(df["vault_label"], errors="coerce") if not df.empty else pd.Series(dtype=float)
    mean_missing = int(df[["cct_mean_um", "acd_epi_mean_mm", "acd_endo_mean_mm", "clr_mean_um", "ata_mean_mm"]].isna().any(axis=1).sum()) if not df.empty else 0
    selected_records = checked_df[checked_df["sample_id"].isin(df["sample_id"])]
    leakage = bool((selected_records["has_vault"] | selected_records["vault_um"].notna()).any()) if not selected_records.empty else False
    excluded = bool(selected_records["notes"].str.contains("exclude_from_preop_measurement_baseline", case=False, na=False).any()) if not selected_records.empty else False
    return {
        "duplicate_sample_id": int(df["sample_id"].duplicated().sum()) if not df.empty else 0,
        "invalid_vault_label": int((labels.isna() | (labels <= 0)).sum()) if not df.empty else 0,
        "missing_mean_features": mean_missing,
        "leakage_risk": leakage,
        "excluded_record_present": excluded,
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_summary_md(
    path: Path,
    ready_df: pd.DataFrame,
    strict_df: pd.DataFrame,
    ready_validation: dict[str, object],
    strict_validation: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Batch 02 preop measurement-only manifest 构建总结",
        "",
        "本步骤基于 batch_02 已验证的 true preoperative CASIA2 2DAnalysis measurement 表，"
        "构建 preop measurement-only POD1 vault regression manifest。当前不训练模型，也不与 batch_01 合并。",
        "",
        "This is a true preoperative measurement-only manifest. Postoperative 2DAnalysis measurements must not be used as input features.",
        "",
        "## 输出规模",
        "",
        f"- ready manifest rows: {len(ready_df)}",
        f"- strict manifest rows: {len(strict_df)}",
        "",
        "## ready manifest",
        "",
        f"- measurement_ready_status distribution: {ready_df['measurement_ready_status'].value_counts(dropna=False).to_dict() if not ready_df.empty else {}}",
        f"- label_qc_flag distribution: {ready_df['label_qc_flag'].value_counts(dropna=False).to_dict() if not ready_df.empty else {}}",
        f"- validation: {ready_validation}",
        "",
        "## strict manifest",
        "",
        f"- measurement_ready_status distribution: {strict_df['measurement_ready_status'].value_counts(dropna=False).to_dict() if not strict_df.empty else {}}",
        f"- label_qc_flag distribution: {strict_df['label_qc_flag'].value_counts(dropna=False).to_dict() if not strict_df.empty else {}}",
        f"- validation: {strict_validation}",
        "",
        "## 说明",
        "",
        "- ready 保留 measurement_ready 与 measurement_ready_with_confirmed_outlier，并要求 POD1 label valid/verified。",
        "- strict 仅保留 measurement_ready 且 label_qc_flag == ok。",
        "- single_valid_preop_measurement_record 样本允许进入 ready，并在 notes 中标记。",
        "- confirmed outlier 样本允许进入 ready，但不进入 strict。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def main() -> None:
    args = parse_args()
    checked_in = resolve_project_path(args.checked_in)
    validation_report_in = resolve_project_path(args.validation_report_in)
    ready_as_oct_manifest_in = resolve_project_path(args.ready_as_oct_manifest_in)
    strict_as_oct_manifest_in = resolve_project_path(args.strict_as_oct_manifest_in)
    labels_in = resolve_project_path(args.labels_in)
    ready_out = resolve_project_path(args.ready_out)
    strict_out = resolve_project_path(args.strict_out)
    summary_md_out = resolve_project_path(args.summary_md_out)

    checked_df = prepare_checked(pd.read_csv(checked_in))
    validation_df = pd.read_csv(validation_report_in)
    ready_as_oct = pd.read_csv(ready_as_oct_manifest_in)
    strict_as_oct = pd.read_csv(strict_as_oct_manifest_in)
    labels_df = pd.read_csv(labels_in)

    aggregated = aggregate_measurements(checked_df)
    all_ready_candidates = build_manifest(aggregated, validation_df, ready_as_oct, labels_df)

    ready_df = all_ready_candidates[
        all_ready_candidates["measurement_ready_status"].isin(READY_STATUSES)
        & all_ready_candidates["label_status"].astype(str).str.lower().eq("valid")
        & all_ready_candidates["verify_status"].astype(str).str.lower().eq("verified")
    ].copy()
    strict_base = build_manifest(aggregated, validation_df, strict_as_oct, labels_df)
    strict_df = strict_base[
        strict_base["measurement_ready_status"].eq("measurement_ready")
        & strict_base["label_qc_flag"].eq("ok")
        & strict_base["label_status"].astype(str).str.lower().eq("valid")
        & strict_base["verify_status"].astype(str).str.lower().eq("verified")
    ].copy()

    write_csv(ready_df, ready_out)
    write_csv(strict_df, strict_out)

    ready_validation = validate_output(ready_df, checked_df)
    strict_validation = validate_output(strict_df, checked_df)
    write_summary_md(summary_md_out, ready_df, strict_df, ready_validation, strict_validation)

    ready_status_dist = ready_df["measurement_ready_status"].value_counts(dropna=False).to_dict() if not ready_df.empty else {}
    ready_qc_dist = ready_df["label_qc_flag"].value_counts(dropna=False).to_dict() if not ready_df.empty else {}
    strict_status_dist = strict_df["measurement_ready_status"].value_counts(dropna=False).to_dict() if not strict_df.empty else {}
    strict_qc_dist = strict_df["label_qc_flag"].value_counts(dropna=False).to_dict() if not strict_df.empty else {}
    single_ready_count = int(ready_df["notes"].str.contains("single_valid_preop_measurement_record", na=False).sum()) if not ready_df.empty else 0
    patient_068_excluded = "patient_068_OD_20250624" not in set(ready_df["sample_id"]) and "patient_068_OD_20250624" not in set(strict_df["sample_id"])
    leakage = bool(ready_validation["leakage_risk"] or strict_validation["leakage_risk"])

    print(f"Ready manifest rows: {len(ready_df)}")
    print(f"Strict manifest rows: {len(strict_df)}")
    print(f"Ready measurement_ready_status distribution: {ready_status_dist}")
    print(f"Strict measurement_ready_status distribution: {strict_status_dist}")
    print(f"Ready label_qc_flag distribution: {ready_qc_dist}")
    print(f"Strict label_qc_flag distribution: {strict_qc_dist}")
    print(f"single_valid_preop_measurement_record sample count: {single_ready_count}")
    print(f"patient_068_OD_20250624 excluded: {patient_068_excluded}")
    print(f"Leakage risk present: {leakage}")
    print(f"Outputs: {format_paths([ready_out, strict_out, summary_md_out])}")


if __name__ == "__main__":
    main()

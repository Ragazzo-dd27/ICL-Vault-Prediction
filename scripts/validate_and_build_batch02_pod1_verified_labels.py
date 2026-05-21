"""Validate batch_02 POD1 manual review and build verified eye-level labels.

batch_02 is processed independently before merging with batch_01. POD1
2DAnalysis measurements are used only as label source, not as preoperative
input features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SORT_COLUMNS = ["patient_uid", "eye", "sample_id", "exam_date", "exam_time", "analysis_index", "image_path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate batch_02 POD1 checked table and build verified labels.")
    parser.add_argument(
        "--checked_in",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_02_pod1_manual_review_checked_v2.csv",
        help="Input batch_02 POD1 manual review checked_v2 CSV.",
    )
    parser.add_argument(
        "--labels_in",
        type=str,
        default="data/manifests/vault_label_candidates_batch_02.csv",
        help="Input batch_02 initial vault label candidate CSV.",
    )
    parser.add_argument(
        "--overview_in",
        type=str,
        default="data/interim/batch_02_eye_level_sample_overview.csv",
        help="Input batch_02 eye-level sample overview CSV.",
    )
    parser.add_argument(
        "--validation_report_out",
        type=str,
        default="artifacts/reports/batch_02_data_curation/batch_02_pod1_manual_review_validation_report.csv",
        help="Output sample-level validation report CSV.",
    )
    parser.add_argument(
        "--problem_records_out",
        type=str,
        default="artifacts/reports/batch_02_data_curation/batch_02_pod1_manual_review_problem_records.csv",
        help="Output problematic record-level CSV.",
    )
    parser.add_argument(
        "--verified_labels_out",
        type=str,
        default="data/manifests/vault_label_candidates_batch_02_pod1_verified.csv",
        help="Output eye-level verified POD1 vault label CSV.",
    )
    parser.add_argument(
        "--summary_md_out",
        type=str,
        default="artifacts/reports/batch_02_data_curation/batch_02_pod1_verified_label_summary.md",
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


def normalize_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sortable = df.copy()
    sortable["_analysis_index_sort"] = normalize_numeric_series(
        sortable.get("analysis_index", pd.Series(dtype=object))
    )
    by: list[str] = []
    for column in SORT_COLUMNS:
        if column in sortable.columns:
            by.append("_analysis_index_sort" if column == "analysis_index" else column)
    return sortable.sort_values(by=by, kind="stable", na_position="last").drop(
        columns=["_analysis_index_sort"], errors="ignore"
    )


def join_unique(values: Iterable[object]) -> str:
    items = sorted({str(value).strip() for value in values if str(value).strip() and str(value).lower() != "nan"})
    return ";".join(items)


def prepare_checked(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["postop_day"] = normalize_numeric_series(out["postop_day"])
    out["vault_um"] = normalize_numeric_series(out["vault_um"])
    out["has_vault"] = normalize_bool_series(out["has_vault"])
    out["verify_status"] = out["verify_status"].fillna("").astype(str).str.strip().str.lower()
    out["extraction_method"] = out["extraction_method"].fillna("").astype(str).str.strip().str.lower()
    out["notes"] = out.get("notes", pd.Series("", index=out.index)).fillna("").astype(str)
    out["record_is_pod1"] = out["postop_day"] == 1
    out["record_is_verified"] = out["verify_status"].eq("verified")
    out["record_is_manual_verified"] = out["extraction_method"].eq("manual_verified")
    out["vault_is_positive"] = out["vault_um"].notna() & (out["vault_um"] > 0)
    out["valid_vault_record"] = (
        out["record_is_pod1"]
        & out["record_is_verified"]
        & out["record_is_manual_verified"]
        & out["has_vault"]
        & out["vault_is_positive"]
    )
    return out


def collect_record_issues(row: pd.Series) -> List[str]:
    issues: list[str] = []
    if float(row.get("postop_day", np.nan)) != 1:
        issues.append("not_pod1")
    if str(row.get("verify_status", "")) in {"pending", "uncertain"}:
        issues.append("manual_review_needed")
    elif str(row.get("verify_status", "")) != "verified":
        issues.append("invalid_verify_status")
    if str(row.get("extraction_method", "")) != "manual_verified":
        issues.append("invalid_extraction_method")
    vault_um = row.get("vault_um", np.nan)
    has_vault = bool(row.get("has_vault", False))
    if has_vault and (pd.isna(vault_um) or float(vault_um) <= 0):
        issues.append("has_vault_without_positive_vault")
    if (not has_vault) and pd.notna(vault_um):
        issues.append("has_vault_false_but_vault_present")
    if pd.notna(vault_um) and float(vault_um) <= 0:
        issues.append("fatal_nonpositive_vault")
    return issues


def build_problem_records(checked_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in checked_df.iterrows():
        issues = collect_record_issues(row)
        if not issues:
            continue
        record = row.to_dict()
        record["problem_flags"] = ";".join(issues)
        record["problem_severity"] = "fatal" if "fatal_nonpositive_vault" in issues else "warning"
        rows.append(record)
    if not rows:
        return pd.DataFrame(columns=[*checked_df.columns, "problem_flags", "problem_severity"])
    return sort_rows(pd.DataFrame(rows))


def qc_flag_for(num_valid: int, vault_range: float | None, has_review_needed: bool) -> str:
    if has_review_needed:
        return "manual_review_needed"
    if num_valid == 0:
        return "manual_review_needed"
    if num_valid == 1:
        return "single_valid_pod1_vault"
    if vault_range is not None and vault_range > 100:
        return "large_between_scan_difference"
    return "ok"


def build_outputs(
    checked_df: pd.DataFrame,
    overview_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overview_ids = overview_df[["sample_id", "patient_uid", "eye"]].drop_duplicates() if not overview_df.empty else pd.DataFrame()
    sample_keys = checked_df[["sample_id", "patient_uid", "eye"]].drop_duplicates()
    all_samples = pd.concat([overview_ids, sample_keys], ignore_index=True).drop_duplicates(subset=["sample_id"])

    validation_rows: List[Dict[str, object]] = []
    label_rows: List[Dict[str, object]] = []
    grouped = {sample_id: group.copy() for sample_id, group in checked_df.groupby("sample_id", dropna=False)}

    for _, sample in all_samples.sort_values(["patient_uid", "eye", "sample_id"], kind="stable").iterrows():
        sample_id = str(sample["sample_id"])
        patient_uid = str(sample["patient_uid"])
        eye = str(sample["eye"])
        group = grouped.get(sample_id, pd.DataFrame(columns=checked_df.columns))
        pod1_group = group[group["record_is_pod1"]] if not group.empty else group
        valid = pod1_group[pod1_group["valid_vault_record"]] if not pod1_group.empty else pod1_group
        vault_values = valid["vault_um"].dropna().astype(float).to_numpy()
        num_records = int(len(pod1_group))
        num_valid = int(len(vault_values))
        has_uncertain_pending = bool(pod1_group["verify_status"].isin(["uncertain", "pending"]).any()) if not pod1_group.empty else False
        has_nonpositive = bool(
            ((pod1_group["vault_um"].notna()) & (pod1_group["vault_um"] <= 0)).any()
        ) if not pod1_group.empty else False
        has_has_vault_mismatch = bool(
            (pod1_group["has_vault"] & (~pod1_group["vault_is_positive"])).any()
        ) if not pod1_group.empty else False

        vault_mean = float(np.mean(vault_values)) if num_valid else np.nan
        vault_median = float(np.median(vault_values)) if num_valid else np.nan
        vault_min = float(np.min(vault_values)) if num_valid else np.nan
        vault_max = float(np.max(vault_values)) if num_valid else np.nan
        vault_range = float(vault_max - vault_min) if num_valid else np.nan
        qc_flag = qc_flag_for(num_valid, vault_range if num_valid else None, has_uncertain_pending)

        if num_valid >= 1 and not has_uncertain_pending and not has_nonpositive and not has_has_vault_mismatch:
            label_status = "valid"
            verify_status = "verified"
        elif has_uncertain_pending:
            label_status = "manual_review_needed"
            verify_status = "needs_review"
        else:
            label_status = "invalid"
            verify_status = "needs_review"

        problem_flags: list[str] = []
        if has_uncertain_pending:
            problem_flags.append("manual_review_needed")
        if has_nonpositive:
            problem_flags.append("fatal_nonpositive_vault")
        if has_has_vault_mismatch:
            problem_flags.append("has_vault_without_positive_vault")
        if num_valid == 0:
            problem_flags.append("no_valid_positive_vault")
        if qc_flag == "large_between_scan_difference":
            problem_flags.append("large_between_scan_difference")
        if qc_flag == "single_valid_pod1_vault":
            problem_flags.append("single_valid_pod1_vault")

        source_paths = join_unique(valid["image_path"]) if not valid.empty else join_unique(pod1_group.get("image_path", []))
        crop_paths = join_unique(valid["measurement_crop_path"]) if not valid.empty else join_unique(
            pod1_group.get("measurement_crop_path", [])
        )
        notes = [
            "batch_02 POD1 verified label construction",
            "POD1 2DAnalysis measurements are used only as label source, not as preoperative input features",
        ]
        if problem_flags:
            notes.append("flags: " + ";".join(problem_flags))

        validation_rows.append(
            {
                "sample_id": sample_id,
                "patient_uid": patient_uid,
                "eye": eye,
                "num_pod1_records": num_records,
                "num_valid_vault_records": num_valid,
                "has_uncertain_or_pending": has_uncertain_pending,
                "has_nonpositive_vault": has_nonpositive,
                "has_vault_mismatch": has_has_vault_mismatch,
                "pod1_vault_range_um": vault_range if num_valid else "",
                "label_qc_flag": qc_flag,
                "label_status": label_status,
                "verify_status": verify_status,
                "problem_flags": ";".join(problem_flags),
            }
        )
        label_rows.append(
            {
                "sample_id": sample_id,
                "patient_uid": patient_uid,
                "eye": eye,
                "pod1_vault_mean_um": vault_mean if num_valid else "",
                "pod1_vault_median_um": vault_median if num_valid else "",
                "pod1_vault_min_um": vault_min if num_valid else "",
                "pod1_vault_max_um": vault_max if num_valid else "",
                "pod1_vault_range_um": vault_range if num_valid else "",
                "num_pod1_records": num_records,
                "num_valid_vault_records": num_valid,
                "label_qc_flag": qc_flag,
                "label_status": label_status,
                "verify_status": verify_status,
                "source_image_paths": source_paths,
                "measurement_crop_paths": crop_paths,
                "notes": " | ".join(notes),
            }
        )

    return pd.DataFrame(validation_rows), pd.DataFrame(label_rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_summary_md(
    path: Path,
    checked_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    problem_df: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    qc_dist = labels_df["label_qc_flag"].value_counts(dropna=False).to_dict() if not labels_df.empty else {}
    status_dist = labels_df["label_status"].value_counts(dropna=False).to_dict() if not labels_df.empty else {}
    large_samples = labels_df.loc[
        labels_df["label_qc_flag"].eq("large_between_scan_difference"), "sample_id"
    ].tolist()
    single_samples = labels_df.loc[labels_df["label_qc_flag"].eq("single_valid_pod1_vault"), "sample_id"].tolist()
    needs_review = labels_df[labels_df["verify_status"].ne("verified")]

    lines = [
        "# Batch 02 POD1 verified label 构建总结",
        "",
        "本步骤只验证 batch_02 POD1 manual review checked_v2 表并构建 eye-level verified vault label table；"
        "不训练模型，也不与 batch_01 合并。",
        "",
        "POD1 2DAnalysis measurements are used only as label source, not as preoperative input features.",
        "",
        "## 输入检查",
        "",
        f"- manual review rows: {len(checked_df)}",
        f"- sample count: {labels_df['sample_id'].nunique() if not labels_df.empty else 0}",
        f"- verified valid label samples: {int(labels_df['label_status'].eq('valid').sum()) if not labels_df.empty else 0}",
        f"- needs_review / invalid samples: {len(needs_review)}",
        f"- problem records: {len(problem_df)}",
        "",
        "## QC flag 分布",
        "",
        str(qc_dist),
        "",
        "## Label status 分布",
        "",
        str(status_dist),
        "",
        "## 需要注意的样本",
        "",
        f"- range > 100 samples: {', '.join(large_samples) if large_samples else 'none'}",
        f"- single valid POD1 vault samples: {', '.join(single_samples) if single_samples else 'none'}",
        "",
        "## 输出",
        "",
        "- artifacts/reports/batch_02_data_curation/batch_02_pod1_manual_review_validation_report.csv",
        "- artifacts/reports/batch_02_data_curation/batch_02_pod1_manual_review_problem_records.csv",
        "- data/manifests/vault_label_candidates_batch_02_pod1_verified.csv",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def main() -> None:
    args = parse_args()
    checked_in = resolve_project_path(args.checked_in)
    labels_in = resolve_project_path(args.labels_in)
    overview_in = resolve_project_path(args.overview_in)
    validation_report_out = resolve_project_path(args.validation_report_out)
    problem_records_out = resolve_project_path(args.problem_records_out)
    verified_labels_out = resolve_project_path(args.verified_labels_out)
    summary_md_out = resolve_project_path(args.summary_md_out)

    checked_df = prepare_checked(pd.read_csv(checked_in))
    _initial_labels_df = pd.read_csv(labels_in)
    overview_df = pd.read_csv(overview_in)

    checked_pod1 = checked_df[checked_df["record_is_pod1"]].copy()
    problem_df = build_problem_records(checked_pod1)
    validation_df, verified_df = build_outputs(checked_pod1, overview_df)

    write_csv(validation_df, validation_report_out)
    write_csv(problem_df, problem_records_out)
    write_csv(verified_df, verified_labels_out)
    write_summary_md(summary_md_out, checked_pod1, validation_df, verified_df, problem_df)

    valid_count = int(verified_df["label_status"].eq("valid").sum()) if not verified_df.empty else 0
    needs_review_count = int(verified_df["verify_status"].ne("verified").sum()) if not verified_df.empty else 0
    qc_dist = verified_df["label_qc_flag"].value_counts(dropna=False).to_dict() if not verified_df.empty else {}
    large_samples = verified_df.loc[
        verified_df["label_qc_flag"].eq("large_between_scan_difference"), "sample_id"
    ].tolist()
    single_samples = verified_df.loc[
        verified_df["label_qc_flag"].eq("single_valid_pod1_vault"), "sample_id"
    ].tolist()

    print(f"Manual review rows: {len(checked_pod1)}")
    print(f"Sample count: {verified_df['sample_id'].nunique() if not verified_df.empty else 0}")
    print(f"Verified valid label count: {valid_count}")
    print(f"Invalid / needs_review sample count: {needs_review_count}")
    print(f"QC flag distribution: {qc_dist}")
    print(f"Range >100 samples: {', '.join(large_samples) if large_samples else 'none'}")
    print(f"single_valid_pod1_vault samples: {', '.join(single_samples) if single_samples else 'none'}")
    print(
        "Outputs: "
        f"{format_paths([validation_report_out, problem_records_out, verified_labels_out, summary_md_out])}"
    )


if __name__ == "__main__":
    main()

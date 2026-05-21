"""Validate batch_02 true preoperative CASIA2 2DAnalysis measurements.

This table is for true preoperative 2DAnalysis measurements only.
Postoperative 2DAnalysis measurements must not be used as input features.

The script does not modify inputs, does not train models, and keeps batch_02
independent from batch_01.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMNS = ["cct_um", "acd_epi_mm", "acd_endo_mm", "clr_um", "ata_mm"]
SORT_COLUMNS = ["patient_uid", "eye", "sample_id", "exam_date", "exam_time", "analysis_index", "image_path"]
RANGES = {
    "cct_um": (300.0, 800.0),
    "acd_epi_mm": (1.0, 7.0),
    "acd_endo_mm": (1.0, 7.0),
    "ata_mm": (6.0, 15.0),
    "clr_um": (-6000.0, 3000.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate batch_02 preop measurement checked table.")
    parser.add_argument(
        "--checked_in",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_02_preop_manual_review_priority_ready_checked_v2.csv",
        help="Input batch_02 preop manual review priority ready checked_v2 CSV.",
    )
    parser.add_argument(
        "--ready_manifest_in",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_02_ready.csv",
        help="Input batch_02 AS-OCT-only ready manifest.",
    )
    parser.add_argument(
        "--strict_manifest_in",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_02_strict.csv",
        help="Input batch_02 AS-OCT-only strict manifest.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/reports/batch_02_data_curation/preop_measurements",
        help="Output directory for validation reports.",
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


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sortable = df.copy()
    sortable["_analysis_index_sort"] = pd.to_numeric(sortable.get("analysis_index"), errors="coerce")
    by: list[str] = []
    for column in SORT_COLUMNS:
        if column in sortable.columns:
            by.append("_analysis_index_sort" if column == "analysis_index" else column)
    return sortable.sort_values(by=by, kind="stable", na_position="last").drop(
        columns=["_analysis_index_sort"], errors="ignore"
    )


def prepare_checked(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["verify_status"] = normalize_text_series(out["verify_status"]).str.lower()
    out["extraction_method"] = normalize_text_series(out["extraction_method"]).str.lower()
    out["has_vault"] = normalize_bool_series(out["has_vault"])
    out["vault_raw_text"] = normalize_text_series(out.get("vault_raw_text", pd.Series("", index=out.index)))
    out["notes"] = normalize_text_series(out.get("notes", pd.Series("", index=out.index)))
    for column in KEY_COLUMNS + ["vault_um"]:
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
    out["leakage_risk"] = out["has_vault"] | out["vault_um"].notna()
    out["complete_key_measurements"] = out[KEY_COLUMNS].notna().all(axis=1)
    return out


def record_problem_flags(row: pd.Series) -> List[str]:
    flags: list[str] = []
    notes = str(row.get("notes", "")).lower()
    if bool(row.get("excluded_from_preop_baseline", False)):
        flags.append("excluded_from_preop_measurement_baseline")
    if str(row.get("verify_status", "")) != "verified":
        flags.append("verify_status_not_verified")
    if str(row.get("extraction_method", "")) != "manual_verified":
        flags.append("extraction_method_not_manual_verified")
    if bool(row.get("leakage_risk", False)):
        flags.append("leakage_risk")
    vault_raw_text = str(row.get("vault_raw_text", "")).strip()
    if vault_raw_text and vault_raw_text != "---" and vault_raw_text.lower() != "nan":
        flags.append("vault_raw_text_present")
    if bool(row.get("candidate_record", False)):
        missing = [column for column in KEY_COLUMNS if pd.isna(row.get(column))]
        if missing:
            flags.append("missing_key_measurement:" + ",".join(missing))
        for column, (low, high) in RANGES.items():
            value = row.get(column)
            if pd.isna(value):
                continue
            if float(value) < low or float(value) > high:
                if "outlier_confirmed_from_crop" in notes:
                    flags.append(f"confirmed_outlier:{column}")
                else:
                    flags.append(f"outlier_recheck_needed:{column}")
        if all(pd.notna(row.get(column)) for column in ("acd_epi_mm", "acd_endo_mm", "cct_um")):
            diff_mm = float(row["acd_epi_mm"]) - float(row["acd_endo_mm"])
            cct_mm = float(row["cct_um"]) / 1000.0
            if abs(diff_mm - cct_mm) > 0.1:
                if "acd_cct_difference_confirmed_from_crop" in notes:
                    flags.append("confirmed_acd_cct_difference")
                else:
                    flags.append("acd_cct_difference_recheck_needed")
    return flags


def build_problem_records(checked_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in checked_df.iterrows():
        flags = record_problem_flags(row)
        if not flags:
            continue
        record = row.to_dict()
        record["problem_flags"] = ";".join(flags)
        if any(flag == "leakage_risk" for flag in flags):
            severity = "fatal"
        elif any("recheck_needed" in flag or "missing_key_measurement" in flag for flag in flags):
            severity = "warning"
        else:
            severity = "info"
        record["problem_severity"] = severity
        rows.append(record)
    if not rows:
        return pd.DataFrame(columns=[*checked_df.columns, "problem_flags", "problem_severity"])
    return sort_rows(pd.DataFrame(rows))


def sample_problem_summary(group: pd.DataFrame) -> dict[str, object]:
    candidates = group[group["candidate_record"]].copy()
    complete = candidates[candidates["complete_key_measurements"]].copy()
    flags_by_record = [record_problem_flags(row) for _, row in candidates.iterrows()]
    flat_flags = [flag for flags in flags_by_record for flag in flags]
    has_confirmed_outlier = any(flag.startswith("confirmed_outlier") for flag in flat_flags)
    has_confirmed_acd_cct = any(flag == "confirmed_acd_cct_difference" for flag in flat_flags)
    has_unconfirmed_outlier = any(flag.startswith("outlier_recheck_needed") for flag in flat_flags)
    has_acd_warning = any(flag == "acd_cct_difference_recheck_needed" for flag in flat_flags)
    has_missing_key = bool(len(candidates) == 0 or (~candidates["complete_key_measurements"]).any())
    has_leakage = bool(group["leakage_risk"].any())
    return {
        "num_preop_records": int(len(group)),
        "num_candidate_records": int(len(candidates)),
        "num_complete_preop_records": int(len(complete)),
        "has_single_valid_preop_record": int(len(complete)) == 1,
        "has_missing_key_measurement": has_missing_key,
        "has_leakage_risk": has_leakage,
        "has_unconfirmed_outlier": has_unconfirmed_outlier,
        "has_acd_cct_difference_warning": has_acd_warning,
        "has_confirmed_outlier": has_confirmed_outlier,
        "has_confirmed_acd_cct_difference": has_confirmed_acd_cct,
        "problem_flags": ";".join(sorted(set(flat_flags))),
    }


def build_validation_report(ready_manifest: pd.DataFrame, checked_df: pd.DataFrame) -> pd.DataFrame:
    grouped = {sample_id: group.copy() for sample_id, group in checked_df.groupby("sample_id", dropna=False)}
    rows: list[dict[str, object]] = []
    for _, sample in ready_manifest.sort_values(["patient_uid", "eye", "sample_id"], kind="stable").iterrows():
        sample_id = str(sample["sample_id"])
        group = grouped.get(sample_id, pd.DataFrame(columns=checked_df.columns))
        if group.empty:
            summary = {
                "num_preop_records": 0,
                "num_candidate_records": 0,
                "num_complete_preop_records": 0,
                "has_single_valid_preop_record": False,
                "has_missing_key_measurement": True,
                "has_leakage_risk": False,
                "has_unconfirmed_outlier": False,
                "has_acd_cct_difference_warning": False,
                "has_confirmed_outlier": False,
                "has_confirmed_acd_cct_difference": False,
                "problem_flags": "missing_preop_measurement",
            }
            status = "missing_preop_measurement"
            notes = ["missing preop measurement records"]
        else:
            summary = sample_problem_summary(group)
            notes = [
                "batch_02 true preop measurement validation",
                "postoperative 2DAnalysis measurements must not be used as input features",
            ]
            if summary["has_leakage_risk"]:
                status = "leakage_risk"
                notes.append("leakage_risk")
            elif summary["num_complete_preop_records"] == 0:
                status = "manual_review_needed"
                notes.append("no complete candidate preop measurement record")
            elif summary["has_unconfirmed_outlier"] or summary["has_acd_cct_difference_warning"]:
                status = "manual_review_needed"
                notes.append("unconfirmed outlier or ACD-CCT difference requires review")
            elif summary["has_confirmed_outlier"] or summary["has_confirmed_acd_cct_difference"]:
                status = "measurement_ready_with_confirmed_outlier"
                notes.append("confirmed outlier or confirmed ACD-CCT difference present")
            else:
                status = "measurement_ready"
            if summary["has_single_valid_preop_record"]:
                notes.append("single_valid_preop_measurement_record")
            if summary["problem_flags"]:
                notes.append("flags: " + str(summary["problem_flags"]))

        rows.append(
            {
                "sample_id": sample_id,
                "patient_uid": sample["patient_uid"],
                "eye": sample["eye"],
                "num_preop_records": summary["num_preop_records"],
                "num_candidate_records": summary["num_candidate_records"],
                "num_complete_preop_records": summary["num_complete_preop_records"],
                "has_single_valid_preop_record": bool(summary["has_single_valid_preop_record"]),
                "has_missing_key_measurement": bool(summary["has_missing_key_measurement"]),
                "has_leakage_risk": bool(summary["has_leakage_risk"]),
                "has_unconfirmed_outlier": bool(summary["has_unconfirmed_outlier"]),
                "has_acd_cct_difference_warning": bool(summary["has_acd_cct_difference_warning"]),
                "measurement_ready_status": status,
                "notes": " | ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_summary_md(
    path: Path,
    ready_manifest: pd.DataFrame,
    strict_manifest: pd.DataFrame,
    checked_df: pd.DataFrame,
    report_df: pd.DataFrame,
    problem_df: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_dist = report_df["measurement_ready_status"].value_counts(dropna=False).to_dict()
    single_samples = report_df.loc[report_df["has_single_valid_preop_record"], "sample_id"].tolist()
    lines = [
        "# Batch 02 术前 measurement 验证总结",
        "",
        "本报告系统检查 batch_02 ready 80 samples 对应的 true preoperative CASIA2 2DAnalysis measurement 表，"
        "用于后续 preop measurement-only baseline 和 AS-OCT + measurement fusion baseline。",
        "",
        "This table is for true preoperative 2DAnalysis measurements only. "
        "Postoperative 2DAnalysis measurements must not be used as input features.",
        "",
        "## 输入规模",
        "",
        f"- ready manifest samples: {len(ready_manifest)}",
        f"- strict manifest samples: {len(strict_manifest)}",
        f"- checked records: {len(checked_df)}",
        f"- verified candidate records: {int(checked_df['candidate_record'].sum())}",
        f"- complete candidate records: {int((checked_df['candidate_record'] & checked_df['complete_key_measurements']).sum())}",
        "",
        "## sample-level 状态",
        "",
        f"- measurement_ready_status distribution: {status_dist}",
        f"- single_valid_preop_measurement_record samples: {len(single_samples)}",
        f"- problem records: {len(problem_df)}",
        "",
        "## 说明",
        "",
        "- measurement_ready 表示至少一条完整 verified/manual_verified 候选记录，且无泄漏、无未确认 outlier、无未确认 ACD-CCT 差异。",
        "- measurement_ready_with_confirmed_outlier 表示存在已人工确认的 outlier 或 ACD-CCT 差异，但无泄漏风险。",
        "- manual_review_needed 表示缺关键字段、存在未确认 outlier 或未确认 ACD-CCT 差异。",
        "- leakage_risk 表示术前候选记录中出现 vault 值或 has_vault=True，应禁止进入 baseline。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    checked_in = resolve_project_path(args.checked_in)
    ready_manifest_in = resolve_project_path(args.ready_manifest_in)
    strict_manifest_in = resolve_project_path(args.strict_manifest_in)

    report_out = output_dir / "batch_02_preop_measurement_validation_report.csv"
    problem_out = output_dir / "batch_02_preop_measurement_problem_records.csv"
    summary_out = output_dir / "batch_02_preop_measurement_validation_summary.md"

    checked_df = prepare_checked(pd.read_csv(checked_in))
    ready_manifest = pd.read_csv(ready_manifest_in)
    strict_manifest = pd.read_csv(strict_manifest_in)

    report_df = build_validation_report(ready_manifest, checked_df)
    problem_df = build_problem_records(checked_df)

    write_csv(report_df, report_out)
    write_csv(problem_df, problem_out)
    write_summary_md(summary_out, ready_manifest, strict_manifest, checked_df, report_df, problem_df)

    verified_candidates = int(checked_df["candidate_record"].sum())
    complete_candidates = int((checked_df["candidate_record"] & checked_df["complete_key_measurements"]).sum())
    status_counts = report_df["measurement_ready_status"].value_counts(dropna=False).to_dict()
    single_count = int(report_df["has_single_valid_preop_record"].sum())

    print(f"Ready manifest samples: {len(ready_manifest)}")
    print(f"Checked records: {len(checked_df)}")
    print(f"Verified candidate records: {verified_candidates}")
    print(f"Complete candidate records: {complete_candidates}")
    print(f"measurement_ready samples: {status_counts.get('measurement_ready', 0)}")
    print(
        "measurement_ready_with_confirmed_outlier samples: "
        f"{status_counts.get('measurement_ready_with_confirmed_outlier', 0)}"
    )
    print(f"manual_review_needed samples: {status_counts.get('manual_review_needed', 0)}")
    print(f"leakage_risk samples: {status_counts.get('leakage_risk', 0)}")
    print(f"single_valid_preop_measurement_record samples: {single_count}")
    print(f"Outputs: {format_paths([report_out, problem_out, summary_out])}")


if __name__ == "__main__":
    main()

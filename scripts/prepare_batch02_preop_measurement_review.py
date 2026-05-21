"""Prepare batch_02 true preoperative CASIA2 2DAnalysis review tables.

This table is for true preoperative 2DAnalysis measurements only.
Postoperative 2DAnalysis measurements must not be used as input features.

The script does not train models, does not run OCR, does not modify inputs,
and keeps batch_02 independent from batch_01.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SORT_COLUMNS = ["patient_uid", "eye", "exam_date", "exam_time", "analysis_index", "image_path"]
REVIEW_COLUMNS = [
    "sample_id",
    "patient_uid",
    "eye",
    "split",
    "exam_date",
    "exam_time",
    "analysis_index",
    "image_path",
    "measurement_crop_path",
    "cct_um",
    "acd_epi_mm",
    "acd_endo_mm",
    "vault_um",
    "clr_um",
    "ata_mm",
    "has_vault",
    "vault_raw_text",
    "extraction_method",
    "verify_status",
    "notes",
]
KEY_MEASUREMENT_COLUMNS = ["cct_um", "acd_epi_mm", "acd_endo_mm", "clr_um", "ata_mm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare batch_02 preop measurement manual-review tables.")
    parser.add_argument(
        "--measurements_in",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_02_initial.csv",
        help="Input batch_02 initial CASIA2 2DAnalysis measurement CSV.",
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
        "--full_manifest_in",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_02_full.csv",
        help="Input batch_02 AS-OCT-only full manifest.",
    )
    parser.add_argument(
        "--preop_out",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_02_preop_manual_review.csv",
        help="Output CSV with all batch_02 preop 2DAnalysis records.",
    )
    parser.add_argument(
        "--priority_ready_out",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_02_preop_manual_review_priority_ready.csv",
        help="Output CSV with ready-sample preop records.",
    )
    parser.add_argument(
        "--priority_strict_out",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_02_preop_manual_review_priority_strict.csv",
        help="Output CSV with strict-sample preop records.",
    )
    parser.add_argument(
        "--status_out",
        type=str,
        default="artifacts/reports/batch_02_data_curation/batch_02_preop_measurement_review_status.csv",
        help="Output ready-sample preop measurement status CSV.",
    )
    parser.add_argument(
        "--summary_md_out",
        type=str,
        default="artifacts/reports/batch_02_data_curation/batch_02_preop_measurement_review_summary.md",
        help="Output Chinese Markdown summary.",
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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def split_sample_key(df: pd.DataFrame) -> pd.DataFrame:
    return df[["sample_id", "patient_uid", "eye", "split", "label_qc_flag"]].drop_duplicates(
        subset=["sample_id"]
    )


def prepare_measurements(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_preop"] = normalize_bool_series(out["is_preop"])
    out["is_postop"] = normalize_bool_series(out["is_postop"])
    out["has_vault"] = normalize_bool_series(out["has_vault"])
    for column in KEY_MEASUREMENT_COLUMNS + ["vault_um"]:
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["vault_raw_text"] = out.get("vault_raw_text", pd.Series("", index=out.index)).fillna("").astype(str)
    out["extraction_method"] = out.get("extraction_method", pd.Series("", index=out.index)).fillna("").astype(str)
    out["verify_status"] = out.get("verify_status", pd.Series("", index=out.index)).fillna("").astype(str)
    out["notes"] = out.get("notes", pd.Series("", index=out.index)).fillna("").astype(str)
    return out


def attach_sample_info(preop_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    sample_info = split_sample_key(manifest_df)
    merged = preop_df.merge(sample_info, on=["patient_uid", "eye"], how="left")
    if "sample_id" not in merged.columns:
        merged["sample_id"] = ""
    if "split" not in merged.columns:
        merged["split"] = ""
    if "label_qc_flag" not in merged.columns:
        merged["label_qc_flag"] = ""
    merged["sample_id"] = merged["sample_id"].fillna("")
    merged["split"] = merged["split"].fillna("")
    return merged


def ensure_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    preop_vault_mask = out["has_vault"] | out["vault_um"].notna()
    if preop_vault_mask.any():
        out.loc[preop_vault_mask, "notes"] = out.loc[preop_vault_mask, "notes"].astype(str) + (
            " | warning_preop_vault_value_present"
        )
    for column in REVIEW_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return sort_rows(out[REVIEW_COLUMNS])


def build_priority(preop_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    keys = manifest_df[["patient_uid", "eye"]].drop_duplicates()
    priority = preop_df.merge(keys, on=["patient_uid", "eye"], how="inner")
    priority = attach_sample_info(priority.drop(columns=["sample_id", "split", "label_qc_flag"], errors="ignore"), manifest_df)
    return ensure_review_columns(priority)


def has_all_key_measurements(group: pd.DataFrame) -> dict[str, bool]:
    return {column: bool(group[column].notna().any()) for column in KEY_MEASUREMENT_COLUMNS}


def build_status(ready_manifest: pd.DataFrame, priority_ready: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict[str, object]] = []
    grouped = {sample_id: group.copy() for sample_id, group in priority_ready.groupby("sample_id", dropna=False)}

    for _, sample in ready_manifest.sort_values(["patient_uid", "eye", "sample_id"], kind="stable").iterrows():
        sample_id = str(sample["sample_id"])
        group = grouped.get(sample_id, pd.DataFrame(columns=priority_ready.columns))
        num_preop_records = int(len(group))
        has_flags = has_all_key_measurements(group) if num_preop_records else {
            column: False for column in KEY_MEASUREMENT_COLUMNS
        }
        has_any_vault = bool((group["has_vault"] | group["vault_um"].notna()).any()) if num_preop_records else False
        all_verified = bool(group["verify_status"].fillna("").astype(str).str.lower().eq("verified").all()) if num_preop_records else False

        notes: list[str] = [
            "batch_02 preop measurement review status",
            "postoperative 2DAnalysis measurements must not be used as input features",
        ]
        if num_preop_records == 0:
            status = "missing_preop_2danalysis"
            notes.append("missing preop 2DAnalysis records")
        elif all(has_flags.values()) and all_verified:
            status = "measurement_ready"
        else:
            status = "manual_review_needed"
            missing = [column for column, present in has_flags.items() if not present]
            if missing:
                notes.append("missing key measurements: " + ",".join(missing))
            if not all_verified:
                notes.append("verify_status is not verified for all preop records")
        if has_any_vault:
            notes.append("warning_preop_vault_value_present")

        rows.append(
            {
                "sample_id": sample_id,
                "patient_uid": sample["patient_uid"],
                "eye": sample["eye"],
                "split": sample.get("split", ""),
                "label_qc_flag": sample.get("label_qc_flag", ""),
                "num_preop_records": num_preop_records,
                "has_cct": has_flags["cct_um"],
                "has_acd_epi": has_flags["acd_epi_mm"],
                "has_acd_endo": has_flags["acd_endo_mm"],
                "has_clr": has_flags["clr_um"],
                "has_ata": has_flags["ata_mm"],
                "measurement_ready_status": status,
                "notes": " | ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def write_summary_md(
    path: Path,
    preop_df: pd.DataFrame,
    ready_manifest: pd.DataFrame,
    strict_manifest: pd.DataFrame,
    priority_ready: pd.DataFrame,
    priority_strict: pd.DataFrame,
    status_df: pd.DataFrame,
    found_preop_vault: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_dist = status_df["measurement_ready_status"].value_counts(dropna=False).to_dict() if not status_df.empty else {}
    lines = [
        "# Batch 02 术前 2DAnalysis measurement review 准备总结",
        "",
        "本步骤只准备 batch_02 真正术前 CASIA2 2DAnalysis measurement 的人工核对表，"
        "不训练模型，也不与 batch_01 合并。",
        "",
        "This table is for true preoperative 2DAnalysis measurements only. "
        "Postoperative 2DAnalysis measurements must not be used as input features.",
        "",
        "## 输出规模",
        "",
        f"- all preop 2DAnalysis records: {len(preop_df)}",
        f"- ready samples: {len(ready_manifest)}",
        f"- strict samples: {len(strict_manifest)}",
        f"- priority_ready preop records: {len(priority_ready)}",
        f"- priority_strict preop records: {len(priority_strict)}",
        "",
        "## 当前核对状态",
        "",
        f"- measurement_ready_status distribution: {status_dist}",
        f"- preop vault value found: {found_preop_vault}",
        "",
        "## 说明",
        "",
        "- `priority_ready` 是第一优先人工核对表，对应 batch_02 ready manifest 的 80 个样本。",
        "- `priority_strict` 对应 strict manifest 的 79 个样本。",
        "- 如果术前记录中出现 vault 值，脚本只做 warning 标记，不自动删除。",
        "- initial CSV 中的 measurement 字段为空时会原样保留，等待人工核对。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def main() -> None:
    args = parse_args()
    measurements_in = resolve_project_path(args.measurements_in)
    ready_manifest_in = resolve_project_path(args.ready_manifest_in)
    strict_manifest_in = resolve_project_path(args.strict_manifest_in)
    full_manifest_in = resolve_project_path(args.full_manifest_in)
    preop_out = resolve_project_path(args.preop_out)
    priority_ready_out = resolve_project_path(args.priority_ready_out)
    priority_strict_out = resolve_project_path(args.priority_strict_out)
    status_out = resolve_project_path(args.status_out)
    summary_md_out = resolve_project_path(args.summary_md_out)

    measurements_df = prepare_measurements(pd.read_csv(measurements_in))
    ready_manifest = pd.read_csv(ready_manifest_in)
    strict_manifest = pd.read_csv(strict_manifest_in)
    _full_manifest = pd.read_csv(full_manifest_in)

    preop_raw = measurements_df[measurements_df["is_preop"]].copy()
    preop_with_ready_info = attach_sample_info(preop_raw, ready_manifest)
    preop_review = ensure_review_columns(preop_with_ready_info)
    priority_ready = build_priority(preop_raw, ready_manifest)
    priority_strict = build_priority(preop_raw, strict_manifest)
    status_df = build_status(ready_manifest, priority_ready)

    found_preop_vault = bool((preop_raw["has_vault"] | preop_raw["vault_um"].notna()).any())

    write_csv(preop_review, preop_out)
    write_csv(priority_ready, priority_ready_out)
    write_csv(priority_strict, priority_strict_out)
    write_csv(status_df, status_out)
    write_summary_md(
        summary_md_out,
        preop_df=preop_review,
        ready_manifest=ready_manifest,
        strict_manifest=strict_manifest,
        priority_ready=priority_ready,
        priority_strict=priority_strict,
        status_df=status_df,
        found_preop_vault=found_preop_vault,
    )

    status_counts = status_df["measurement_ready_status"].value_counts(dropna=False).to_dict()
    print(f"batch_02 all preop records: {len(preop_review)}")
    print(f"Ready samples: {len(ready_manifest)}")
    print(f"Strict samples: {len(strict_manifest)}")
    print(f"Priority ready preop review records: {len(priority_ready)}")
    print(f"Priority strict preop review records: {len(priority_strict)}")
    print(f"measurement_ready samples: {status_counts.get('measurement_ready', 0)}")
    print(f"manual_review_needed samples: {status_counts.get('manual_review_needed', 0)}")
    print(f"missing_preop_2danalysis samples: {status_counts.get('missing_preop_2danalysis', 0)}")
    print(f"Found preop vault value: {found_preop_vault}")
    print(f"Outputs: {format_paths([preop_out, priority_ready_out, priority_strict_out, status_out, summary_md_out])}")


if __name__ == "__main__":
    main()

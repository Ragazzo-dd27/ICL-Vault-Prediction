"""Prepare batch_02 manual-review tables from initial CASIA2 2DAnalysis CSVs.

batch_02 is processed independently before merging with batch_01. This script
does not run OCR, does not train a model, and does not modify the initial CSVs.
Postoperative 2DAnalysis measurements are label-source/review material only;
they must not be used as preoperative input features.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SORT_COLUMNS = ["patient_uid", "eye", "exam_date", "exam_time", "analysis_index", "image_path"]
MEASUREMENT_FIELDS = ["cct_um", "acd_epi_mm", "acd_endo_mm", "vault_um", "clr_um", "ata_mm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare batch_02 POD1 manual-review tables.")
    parser.add_argument(
        "--measurements_in",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_02_initial.csv",
        help="Input batch_02 initial CASIA2 2DAnalysis measurement CSV.",
    )
    parser.add_argument(
        "--labels_in",
        type=str,
        default="data/manifests/vault_label_candidates_batch_02.csv",
        help="Input batch_02 vault label candidate CSV.",
    )
    parser.add_argument(
        "--summary_in",
        type=str,
        default="data/manifests/real_export_batch_02_summary.csv",
        help="Input batch_02 patient-level export summary CSV.",
    )
    parser.add_argument(
        "--manifest_in",
        type=str,
        default="data/manifests/real_export_batch_02_manifest_initial.csv",
        help="Input batch_02 initial export manifest CSV.",
    )
    parser.add_argument(
        "--pod1_out",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_02_pod1_manual_review.csv",
        help="Output CSV for POD1 vault manual review.",
    )
    parser.add_argument(
        "--overview_out",
        type=str,
        default="data/interim/batch_02_eye_level_sample_overview.csv",
        help="Output CSV for patient-eye sample overview.",
    )
    parser.add_argument(
        "--unclassified_out",
        type=str,
        default="data/interim/batch_02_unclassified_2danalysis_records.csv",
        help="Output CSV for unclassified 2DAnalysis records.",
    )
    parser.add_argument(
        "--summary_md_out",
        type=str,
        default="artifacts/reports/batch_02_data_curation/batch_02_initial_review_summary.md",
        help="Output Markdown summary for batch_02 initial review.",
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


def sort_measurement_rows(df: pd.DataFrame) -> pd.DataFrame:
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
    sorted_df = sortable.sort_values(by=by, kind="stable", na_position="last")
    return sorted_df.drop(columns=["_analysis_index_sort"], errors="ignore")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def sample_id_for(patient_uid: str, eye: str, preop_exam_date: str) -> str:
    if preop_exam_date:
        return f"{patient_uid}_{eye}_{preop_exam_date.replace('-', '')}"
    return f"{patient_uid}_{eye}"


def join_unique(values: Iterable[object]) -> str:
    items = sorted({str(value).strip() for value in values if str(value).strip() and str(value).lower() != "nan"})
    return ";".join(items)


def has_nonempty_path(values: Iterable[object]) -> bool:
    return bool(join_unique(values))


def maybe_contains_identifying_raw_subdir(*dfs: pd.DataFrame) -> bool:
    pattern = re.compile(r"[\u4e00-\u9fff]")
    for df in dfs:
        for column in ("source_patient_folder", "image_path", "measurement_crop_path"):
            if column not in df.columns:
                continue
            if df[column].fillna("").astype(str).map(lambda value: bool(pattern.search(value))).any():
                return True
    return False


def prepare_measurements(measurements_df: pd.DataFrame) -> pd.DataFrame:
    df = measurements_df.copy()
    df["is_preop"] = normalize_bool_series(df["is_preop"])
    df["is_postop"] = normalize_bool_series(df["is_postop"])
    df["postop_day"] = normalize_numeric_series(df["postop_day"])
    for field in MEASUREMENT_FIELDS:
        if field not in df.columns:
            df[field] = pd.NA
        df[field] = pd.to_numeric(df[field], errors="coerce")
    if "has_vault" not in df.columns:
        df["has_vault"] = False
    df["has_vault"] = normalize_bool_series(df["has_vault"])
    if "vault_raw_text" not in df.columns:
        df["vault_raw_text"] = ""
    if "extraction_method" not in df.columns:
        df["extraction_method"] = "manual_pending"
    if "verify_status" not in df.columns:
        df["verify_status"] = "pending"
    return df


def build_preop_lookup(measurements_df: pd.DataFrame) -> Dict[tuple[str, str], str]:
    lookup: Dict[tuple[str, str], str] = {}
    for (patient_uid, eye), group in measurements_df.groupby(["patient_uid", "eye"], dropna=False):
        dates = sorted(
            str(value)
            for value in group.loc[group["is_preop"], "exam_date"].dropna().unique()
            if str(value).strip()
        )
        lookup[(str(patient_uid), str(eye))] = dates[0] if dates else ""
    return lookup


def build_pod1_review_df(measurements_df: pd.DataFrame, preop_lookup: Dict[tuple[str, str], str]) -> pd.DataFrame:
    pod1_df = measurements_df[(measurements_df["is_postop"]) & (measurements_df["postop_day"] == 1)].copy()
    if pod1_df.empty:
        return pod1_df

    pod1_df["sample_id"] = [
        sample_id_for(str(row.patient_uid), str(row.eye), preop_lookup.get((str(row.patient_uid), str(row.eye)), ""))
        for row in pod1_df.itertuples(index=False)
    ]
    pod1_df["extraction_method"] = "manual_pending"
    pod1_df["verify_status"] = "pending"
    pod1_df["vault_um"] = pd.NA
    pod1_df["has_vault"] = False
    pod1_df["vault_raw_text"] = pod1_df["vault_raw_text"].fillna("").replace({"nan": ""})

    columns = [
        "sample_id",
        "patient_uid",
        "eye",
        "exam_date",
        "exam_time",
        "analysis_index",
        "image_path",
        "measurement_crop_path",
        "postop_day",
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
    for column in columns:
        if column not in pod1_df.columns:
            pod1_df[column] = ""
    return sort_measurement_rows(pod1_df[columns])


def build_manifest_lookup(manifest_df: pd.DataFrame) -> Dict[tuple[str, str], Dict[str, object]]:
    manifest = manifest_df.copy()
    if "has_oct_raw" in manifest.columns:
        manifest["has_oct_raw"] = normalize_bool_series(manifest["has_oct_raw"])
    if "has_ubm_horizontal" in manifest.columns:
        manifest["has_ubm_horizontal"] = normalize_bool_series(manifest["has_ubm_horizontal"])
    if "has_ubm_vertical" in manifest.columns:
        manifest["has_ubm_vertical"] = normalize_bool_series(manifest["has_ubm_vertical"])
    lookup: Dict[tuple[str, str], Dict[str, object]] = {}
    for (patient_uid, eye), group in manifest.groupby(["patient_uid", "eye"], dropna=False):
        key = (str(patient_uid), str(eye))
        preop_group = group.sort_values(["date", "time"], kind="stable", na_position="last")
        lookup[key] = {
            "has_oct_raw": bool(preop_group.get("has_oct_raw", pd.Series(dtype=bool)).any()),
            "has_standard_ubm_horizontal": bool(
                preop_group.get("has_ubm_horizontal", pd.Series(dtype=bool)).any()
            ),
            "has_standard_ubm_vertical": bool(preop_group.get("has_ubm_vertical", pd.Series(dtype=bool)).any()),
            "has_ubm_unknown": has_nonempty_path(preop_group.get("ubm_unknown_paths", pd.Series(dtype=object))),
        }
    return lookup


def build_patient_ubm_lookup(summary_df: pd.DataFrame) -> Dict[str, Dict[str, bool]]:
    summary = summary_df.copy()
    lookup: Dict[str, Dict[str, bool]] = {}
    for _, row in summary.iterrows():
        lookup[str(row["patient_uid"])] = {
            "has_standard_ubm_horizontal": int(row.get("num_ubm_horizontal_images", 0) or 0) > 0,
            "has_standard_ubm_vertical": int(row.get("num_ubm_vertical_images", 0) or 0) > 0,
            "has_ubm_unknown": int(row.get("num_ubm_unknown_images", 0) or 0) > 0,
        }
    return lookup


def build_overview(
    measurements_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    manifest_lookup = build_manifest_lookup(manifest_df)
    patient_ubm_lookup = build_patient_ubm_lookup(summary_df)
    grouped = measurements_df[measurements_df["eye"].isin(["OD", "OS"])].groupby(["patient_uid", "eye"], dropna=False)
    eye_counts_by_patient = (
        measurements_df[measurements_df["eye"].isin(["OD", "OS"])]
        .groupby("patient_uid")["eye"]
        .nunique()
        .to_dict()
    )

    for (patient_uid, eye), group in grouped:
        patient_uid = str(patient_uid)
        eye = str(eye)
        preop_dates = sorted(
            str(value)
            for value in group.loc[group["is_preop"], "exam_date"].dropna().unique()
            if str(value).strip()
        )
        preop_exam_date = preop_dates[0] if preop_dates else ""
        sample_id = sample_id_for(patient_uid, eye, preop_exam_date)
        postop_days = group.loc[group["is_postop"], "postop_day"].dropna()
        num_pod1 = int(((group["is_postop"]) & (group["postop_day"] == 1)).sum())
        has_pod1 = num_pod1 > 0
        manifest_info = manifest_lookup.get((patient_uid, eye), {})
        patient_ubm = patient_ubm_lookup.get(patient_uid, {})
        has_oct_raw = bool(manifest_info.get("has_oct_raw", False))
        has_h = bool(patient_ubm.get("has_standard_ubm_horizontal", manifest_info.get("has_standard_ubm_horizontal", False)))
        has_v = bool(patient_ubm.get("has_standard_ubm_vertical", manifest_info.get("has_standard_ubm_vertical", False)))
        has_unknown = bool(patient_ubm.get("has_ubm_unknown", manifest_info.get("has_ubm_unknown", False)))

        notes = [
            "batch_02 manual review overview only",
            "POD1 postoperative measurements are label-source material, not preop input features",
        ]
        status = "pod1_review_ready"
        if not has_pod1:
            status = "missing_pod1_candidate"
            notes.append("missing POD1 candidate")
        elif not has_oct_raw:
            status = "missing_preop_oct"
            notes.append("missing preop OCT raw")

        if int(eye_counts_by_patient.get(patient_uid, 0)) == 1:
            status = "one_eye_only" if status == "pod1_review_ready" else status
            notes.append("one_eye_only")
        if num_pod1 > 2:
            status = "special_pod1_more_than_two_records" if status == "pod1_review_ready" else status
            notes.append("special_pod1_more_than_two_records")
        if not has_h or not has_v:
            notes.append("missing standard UBM orientation")
        if has_unknown:
            notes.append("UBM unknown orientation exists")

        rows.append(
            {
                "sample_id": sample_id,
                "patient_uid": patient_uid,
                "eye": eye,
                "preop_exam_date": preop_exam_date,
                "num_preop_2d_analysis": int(group["is_preop"].sum()),
                "num_pod1_2d_analysis": num_pod1,
                "has_pod1_candidate": has_pod1,
                "num_all_postop_candidates": int(group["is_postop"].sum()),
                "min_postop_day": int(postop_days.min()) if not postop_days.empty else "",
                "max_postop_day": int(postop_days.max()) if not postop_days.empty else "",
                "has_oct_raw": has_oct_raw,
                "has_standard_ubm_horizontal": has_h,
                "has_standard_ubm_vertical": has_v,
                "has_ubm_unknown": has_unknown,
                "sample_status": status,
                "notes": " | ".join(notes),
            }
        )

    overview_df = pd.DataFrame(rows)
    if overview_df.empty:
        return overview_df
    return overview_df.sort_values(["patient_uid", "eye", "preop_exam_date", "sample_id"], kind="stable")


def build_unclassified_df(measurements_df: pd.DataFrame) -> pd.DataFrame:
    unclassified = measurements_df[
        (measurements_df["eye"].fillna("").astype(str).str.lower() == "unknown")
        | (measurements_df["exam_date"].fillna("").astype(str).str.strip() == "")
        | ((~measurements_df["is_preop"]) & (~measurements_df["is_postop"]))
    ].copy()
    return sort_measurement_rows(unclassified)


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def write_summary_md(
    path: Path,
    summary_df: pd.DataFrame,
    measurements_df: pd.DataFrame,
    pod1_df: pd.DataFrame,
    overview_df: pd.DataFrame,
    unclassified_df: pd.DataFrame,
    has_possible_identifying_subdir: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pod1_dist = overview_df["num_pod1_2d_analysis"].value_counts().sort_index().to_dict() if not overview_df.empty else {}
    status_dist = overview_df["sample_status"].value_counts().to_dict() if not overview_df.empty else {}
    missing_ubm = overview_df[
        (~overview_df["has_standard_ubm_horizontal"].astype(bool)) | (~overview_df["has_standard_ubm_vertical"].astype(bool))
    ] if not overview_df.empty else overview_df
    patient_088 = overview_df[overview_df["patient_uid"] == "patient_088"] if not overview_df.empty else overview_df
    patient_088_line = (
        "patient_088 当前未形成有效 POD1 label candidate。"
        if patient_088.empty or not bool(patient_088["has_pod1_candidate"].any())
        else "patient_088 当前存在 POD1 label candidate。"
    )

    lines = [
        "# Batch 02 初始人工核对准备总结",
        "",
        "batch_02 当前作为独立数据批次处理，尚未与 batch_01 合并。本步骤只生成人工核对表，不训练模型。",
        "",
        "## 数据规模",
        "",
        f"- patients: {summary_df['patient_uid'].nunique()}",
        f"- total images: {int(summary_df['num_all_images'].sum())}",
        f"- OCT raw images: {int(summary_df['num_oct_raw_images'].sum())}",
        f"- OCT 2DAnalysis images: {int(summary_df['num_oct_2d_analysis_images'].sum())}",
        f"- UBM horizontal images: {int(summary_df['num_ubm_horizontal_images'].sum())}",
        f"- UBM vertical images: {int(summary_df['num_ubm_vertical_images'].sum())}",
        f"- UBM unknown images: {int(summary_df['num_ubm_unknown_images'].sum())}",
        f"- initial 2DAnalysis records: {len(measurements_df)}",
        f"- preop records: {int(measurements_df['is_preop'].sum())}",
        f"- postop records: {int(measurements_df['is_postop'].sum())}",
        f"- POD1 manual review records: {len(pod1_df)}",
        f"- eye-level candidate samples: {len(overview_df)}",
        f"- unclassified 2DAnalysis records: {len(unclassified_df)}",
        "",
        "## POD1 manual review",
        "",
        "POD1 manual review 表只用于人工核对术后第 1 天 vault 标签。表中保留 CCT、ACD、CLR、ATA 字段，"
        "但这些是术后测量值，不能作为术前 measurement baseline 的输入特征。",
        "",
        f"- POD1 records per sample distribution: {pod1_dist}",
        f"- sample_status distribution: {status_dist}",
        f"- {patient_088_line}",
        "",
        "## UBM 与特殊情况",
        "",
        f"- standard UBM 缺失的 patient 数: {missing_ubm['patient_uid'].nunique() if not missing_ubm.empty else 0}",
        "- 如果 standard UBM 缺失但 UBM unknown 存在，后续需要人工确认方向。",
        "",
        "## 脱敏提醒",
        "",
        (
            "- some raw subdirectory names may still need de-identification check"
            if has_possible_identifying_subdir
            else "- 未在当前导出路径字段中检测到明显中文原始子目录名。"
        ),
        "",
        "## 输出文件",
        "",
        "- data/interim/casia2_2d_measurements_batch_02_pod1_manual_review.csv",
        "- data/interim/batch_02_eye_level_sample_overview.csv",
        "- data/interim/batch_02_unclassified_2danalysis_records.csv",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    measurements_in = resolve_project_path(args.measurements_in)
    labels_in = resolve_project_path(args.labels_in)
    summary_in = resolve_project_path(args.summary_in)
    manifest_in = resolve_project_path(args.manifest_in)
    pod1_out = resolve_project_path(args.pod1_out)
    overview_out = resolve_project_path(args.overview_out)
    unclassified_out = resolve_project_path(args.unclassified_out)
    summary_md_out = resolve_project_path(args.summary_md_out)

    measurements_df = prepare_measurements(pd.read_csv(measurements_in))
    labels_df = pd.read_csv(labels_in)
    summary_df = pd.read_csv(summary_in)
    manifest_df = pd.read_csv(manifest_in)

    preop_lookup = build_preop_lookup(measurements_df)
    pod1_df = build_pod1_review_df(measurements_df, preop_lookup)
    overview_df = build_overview(measurements_df, manifest_df, summary_df)
    unclassified_df = build_unclassified_df(measurements_df)

    write_csv(pod1_df, pod1_out)
    write_csv(overview_df, overview_out)
    write_csv(unclassified_df, unclassified_out)

    has_possible_identifying_subdir = maybe_contains_identifying_raw_subdir(
        measurements_df, labels_df, summary_df, manifest_df
    )
    write_summary_md(
        summary_md_out,
        summary_df=summary_df,
        measurements_df=measurements_df,
        pod1_df=pod1_df,
        overview_df=overview_df,
        unclassified_df=unclassified_df,
        has_possible_identifying_subdir=has_possible_identifying_subdir,
    )

    pod1_dist = overview_df["num_pod1_2d_analysis"].value_counts().sort_index().to_dict() if not overview_df.empty else {}
    patient_088 = overview_df[overview_df["patient_uid"] == "patient_088"] if not overview_df.empty else overview_df
    patient_088_missing = patient_088.empty or not bool(patient_088["has_pod1_candidate"].any())
    one_eye_patients = sorted(
        overview_df.loc[overview_df["notes"].str.contains("one_eye_only", na=False), "patient_uid"].unique().tolist()
    ) if not overview_df.empty else []
    missing_standard_unknown = overview_df[
        (
            (~overview_df["has_standard_ubm_horizontal"].astype(bool))
            | (~overview_df["has_standard_ubm_vertical"].astype(bool))
        )
        & (overview_df["has_ubm_unknown"].astype(bool))
    ] if not overview_df.empty else overview_df
    missing_standard_unknown_patients = sorted(missing_standard_unknown["patient_uid"].unique().tolist()) if not missing_standard_unknown.empty else []

    print(f"batch_02 patients: {summary_df['patient_uid'].nunique()}")
    print(f"Total images: {int(summary_df['num_all_images'].sum())}")
    print(f"OCT raw images: {int(summary_df['num_oct_raw_images'].sum())}")
    print(f"OCT 2DAnalysis images: {int(summary_df['num_oct_2d_analysis_images'].sum())}")
    print(f"UBM horizontal images: {int(summary_df['num_ubm_horizontal_images'].sum())}")
    print(f"UBM vertical images: {int(summary_df['num_ubm_vertical_images'].sum())}")
    print(f"UBM unknown images: {int(summary_df['num_ubm_unknown_images'].sum())}")
    print(f"Initial 2DAnalysis records: {len(measurements_df)}")
    print(f"Preop records: {int(measurements_df['is_preop'].sum())}")
    print(f"Postop records: {int(measurements_df['is_postop'].sum())}")
    print(f"POD1 manual review records: {len(pod1_df)}")
    print(f"Eye-level candidate samples: {len(overview_df)}")
    print(f"POD1 records per sample distribution: {pod1_dist}")
    print(f"patient_088 missing valid label candidate: {patient_088_missing}")
    print(f"One-eye-only patients: {', '.join(one_eye_patients) if one_eye_patients else 'none'}")
    print(
        "Standard UBM missing but UBM unknown exists patients: "
        f"{', '.join(missing_standard_unknown_patients) if missing_standard_unknown_patients else 'none'}"
    )
    print(f"Unclassified records: {len(unclassified_df)}")
    print(f"Outputs: {format_paths([pod1_out, overview_out, unclassified_out, summary_md_out])}")


if __name__ == "__main__":
    main()

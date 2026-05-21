"""Build batch_02 POD1 formal draft and AS-OCT-only manifests.

batch_02 is processed independently before merging with batch_01. POD1
2DAnalysis measurements are used only as label source, not as preoperative
input features. This script does not train models and does not modify inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path("data/raw/real_export_batch_02/patient")
LABEL_SOURCE = "POD1_CASIA2_2DAnalysis_manual_verified"
INPUT_STRATEGY = "as_oct_primary_image_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build batch_02 POD1 AS-OCT-only manifests.")
    parser.add_argument(
        "--labels_in",
        type=str,
        default="data/manifests/vault_label_candidates_batch_02_pod1_verified.csv",
        help="Input batch_02 eye-level verified POD1 label CSV.",
    )
    parser.add_argument(
        "--measurements_in",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_02_initial.csv",
        help="Input batch_02 initial CASIA2 2DAnalysis measurement CSV.",
    )
    parser.add_argument(
        "--manifest_in",
        type=str,
        default="data/manifests/real_export_batch_02_manifest_initial.csv",
        help="Input batch_02 real export initial manifest CSV.",
    )
    parser.add_argument(
        "--summary_in",
        type=str,
        default="data/manifests/real_export_batch_02_summary.csv",
        help="Input batch_02 real export summary CSV.",
    )
    parser.add_argument(
        "--formal_out",
        type=str,
        default="data/manifests/formal_vault_manifest_batch_02_pod1_draft.csv",
        help="Output formal POD1 manifest draft CSV.",
    )
    parser.add_argument(
        "--full_out",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_02_full.csv",
        help="Output AS-OCT-only full manifest CSV.",
    )
    parser.add_argument(
        "--ready_out",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_02_ready.csv",
        help="Output AS-OCT-only ready manifest CSV.",
    )
    parser.add_argument(
        "--strict_out",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_02_strict.csv",
        help="Output AS-OCT-only strict manifest CSV.",
    )
    parser.add_argument(
        "--summary_md_out",
        type=str,
        default="artifacts/reports/batch_02_data_curation/batch_02_as_oct_manifest_summary.md",
        help="Output Markdown summary.",
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def normalize_date(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def split_paths(value: object) -> List[str]:
    if pd.isna(value):
        return []
    paths = [item.strip() for item in str(value).split(";") if item.strip() and item.strip().lower() != "nan"]
    return sorted(dict.fromkeys(paths))


def to_raw_relative_path(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return path.as_posix()
    normalized = path.as_posix()
    if normalized.startswith("data/"):
        return normalized
    return (RAW_ROOT / normalized).as_posix()


def join_paths(paths: Iterable[str]) -> str:
    return ";".join(sorted(dict.fromkeys(path for path in paths if path)))


def path_exists(path_text: object) -> bool:
    if pd.isna(path_text):
        return False
    text = str(path_text).strip()
    if not text:
        return False
    return resolve_project_path(text).exists()


def eye_to_side(eye: object) -> str:
    token = str(eye).strip().upper()
    if token == "OD":
        return "R"
    if token == "OS":
        return "L"
    return ""


def prepare_measurements(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_preop"] = normalize_bool_series(out["is_preop"])
    out["is_postop"] = normalize_bool_series(out["is_postop"])
    out["exam_date"] = out["exam_date"].map(normalize_date)
    out["analysis_index_sort"] = pd.to_numeric(out.get("analysis_index"), errors="coerce")
    return out


def build_preop_lookup(measurements_df: pd.DataFrame) -> Dict[tuple[str, str], Dict[str, object]]:
    lookup: Dict[tuple[str, str], Dict[str, object]] = {}
    preop = measurements_df[measurements_df["is_preop"]].copy()
    for (patient_uid, eye), group in preop.groupby(["patient_uid", "eye"], dropna=False):
        group = group.sort_values(["exam_date", "exam_time", "analysis_index_sort", "image_path"], kind="stable")
        dates = [date for date in group["exam_date"].dropna().astype(str).unique() if date]
        preop_exam_date = sorted(dates)[0] if dates else ""
        lookup[(str(patient_uid), str(eye))] = {
            "preop_exam_date": preop_exam_date,
            "num_preop_2d_analysis": int(len(group)),
            "preop_2d_analysis_paths": join_paths(group["image_path"].fillna("").astype(str).tolist()),
        }
    return lookup


def prepare_manifest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_norm"] = out["date"].map(normalize_date)
    out["has_oct_raw"] = normalize_bool_series(out["has_oct_raw"])
    out["has_oct_2d_analysis"] = normalize_bool_series(out["has_oct_2d_analysis"])
    return out


def build_preop_oct_lookup(manifest_df: pd.DataFrame) -> Dict[tuple[str, str, str], Dict[str, object]]:
    lookup: Dict[tuple[str, str, str], Dict[str, object]] = {}
    for (patient_uid, eye, date_norm), group in manifest_df.groupby(["patient_uid", "eye", "date_norm"], dropna=False):
        if str(eye) not in {"OD", "OS"} or not str(date_norm):
            continue
        raw_paths: list[str] = []
        analysis_paths: list[str] = []
        for value in group.get("oct_raw_paths", pd.Series(dtype=object)):
            raw_paths.extend(to_raw_relative_path(path) for path in split_paths(value))
        for value in group.get("oct_2d_analysis_paths", pd.Series(dtype=object)):
            analysis_paths.extend(to_raw_relative_path(path) for path in split_paths(value))
        raw_paths = sorted(dict.fromkeys(raw_paths))
        analysis_paths = sorted(dict.fromkeys(analysis_paths))
        lookup[(str(patient_uid), str(eye), str(date_norm))] = {
            "preop_as_oct_raw_paths": join_paths(raw_paths),
            "preop_as_oct_2d_analysis_paths": join_paths(analysis_paths),
            "has_preop_as_oct_raw": bool(raw_paths),
            "has_preop_2d_analysis": bool(analysis_paths),
            "num_preop_as_oct_raw": len(raw_paths),
            "oct_path": raw_paths[0] if raw_paths else "",
        }
    return lookup


def build_formal_manifest(
    labels_df: pd.DataFrame,
    preop_lookup: Dict[tuple[str, str], Dict[str, object]],
    oct_lookup: Dict[tuple[str, str, str], Dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, label in labels_df.sort_values(["patient_uid", "eye", "sample_id"], kind="stable").iterrows():
        patient_uid = str(label["patient_uid"])
        eye = str(label["eye"])
        preop_info = preop_lookup.get((patient_uid, eye), {})
        preop_exam_date = str(preop_info.get("preop_exam_date", ""))
        oct_info = oct_lookup.get((patient_uid, eye, preop_exam_date), {})
        vault_label = pd.to_numeric(pd.Series([label.get("pod1_vault_mean_um")]), errors="coerce").iloc[0]
        oct_path = str(oct_info.get("oct_path", ""))
        qc_flag = str(label.get("label_qc_flag", ""))
        notes = [
            "batch_02 POD1 formal manifest draft, not a training manifest",
            "POD1 2DAnalysis measurements are label source only, not preoperative input features",
            "UBM is not used in AS-OCT-only baseline",
        ]
        if qc_flag == "large_between_scan_difference":
            notes.append("label_qc_large_difference")
        if qc_flag == "single_valid_pod1_vault":
            notes.append("single_valid_pod1_vault")
        if not oct_path:
            notes.append("missing_preop_as_oct")

        training_ready_status = (
            "image_label_ready"
            if oct_path and pd.notna(vault_label) and float(vault_label) > 0
            else "missing_preop_as_oct"
        )

        rows.append(
            {
                "sample_id": label["sample_id"],
                "patient_id": patient_uid,
                "patient_uid": patient_uid,
                "eye_side": eye_to_side(eye),
                "eye": eye,
                "split": "",
                "oct_path": oct_path,
                "oct_paths": str(oct_info.get("preop_as_oct_raw_paths", "")),
                "has_oct": bool(oct_path),
                "has_ubm": False,
                "ubm_path": "",
                "ubm_alignment_status": "not_used_in_as_oct_only_baseline",
                "vault_label": float(vault_label) if pd.notna(vault_label) else "",
                "pod1_vault_mean_um": label.get("pod1_vault_mean_um", ""),
                "pod1_vault_median_um": label.get("pod1_vault_median_um", ""),
                "pod1_vault_min_um": label.get("pod1_vault_min_um", ""),
                "pod1_vault_max_um": label.get("pod1_vault_max_um", ""),
                "pod1_vault_range_um": label.get("pod1_vault_range_um", ""),
                "num_pod1_records": label.get("num_pod1_records", ""),
                "num_valid_vault_records": label.get("num_valid_vault_records", ""),
                "label_qc_flag": qc_flag,
                "label_status": label.get("label_status", ""),
                "verify_status": label.get("verify_status", ""),
                "training_ready_status": training_ready_status,
                "input_strategy": INPUT_STRATEGY,
                "label_source": LABEL_SOURCE,
                "preop_exam_date": preop_exam_date,
                "num_preop_2d_analysis": int(preop_info.get("num_preop_2d_analysis", 0) or 0),
                "num_preop_as_oct_raw": int(oct_info.get("num_preop_as_oct_raw", 0) or 0),
                "preop_2d_analysis_paths": str(preop_info.get("preop_2d_analysis_paths", "")),
                "preop_as_oct_2d_analysis_paths": str(oct_info.get("preop_as_oct_2d_analysis_paths", "")),
                "source_image_paths": label.get("source_image_paths", ""),
                "measurement_crop_paths": label.get("measurement_crop_paths", ""),
                "notes": " | ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def as_oct_columns() -> List[str]:
    return [
        "sample_id",
        "patient_id",
        "patient_uid",
        "eye_side",
        "eye",
        "split",
        "oct_path",
        "oct_paths",
        "has_oct",
        "has_ubm",
        "ubm_path",
        "ubm_alignment_status",
        "vault_label",
        "pod1_vault_mean_um",
        "pod1_vault_median_um",
        "pod1_vault_min_um",
        "pod1_vault_max_um",
        "pod1_vault_range_um",
        "num_pod1_records",
        "num_valid_vault_records",
        "label_qc_flag",
        "label_status",
        "verify_status",
        "training_ready_status",
        "input_strategy",
        "label_source",
        "preop_exam_date",
        "num_preop_2d_analysis",
        "num_preop_as_oct_raw",
        "notes",
    ]


def validate_manifest(df: pd.DataFrame) -> Dict[str, object]:
    labels = pd.to_numeric(df["vault_label"], errors="coerce") if not df.empty else pd.Series(dtype=float)
    missing_oct = int(df["oct_path"].fillna("").astype(str).str.strip().eq("").sum()) if not df.empty else 0
    nonexistent_oct = int((~df["oct_path"].map(path_exists)).sum()) if not df.empty else 0
    invalid_labels = int((labels.isna() | (labels <= 0)).sum()) if not df.empty else 0
    duplicate_samples = int(df["sample_id"].duplicated().sum()) if not df.empty else 0
    return {
        "rows": len(df),
        "missing_preop_as_oct": missing_oct,
        "nonexistent_oct_path": nonexistent_oct,
        "invalid_vault_label": invalid_labels,
        "duplicate_sample_id": duplicate_samples,
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_summary_md(
    path: Path,
    formal_df: pd.DataFrame,
    full_df: pd.DataFrame,
    ready_df: pd.DataFrame,
    strict_df: pd.DataFrame,
    validation: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    qc_dist = full_df["label_qc_flag"].value_counts(dropna=False).to_dict() if not full_df.empty else {}
    large_samples = full_df.loc[full_df["label_qc_flag"].eq("large_between_scan_difference"), "sample_id"].tolist()
    single_samples = full_df.loc[full_df["label_qc_flag"].eq("single_valid_pod1_vault"), "sample_id"].tolist()
    lines = [
        "# Batch 02 POD1 AS-OCT-only manifest 构建总结",
        "",
        "batch_02 当前独立处理，尚未与 batch_01 合并。本步骤只构建 formal draft 和 AS-OCT-only manifest，不训练模型。",
        "",
        "POD1 2DAnalysis measurements are used only as label source, not as preoperative input features.",
        "",
        "## 输出规模",
        "",
        f"- formal manifest rows: {len(formal_df)}",
        f"- AS-OCT full rows: {len(full_df)}",
        f"- AS-OCT ready rows: {len(ready_df)}",
        f"- AS-OCT strict rows: {len(strict_df)}",
        "",
        "## 数据检查",
        "",
        f"- missing_preop_as_oct samples: {validation['missing_preop_as_oct']}",
        f"- nonexistent oct_path samples: {validation['nonexistent_oct_path']}",
        f"- invalid vault_label samples: {validation['invalid_vault_label']}",
        f"- duplicate sample_id count: {validation['duplicate_sample_id']}",
        "",
        "## Label QC",
        "",
        f"- label_qc_flag distribution in full: {qc_dist}",
        f"- single_valid_pod1_vault samples: {', '.join(single_samples) if single_samples else 'none'}",
        f"- large_between_scan_difference samples: {', '.join(large_samples) if large_samples else 'none'}",
        "",
        "## 说明",
        "",
        "- full manifest 保留所有 valid + verified label，包括 large_between_scan_difference。",
        "- ready manifest 保留 ok 与 single_valid_pod1_vault，排除 large_between_scan_difference 与 manual_review_needed。",
        "- strict manifest 只保留 ok。",
        "- split 暂时为空，后续合并 batch_01 + batch_02 后再统一生成 patient-level split。",
        "- UBM 在当前 AS-OCT-only manifest 中被显式禁用。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def main() -> None:
    args = parse_args()
    labels_in = resolve_project_path(args.labels_in)
    measurements_in = resolve_project_path(args.measurements_in)
    manifest_in = resolve_project_path(args.manifest_in)
    summary_in = resolve_project_path(args.summary_in)
    formal_out = resolve_project_path(args.formal_out)
    full_out = resolve_project_path(args.full_out)
    ready_out = resolve_project_path(args.ready_out)
    strict_out = resolve_project_path(args.strict_out)
    summary_md_out = resolve_project_path(args.summary_md_out)

    labels_df = pd.read_csv(labels_in)
    measurements_df = prepare_measurements(pd.read_csv(measurements_in))
    manifest_df = prepare_manifest(pd.read_csv(manifest_in))
    _summary_df = pd.read_csv(summary_in)

    preop_lookup = build_preop_lookup(measurements_df)
    oct_lookup = build_preop_oct_lookup(manifest_df)
    formal_df = build_formal_manifest(labels_df, preop_lookup, oct_lookup)

    valid_verified = formal_df[
        formal_df["label_status"].astype(str).str.lower().eq("valid")
        & formal_df["verify_status"].astype(str).str.lower().eq("verified")
    ].copy()
    full_df = valid_verified.copy()
    ready_df = full_df[full_df["label_qc_flag"].isin(["ok", "single_valid_pod1_vault"])].copy()
    strict_df = full_df[full_df["label_qc_flag"].eq("ok")].copy()

    write_csv(formal_df, formal_out)
    write_csv(full_df[as_oct_columns()], full_out)
    write_csv(ready_df[as_oct_columns()], ready_out)
    write_csv(strict_df[as_oct_columns()], strict_out)

    full_validation = validate_manifest(full_df)
    write_summary_md(summary_md_out, formal_df, full_df, ready_df, strict_df, full_validation)

    qc_dist = full_df["label_qc_flag"].value_counts(dropna=False).to_dict() if not full_df.empty else {}
    single_samples = full_df.loc[full_df["label_qc_flag"].eq("single_valid_pod1_vault"), "sample_id"].tolist()
    large_samples = full_df.loc[full_df["label_qc_flag"].eq("large_between_scan_difference"), "sample_id"].tolist()
    manual_review_excluded = "patient_063_OD_20250115" not in set(full_df["sample_id"])

    print(f"Formal manifest rows: {len(formal_df)}")
    print(f"Full manifest rows: {len(full_df)}")
    print(f"Ready manifest rows: {len(ready_df)}")
    print(f"Strict manifest rows: {len(strict_df)}")
    print(f"Missing preop AS-OCT samples: {full_validation['missing_preop_as_oct']}")
    print(f"Nonexistent oct_path samples: {full_validation['nonexistent_oct_path']}")
    print(f"Label QC flag distribution: {qc_dist}")
    print(f"single_valid_pod1_vault samples: {', '.join(single_samples) if single_samples else 'none'}")
    print(f"large_between_scan_difference samples: {', '.join(large_samples) if large_samples else 'none'}")
    print(f"manual_review_needed excluded: {manual_review_excluded}")
    print(f"patient_088 excluded: {'patient_088' not in set(full_df['patient_uid'])}")
    print(f"patient_060_OD_20250807 in ready: {'patient_060_OD_20250807' in set(ready_df['sample_id'])}")
    print(f"patient_060_OD_20250807 in strict: {'patient_060_OD_20250807' in set(strict_df['sample_id'])}")
    print(f"Outputs: {format_paths([formal_out, full_out, ready_out, strict_out, summary_md_out])}")


if __name__ == "__main__":
    main()

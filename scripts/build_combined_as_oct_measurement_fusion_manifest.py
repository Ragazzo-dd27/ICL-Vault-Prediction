"""Build combined AS-OCT + preop measurement fusion manifests.

This fusion manifest uses preoperative AS-OCT image and true preoperative
2DAnalysis measurements only. Postoperative 2DAnalysis measurements must not
be used as input features.

The script does not modify input manifests, does not modify training code, does
not train models, and only checks image path existence without reading images.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEAN_FEATURES = ["cct_mean_um", "acd_epi_mean_mm", "acd_endo_mean_mm", "clr_mean_um", "ata_mean_mm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined AS-OCT + measurement fusion manifests.")
    parser.add_argument(
        "--as_oct_in",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_combined_strict.csv",
        help="Input combined AS-OCT strict manifest.",
    )
    parser.add_argument(
        "--measurement_ready_in",
        type=str,
        default="data/manifests/vault_preop_measurement_only_pod1_manifest_combined_ready.csv",
        help="Input combined measurement ready manifest.",
    )
    parser.add_argument(
        "--measurement_strict_in",
        type=str,
        default="data/manifests/vault_preop_measurement_only_pod1_manifest_combined_strict.csv",
        help="Input combined measurement strict manifest.",
    )
    parser.add_argument(
        "--fusion_ready_out",
        type=str,
        default="data/manifests/vault_as_oct_plus_preop_measurement_pod1_manifest_combined_ready.csv",
        help="Output combined fusion ready manifest.",
    )
    parser.add_argument(
        "--fusion_strict_out",
        type=str,
        default="data/manifests/vault_as_oct_plus_preop_measurement_pod1_manifest_combined_strict.csv",
        help="Output combined fusion strict manifest.",
    )
    parser.add_argument(
        "--summary_md_out",
        type=str,
        default="artifacts/reports/combined_batch_01_02/fusion_manifest_summary.md",
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


def ensure_global_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "global_sample_id" not in out.columns:
        out["global_sample_id"] = out["batch_id"].astype(str) + "__" + out["sample_id"].astype(str)
    if "global_patient_uid" not in out.columns:
        out["global_patient_uid"] = out["batch_id"].astype(str) + "__" + out["patient_uid"].astype(str)
    return out


def path_exists(path_text: object) -> bool:
    if pd.isna(path_text):
        return False
    text = str(path_text).strip()
    if not text or text.lower() == "nan":
        return False
    return resolve_project_path(text).exists()


def validate_patient_split(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    return bool((df.groupby("global_patient_uid")["split"].nunique() <= 1).all())


def build_fusion(as_oct_df: pd.DataFrame, measurement_df: pd.DataFrame) -> pd.DataFrame:
    as_oct = ensure_global_ids(as_oct_df).copy()
    measurement = ensure_global_ids(measurement_df).copy()

    as_oct_cols = [
        "global_sample_id",
        "sample_id",
        "batch_id",
        "patient_id",
        "patient_uid",
        "global_patient_uid",
        "eye_side",
        "eye",
        "split",
        "oct_path",
        "has_oct",
        "has_ubm",
        "vault_label",
        "pod1_vault_mean_um",
        "pod1_vault_range_um",
        "label_qc_flag",
        "notes",
    ]
    measurement_cols = [
        "global_sample_id",
        "measurement_ready_status",
        "measurement_input_status",
        "cct_mean_um",
        "acd_epi_mean_mm",
        "acd_endo_mean_mm",
        "clr_mean_um",
        "ata_mean_mm",
        "cct_scan1_um",
        "cct_scan2_um",
        "acd_epi_scan1_mm",
        "acd_epi_scan2_mm",
        "acd_endo_scan1_mm",
        "acd_endo_scan2_mm",
        "clr_scan1_um",
        "clr_scan2_um",
        "ata_scan1_mm",
        "ata_scan2_mm",
        "num_preop_measurement_records",
        "num_complete_preop_measurement_records",
        "measurement_source_images",
        "measurement_crop_paths",
        "notes",
    ]
    for col in as_oct_cols:
        if col not in as_oct.columns:
            as_oct[col] = ""
    for col in measurement_cols:
        if col not in measurement.columns:
            measurement[col] = ""

    merged = as_oct[as_oct_cols].merge(
        measurement[measurement_cols],
        on="global_sample_id",
        how="inner",
        suffixes=("_as_oct", "_measurement"),
    )
    merged["patient_id"] = merged["global_patient_uid"]
    notes = []
    for _, row in merged.iterrows():
        parts = [
            "combined AS-OCT + true preop measurement fusion manifest",
            f"batch_id={row['batch_id']}",
            "AS-OCT image and preoperative 2DAnalysis measurements are inputs",
            "postoperative 2DAnalysis measurements are label source only",
        ]
        as_oct_note = str(row.get("notes_as_oct", "")).strip()
        meas_note = str(row.get("notes_measurement", "")).strip()
        if as_oct_note and as_oct_note.lower() != "nan":
            parts.append("as_oct_notes: " + as_oct_note)
        if meas_note and meas_note.lower() != "nan":
            parts.append("measurement_notes: " + meas_note)
        notes.append(" | ".join(parts))
    merged["notes"] = notes

    output_columns = [
        "global_sample_id",
        "sample_id",
        "batch_id",
        "patient_id",
        "patient_uid",
        "global_patient_uid",
        "eye_side",
        "eye",
        "split",
        "oct_path",
        "has_oct",
        "vault_label",
        "pod1_vault_mean_um",
        "pod1_vault_range_um",
        "label_qc_flag",
        "measurement_ready_status",
        "measurement_input_status",
        "cct_mean_um",
        "acd_epi_mean_mm",
        "acd_endo_mean_mm",
        "clr_mean_um",
        "ata_mean_mm",
        "cct_scan1_um",
        "cct_scan2_um",
        "acd_epi_scan1_mm",
        "acd_epi_scan2_mm",
        "acd_endo_scan1_mm",
        "acd_endo_scan2_mm",
        "clr_scan1_um",
        "clr_scan2_um",
        "ata_scan1_mm",
        "ata_scan2_mm",
        "num_preop_measurement_records",
        "num_complete_preop_measurement_records",
        "measurement_source_images",
        "measurement_crop_paths",
        "notes",
    ]
    for column in output_columns:
        if column not in merged.columns:
            merged[column] = ""
    return merged[output_columns].sort_values(["split", "batch_id", "global_patient_uid", "eye"], kind="stable")


def validate_fusion(df: pd.DataFrame) -> Dict[str, object]:
    labels = pd.to_numeric(df["vault_label"], errors="coerce")
    features_missing = df[MEAN_FEATURES].apply(pd.to_numeric, errors="coerce").isna().any(axis=1)
    has_oct = normalize_bool_series(df["has_oct"])
    return {
        "rows": len(df),
        "global_sample_id_duplicates": int(df["global_sample_id"].duplicated().sum()),
        "empty_split": int(df["split"].fillna("").astype(str).str.strip().eq("").sum()),
        "patient_cross_split": not validate_patient_split(df),
        "nonexistent_oct_path": int((~df["oct_path"].map(path_exists)).sum()),
        "invalid_vault_label": int((labels.isna() | (labels <= 0)).sum()),
        "missing_mean_measurement_features": int(features_missing.sum()),
        "has_oct_false": int((~has_oct).sum()),
        "has_ubm_used": False,
        "postop_measurement_input_risk": 0,
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def split_counts(df: pd.DataFrame) -> Dict[str, int]:
    return df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0).astype(int).to_dict()


def write_summary(
    path: Path,
    input_counts: Dict[str, int],
    ready_df: pd.DataFrame,
    strict_df: pd.DataFrame,
    validations: Dict[str, Dict[str, object]],
    ready_intersection_info: Dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Combined AS-OCT + preop measurement fusion manifest summary",
        "",
        "本步骤构建 batch_01 + batch_02 combined AS-OCT image + true preop measurement fusion manifest。"
        "不训练模型，不读取图像内容，只检查路径是否存在。",
        "",
        "This fusion manifest uses preoperative AS-OCT image and true preoperative 2DAnalysis measurements only. "
        "Postoperative 2DAnalysis measurements must not be used as input features.",
        "",
        "## 输入文件行数",
        "",
    ]
    for name, count in input_counts.items():
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            "",
            "## 输出行数",
            "",
            f"- fusion ready: {len(ready_df)}",
            f"- fusion strict: {len(strict_df)}",
            "",
            "## 交集信息",
            "",
        ]
    )
    for name, count in ready_intersection_info.items():
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            "",
            "## train/val/test 分布",
            "",
            f"- fusion ready: {split_counts(ready_df)}",
            f"- fusion strict: {split_counts(strict_df)}",
            "",
            "## batch 样本贡献",
            "",
            f"- fusion ready: {ready_df['batch_id'].value_counts(dropna=False).to_dict()}",
            f"- fusion strict: {strict_df['batch_id'].value_counts(dropna=False).to_dict()}",
            "",
            "## measurement_ready_status 分布",
            "",
            f"- fusion ready: {ready_df['measurement_ready_status'].value_counts(dropna=False).to_dict()}",
            f"- fusion strict: {strict_df['measurement_ready_status'].value_counts(dropna=False).to_dict()}",
            "",
            "## label_qc_flag 分布",
            "",
            f"- fusion ready: {ready_df['label_qc_flag'].value_counts(dropna=False).to_dict()}",
            f"- fusion strict: {strict_df['label_qc_flag'].value_counts(dropna=False).to_dict()}",
            "",
            "## 数据检查",
            "",
        ]
    )
    for name, validation in validations.items():
        lines.append(f"### {name}")
        for key, value in validation.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(
        [
            "## 说明与下一步",
            "",
            "- UBM 不作为当前 fusion 输入。",
            "- measurement_source_images 来自 true preop measurement manifest；术后 2DAnalysis 不作为输入特征。",
            "- 下一步建议运行 fusion dataset smoke test，确认 Dataset/DataLoader 能同时返回 resized OCT image 与 measurement feature tensor。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def main() -> None:
    args = parse_args()
    as_oct_in = resolve_project_path(args.as_oct_in)
    measurement_ready_in = resolve_project_path(args.measurement_ready_in)
    measurement_strict_in = resolve_project_path(args.measurement_strict_in)
    fusion_ready_out = resolve_project_path(args.fusion_ready_out)
    fusion_strict_out = resolve_project_path(args.fusion_strict_out)
    summary_md_out = resolve_project_path(args.summary_md_out)

    as_oct_df = ensure_global_ids(pd.read_csv(as_oct_in))
    measurement_ready_df = ensure_global_ids(pd.read_csv(measurement_ready_in))
    measurement_strict_df = ensure_global_ids(pd.read_csv(measurement_strict_in))

    ready_df = build_fusion(as_oct_df, measurement_ready_df)
    strict_df = build_fusion(as_oct_df, measurement_strict_df)

    ready_intersection_info = {
        "as_oct_strict_samples": len(as_oct_df),
        "measurement_ready_samples": len(measurement_ready_df),
        "measurement_strict_samples": len(measurement_strict_df),
        "ready_intersection": len(ready_df),
        "strict_intersection": len(strict_df),
        "measurement_ready_not_in_as_oct_strict": len(
            set(measurement_ready_df["global_sample_id"]) - set(as_oct_df["global_sample_id"])
        ),
        "as_oct_strict_not_in_measurement_ready": len(
            set(as_oct_df["global_sample_id"]) - set(measurement_ready_df["global_sample_id"])
        ),
    }
    validations = {
        "fusion_ready": validate_fusion(ready_df),
        "fusion_strict": validate_fusion(strict_df),
    }

    write_csv(ready_df, fusion_ready_out)
    write_csv(strict_df, fusion_strict_out)
    write_summary(
        summary_md_out,
        input_counts={
            "as_oct_strict": len(as_oct_df),
            "measurement_ready": len(measurement_ready_df),
            "measurement_strict": len(measurement_strict_df),
        },
        ready_df=ready_df,
        strict_df=strict_df,
        validations=validations,
        ready_intersection_info=ready_intersection_info,
    )

    print(f"Fusion ready rows: {len(ready_df)}")
    print(f"Fusion strict rows: {len(strict_df)}")
    print(f"Fusion ready split distribution: {split_counts(ready_df)}")
    print(f"Fusion strict split distribution: {split_counts(strict_df)}")
    print(f"Fusion ready batch contribution: {ready_df['batch_id'].value_counts(dropna=False).to_dict()}")
    print(f"Fusion strict batch contribution: {strict_df['batch_id'].value_counts(dropna=False).to_dict()}")
    print(f"Fusion ready measurement_ready_status: {ready_df['measurement_ready_status'].value_counts(dropna=False).to_dict()}")
    print(f"Fusion strict measurement_ready_status: {strict_df['measurement_ready_status'].value_counts(dropna=False).to_dict()}")
    print(f"Fusion ready label_qc_flag: {ready_df['label_qc_flag'].value_counts(dropna=False).to_dict()}")
    print(f"Fusion strict label_qc_flag: {strict_df['label_qc_flag'].value_counts(dropna=False).to_dict()}")
    print(f"Fusion ready validation: {validations['fusion_ready']}")
    print(f"Fusion strict validation: {validations['fusion_strict']}")
    print(f"Outputs: {format_paths([fusion_ready_out, fusion_strict_out, summary_md_out])}")


if __name__ == "__main__":
    main()

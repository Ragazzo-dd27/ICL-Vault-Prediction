"""Prepare a review package for top AS-OCT-only POD1 prediction errors.

The package copies selected preoperative AS-OCT inputs and POD1 measurement
crops into per-sample folders for manual review. It does not modify source
data, prediction files, manifests, or checkpoints.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    "artifacts/reports/as_oct_pod1_baseline_batch_01/error_analysis/top_error_review_package"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare top-error manual review package.")
    parser.add_argument(
        "--top_error",
        type=str,
        default="artifacts/reports/as_oct_pod1_baseline_batch_01/error_analysis/top_error_samples.csv",
    )
    parser.add_argument(
        "--error_summary",
        type=str,
        default="artifacts/reports/as_oct_pod1_baseline_batch_01/error_analysis/test_error_summary_by_sample.csv",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean_strict.csv",
    )
    parser.add_argument(
        "--pod1_checked",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_pod1_manual_review_checked.csv",
    )
    parser.add_argument(
        "--verified_labels",
        type=str,
        default="data/manifests/vault_label_candidates_batch_01_pod1_verified.csv",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_k", type=int, default=10)
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


def warn(message: str, warnings: List[str]) -> None:
    warnings.append(message)
    print(f"Warning: {message}")


def load_top_errors(top_error_path: Path, error_summary_path: Path, top_k: int) -> pd.DataFrame:
    if top_error_path.exists():
        df = pd.read_csv(top_error_path)
    elif error_summary_path.exists():
        df = pd.read_csv(error_summary_path).sort_values(
            "mean_abs_error_um",
            ascending=False,
            kind="stable",
        ).head(top_k)
    else:
        raise FileNotFoundError(f"Missing both top error files: {top_error_path}, {error_summary_path}")
    return df.sort_values("mean_abs_error_um", ascending=False, kind="stable").head(top_k).copy()


def split_paths(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def resolve_existing_path(value: object) -> Path | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    if path.is_absolute():
        if path.exists():
            return path
    else:
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            return candidate

    normalized = text.replace("\\", "/")
    alternates = []
    if "/real_export_batch_01/patients/" in normalized:
        alternates.append(normalized.replace("/real_export_batch_01/patients/", "/real_export_batch_01/patient/"))
    if "/real_export_batch_01/patient/" in normalized:
        alternates.append(normalized.replace("/real_export_batch_01/patient/", "/real_export_batch_01/patients/"))
    for alternate in alternates:
        alternate_path = Path(alternate)
        if alternate_path.is_absolute() and alternate_path.exists():
            return alternate_path
        candidate = PROJECT_ROOT / alternate_path
        if candidate.exists():
            return candidate
    return None


def copy_file(src: Path | None, dst: Path, warnings: List[str], description: str) -> bool:
    if src is None or not src.exists():
        warn(f"missing {description}: {src}", warnings)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def vault_range_group(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if numeric < 500:
        return "low vault (<500 um)"
    if numeric <= 800:
        return "medium vault (500-800 um)"
    return "high vault (>800 um)"


def write_readme(sample_dir: Path, sample_id: str) -> None:
    text = f"""# {sample_id} 人工复查提示

本目录用于复查 AS-OCT-only POD1 baseline 的高误差样本，不包含真实患者姓名。

建议复查顺序：

1. 检查 `preop_as_oct_input.jpg` 是否为正确的术前 AS-OCT raw 图像。
2. 检查 `pod1_measurement_crops/` 中 POD1 2DAnalysis measurement crop 的 vault 数值是否录入正确。
3. 检查 POD1 scan1 与 scan2 的 vault 是否差异较大。
4. 检查 AS-OCT 图像质量、方向、裁切和是否存在异常伪影。
5. 判断该样本是否需要在后续表格中标记为 `review_needed`。

请将人工复查结论记录到后续 review 表中，不要直接修改原始 manifest 或 prediction 文件。
"""
    (sample_dir / "README.md").write_text(text, encoding="utf-8")


def find_sample_manifest(manifest_df: pd.DataFrame, sample_id: str) -> Dict[str, object]:
    matches = manifest_df[manifest_df["sample_id"] == sample_id]
    if matches.empty:
        return {}
    return matches.iloc[0].to_dict()


def find_verified_label(verified_df: pd.DataFrame, sample_id: str) -> Dict[str, object]:
    matches = verified_df[verified_df["sample_id"] == sample_id]
    if matches.empty:
        return {}
    return matches.iloc[0].to_dict()


def find_checked_rows(checked_df: pd.DataFrame, patient_uid: str, eye: str) -> pd.DataFrame:
    postop_day = pd.to_numeric(checked_df["postop_day"], errors="coerce")
    return checked_df[
        (checked_df["patient_uid"].astype(str) == patient_uid)
        & (checked_df["eye"].astype(str) == eye)
        & (postop_day == 1)
    ].copy()


def prepare_sample_package(
    row: Dict[str, object],
    manifest_df: pd.DataFrame,
    checked_df: pd.DataFrame,
    verified_df: pd.DataFrame,
    output_dir: Path,
    warnings: List[str],
) -> Dict[str, object]:
    sample_id = str(row["sample_id"])
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    manifest_row = find_sample_manifest(manifest_df, sample_id)
    verified_row = find_verified_label(verified_df, sample_id)
    patient_uid = str(manifest_row.get("patient_uid", row.get("patient_id", "")))
    eye = str(manifest_row.get("eye", ""))
    if not eye:
        eye = "OD" if row.get("eye_side") == "R" else "OS" if row.get("eye_side") == "L" else ""

    copied: List[str] = []
    missing: List[str] = []
    oct_src = resolve_existing_path(manifest_row.get("oct_path", row.get("oct_path", "")))
    if copy_file(oct_src, sample_dir / "preop_as_oct_input.jpg", warnings, f"preop AS-OCT for {sample_id}"):
        copied.append("preop_as_oct_input.jpg")
    else:
        missing.append("preop_as_oct_input.jpg")

    crop_dir = sample_dir / "pod1_measurement_crops"
    crop_paths = split_paths(verified_row.get("measurement_crop_paths", ""))
    if not crop_paths:
        checked_rows = find_checked_rows(checked_df, patient_uid=patient_uid, eye=eye)
        crop_paths = checked_rows["measurement_crop_path"].dropna().astype(str).tolist()
    copied_crops: List[str] = []
    for crop_path in crop_paths:
        crop_src = resolve_existing_path(crop_path)
        crop_name = Path(crop_path).name or "measurement_crop.png"
        if copy_file(crop_src, crop_dir / crop_name, warnings, f"POD1 measurement crop for {sample_id}"):
            copied_crops.append(f"pod1_measurement_crops/{crop_name}")
        else:
            missing.append(crop_path)

    review_info = {
        "sample_id": sample_id,
        "patient_id": row.get("patient_id", ""),
        "patient_uid": patient_uid,
        "eye_side": row.get("eye_side", ""),
        "eye": eye,
        "vault_label_um": row.get("vault_label_um", ""),
        "mean_abs_error_um": row.get("mean_abs_error_um", ""),
        "mean_signed_error_um": row.get("mean_signed_error_um", ""),
        "worst_run_name": row.get("worst_run_name", ""),
        "max_abs_error_um": row.get("max_abs_error_um", ""),
        "oct_path": manifest_row.get("oct_path", row.get("oct_path", "")),
        "pod1_vault_scan1_um": verified_row.get("pod1_vault_scan1_um", ""),
        "pod1_vault_scan2_um": verified_row.get("pod1_vault_scan2_um", ""),
        "pod1_vault_mean_um": verified_row.get("pod1_vault_mean_um", ""),
        "pod1_vault_range_um": verified_row.get("pod1_vault_range_um", ""),
        "label_qc_flag": verified_row.get("qc_flag", manifest_row.get("label_qc_flag", "")),
        "measurement_crop_paths": verified_row.get("measurement_crop_paths", ";".join(crop_paths)),
        "copied_files": ";".join(copied + copied_crops),
        "missing_files": ";".join(missing),
        "notes": row.get("notes", ""),
    }
    pd.DataFrame([review_info]).to_csv(sample_dir / "review_info.csv", index=False, encoding="utf-8")
    write_readme(sample_dir, sample_id)

    print(f"{sample_id}: copied {', '.join(copied + copied_crops) if copied or copied_crops else 'none'}")
    if missing:
        print(f"{sample_id}: missing {', '.join(missing)}")

    return {
        "sample_id": sample_id,
        "patient_id": row.get("patient_id", ""),
        "eye_side": row.get("eye_side", ""),
        "vault_label_um": row.get("vault_label_um", ""),
        "mean_abs_error_um": row.get("mean_abs_error_um", ""),
        "mean_signed_error_um": row.get("mean_signed_error_um", ""),
        "vault_range_group": vault_range_group(row.get("vault_label_um", "")),
        "review_folder": relative_path(sample_dir),
        "review_status": "pending",
        "notes": "",
    }


def main() -> None:
    args = parse_args()
    top_error_path = resolve_project_path(args.top_error)
    error_summary_path = resolve_project_path(args.error_summary)
    manifest_path = resolve_project_path(args.manifest)
    pod1_checked_path = resolve_project_path(args.pod1_checked)
    verified_labels_path = resolve_project_path(args.verified_labels)
    output_dir = resolve_project_path(args.output_dir)
    warnings: List[str] = []

    top_errors = load_top_errors(top_error_path, error_summary_path, top_k=args.top_k)
    manifest_df = pd.read_csv(manifest_path)
    checked_df = pd.read_csv(pod1_checked_path)
    verified_df = pd.read_csv(verified_labels_path)

    index_rows: List[Dict[str, object]] = []
    for row in top_errors.to_dict(orient="records"):
        index_rows.append(
            prepare_sample_package(
                row=row,
                manifest_df=manifest_df,
                checked_df=checked_df,
                verified_df=verified_df,
                output_dir=output_dir,
                warnings=warnings,
            )
        )

    index_path = output_dir / "top_error_review_index.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index_rows).to_csv(index_path, index=False, encoding="utf-8")

    print(f"Processed top error samples: {len(index_rows)}")
    print(f"Missing files: {len(warnings)}")
    for item in warnings:
        print(f"  {item}")
    print(f"Review package output: {relative_path(output_dir)}")
    print(f"Review index: {relative_path(index_path)}")


if __name__ == "__main__":
    main()

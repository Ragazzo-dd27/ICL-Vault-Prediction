"""Build combined batch_01 + batch_02 POD1 manifests.

The combined split is patient-level based on global_patient_uid. This script
does not modify source manifests, does not modify training code, and does not
train models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MEAN_FEATURES = ["cct_mean_um", "acd_epi_mean_mm", "acd_endo_mean_mm", "clr_mean_um", "ata_mean_mm"]


INPUT_DEFAULTS = {
    "as_oct_b01_clean": "data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean_strict.csv",
    "as_oct_b02_ready": "data/manifests/vault_as_oct_only_pod1_manifest_batch_02_ready.csv",
    "as_oct_b02_strict": "data/manifests/vault_as_oct_only_pod1_manifest_batch_02_strict.csv",
    "meas_b01_ready": "data/manifests/vault_preop_measurement_only_pod1_manifest_batch_01_ready.csv",
    "meas_b01_strict": "data/manifests/vault_preop_measurement_only_pod1_manifest_batch_01_strict.csv",
    "meas_b02_ready": "data/manifests/vault_preop_measurement_only_pod1_manifest_batch_02_ready.csv",
    "meas_b02_strict": "data/manifests/vault_preop_measurement_only_pod1_manifest_batch_02_strict.csv",
}


OUTPUT_DEFAULTS = {
    "as_oct_ready": "data/manifests/vault_as_oct_only_pod1_manifest_combined_ready.csv",
    "as_oct_strict": "data/manifests/vault_as_oct_only_pod1_manifest_combined_strict.csv",
    "measurement_ready": "data/manifests/vault_preop_measurement_only_pod1_manifest_combined_ready.csv",
    "measurement_strict": "data/manifests/vault_preop_measurement_only_pod1_manifest_combined_strict.csv",
    "split": "data/splits/pod1_combined_patient_level_split.csv",
    "summary": "artifacts/reports/combined_batch_01_02/combined_manifest_summary.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined batch_01 + batch_02 manifests.")
    for name, default in INPUT_DEFAULTS.items():
        parser.add_argument(f"--{name}", type=str, default=default)
    for name, default in OUTPUT_DEFAULTS.items():
        parser.add_argument(f"--{name}_out" if name != "split" and name != "summary" else f"--{name}_out", type=str, default=default)
    parser.add_argument("--seed", type=int, default=SEED)
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


def path_exists(path_text: object) -> bool:
    if pd.isna(path_text):
        return False
    text = str(path_text).strip()
    if not text or text.lower() == "nan":
        return False
    return resolve_project_path(text).exists()


def load_manifest(path: Path, batch_id: str, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["batch_id"] = batch_id
    df["source_manifest"] = source_name
    df["global_sample_id"] = df["batch_id"] + "__" + df["sample_id"].astype(str)
    df["global_patient_uid"] = df["batch_id"] + "__" + df["patient_uid"].astype(str)
    df["patient_id"] = df["global_patient_uid"]
    return df


def append_notes(df: pd.DataFrame, text: str) -> pd.DataFrame:
    out = df.copy()
    if "notes" not in out.columns:
        out["notes"] = ""
    out["notes"] = out["notes"].fillna("").astype(str)
    suffix = out["batch_id"].astype(str).map(lambda batch_id: f"batch_id={batch_id} | {text}")
    out["notes"] = out["notes"].where(out["notes"].str.strip().eq(""), out["notes"] + " | ") + suffix
    return out


def collect_all_patients(manifests: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for df in manifests:
        parts.append(df[["batch_id", "patient_uid", "global_patient_uid"]].drop_duplicates())
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["global_patient_uid"])


def build_patient_split(patients_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    patients = patients_df.sort_values("global_patient_uid").reset_index(drop=True)
    shuffled = patients.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(round(n * SPLIT_RATIOS["train"]))
    n_val = int(round(n * SPLIT_RATIOS["val"]))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    shuffled["split"] = splits
    return shuffled.sort_values(["split", "global_patient_uid"], kind="stable").reset_index(drop=True)


def apply_split(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    split_map = dict(zip(split_df["global_patient_uid"], split_df["split"]))
    out = df.copy()
    out["split"] = out["global_patient_uid"].map(split_map).fillna("")
    return out


def combine_pair(batch01_df: pd.DataFrame, batch02_df: pd.DataFrame, split_df: pd.DataFrame, kind: str) -> pd.DataFrame:
    combined = pd.concat([batch01_df, batch02_df], ignore_index=True, sort=False)
    combined = append_notes(combined, f"combined batch_01 + batch_02 {kind} manifest; split regenerated by global_patient_uid")
    combined = apply_split(combined, split_df)
    sort_cols = ["split", "batch_id", "global_patient_uid", "eye", "global_sample_id"]
    sort_cols = [col for col in sort_cols if col in combined.columns]
    return combined.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def validate_patient_split(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    counts = df.groupby("global_patient_uid")["split"].nunique()
    return bool((counts <= 1).all())


def validate_as_oct(df: pd.DataFrame) -> Dict[str, object]:
    labels = pd.to_numeric(df["vault_label"], errors="coerce")
    has_oct = normalize_bool_series(df["has_oct"]) if "has_oct" in df.columns else pd.Series(False, index=df.index)
    has_ubm = normalize_bool_series(df["has_ubm"]) if "has_ubm" in df.columns else pd.Series(False, index=df.index)
    ubm_path_nonempty = df.get("ubm_path", pd.Series("", index=df.index)).fillna("").astype(str).str.strip().ne("")
    return {
        "rows": len(df),
        "global_sample_duplicates": int(df["global_sample_id"].duplicated().sum()),
        "missing_or_invalid_label": int((labels.isna() | (labels <= 0)).sum()),
        "missing_oct_path": int(df["oct_path"].fillna("").astype(str).str.strip().eq("").sum()),
        "nonexistent_oct_path": int((~df["oct_path"].map(path_exists)).sum()),
        "has_oct_false": int((~has_oct).sum()),
        "has_ubm_true": int(has_ubm.sum()),
        "ubm_path_nonempty": int(ubm_path_nonempty.sum()),
        "patient_cross_split": not validate_patient_split(df),
    }


def validate_measurement(df: pd.DataFrame) -> Dict[str, object]:
    labels = pd.to_numeric(df["vault_label"], errors="coerce")
    missing_features = int(df[MEAN_FEATURES].apply(pd.to_numeric, errors="coerce").isna().any(axis=1).sum())
    return {
        "rows": len(df),
        "global_sample_duplicates": int(df["global_sample_id"].duplicated().sum()),
        "missing_or_invalid_label": int((labels.isna() | (labels <= 0)).sum()),
        "missing_mean_features": missing_features,
        "patient_cross_split": not validate_patient_split(df),
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def split_counts(df: pd.DataFrame) -> Dict[str, int]:
    return df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0).astype(int).to_dict()


def markdown_dict(title: str, value: Dict[str, object]) -> List[str]:
    lines = [f"### {title}", ""]
    for key, item in value.items():
        lines.append(f"- {key}: {item}")
    lines.append("")
    return lines


def write_summary(
    path: Path,
    input_counts: Dict[str, int],
    outputs: Dict[str, pd.DataFrame],
    split_df: pd.DataFrame,
    validations: Dict[str, Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    patient_split_counts = split_df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0).astype(int).to_dict()
    lines = [
        "# Batch 01 + Batch 02 combined manifest summary",
        "",
        "本步骤合并 batch_01 与 batch_02 的 AS-OCT-only 和 preop measurement-only manifests，"
        "并基于 `global_patient_uid` 重新生成统一 patient-level split。没有修改原始 manifest，也没有训练模型。",
        "",
        "combined split is patient-level based on global_patient_uid.",
        "",
        "## 输入文件行数",
        "",
    ]
    for name, count in input_counts.items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## 输出 manifest 行数", ""])
    for name, df in outputs.items():
        lines.append(f"- {name}: {len(df)}")
    lines.extend(
        [
            "",
            "## Patient-level split",
            "",
            f"- combined unique patients: {split_df['global_patient_uid'].nunique()}",
            f"- train/val/test patient counts: {patient_split_counts}",
            "",
            "## Manifest split sample counts",
            "",
        ]
    )
    for name, df in outputs.items():
        lines.append(f"- {name}: {split_counts(df)}")
    lines.extend(["", "## Label QC 分布", ""])
    for name, df in outputs.items():
        if "label_qc_flag" in df.columns:
            lines.append(f"- {name}: {df['label_qc_flag'].value_counts(dropna=False).to_dict()}")
    lines.extend(["", "## Measurement readiness 分布", ""])
    for name, df in outputs.items():
        if "measurement_ready_status" in df.columns:
            lines.append(f"- {name}: {df['measurement_ready_status'].value_counts(dropna=False).to_dict()}")
    lines.extend(["", "## Batch 贡献", ""])
    for name, df in outputs.items():
        lines.append(f"- {name}: {df['batch_id'].value_counts(dropna=False).to_dict()}")
    lines.extend(["", "## 数据检查", ""])
    for name, validation in validations.items():
        lines.extend(markdown_dict(name, validation))
    lines.extend(
        [
            "## 下一步",
            "",
            "- 建议先运行 combined AS-OCT-only baseline，复用已通过 smoke test 的 strict AS-OCT-only dataloader/resize 设置。",
            "- 建议随后运行 combined measurement-only baseline，并与 batch_01 pilot、batch_02-only manifest 做对照。",
            "- fusion baseline 应在确认 combined split 与两个输入模态 manifest 对齐后再推进。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def main() -> None:
    args = parse_args()
    input_paths = {name: resolve_project_path(getattr(args, name)) for name in INPUT_DEFAULTS}
    output_paths = {
        "as_oct_ready": resolve_project_path(args.as_oct_ready_out),
        "as_oct_strict": resolve_project_path(args.as_oct_strict_out),
        "measurement_ready": resolve_project_path(args.measurement_ready_out),
        "measurement_strict": resolve_project_path(args.measurement_strict_out),
        "split": resolve_project_path(args.split_out),
        "summary": resolve_project_path(args.summary_out),
    }

    manifests = {
        "as_oct_b01_clean": load_manifest(input_paths["as_oct_b01_clean"], "batch_01", "as_oct_b01_clean"),
        "as_oct_b02_ready": load_manifest(input_paths["as_oct_b02_ready"], "batch_02", "as_oct_b02_ready"),
        "as_oct_b02_strict": load_manifest(input_paths["as_oct_b02_strict"], "batch_02", "as_oct_b02_strict"),
        "meas_b01_ready": load_manifest(input_paths["meas_b01_ready"], "batch_01", "meas_b01_ready"),
        "meas_b01_strict": load_manifest(input_paths["meas_b01_strict"], "batch_01", "meas_b01_strict"),
        "meas_b02_ready": load_manifest(input_paths["meas_b02_ready"], "batch_02", "meas_b02_ready"),
        "meas_b02_strict": load_manifest(input_paths["meas_b02_strict"], "batch_02", "meas_b02_strict"),
    }
    input_counts = {name: len(df) for name, df in manifests.items()}

    all_patients = collect_all_patients(manifests.values())
    split_df = build_patient_split(all_patients, seed=args.seed)

    outputs = {
        "as_oct_ready": combine_pair(manifests["as_oct_b01_clean"], manifests["as_oct_b02_ready"], split_df, "AS-OCT ready"),
        "as_oct_strict": combine_pair(manifests["as_oct_b01_clean"], manifests["as_oct_b02_strict"], split_df, "AS-OCT strict"),
        "measurement_ready": combine_pair(manifests["meas_b01_ready"], manifests["meas_b02_ready"], split_df, "measurement ready"),
        "measurement_strict": combine_pair(manifests["meas_b01_strict"], manifests["meas_b02_strict"], split_df, "measurement strict"),
    }

    validations = {
        "as_oct_ready": validate_as_oct(outputs["as_oct_ready"]),
        "as_oct_strict": validate_as_oct(outputs["as_oct_strict"]),
        "measurement_ready": validate_measurement(outputs["measurement_ready"]),
        "measurement_strict": validate_measurement(outputs["measurement_strict"]),
    }

    write_csv(outputs["as_oct_ready"], output_paths["as_oct_ready"])
    write_csv(outputs["as_oct_strict"], output_paths["as_oct_strict"])
    write_csv(outputs["measurement_ready"], output_paths["measurement_ready"])
    write_csv(outputs["measurement_strict"], output_paths["measurement_strict"])
    write_csv(split_df, output_paths["split"])
    write_summary(output_paths["summary"], input_counts, outputs, split_df, validations)

    print("Input row counts:")
    for name, count in input_counts.items():
        print(f"  {name}: {count}")
    print("Combined manifest rows:")
    for name, df in outputs.items():
        print(f"  {name}: {len(df)}")
    print(f"Combined unique patients: {split_df['global_patient_uid'].nunique()}")
    print(f"Patient split counts: {split_df['split'].value_counts().reindex(['train','val','test'], fill_value=0).to_dict()}")
    for name, validation in validations.items():
        print(f"{name} validation: {validation}")
    print(f"Outputs: {format_paths(output_paths.values())}")


if __name__ == "__main__":
    main()

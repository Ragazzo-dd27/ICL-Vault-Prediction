"""Run AS-OCT-only ImageNet fine-tune on standard repeated patient splits.

This is a wrapper around scripts/train_as_oct_pod1_baseline.py. It reuses the
existing AS-OCT-only training logic and only creates per-split manifest copies
whose split column is mapped from existing repeated patient-level split CSVs.

This script does not run patient_052 forced-test splits, fusion, measurement-only
models, or weighted-loss experiments. It does not modify the source manifest,
existing split CSVs, paper text, or previous experiment outputs.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "data/manifests/vault_as_oct_only_pod1_manifest_combined_strict.csv"
DEFAULT_STANDARD_SPLITS = [
    "data/splits/repeated_patient_split_seed42.csv",
    "data/splits/repeated_patient_split_seed1001.csv",
    "data/splits/repeated_patient_split_seed2002.csv",
    "data/splits/repeated_patient_split_seed2026.csv",
    "data/splits/repeated_patient_split_seed3407.csv",
]
OUTPUT_DIR = PROJECT_ROOT / "artifacts/reports/combined_batch_01_02/repeated_patient_split_stability/as_oct_only_seed42_standard"
SANITY_OUTPUT_DIR = PROJECT_ROOT / "artifacts/reports/combined_batch_01_02/repeated_patient_split_stability/as_oct_sanity_check"
ENSEMBLE_OUTPUT_DIR = PROJECT_ROOT / "artifacts/reports/combined_batch_01_02/repeated_patient_split_stability/as_oct_only_ensemble_pilot"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/train_as_oct_pod1_baseline.py"
PRED_ROOT = PROJECT_ROOT / "artifacts/predictions/as_oct_pod1_baseline_batch_01"
LOG_ROOT = PROJECT_ROOT / "artifacts/logs/as_oct_pod1_baseline_batch_01"
REPORT_ROOT = PROJECT_ROOT / "artifacts/reports/as_oct_pod1_baseline_batch_01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate AS-OCT-only ImageNet fine-tune across standard repeated patient-level splits."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--split_files",
        default=",".join(DEFAULT_STANDARD_SPLITS),
        help="Comma-separated standard repeated split CSV files. Forced-test split files are intentionally not included.",
    )
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR.relative_to(PROJECT_ROOT).as_posix()))
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="", help="Forwarded to train_as_oct_pod1_baseline.py.")
    parser.add_argument("--split_seeds", default="", help="Optional comma-separated subset, e.g. 42 or 42,1001.")
    parser.add_argument("--run_suffix", default="", help="Optional suffix for run_name, useful for smoke runs.")
    parser.add_argument("--dry_run", action="store_true", help="Only prepare manifests and print commands; do not train.")
    parser.add_argument("--force", action="store_true", help="Run training even if prediction output already exists.")
    parser.add_argument("--low_threshold", type=float, default=500.0)
    parser.add_argument("--high_threshold", type=float, default=800.0)
    parser.add_argument(
        "--sanity_original_split",
        action="store_true",
        help="Run the original-split sanity reproduction instead of repeated split evaluation.",
    )
    parser.add_argument(
        "--sanity_output_dir",
        default=str(SANITY_OUTPUT_DIR.relative_to(PROJECT_ROOT).as_posix()),
    )
    parser.add_argument(
        "--sanity_run_name",
        default="as_oct_original_split_reproduce_modelseed42_sanity",
    )
    parser.add_argument(
        "--original_reference_run",
        default="combined_as_oct_strict_imagenet_seed42_e30",
        help="Existing original seed42 run used as the reference in the sanity summary.",
    )
    parser.add_argument("--original_reference_mae", type=float, default=111.55)
    parser.add_argument(
        "--ensemble_pilot",
        action="store_true",
        help="Run 3-seed ensemble pilot for selected standard repeated splits.",
    )
    parser.add_argument("--ensemble_split_seeds", default="2026,3407")
    parser.add_argument("--ensemble_model_seeds", default="42,2026,3407")
    parser.add_argument("--ensemble_output_dir", default=str(ENSEMBLE_OUTPUT_DIR.relative_to(PROJECT_ROOT).as_posix()))
    parser.add_argument("--measurement_only_ridge_reference_mae", type=float, default=148.41)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_split_files(text: str) -> List[Path]:
    return [resolve_path(item.strip()) for item in text.split(",") if item.strip()]


def parse_seed_from_split_path(path: Path) -> int:
    name = path.name
    marker = "seed"
    if marker not in name:
        raise ValueError(f"Could not parse split seed from {name}")
    tail = name.split(marker, 1)[1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse split seed from {name}")
    return int(digits)


def requested_seed_filter(text: str) -> set[int] | None:
    if not text.strip():
        return None
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def prepare_manifest(base_manifest: pd.DataFrame, split_path: Path) -> pd.DataFrame:
    split_df = pd.read_csv(split_path)
    if "global_patient_uid" not in base_manifest.columns:
        raise KeyError("Base manifest must contain global_patient_uid.")
    if "global_patient_uid" not in split_df.columns:
        if "patient_id" in split_df.columns:
            split_df["global_patient_uid"] = split_df["patient_id"].astype(str)
        elif "patient_uid" in split_df.columns:
            split_df["global_patient_uid"] = split_df["patient_uid"].astype(str)
        else:
            raise KeyError(f"Split file lacks global_patient_uid/patient_id/patient_uid: {split_path}")
    patient_split = split_df[["global_patient_uid", "split"]].drop_duplicates()
    leaked = patient_split.groupby("global_patient_uid")["split"].nunique()
    leaked = leaked[leaked > 1]
    if not leaked.empty:
        raise ValueError(f"Patient leakage in split file {split_path}: {leaked.index.tolist()[:5]}")

    out = base_manifest.drop(columns=["split"], errors="ignore").merge(patient_split, on="global_patient_uid", how="left")
    if out["split"].isna().any():
        missing = sorted(out.loc[out["split"].isna(), "global_patient_uid"].unique())
        raise ValueError(f"Patients in manifest missing from split file {split_path}: {missing[:10]}")
    if (out.groupby("global_patient_uid")["split"].nunique() > 1).any():
        raise ValueError(f"Patient leakage after applying split file {split_path}.")
    return out


def vault_range(labels: pd.Series, low_threshold: float, high_threshold: float) -> pd.Series:
    values = pd.to_numeric(labels, errors="coerce")
    return pd.Series(
        np.select([values < low_threshold, values <= high_threshold], ["low", "medium"], default="high"),
        index=labels.index,
    )


def sample_count_rows(split_seed: int, split_path: Path, df: pd.DataFrame, low_threshold: float, high_threshold: float) -> List[Dict[str, object]]:
    out = df.copy()
    out["vault_range"] = vault_range(out["vault_label"], low_threshold, high_threshold)
    rows = []
    for split in ["train", "val", "test"]:
        group = out[out["split"] == split]
        rows.append(
            {
                "split_seed": split_seed,
                "model_seed": None,
                "split_file": relative_path(split_path),
                "split": split,
                "n_samples": len(group),
                "n_patients": group["global_patient_uid"].nunique(),
                "n_low": int((group["vault_range"] == "low").sum()),
                "n_medium": int((group["vault_range"] == "medium").sum()),
                "n_high": int((group["vault_range"] == "high").sum()),
            }
        )
    return rows


def run_name_for(split_seed: int, model_seed: int, suffix: str = "") -> str:
    suffix = suffix.strip()
    return f"as_oct_repeated_split{split_seed}_modelseed{model_seed}{suffix}"


def build_train_command(args: argparse.Namespace, run_name: str, manifest_path: Path) -> List[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--manifest",
        relative_path(manifest_path),
        "--image_size",
        str(args.image_size),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--seed",
        str(args.model_seed),
        "--num_workers",
        str(args.num_workers),
        "--pretrained",
        "--loss_weight_mode",
        "none",
        "--run_name",
        run_name,
    ]
    if args.device:
        command.extend(["--device", args.device])
    return command


def args_with_model_seed(args: argparse.Namespace, model_seed: int) -> argparse.Namespace:
    copied = Namespace(**vars(args))
    copied.model_seed = model_seed
    return copied


def build_key_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "model": "torchvision.models.resnet18",
        "pretrained": "ImageNet",
        "freeze_backbone": False,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "scheduler": "none",
        "augmentation": "Resize((image_size, image_size)) + ToTensor()",
        "label_normalize": True,
        "best_checkpoint_selection": "lowest val_mae_um",
        "loss_weight_mode": "none",
        "model_seed": args.model_seed,
    }


def prediction_paths(run_name: str) -> Dict[str, Path]:
    return {
        "train_log": LOG_ROOT / run_name / "train_log.csv",
        "val_predictions": PRED_ROOT / run_name / "val_predictions.csv",
        "test_predictions": PRED_ROOT / run_name / "test_predictions.csv",
        "range_metrics": REPORT_ROOT / run_name / "range_metrics.csv",
    }


def prediction_columns(df: pd.DataFrame) -> tuple[str, str]:
    label_candidates = ["vault_label_um", "vault_label", "label", "y_true", "target"]
    pred_candidates = ["pred_vault_um", "pred_um", "prediction", "pred", "y_pred"]
    label_col = next((col for col in label_candidates if col in df.columns), None)
    pred_col = next((col for col in pred_candidates if col in df.columns), None)
    if label_col is None or pred_col is None:
        raise KeyError(f"Could not detect label/prediction columns from: {list(df.columns)}")
    return label_col, pred_col


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    signed = y_pred - y_true
    return {
        "mae_um": float(mean_absolute_error(y_true, y_pred)),
        "rmse_um": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "mean_signed_error_um": float(np.mean(signed)),
    }


def collect_predictions(split_seed: int, model_seed: int, run_name: str, low_threshold: float, high_threshold: float) -> tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    paths = prediction_paths(run_name)
    test_path = paths["test_predictions"]
    val_path = paths["val_predictions"]
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test predictions for {run_name}: {test_path}")
    test_df = pd.read_csv(test_path)
    label_col, pred_col = prediction_columns(test_df)
    y_true = pd.to_numeric(test_df[label_col], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(test_df[pred_col], errors="coerce").to_numpy(dtype=float)
    metrics = overall_metrics(y_true, y_pred)

    best_val_mae = np.nan
    if paths["train_log"].exists():
        log_df = pd.read_csv(paths["train_log"])
        if "val_mae_um" in log_df.columns and not log_df.empty:
            best_val_mae = float(pd.to_numeric(log_df["val_mae_um"], errors="coerce").min())

    row = {
        "split_seed": split_seed,
        "model_seed": model_seed,
        "run_name": run_name,
        "n_test": len(test_df),
        "best_val_mae_um": best_val_mae,
        **metrics,
        "test_predictions_path": relative_path(test_path),
        "train_log_path": relative_path(paths["train_log"]),
    }

    pred_out = test_df.copy()
    pred_out["split_seed"] = split_seed
    pred_out["model_seed"] = model_seed
    pred_out["run_name"] = run_name
    pred_out["vault_label_um"] = y_true
    pred_out["pred_vault_um"] = y_pred
    pred_out["signed_error_um"] = y_pred - y_true
    pred_out["abs_error_um"] = np.abs(y_pred - y_true)
    pred_out["vault_range"] = vault_range(pred_out["vault_label_um"], low_threshold, high_threshold)

    if paths["range_metrics"].exists():
        range_df = pd.read_csv(paths["range_metrics"])
        range_df = range_df[range_df["split"].eq("test")].copy()
        range_df["split_seed"] = split_seed
        range_df["model_seed"] = model_seed
        range_df["run_name"] = run_name
    else:
        range_df = range_metrics_from_predictions(pred_out, split_seed, model_seed, run_name)
    return row, pred_out, range_df


def range_metrics_from_predictions(pred_df: pd.DataFrame, split_seed: int, model_seed: int, run_name: str) -> pd.DataFrame:
    rows = []
    for range_name in ["low", "medium", "high"]:
        group = pred_df[pred_df["vault_range"] == range_name]
        if group.empty:
            rows.append(
                {
                    "split": "test",
                    "vault_range": range_name,
                    "n_samples": 0,
                    "mae_um": np.nan,
                    "rmse_um": np.nan,
                    "mean_signed_error_um": np.nan,
                    "overestimation_count": 0,
                    "split_seed": split_seed,
                    "model_seed": model_seed,
                    "run_name": run_name,
                }
            )
            continue
        y_true = group["vault_label_um"].to_numpy(dtype=float)
        y_pred = group["pred_vault_um"].to_numpy(dtype=float)
        signed = y_pred - y_true
        rows.append(
            {
                "split": "test",
                "vault_range": range_name,
                "n_samples": len(group),
                "mae_um": float(np.mean(np.abs(signed))),
                "rmse_um": float(np.sqrt(np.mean(signed**2))),
                "mean_signed_error_um": float(np.mean(signed)),
                "overestimation_count": int((signed > 0).sum()),
                "split_seed": split_seed,
                "model_seed": model_seed,
                "run_name": run_name,
            }
        )
    return pd.DataFrame(rows)


def range_metrics_for_split(pred_df: pd.DataFrame, split_name: str, low_threshold: float, high_threshold: float) -> pd.DataFrame:
    out = pred_df.copy()
    out["vault_range"] = vault_range(out["vault_label_um"], low_threshold, high_threshold)
    rows = []
    for range_name in ["low", "medium", "high"]:
        group = out[out["vault_range"] == range_name]
        if group.empty:
            rows.append(
                {
                    "split": split_name,
                    "vault_range": range_name,
                    "n_samples": 0,
                    "mae_um": np.nan,
                    "rmse_um": np.nan,
                    "mean_signed_error_um": np.nan,
                    "median_abs_error_um": np.nan,
                    "overestimation_count": 0,
                    "underestimation_count": 0,
                }
            )
            continue
        y_true = group["vault_label_um"].to_numpy(dtype=float)
        y_pred = group["pred_vault_um"].to_numpy(dtype=float)
        signed = y_pred - y_true
        rows.append(
            {
                "split": split_name,
                "vault_range": range_name,
                "n_samples": len(group),
                "mae_um": float(np.mean(np.abs(signed))),
                "rmse_um": float(np.sqrt(np.mean(signed**2))),
                "mean_signed_error_um": float(np.mean(signed)),
                "median_abs_error_um": float(np.median(np.abs(signed))),
                "overestimation_count": int((signed > 0).sum()),
                "underestimation_count": int((signed < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def load_prediction_file(path: Path, split_name: str, low_threshold: float, high_threshold: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    label_col, pred_col = prediction_columns(df)
    out = df.copy()
    out["vault_label_um"] = pd.to_numeric(out[label_col], errors="coerce")
    out["pred_vault_um"] = pd.to_numeric(out[pred_col], errors="coerce")
    out["signed_error_um"] = out["pred_vault_um"] - out["vault_label_um"]
    out["abs_error_um"] = out["signed_error_um"].abs()
    out["split"] = split_name
    out["vault_range"] = vault_range(out["vault_label_um"], low_threshold, high_threshold)
    return out


def sanity_outputs_exist(run_name: str) -> bool:
    paths = prediction_paths(run_name)
    return paths["test_predictions"].exists() and paths["val_predictions"].exists() and paths["train_log"].exists()


def collect_sanity_outputs(args: argparse.Namespace, output_dir: Path, manifest_path: Path) -> bool:
    run_name = args.sanity_run_name
    paths = prediction_paths(run_name)
    if not paths["test_predictions"].exists():
        return False

    test_predictions = load_prediction_file(
        paths["test_predictions"],
        split_name="test",
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    val_predictions = (
        load_prediction_file(paths["val_predictions"], "val", args.low_threshold, args.high_threshold)
        if paths["val_predictions"].exists()
        else pd.DataFrame()
    )
    y_true = test_predictions["vault_label_um"].to_numpy(dtype=float)
    y_pred = test_predictions["pred_vault_um"].to_numpy(dtype=float)
    metrics = overall_metrics(y_true, y_pred)
    best_val_mae = np.nan
    if paths["train_log"].exists():
        log_df = pd.read_csv(paths["train_log"])
        if "val_mae_um" in log_df.columns and not log_df.empty:
            best_val_mae = float(pd.to_numeric(log_df["val_mae_um"], errors="coerce").min())

    manifest_df = pd.read_csv(manifest_path)
    counts = manifest_df["split"].value_counts()
    overall = pd.DataFrame(
        [
            {
                "run_name": run_name,
                "model_seed": args.model_seed,
                "n_train": int(counts.get("train", 0)),
                "n_val": int(counts.get("val", 0)),
                "n_test": int(counts.get("test", 0)),
                "best_val_mae_um": best_val_mae,
                **metrics,
                "original_reference_run": args.original_reference_run,
                "original_reference_mae_um": args.original_reference_mae,
                "mae_delta_vs_reference_um": metrics["mae_um"] - args.original_reference_mae,
            }
        ]
    )
    range_frames = []
    if not val_predictions.empty:
        range_frames.append(range_metrics_for_split(val_predictions, "val", args.low_threshold, args.high_threshold))
    range_frames.append(range_metrics_for_split(test_predictions, "test", args.low_threshold, args.high_threshold))
    range_df = pd.concat(range_frames, ignore_index=True)
    predictions = pd.concat([df for df in [val_predictions, test_predictions] if not df.empty], ignore_index=True)

    overall_path = output_dir / "sanity_original_split_overall_metrics.csv"
    range_path = output_dir / "sanity_original_split_range_metrics.csv"
    predictions_path = output_dir / "sanity_original_split_predictions.csv"
    summary_path = output_dir / "sanity_original_split_summary.md"
    overall.to_csv(overall_path, index=False, encoding="utf-8")
    range_df.to_csv(range_path, index=False, encoding="utf-8")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    write_sanity_summary(summary_path, args, manifest_path, output_dir / "manifests" / f"{run_name}_manifest.csv", overall, range_df)
    return True


def write_sanity_summary(
    path: Path,
    args: argparse.Namespace,
    source_manifest_path: Path,
    run_manifest_path: Path,
    overall: pd.DataFrame | None,
    range_df: pd.DataFrame | None,
    command: List[str] | None = None,
) -> None:
    config = build_key_config(args)
    source_df = pd.read_csv(source_manifest_path)
    counts = source_df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0)
    lines = [
        "# AS-OCT original-split sanity reproduction",
        "",
        "本步骤用于确认 repeated split wrapper 复用的 AS-OCT-only ImageNet fine-tune 配置是否与原始主实验一致。",
        "本 sanity run 使用当前 combined AS-OCT strict manifest 中已有的原始 train/val/test split，不使用 repeated split CSV，也不包含 patient_052 forced-test split。",
        "",
        f"- 原始 split / manifest 路径: `{relative_path(source_manifest_path)}`",
        f"- sanity run manifest 路径: `{relative_path(run_manifest_path)}`",
        f"- sanity run name: `{args.sanity_run_name}`",
        f"- 原始记录 seed42 test MAE 约: {args.original_reference_mae:.2f} um",
        f"- train/val/test 样本数: {int(counts['train'])} / {int(counts['val'])} / {int(counts['test'])}",
        "",
        "## Key Training Configuration",
        "",
    ]
    for key, value in config.items():
        lines.append(f"- {key}: {value}")

    if command is not None:
        lines.extend(["", "## Training Command", "", "```powershell", " ".join(command), "```", ""])

    lines.extend(["", "## Sanity Reproduction Metrics", ""])
    if overall is None or overall.empty:
        lines.append("本次尚未完成训练，因此暂无 reproduction metrics。运行非 dry-run 后会生成 CSV 和本节指标。")
    else:
        row = overall.iloc[0]
        close = abs(float(row["mae_delta_vs_reference_um"])) <= 10.0
        lines.extend(
            [
                f"- 本次 sanity reproduction test MAE: {row['mae_um']:.2f} um",
                f"- RMSE: {row['rmse_um']:.2f} um",
                f"- R2: {row['r2']:.4f}",
                f"- 与原始记录 MAE 差异: {row['mae_delta_vs_reference_um']:.2f} um",
                f"- 是否接近原始 seed42 MAE: {'yes' if close else 'no / needs investigation'}",
            ]
        )
    lines.extend(
        [
            "",
            "## If Not Close, Possible Reasons",
            "",
            "- 训练参数不同，例如 epoch、batch size、learning rate、weight decay。",
            "- 数据增强或 image preprocessing 不一致。",
            "- label normalization 不一致。",
            "- best checkpoint selection 不一致。",
            "- 输入图像过滤或 manifest 行集合不同。",
            "- split 映射不同。",
            "- GPU/cuDNN 非完全确定性导致的小幅随机波动。",
            "",
            "本步骤不覆盖已有主实验 checkpoint / prediction，不修改原始 manifest、split、prediction、checkpoint 或论文正文。",
            "",
        ]
    )
    if range_df is not None and not range_df.empty:
        lines.extend(["## Range Metrics Preview", ""])
        lines.extend(md_table(range_df, ["split", "vault_range", "n_samples", "mae_um", "mean_signed_error_um"]))
    path.write_text("\n".join(lines), encoding="utf-8")


def run_sanity_original_split(args: argparse.Namespace) -> None:
    manifest_path = resolve_path(args.manifest)
    output_dir = resolve_path(args.sanity_output_dir)
    manifest_dir = output_dir / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = pd.read_csv(manifest_path)
    run_manifest_path = manifest_dir / f"{args.sanity_run_name}_manifest.csv"
    source_manifest.to_csv(run_manifest_path, index=False, encoding="utf-8")

    command = build_train_command(args, args.sanity_run_name, run_manifest_path)
    paths = prediction_paths(args.sanity_run_name)

    print("AS-OCT original-split sanity reproduction")
    print(f"Source manifest: {manifest_path}")
    print(f"Run manifest: {run_manifest_path}")
    print(f"Run name: {args.sanity_run_name}")
    print(f"Train/val/test samples: {source_manifest['split'].value_counts().reindex(['train','val','test'], fill_value=0).to_dict()}")
    print(f"Reference original seed42 MAE: {args.original_reference_mae:.2f} um")

    if args.dry_run:
        print(f"[dry-run] {' '.join(command)}")
        write_sanity_summary(
            output_dir / "sanity_original_split_summary.md",
            args,
            manifest_path,
            run_manifest_path,
            overall=None,
            range_df=None,
            command=command,
        )
        print(f"Dry-run summary: {output_dir / 'sanity_original_split_summary.md'}")
        return

    if paths["test_predictions"].exists() and not args.force:
        print(f"Existing sanity predictions found, skip training: {relative_path(paths['test_predictions'])}")
    else:
        print("Running sanity reproduction training...")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    ok = collect_sanity_outputs(args, output_dir, manifest_path)
    if not ok:
        raise FileNotFoundError(f"Sanity test predictions were not produced: {paths['test_predictions']}")
    overall = pd.read_csv(output_dir / "sanity_original_split_overall_metrics.csv")
    print("Sanity reproduction metrics:")
    print(overall[["run_name", "n_train", "n_val", "n_test", "best_val_mae_um", "mae_um", "rmse_um", "r2", "mae_delta_vs_reference_um"]].to_string(index=False))
    print("Output files:")
    for path in [
        output_dir / "sanity_original_split_overall_metrics.csv",
        output_dir / "sanity_original_split_range_metrics.csv",
        output_dir / "sanity_original_split_predictions.csv",
        output_dir / "sanity_original_split_summary.md",
    ]:
        print(path)


def ensemble_key_columns(df: pd.DataFrame) -> List[str]:
    preferred = ["sample_id", "patient_id", "eye_side"]
    return [col for col in preferred if col in df.columns]


def load_test_predictions_for_run(run_name: str, split_seed: int, model_seed: int) -> pd.DataFrame:
    path = prediction_paths(run_name)["test_predictions"]
    if not path.exists():
        raise FileNotFoundError(f"Missing test predictions for {run_name}: {path}")
    pred = load_prediction_file(path, "test", low_threshold=500.0, high_threshold=800.0)
    pred["split_seed"] = split_seed
    pred["model_seed"] = model_seed
    pred["run_name"] = run_name
    return pred


def ensemble_predictions_for_split(
    split_seed: int,
    model_seeds: List[int],
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    key_cols: List[str] | None = None
    seed_pred_cols = []
    seed_abs_cols = []
    for model_seed in model_seeds:
        run_name = run_name_for(split_seed, model_seed)
        pred = load_prediction_file(prediction_paths(run_name)["test_predictions"], "test", low_threshold, high_threshold)
        if key_cols is None:
            key_cols = ensemble_key_columns(pred)
            if not key_cols:
                raise KeyError(f"Could not determine ensemble alignment keys for {run_name}.")
        pred_col = f"pred_modelseed{model_seed}_um"
        abs_col = f"abs_error_modelseed{model_seed}_um"
        signed_col = f"signed_error_modelseed{model_seed}_um"
        keep_cols = key_cols + ["vault_label_um", "label_qc_flag", "oct_path"]
        keep_cols = [col for col in keep_cols if col in pred.columns]
        subset = pred[keep_cols].copy()
        subset[pred_col] = pred["pred_vault_um"].to_numpy(dtype=float)
        subset[signed_col] = subset[pred_col] - subset["vault_label_um"]
        subset[abs_col] = subset[signed_col].abs()
        seed_pred_cols.append(pred_col)
        seed_abs_cols.append(abs_col)
        if merged is None:
            merged = subset
        else:
            merge_cols = key_cols + ["vault_label_um"]
            merged = merged.merge(
                subset.drop(columns=[col for col in ["label_qc_flag", "oct_path"] if col in subset.columns]),
                on=merge_cols,
                how="inner",
            )
    if merged is None or key_cols is None:
        raise RuntimeError(f"No predictions loaded for split {split_seed}.")
    merged["split_seed"] = split_seed
    merged["model_seeds"] = ",".join(str(seed) for seed in model_seeds)
    merged["ensemble_pred_um"] = merged[seed_pred_cols].mean(axis=1)
    merged["ensemble_signed_error_um"] = merged["ensemble_pred_um"] - merged["vault_label_um"]
    merged["ensemble_abs_error_um"] = merged["ensemble_signed_error_um"].abs()
    merged["vault_range"] = vault_range(merged["vault_label_um"], low_threshold, high_threshold)
    return merged


def metrics_from_prediction_frame(df: pd.DataFrame, pred_col: str, label_col: str = "vault_label_um") -> Dict[str, float]:
    y_true = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    return overall_metrics(y_true, y_pred)


def ensemble_range_metrics(
    split_seed: int,
    model_seeds: List[int],
    pred_df: pd.DataFrame,
    method: str = "ensemble",
) -> pd.DataFrame:
    rows = []
    for range_name in ["low", "medium", "high"]:
        group = pred_df[pred_df["vault_range"].eq(range_name)]
        if group.empty:
            rows.append(
                {
                    "split_seed": split_seed,
                    "model_seeds": ",".join(str(seed) for seed in model_seeds),
                    "method": method,
                    "vault_range": range_name,
                    "n": 0,
                    "mae_um": np.nan,
                    "rmse_um": np.nan,
                    "mean_signed_error_um": np.nan,
                    "overestimation_count": 0,
                    "underestimation_count": 0,
                }
            )
            continue
        y_true = group["vault_label_um"].to_numpy(dtype=float)
        y_pred = group["ensemble_pred_um"].to_numpy(dtype=float)
        signed = y_pred - y_true
        rows.append(
            {
                "split_seed": split_seed,
                "model_seeds": ",".join(str(seed) for seed in model_seeds),
                "method": method,
                "vault_range": range_name,
                "n": len(group),
                "mae_um": float(np.mean(np.abs(signed))),
                "rmse_um": float(np.sqrt(np.mean(signed**2))),
                "mean_signed_error_um": float(np.mean(signed)),
                "overestimation_count": int((signed > 0).sum()),
                "underestimation_count": int((signed < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_ensemble_summary(
    path: Path,
    split_seeds: List[int],
    model_seeds: List[int],
    single_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    range_df: pd.DataFrame,
    measurement_ref: float,
) -> None:
    lines = [
        "# AS-OCT repeated split 3-seed ensemble pilot",
        "",
        "本实验只针对两个代表性 standard repeated splits 进行 AS-OCT-only 3-seed ensemble pilot。",
        "本实验不包含 patient_052 forced-test split，不包含 fusion，不包含 measurement-only 训练，不包含 weighted loss，也不修改论文正文。",
        "",
        f"- split seeds: {', '.join(str(seed) for seed in split_seeds)}",
        f"- model seeds: {', '.join(str(seed) for seed in model_seeds)}",
        "- training config: ResNet18 ImageNet fine-tune, image_size=224, batch_size=8, epochs=30, lr=1e-4, weight_decay=1e-4, AdamW, no scheduler, label normalization, best checkpoint by lowest val MAE.",
        f"- measurement-only Ridge reference MAE: {measurement_ref:.2f} um",
        "",
        "## Ensemble Metrics",
        "",
    ]
    lines.extend(
        md_table(
            ensemble_df,
            [
                "split_seed",
                "n_test",
                "single_seed42_mae",
                "ensemble_mae",
                "ensemble_rmse",
                "ensemble_r2",
                "ensemble_mean_signed_error",
                "delta_mae_vs_seed42",
                "measurement_only_ridge_reference_mae",
            ],
        )
    )
    lines.extend(["## Single-seed Metrics", ""])
    lines.extend(md_table(single_df, ["split_seed", "model_seed", "test_mae_um", "test_rmse_um", "test_r2", "best_val_mae_um", "run_name"]))
    lines.extend(["## Range Metrics", ""])
    lines.extend(md_table(range_df, ["split_seed", "vault_range", "n", "mae_um", "mean_signed_error_um", "overestimation_count", "underestimation_count"]))

    for _, row in ensemble_df.iterrows():
        split_seed = int(row["split_seed"])
        delta = float(row["delta_mae_vs_seed42"])
        better = delta < 0
        lines.append(
            f"- split_seed{split_seed}: 3-seed ensemble {'improved' if better else 'did not improve'} vs seed42 "
            f"(delta MAE {delta:.2f} um)."
        )
    low = range_df[range_df["vault_range"].eq("low")]
    high = range_df[range_df["vault_range"].eq("high")]
    if not low.empty:
        lines.append(f"- low-vault mean signed error across pilot splits: {low['mean_signed_error_um'].mean():.2f} um.")
    if not high.empty:
        lines.append(f"- high-vault mean signed error across pilot splits: {high['mean_signed_error_um'].mean():.2f} um.")
    best_ensemble = ensemble_df["ensemble_mae"].min()
    if best_ensemble < measurement_ref:
        lines.append("- 至少一个 ensemble split 的 MAE 低于 measurement-only Ridge reference。")
    else:
        lines.append("- 两个 ensemble split 均未低于 measurement-only Ridge reference。")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 本实验用于判断少量 repeated split 上 seed ensemble 是否能降低 split 波动。",
            "- 是否值得扩展到全部 5 个 standard repeated splits，应结合两个 pilot split 的 delta MAE、range-level error 和训练成本决定。",
            "- 本实验不替代原始主结果。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ensemble_pilot(args: argparse.Namespace) -> None:
    split_seed_values = parse_int_list(args.ensemble_split_seeds)
    model_seed_values = parse_int_list(args.ensemble_model_seeds)
    if 42 not in model_seed_values:
        raise ValueError("Ensemble pilot expects existing model_seed42 to be included.")
    output_dir = resolve_path(args.ensemble_output_dir)
    manifest_dir = output_dir / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    split_files = {parse_seed_from_split_path(path): path for path in parse_split_files(args.split_files)}
    base_manifest = pd.read_csv(resolve_path(args.manifest))
    single_rows = []

    for split_seed in split_seed_values:
        if split_seed not in split_files:
            raise FileNotFoundError(f"Split seed {split_seed} is not in --split_files.")
        split_path = split_files[split_seed]
        split_manifest = prepare_manifest(base_manifest, split_path)
        for model_seed in model_seed_values:
            run_name = run_name_for(split_seed, model_seed)
            run_manifest_path = manifest_dir / f"{run_name}_manifest.csv"
            split_manifest.to_csv(run_manifest_path, index=False, encoding="utf-8")
            train_args = args_with_model_seed(args, model_seed)
            command = build_train_command(train_args, run_name, run_manifest_path)
            pred_path = prediction_paths(run_name)["test_predictions"]
            if args.dry_run:
                print(f"[dry-run] {' '.join(command)}")
                continue
            if pred_path.exists() and not args.force:
                print(f"Existing predictions found, skip training: {relative_path(pred_path)}")
            else:
                print(f"Running AS-OCT ensemble pilot training: split={split_seed}, model_seed={model_seed}")
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            row, _, _ = collect_predictions(split_seed, model_seed, run_name, args.low_threshold, args.high_threshold)
            row["model_seed"] = model_seed
            row["test_mae_um"] = row.pop("mae_um")
            row["test_rmse_um"] = row.pop("rmse_um")
            row["test_r2"] = row.pop("r2")
            row["test_mean_signed_error_um"] = row.pop("mean_signed_error_um")
            single_rows.append(row)

    if args.dry_run:
        print(f"Dry-run complete. Ensemble pilot output dir: {output_dir}")
        return

    single_df = pd.DataFrame(single_rows).sort_values(["split_seed", "model_seed"])
    ensemble_rows = []
    ensemble_prediction_frames = []
    range_frames = []
    for split_seed in split_seed_values:
        pred_df = ensemble_predictions_for_split(split_seed, model_seed_values, args.low_threshold, args.high_threshold)
        seed42 = single_df[(single_df["split_seed"].eq(split_seed)) & (single_df["model_seed"].eq(42))]
        if seed42.empty:
            raise ValueError(f"Missing seed42 metric for split {split_seed}.")
        metrics = metrics_from_prediction_frame(pred_df, "ensemble_pred_um")
        ensemble_rows.append(
            {
                "split_seed": split_seed,
                "model_seeds": ",".join(str(seed) for seed in model_seed_values),
                "n_test": len(pred_df),
                "single_seed42_mae": float(seed42.iloc[0]["test_mae_um"]),
                "ensemble_mae": metrics["mae_um"],
                "ensemble_rmse": metrics["rmse_um"],
                "ensemble_r2": metrics["r2"],
                "ensemble_mean_signed_error": metrics["mean_signed_error_um"],
                "delta_mae_vs_seed42": metrics["mae_um"] - float(seed42.iloc[0]["test_mae_um"]),
                "measurement_only_ridge_reference_mae": args.measurement_only_ridge_reference_mae,
            }
        )
        ensemble_prediction_frames.append(pred_df)
        range_frames.append(ensemble_range_metrics(split_seed, model_seed_values, pred_df))

    ensemble_df = pd.DataFrame(ensemble_rows).sort_values("split_seed")
    ensemble_predictions = pd.concat(ensemble_prediction_frames, ignore_index=True)
    range_df = pd.concat(range_frames, ignore_index=True)

    single_path = output_dir / "as_oct_ensemble_pilot_single_seed_metrics.csv"
    ensemble_path = output_dir / "as_oct_ensemble_pilot_ensemble_metrics.csv"
    range_path = output_dir / "as_oct_ensemble_pilot_range_metrics.csv"
    predictions_path = output_dir / "as_oct_ensemble_pilot_predictions.csv"
    summary_path = output_dir / "as_oct_ensemble_pilot_summary.md"
    single_df.to_csv(single_path, index=False, encoding="utf-8")
    ensemble_df.to_csv(ensemble_path, index=False, encoding="utf-8")
    range_df.to_csv(range_path, index=False, encoding="utf-8")
    ensemble_predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    write_ensemble_summary(
        summary_path,
        split_seed_values,
        model_seed_values,
        single_df,
        ensemble_df,
        range_df,
        args.measurement_only_ridge_reference_mae,
    )

    print("\nAS-OCT ensemble pilot metrics:")
    print(ensemble_df.to_string(index=False))
    print("Output files:")
    for path in [single_path, ensemble_path, range_path, predictions_path, summary_path]:
        print(path)


def md_table(df: pd.DataFrame, columns: List[str] | None = None) -> List[str]:
    if columns is not None:
        df = df[columns]
    if df.empty:
        return ["_None_", ""]
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        else:
            text_df[col] = text_df[col].fillna("").astype(str)
    lines = ["| " + " | ".join(text_df.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(text_df.columns)) + " |")
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text_df.columns) + " |")
    lines.append("")
    return lines


def write_summary(
    path: Path,
    manifest_path: Path,
    split_paths: List[Path],
    sample_counts: pd.DataFrame,
    overall: pd.DataFrame,
    range_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# AS-OCT-only repeated split stability evaluation",
        "",
        "本步骤评估 AS-OCT-only ImageNet fine-tune baseline 在 standard repeated patient-level splits 下的稳定性。",
        "本分析不包含 patient_052 forced-test split，不包含 fusion，不包含 measurement-only，不包含 weighted loss，也不替代原始主结果。",
        "本步骤不修改论文正文。",
        "",
        f"- 输入 manifest: `{relative_path(manifest_path)}`",
        "- 模型: ResNet18 ImageNet fine-tune",
        "- model seed: 42",
        "",
        "## Standard Repeated Split Files",
        "",
    ]
    for split_path in split_paths:
        lines.append(f"- `{relative_path(split_path)}`")

    lines.extend(["", "## Sample Counts", ""])
    lines.extend(md_table(sample_counts, ["split_seed", "split", "n_samples", "n_patients", "n_low", "n_medium", "n_high"]))

    lines.extend(["## Overall Test Metrics", ""])
    lines.extend(md_table(overall, ["split_seed", "n_train", "n_val", "n_test", "best_val_mae_um", "mae_um", "rmse_um", "r2", "mean_signed_error_um"]))

    mae_mean = overall["mae_um"].mean() if not overall.empty else np.nan
    mae_std = overall["mae_um"].std(ddof=1) if len(overall) > 1 else np.nan
    lines.extend(
        [
            "## Stability Summary",
            "",
            f"- 5 个 standard repeated splits 的 test MAE mean ± std: {mae_mean:.2f} ± {mae_std:.2f} um",
            "- measurement-only Ridge standard repeated split MAE = 148.41 ± 37.08 um，可作为结构化术前参数 baseline 的参照。",
            "",
            "## Vault Range Error Pattern",
            "",
        ]
    )
    range_summary = (
        range_metrics.groupby("vault_range")
        .agg(
            n_mean=("n_samples", "mean"),
            mae_mean=("mae_um", "mean"),
            mae_std=("mae_um", "std"),
            signed_mean=("mean_signed_error_um", "mean"),
        )
        .reset_index()
    )
    lines.extend(md_table(range_summary, ["vault_range", "n_mean", "mae_mean", "mae_std", "signed_mean"]))
    lines.extend(
        [
            "完整 range-level 指标见 `as_oct_repeated_split_range_metrics.csv`。",
            "",
            "## Notes",
            "",
            "- split 来自已有 repeated patient-level split CSV，没有重新随机划分。",
            "- 原始 manifest 没有被修改；每个 split 只生成一个临时 manifest 副本用于训练。",
            "- 输出 run name 均为 `as_oct_repeated_split{seed}_modelseed42`，避免覆盖已有主实验。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.sanity_original_split:
        run_sanity_original_split(args)
        return
    if args.ensemble_pilot:
        run_ensemble_pilot(args)
        return

    manifest_path = resolve_path(args.manifest)
    output_dir = resolve_path(args.output_dir)
    manifest_dir = output_dir / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    split_paths = parse_split_files(args.split_files)
    forced_paths = [path for path in split_paths if "patient052test" in path.name]
    if forced_paths:
        raise ValueError(f"Forced-test split files are not allowed in this script: {forced_paths}")
    selected_seeds = requested_seed_filter(args.split_seeds)
    if selected_seeds is not None:
        split_paths = [path for path in split_paths if parse_seed_from_split_path(path) in selected_seeds]
    if not split_paths:
        raise ValueError("No standard split files selected.")

    base_manifest = pd.read_csv(manifest_path)
    overall_rows = []
    prediction_frames = []
    range_frames = []
    sample_count_rows_all = []

    print(f"Input manifest: {manifest_path}")
    print(f"Selected split files: {[relative_path(path) for path in split_paths]}")
    print(f"Output directory: {output_dir}")

    for split_path in split_paths:
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        split_seed = parse_seed_from_split_path(split_path)
        run_name = run_name_for(split_seed, args.model_seed, args.run_suffix)
        split_manifest = prepare_manifest(base_manifest, split_path)
        temp_manifest_path = manifest_dir / f"{run_name}_manifest.csv"
        split_manifest.to_csv(temp_manifest_path, index=False, encoding="utf-8")
        sample_count_rows_all.extend(sample_count_rows(split_seed, split_path, split_manifest, args.low_threshold, args.high_threshold))

        paths = prediction_paths(run_name)
        command = build_train_command(args, run_name, temp_manifest_path)
        if args.dry_run:
            print(f"[dry-run] {' '.join(command)}")
            continue
        if paths["test_predictions"].exists() and not args.force:
            print(f"Existing predictions found, skip training: {relative_path(paths['test_predictions'])}")
        else:
            print(f"Running AS-OCT training: {run_name}")
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        row, pred_df, range_df = collect_predictions(split_seed, args.model_seed, run_name, args.low_threshold, args.high_threshold)
        counts = split_manifest["split"].value_counts()
        row["n_train"] = int(counts.get("train", 0))
        row["n_val"] = int(counts.get("val", 0))
        row["n_test"] = int(counts.get("test", 0))
        overall_rows.append(row)
        prediction_frames.append(pred_df)
        range_frames.append(range_df)

    sample_counts = pd.DataFrame(sample_count_rows_all)
    sample_counts["model_seed"] = args.model_seed
    sample_counts_path = output_dir / "as_oct_repeated_split_sample_counts.csv"
    sample_counts.to_csv(sample_counts_path, index=False, encoding="utf-8")

    if args.dry_run:
        print(f"Dry run complete. Temporary manifests and sample counts are under {relative_path(output_dir)}")
        print(sample_counts_path)
        return

    overall = pd.DataFrame(overall_rows).sort_values("split_seed")
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    range_metrics = pd.concat(range_frames, ignore_index=True) if range_frames else pd.DataFrame()

    overall_path = output_dir / "as_oct_repeated_split_overall_metrics.csv"
    range_path = output_dir / "as_oct_repeated_split_range_metrics.csv"
    predictions_path = output_dir / "as_oct_repeated_split_predictions.csv"
    summary_path = output_dir / "as_oct_repeated_split_summary.md"
    overall.to_csv(overall_path, index=False, encoding="utf-8")
    range_metrics.to_csv(range_path, index=False, encoding="utf-8")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    write_summary(summary_path, manifest_path, split_paths, sample_counts, overall, range_metrics)

    print("\nAS-OCT repeated split overall metrics:")
    print(overall[["split_seed", "n_train", "n_val", "n_test", "mae_um", "rmse_um", "r2"]].to_string(index=False))
    print(f"\nMean +/- std MAE: {overall['mae_um'].mean():.2f} +/- {overall['mae_um'].std(ddof=1):.2f} um")
    print("\nOutput files:")
    for path in [overall_path, range_path, predictions_path, sample_counts_path, summary_path]:
        print(path)


if __name__ == "__main__":
    main()

"""Train combined v4 AS-OCT + preop measurement concat fusion baseline.

This launcher reuses the existing fusion baseline implementation, but writes
seed-specific formal v4 outputs directly into the requested report/checkpoint
directories. It trains only the AS-OCT + true preoperative measurement concat
fusion model on an existing split manifest.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_as_oct_measurement_fusion_baseline import (  # noqa: E402
    FEATURE_COLUMNS,
    build_model,
    build_transform,
    compute_feature_stats,
    compute_label_stats,
    evaluate,
    fusion_collate,
    make_dataset_class,
    read_manifest,
    require_torch_stack,
    save_checkpoint,
    train_one_epoch,
)


DEFAULT_MANIFEST = (
    "data/manifests/"
    "vault_as_oct_plus_measurement_pod1_manifest_combined_batch_01_02_03_04_ready_split_seed42.csv"
)
MEASUREMENT_ONLY_RF_MAE_UM = 169.44
AS_OCT_ENSEMBLE_MAE_UM = 194.79


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train combined v4 AS-OCT + measurement concat fusion baseline.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--check_only", action="store_true", help="Run manifest/input checks and exit before model setup.")
    args = parser.parse_args()

    # Fixed protocol knobs for the combined v4 fusion baseline.
    args.image_size = 224
    args.weight_decay = 1e-4
    args.device = ""
    args.pretrained = True
    args.freeze_backbone = False
    args.label_normalize = True
    args.measurement_hidden_dim = 32
    args.fusion_hidden_dim = 128
    args.low_threshold = 500.0
    args.high_threshold = 800.0
    args.measurement_only_rf_mae = MEASUREMENT_ONLY_RF_MAE_UM
    args.as_oct_ensemble_mae = AS_OCT_ENSEMBLE_MAE_UM
    return args


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_all_seeds(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    try:
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(int(worker_seed))
    np.random.seed(int(worker_seed))


def build_seeded_dataloaders(
    df: pd.DataFrame,
    args: argparse.Namespace,
    torch_module: Any,
    Dataset: Any,
    DataLoader: Any,
    transform: Any,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    FusionDataset = make_dataset_class(torch_module, Dataset)
    train_ds = FusionDataset(df, "train", transform, feature_mean, feature_std)
    val_ds = FusionDataset(df, "val", transform, feature_mean, feature_std)
    test_ds = FusionDataset(df, "test", transform, feature_mean, feature_std)
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Train, val, and test datasets must all be non-empty.")

    generator = torch_module.Generator()
    generator.manual_seed(args.seed)
    collate_fn = lambda batch: fusion_collate(torch_module, batch)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def split_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {split: int((df["split"] == split).sum()) for split in ["train", "val", "test"]}


def patient_leakage_count(df: pd.DataFrame) -> int:
    leakage = df.groupby("global_patient_uid")["split"].nunique()
    return int((leakage > 1).sum())


def duplicate_global_sample_count(df: pd.DataFrame) -> int:
    return int(df["global_sample_id"].duplicated().sum())


def write_log_row_flush(path: Path, row: Dict[str, Any], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def regression_metrics_np(preds: pd.Series, labels: pd.Series) -> Dict[str, float]:
    errors = preds.astype(float) - labels.astype(float)
    ss_tot = float(((labels.astype(float) - labels.astype(float).mean()) ** 2).sum())
    ss_res = float((errors**2).sum())
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "r2": float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot),
        "mean_signed_error": float(errors.mean()),
    }


def range_metrics_from_predictions(
    predictions: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    df = predictions.copy()
    df["vault_label_um"] = pd.to_numeric(df["vault_label_um"], errors="coerce")
    df["pred_vault_um"] = pd.to_numeric(df["pred_vault_um"], errors="coerce")
    df["signed_error_um"] = df["pred_vault_um"] - df["vault_label_um"]
    df["abs_error_um"] = df["signed_error_um"].abs()
    df["vault_range"] = "medium"
    df.loc[df["vault_label_um"] < low_threshold, "vault_range"] = "low"
    df.loc[df["vault_label_um"] > high_threshold, "vault_range"] = "high"

    rows: List[Dict[str, Any]] = []
    for vault_range in ["low", "medium", "high"]:
        sub = df[df["vault_range"] == vault_range]
        if sub.empty:
            rows.append(
                {
                    "vault_range": vault_range,
                    "n": 0,
                    "mae_um": float("nan"),
                    "rmse_um": float("nan"),
                    "mean_signed_error_um": float("nan"),
                    "overestimation_count": 0,
                    "underestimation_count": 0,
                }
            )
            continue
        rows.append(
            {
                "vault_range": vault_range,
                "n": int(len(sub)),
                "mae_um": float(sub["abs_error_um"].mean()),
                "rmse_um": float(np.sqrt((sub["signed_error_um"] ** 2).mean())),
                "mean_signed_error_um": float(sub["signed_error_um"].mean()),
                "overestimation_count": int((sub["signed_error_um"] > 0).sum()),
                "underestimation_count": int((sub["signed_error_um"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def prediction_range_ratio(predictions: pd.DataFrame) -> float:
    pred_range = float(predictions["pred_vault_um"].max() - predictions["pred_vault_um"].min())
    label_range = float(predictions["vault_label_um"].max() - predictions["vault_label_um"].min())
    return pred_range / label_range if label_range > 0 else float("nan")


def metric_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def write_summary(
    path: Path,
    args: argparse.Namespace,
    counts: Dict[str, int],
    best_epoch: int,
    best_val_mae: float,
    test_metrics: Dict[str, float],
    test_mean_signed_error: float,
    range_metrics: pd.DataFrame,
    pred_range_ratio: float,
) -> None:
    measurement_delta = test_metrics["mae"] - args.measurement_only_rf_mae
    as_oct_delta = test_metrics["mae"] - args.as_oct_ensemble_mae
    lines: List[str] = [
        f"# Combined v4 AS-OCT + measurement fusion seed{args.seed} baseline",
        "",
        "## Cohort",
        f"- Train / val / test eyes: {counts['train']} / {counts['val']} / {counts['test']}",
        "",
        "## Model structure",
        "- Image branch: ResNet18 ImageNet pretrained backbone, final FC replaced by identity.",
        "- Measurement branch: MLP projection from 5 scaled measurement features to 32 dimensions.",
        "- Fusion: concat image feature + measurement feature, followed by an MLP regression head.",
        "- Loss: label-normalized unweighted MSE.",
        "- Checkpoint selection: best validation MAE.",
        "",
        "## Measurement features",
        *[f"- {name}" for name in FEATURE_COLUMNS],
        "",
        "## Scaler policy",
        "- Measurement scaler is fit on the train split only.",
        "- The train-fitted scaler is then applied to val and test.",
        "",
        "## Training selection",
        f"- Best epoch: {best_epoch}",
        f"- Best val MAE: {best_val_mae:.2f} um",
        "",
        "## Test metrics",
        f"- Test MAE: {test_metrics['mae']:.2f} um",
        f"- Test RMSE: {test_metrics['rmse']:.2f} um",
        f"- Test R2: {test_metrics['r2']:.4f}",
        f"- Test mean signed error: {test_mean_signed_error:.2f} um",
        f"- Prediction range / label range ratio: {pred_range_ratio:.3f}",
        "",
        "## Range metrics",
    ]
    for _, row in range_metrics.iterrows():
        if int(row["n"]) == 0:
            lines.append(f"- {row['vault_range']}: n=0")
        else:
            lines.append(
                "- "
                f"{row['vault_range']}: n={int(row['n'])}, "
                f"MAE={metric_float(row['mae_um']):.2f} um, "
                f"RMSE={metric_float(row['rmse_um']):.2f} um, "
                f"mean signed error={metric_float(row['mean_signed_error_um']):.2f} um, "
                f"over={int(row['overestimation_count'])}, "
                f"under={int(row['underestimation_count'])}"
            )
    lines.extend(
        [
            "",
            "## Baseline comparison",
            f"- Measurement-only RF MAE: {args.measurement_only_rf_mae:.2f} um",
            f"- Fusion minus measurement-only RF MAE: {measurement_delta:+.2f} um",
            f"- AS-OCT-only ensemble MAE: {args.as_oct_ensemble_mae:.2f} um",
            f"- Fusion minus AS-OCT-only ensemble MAE: {as_oct_delta:+.2f} um",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_manifest_for_v4(df: pd.DataFrame) -> Dict[str, int]:
    counts = split_counts(df)
    if any(counts[split] <= 0 for split in ["train", "val", "test"]):
        raise ValueError(f"Train, val, and test splits must all be non-empty. Found: {counts}")
    leakage = patient_leakage_count(df)
    if leakage != 0:
        raise ValueError(f"Patient leakage count is {leakage}")
    duplicate_count = duplicate_global_sample_count(df)
    if duplicate_count != 0:
        raise ValueError(f"Duplicate global_sample_id count is {duplicate_count}")
    feature_na = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").isna().sum()
    if int(feature_na.sum()) != 0:
        raise ValueError(f"Missing/non-numeric measurement fields: {feature_na[feature_na > 0].to_dict()}")
    return counts


def split_patient_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {
        split: int(df.loc[df["split"].astype(str).eq(split), "global_patient_uid"].nunique())
        for split in ["train", "val", "test"]
    }


def main() -> None:
    args = parse_args()
    args.manifest = str(resolve_project_path(args.manifest))
    report_dir = resolve_project_path(args.report_dir)
    checkpoint_dir = resolve_project_path(args.checkpoint_dir)

    manifest_df = read_manifest(Path(args.manifest))
    counts = validate_manifest_for_v4(manifest_df)
    patient_counts = split_patient_counts(manifest_df)
    print(f"Manifest path: {args.manifest}", flush=True)
    print(f"train / val / test eyes: {counts['train']} / {counts['val']} / {counts['test']}", flush=True)
    print(
        f"train / val / test patients: {patient_counts['train']} / {patient_counts['val']} / {patient_counts['test']}",
        flush=True,
    )
    print("patient leakage: 0", flush=True)
    print("global_sample_id duplicate: 0", flush=True)
    print(f"measurement features: {', '.join(FEATURE_COLUMNS)}", flush=True)
    if args.check_only:
        print("Check-only complete; no training started.", flush=True)
        return

    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch, nn, DataLoader, Dataset, model_modules = require_torch_stack()
    models, transforms = model_modules
    set_all_seeds(torch, args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)

    label_stats = compute_label_stats(manifest_df)
    feature_mean, feature_std = compute_feature_stats(manifest_df)
    transform = build_transform(transforms, args.image_size)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_seeded_dataloaders(
        manifest_df, args, torch, Dataset, DataLoader, transform, feature_mean, feature_std
    )
    model = build_model(torch, nn, models, args).to(torch_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters available.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("torch.cuda.get_device_name(0): unavailable", flush=True)
    print(f"Current device: {device}", flush=True)
    print(
        f"train / val / test samples: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}",
        flush=True,
    )
    print(
        f"train / val / test patients: {patient_counts['train']} / {patient_counts['val']} / {patient_counts['test']}",
        flush=True,
    )
    print(f"model seed: {args.seed}", flush=True)
    preview_count = min(args.batch_size, len(train_ds))
    first_batch = fusion_collate(torch, [train_ds[index] for index in range(preview_count)])
    print(f"image tensor shape: {tuple(first_batch['oct_images'].shape)}", flush=True)
    print(f"measurement tensor shape: {tuple(first_batch['measurement_features'].shape)}", flush=True)
    print(f"label tensor shape: {tuple(first_batch['vault_labels'].shape)}", flush=True)
    print(
        "measurement train mean: "
        + ", ".join(f"{name}={float(value):.6g}" for name, value in zip(FEATURE_COLUMNS, feature_mean)),
        flush=True,
    )
    print(
        "measurement train std: "
        + ", ".join(f"{name}={float(value):.6g}" for name, value in zip(FEATURE_COLUMNS, feature_std)),
        flush=True,
    )
    print(f"label train mean/std: {label_stats['mean']:.6g} / {label_stats['std']:.6g}", flush=True)
    print("Model: ResNet18 ImageNet + measurement MLP concat fusion", flush=True)
    print("Measurement scaler: fit on train only, apply to val/test", flush=True)
    print("Entering training loop", flush=True)

    output_prefix = f"fusion_v4_seed{args.seed}"
    latest_path = checkpoint_dir / "latest.pth"
    best_path = checkpoint_dir / "best.pth"
    log_path = report_dir / f"{output_prefix}_training_log.csv"
    overall_metrics_path = report_dir / f"{output_prefix}_overall_metrics.csv"
    range_metrics_path = report_dir / f"{output_prefix}_range_metrics.csv"
    predictions_path = report_dir / f"{output_prefix}_predictions.csv"
    summary_path = report_dir / f"{output_prefix}_summary.md"
    if log_path.exists():
        log_path.unlink()

    best_val_mae = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            torch, model, train_loader, optimizer, criterion, torch_device, label_stats["mean"], label_stats["std"], True
        )
        val_metrics, _ = evaluate(
            torch, model, val_loader, criterion, torch_device, label_stats["mean"], label_stats["std"], True
        )
        log_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae_um": train_metrics["mae"],
            "train_rmse_um": train_metrics["rmse"],
            "val_loss": val_metrics["loss"],
            "val_mae_um": val_metrics["mae"],
            "val_rmse_um": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "lr": optimizer.param_groups[0]["lr"],
            "seed": args.seed,
        }
        write_log_row_flush(log_path, log_row, write_header=(epoch == 1))
        save_checkpoint(
            torch, latest_path, model, optimizer, epoch, best_val_mae, label_stats["mean"], label_stats["std"], feature_mean, feature_std, args
        )
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            save_checkpoint(
                torch, best_path, model, optimizer, epoch, best_val_mae, label_stats["mean"], label_stats["std"], feature_mean, feature_std, args
            )
        print(
            f"epoch {epoch:03d} | train_loss={train_metrics['loss']:.6f} | "
            f"val_mae={val_metrics['mae']:.2f} um",
            flush=True,
        )

    checkpoint = torch.load(best_path, map_location=torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_predictions = evaluate(
        torch, model, test_loader, criterion, torch_device, label_stats["mean"], label_stats["std"], True, collect_predictions=True
    )
    if len(test_predictions) != counts["test"]:
        raise ValueError(f"Expected {counts['test']} test predictions, got {len(test_predictions)}")

    test_predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    checked = regression_metrics_np(test_predictions["pred_vault_um"], test_predictions["vault_label_um"])
    test_mean_signed_error = checked["mean_signed_error"]
    range_metrics = range_metrics_from_predictions(test_predictions, args.low_threshold, args.high_threshold)
    pred_range_ratio = prediction_range_ratio(test_predictions)
    range_metrics.to_csv(range_metrics_path, index=False, encoding="utf-8")

    overall = pd.DataFrame(
        [
            {
                "split": "test",
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_val_mae_um": best_val_mae,
                "mae_um": checked["mae"],
                "rmse_um": checked["rmse"],
                "r2": checked["r2"],
                "mean_signed_error_um": test_mean_signed_error,
                "n_samples": len(test_predictions),
                "prediction_min_um": float(test_predictions["pred_vault_um"].min()),
                "prediction_max_um": float(test_predictions["pred_vault_um"].max()),
                "prediction_mean_um": float(test_predictions["pred_vault_um"].mean()),
                "prediction_std_um": float(test_predictions["pred_vault_um"].std()),
                "label_min_um": float(test_predictions["vault_label_um"].min()),
                "label_max_um": float(test_predictions["vault_label_um"].max()),
                "label_mean_um": float(test_predictions["vault_label_um"].mean()),
                "label_std_um": float(test_predictions["vault_label_um"].std()),
                "prediction_range_label_range_ratio": pred_range_ratio,
                "measurement_only_rf_mae_um": args.measurement_only_rf_mae,
                "as_oct_only_ensemble_mae_um": args.as_oct_ensemble_mae,
            }
        ]
    )
    if not math.isfinite(float(overall.loc[0, "r2"])):
        overall.loc[0, "r2"] = float("nan")
    overall.to_csv(overall_metrics_path, index=False, encoding="utf-8")

    write_summary(
        path=summary_path,
        args=args,
        counts=counts,
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        test_metrics={"mae": checked["mae"], "rmse": checked["rmse"], "r2": checked["r2"]},
        test_mean_signed_error=test_mean_signed_error,
        range_metrics=range_metrics,
        pred_range_ratio=pred_range_ratio,
    )

    print("Training complete", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best val MAE: {best_val_mae:.2f} um", flush=True)
    print(
        f"Test metrics: MAE={checked['mae']:.2f} um, RMSE={checked['rmse']:.2f} um, "
        f"R2={checked['r2']:.4f}, mean signed error={test_mean_signed_error:.2f} um",
        flush=True,
    )


if __name__ == "__main__":
    main()

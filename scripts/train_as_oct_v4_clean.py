"""Train combined v4 AS-OCT-only baseline with seed-specific output names.

This launcher trains only the AS-OCT image baseline on an existing split
manifest. It does not create splits, train measurement-only models, or train
fusion models. Outputs are named as as_oct_v4_seed{seed}_*.csv/md, while
checkpoints remain best.pth/latest.pth inside the requested checkpoint_dir.
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
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_as_oct_pod1_baseline import (  # noqa: E402
    build_model,
    build_transform,
    compute_label_stats,
    evaluate,
    range_metrics_from_predictions,
    read_manifest,
    regression_metrics,
    require_torch_stack,
    save_checkpoint,
    train_one_epoch,
)


DEFAULT_MANIFEST = (
    "data/manifests/"
    "vault_as_oct_only_pod1_manifest_combined_batch_01_02_03_04_strict_split_seed42.csv"
)
MEASUREMENT_ONLY_RF_MAE_UM = 169.44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train combined v4 AS-OCT-only clean baseline.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    # Fixed protocol knobs for the v4 AS-OCT-only baseline.
    args.image_size = 224
    args.weight_decay = 1e-4
    args.device = ""
    args.pretrained = True
    args.freeze_backbone = False
    args.label_normalize = True
    args.loss_weight_mode = "none"
    args.low_threshold = 500.0
    args.high_threshold = 800.0
    args.low_weight = 1.0
    args.medium_weight = 1.0
    args.high_weight = 1.0
    args.measurement_only_rf_mae = MEASUREMENT_ONLY_RF_MAE_UM
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
    args: argparse.Namespace,
    transform: Any,
    DataLoader: Any,
    torch_module: Any,
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    from icl_vault.data.collate import collate_vault_batch
    from icl_vault.data.datasets import VaultDataset

    train_dataset = VaultDataset(args.manifest, split="train", oct_transform=transform)
    val_dataset = VaultDataset(args.manifest, split="val", oct_transform=transform)
    test_dataset = VaultDataset(args.manifest, split="test", oct_transform=transform)

    generator = torch_module.Generator()
    generator.manual_seed(args.seed)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate_vault_batch,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def split_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {split: int((df["split"] == split).sum()) for split in ["train", "val", "test"]}


def missing_image_count(df: pd.DataFrame) -> int:
    count = 0
    for value in df["oct_path"].fillna(""):
        if not value or not resolve_project_path(str(value)).exists():
            count += 1
    return count


def patient_leakage_count(df: pd.DataFrame) -> int:
    patient_col = "global_patient_uid" if "global_patient_uid" in df.columns else "patient_id"
    leakage = df.groupby(patient_col)["split"].nunique()
    return int((leakage > 1).sum())


def global_sample_duplicate_count(df: pd.DataFrame) -> int:
    if "global_sample_id" not in df.columns:
        return 0
    return int(df["global_sample_id"].duplicated().sum())


def write_log_row_flush(path: Path, row: Dict[str, Any], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def metric_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def summarize_bias(range_metrics: pd.DataFrame) -> Dict[str, str]:
    out = {
        "low_vault_overestimation": "not_applicable",
        "high_vault_underestimation": "not_applicable",
    }
    for vault_range, key, sign in [
        ("low", "low_vault_overestimation", 1),
        ("high", "high_vault_underestimation", -1),
    ]:
        row = range_metrics[range_metrics["vault_range"] == vault_range]
        if row.empty or int(row.iloc[0]["n_samples"]) == 0:
            continue
        mean_signed_error = metric_float(row.iloc[0]["mean_signed_error_um"])
        out[key] = "yes" if mean_signed_error * sign > 0 else "no"
    return out


def write_summary(
    path: Path,
    seed: int,
    best_epoch: int,
    best_val_mae: float,
    test_metrics: Dict[str, float],
    test_mean_signed_error: float,
    range_metrics: pd.DataFrame,
    bias: Dict[str, str],
    measurement_only_rf_mae: float,
    missing_images: int,
    patient_leakage: int,
    n_test_predictions: int,
) -> None:
    delta = test_metrics["mae"] - measurement_only_rf_mae
    comparison = "better" if delta < 0 else "worse"
    lines: List[str] = [
        f"# Combined v4 AS-OCT-only seed{seed} baseline",
        "",
        "## Training selection",
        f"- Model seed: {seed}",
        f"- Best epoch: {best_epoch}",
        f"- Best val MAE: {best_val_mae:.2f} um",
        "",
        "## Test metrics",
        f"- Test MAE: {test_metrics['mae']:.2f} um",
        f"- Test RMSE: {test_metrics['rmse']:.2f} um",
        f"- Test R2: {test_metrics['r2']:.4f}",
        f"- Test mean signed error: {test_mean_signed_error:.2f} um",
        f"- Test predictions rows: {n_test_predictions}",
        "",
        "## Range metrics",
    ]
    for _, row in range_metrics.iterrows():
        n_samples = int(row["n_samples"])
        if n_samples == 0:
            lines.append(f"- {row['vault_range']}: n=0")
        else:
            lines.append(
                "- "
                f"{row['vault_range']}: n={n_samples}, "
                f"MAE={metric_float(row['mae_um']):.2f} um, "
                f"RMSE={metric_float(row['rmse_um']):.2f} um, "
                f"mean signed error={metric_float(row['mean_signed_error_um']):.2f} um"
            )
    lines.extend(
        [
            "",
            "## Bias pattern",
            f"- Low-vault overestimation: {bias['low_vault_overestimation']}",
            f"- High-vault underestimation: {bias['high_vault_underestimation']}",
            "",
            "## Measurement-only comparison",
            f"- Measurement-only v4 best RF test MAE: {measurement_only_rf_mae:.2f} um",
            f"- AS-OCT-only test MAE difference: {delta:+.2f} um ({comparison} than measurement-only)",
            "",
            "## Manifest QC",
            f"- Missing image: {missing_images}",
            f"- Patient leakage: {patient_leakage}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.manifest = str(resolve_project_path(args.manifest))
    report_dir = resolve_project_path(args.report_dir)
    checkpoint_dir = resolve_project_path(args.checkpoint_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch, nn, DataLoader, models, transforms = require_torch_stack()
    set_all_seeds(torch, args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)

    manifest_df = read_manifest(Path(args.manifest))
    counts = split_counts(manifest_df)
    missing_images = missing_image_count(manifest_df)
    patient_leakage = patient_leakage_count(manifest_df)
    duplicated_samples = global_sample_duplicate_count(manifest_df)
    if missing_images != 0:
        raise ValueError(f"Manifest has missing images: {missing_images}")
    if patient_leakage != 0:
        raise ValueError(f"Manifest has patient leakage count: {patient_leakage}")
    if duplicated_samples != 0:
        raise ValueError(f"Manifest has duplicate global_sample_id count: {duplicated_samples}")

    if any(counts[split] <= 0 for split in ["train", "val", "test"]):
        raise ValueError(f"Train, val, and test splits must all be non-empty. Found: {counts}")

    label_stats = compute_label_stats(manifest_df)
    transform = build_transform(transforms, args.image_size)
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_seeded_dataloaders(
        args=args,
        transform=transform,
        DataLoader=DataLoader,
        torch_module=torch,
    )

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("torch.cuda.get_device_name(0): unavailable", flush=True)
    print(f"Current device: {device}", flush=True)
    print(
        f"train / val / test samples: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}",
        flush=True,
    )
    print(f"Model seed: {args.seed}", flush=True)

    model = build_model(models, nn, pretrained=True, freeze_backbone=False).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    output_prefix = f"as_oct_v4_seed{args.seed}"
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
            torch_module=torch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch_device,
            label_mean=label_stats["mean"],
            label_std=label_stats["std"],
            label_normalize=True,
            args=args,
        )
        val_metrics, _ = evaluate(
            torch_module=torch,
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=torch_device,
            label_mean=label_stats["mean"],
            label_std=label_stats["std"],
            label_normalize=True,
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
            torch_module=torch,
            path=latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_mae=best_val_mae,
            label_mean=label_stats["mean"],
            label_std=label_stats["std"],
            args=args,
        )
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            save_checkpoint(
                torch_module=torch,
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_mae=best_val_mae,
                label_mean=label_stats["mean"],
                label_std=label_stats["std"],
                args=args,
            )

        print(
            f"epoch {epoch:03d} | train_loss={train_metrics['loss']:.6f} | "
            f"val_mae={val_metrics['mae']:.2f} um",
            flush=True,
        )

    checkpoint = torch.load(best_path, map_location=torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_predictions = evaluate(
        torch_module=torch,
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=torch_device,
        label_mean=label_stats["mean"],
        label_std=label_stats["std"],
        label_normalize=True,
        collect_predictions=True,
    )
    if len(test_predictions) != counts["test"]:
        raise ValueError(f"Expected {counts['test']} test predictions, got {len(test_predictions)}")

    test_predictions["signed_error_um"] = test_predictions["pred_vault_um"] - test_predictions["vault_label_um"]
    test_predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    labels = torch.tensor(test_predictions["vault_label_um"].astype(float).to_numpy())
    preds = torch.tensor(test_predictions["pred_vault_um"].astype(float).to_numpy())
    checked_metrics = regression_metrics(torch, preds, labels)
    if any(abs(checked_metrics[key] - test_metrics[key]) > 1e-4 for key in ["mae", "rmse"]):
        raise ValueError("Metric recomputation mismatch.")

    test_mean_signed_error = float(test_predictions["signed_error_um"].mean())
    range_metrics = range_metrics_from_predictions(
        test_predictions,
        split="test",
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    range_metrics.to_csv(range_metrics_path, index=False, encoding="utf-8")
    bias = summarize_bias(range_metrics)

    overall = pd.DataFrame(
        [
            {
                "split": "test",
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_val_mae_um": best_val_mae,
                "mae_um": test_metrics["mae"],
                "rmse_um": test_metrics["rmse"],
                "r2": test_metrics["r2"],
                "mean_signed_error_um": test_mean_signed_error,
                "n_samples": len(test_predictions),
                "measurement_only_v4_best_rf_test_mae_um": args.measurement_only_rf_mae,
                "missing_image": missing_images,
                "patient_leakage": patient_leakage,
                "global_sample_id_duplicate": duplicated_samples,
            }
        ]
    )
    if not math.isfinite(float(overall.loc[0, "r2"])):
        overall.loc[0, "r2"] = float("nan")
    overall.to_csv(overall_metrics_path, index=False, encoding="utf-8")

    write_summary(
        path=summary_path,
        seed=args.seed,
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        test_metrics=test_metrics,
        test_mean_signed_error=test_mean_signed_error,
        range_metrics=range_metrics,
        bias=bias,
        measurement_only_rf_mae=args.measurement_only_rf_mae,
        missing_images=missing_images,
        patient_leakage=patient_leakage,
        n_test_predictions=len(test_predictions),
    )

    print("Training complete", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best val MAE: {best_val_mae:.2f} um", flush=True)
    print(
        "Test metrics: "
        f"MAE={test_metrics['mae']:.2f} um, RMSE={test_metrics['rmse']:.2f} um, "
        f"R2={test_metrics['r2']:.4f}, mean signed error={test_mean_signed_error:.2f} um",
        flush=True,
    )


if __name__ == "__main__":
    main()

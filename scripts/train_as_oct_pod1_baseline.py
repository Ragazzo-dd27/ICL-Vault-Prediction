"""Train a first AS-OCT-only POD1 vault regression smoke baseline.

This is a real-data AS-OCT-only POD1 smoke/baseline experiment. It uses
preoperative AS-OCT raw images to predict manually verified POD1 vault mean.
It is not the final multimodal model and intentionally ignores UBM,
topography, and clinical features.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts/checkpoints/as_oct_pod1_baseline_batch_01"
LOG_DIR = PROJECT_ROOT / "artifacts/logs/as_oct_pod1_baseline_batch_01"
PREDICTION_DIR = PROJECT_ROOT / "artifacts/predictions/as_oct_pod1_baseline_batch_01"
REPORT_DIR = PROJECT_ROOT / "artifacts/reports/as_oct_pod1_baseline_batch_01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AS-OCT-only POD1 vault regression baseline.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean_strict.csv",
        help="Strict AS-OCT-only POD1 manifest.",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="", help="Default: cuda if available else cpu.")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--label_normalize", dest="label_normalize", action="store_true", default=True)
    parser.add_argument("--no_label_normalize", dest="label_normalize", action="store_false")
    parser.add_argument(
        "--loss_weight_mode",
        choices=["none", "vault_range"],
        default="none",
        help="Optional training loss weighting. Default keeps the original unweighted MSE behavior.",
    )
    parser.add_argument("--low_threshold", type=float, default=500.0)
    parser.add_argument("--high_threshold", type=float, default=800.0)
    parser.add_argument("--low_weight", type=float, default=2.0)
    parser.add_argument("--medium_weight", type=float, default=1.0)
    parser.add_argument("--high_weight", type=float, default=1.0)
    parser.add_argument("--run_name", type=str, default="as_oct_pod1_clean_resnet18")
    return parser.parse_args()


def require_torch_stack() -> Tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
    except Exception as exc:
        raise RuntimeError(
            "torch/torchvision are required for baseline training. "
            "Please run this script in the environment where the Dataset/DataLoader smoke test passed."
        ) from exc

    return torch, nn, DataLoader, models, transforms


def add_src_to_path() -> None:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_output_dirs(run_name: str) -> Tuple[Path, Path, Path]:
    run_name = run_name.strip() or "as_oct_pod1_clean_resnet18"
    return CHECKPOINT_DIR / run_name, LOG_DIR / run_name, PREDICTION_DIR / run_name


def run_report_dir(run_name: str) -> Path:
    run_name = run_name.strip() or "as_oct_pod1_clean_resnet18"
    return REPORT_DIR / run_name


def set_seed(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def build_transform(transforms_module: Any, image_size: int) -> Any:
    return transforms_module.Compose(
        [
            transforms_module.Resize((image_size, image_size)),
            transforms_module.ToTensor(),
        ]
    )


def build_model(models_module: Any, nn_module: Any, pretrained: bool, freeze_backbone: bool) -> Any:
    try:
        weights = models_module.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models_module.resnet18(weights=weights)
    except AttributeError:
        model = models_module.resnet18(pretrained=pretrained)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn_module.Linear(in_features, 1)
    return model


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    required = {"sample_id", "patient_id", "eye_side", "split", "oct_path", "vault_label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    return df


def compute_label_stats(df: pd.DataFrame) -> Dict[str, float]:
    train_labels = pd.to_numeric(df.loc[df["split"] == "train", "vault_label"], errors="coerce").dropna()
    if train_labels.empty:
        raise ValueError("Train split has no numeric vault_label values.")
    label_std = float(train_labels.std())
    if not math.isfinite(label_std) or label_std <= 0:
        label_std = 1.0
    return {
        "mean": float(train_labels.mean()),
        "std": label_std,
        "min": float(train_labels.min()),
        "max": float(train_labels.max()),
    }


def compute_vault_range_counts(
    df: pd.DataFrame,
    split: str,
    low_threshold: float,
    high_threshold: float,
) -> Dict[str, int]:
    labels = pd.to_numeric(df.loc[df["split"] == split, "vault_label"], errors="coerce").dropna()
    return {
        "low": int((labels < low_threshold).sum()),
        "medium": int(((labels >= low_threshold) & (labels <= high_threshold)).sum()),
        "high": int((labels > high_threshold).sum()),
    }


def build_dataloaders(args: argparse.Namespace, transform: Any, DataLoader: Any) -> Tuple[Any, Any, Any, Any, Any, Any]:
    add_src_to_path()
    from icl_vault.data.collate import collate_vault_batch
    from icl_vault.data.datasets import VaultDataset

    train_dataset = VaultDataset(args.manifest, split="train", oct_transform=transform)
    val_dataset = VaultDataset(args.manifest, split="val", oct_transform=transform)
    test_dataset = VaultDataset(args.manifest, split="test", oct_transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_vault_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_vault_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_vault_batch,
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def denormalize(values: Any, label_mean: float, label_std: float, enabled: bool) -> Any:
    if not enabled:
        return values
    return values * label_std + label_mean


def normalize(values: Any, label_mean: float, label_std: float, enabled: bool) -> Any:
    if not enabled:
        return values
    return (values - label_mean) / label_std


def regression_metrics(torch_module: Any, preds_um: Any, labels_um: Any) -> Dict[str, float]:
    errors = preds_um - labels_um
    mae = torch_module.mean(torch_module.abs(errors)).item()
    rmse = torch_module.sqrt(torch_module.mean(errors**2)).item()
    label_mean = torch_module.mean(labels_um)
    ss_tot = torch_module.sum((labels_um - label_mean) ** 2)
    if ss_tot.item() <= 0:
        r2 = float("nan")
    else:
        ss_res = torch_module.sum(errors**2)
        r2 = (1.0 - ss_res / ss_tot).item()
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def get_batch_inputs(batch: Dict[str, Any], device: Any) -> Tuple[Any, Any]:
    oct_images = batch.get("oct_images")
    vault_labels = batch.get("vault_labels")
    if oct_images is None:
        raise ValueError("Batch is missing oct_images; AS-OCT-only baseline cannot train.")
    if vault_labels is None:
        raise ValueError("Batch is missing vault_labels; AS-OCT-only baseline cannot train.")
    return oct_images.to(device), vault_labels.to(device)


def weighted_mse_loss(
    torch_module: Any,
    squared_error: Any,
    labels_um: Any,
    args: argparse.Namespace,
) -> Any:
    """Compute optional vault-range weighted MSE on normalized squared errors."""
    if args.loss_weight_mode == "none":
        return torch_module.mean(squared_error)
    if args.loss_weight_mode != "vault_range":
        raise ValueError(f"Unsupported loss_weight_mode: {args.loss_weight_mode}")

    weights = torch_module.full_like(labels_um, float(args.medium_weight))
    weights = torch_module.where(labels_um < args.low_threshold, torch_module.full_like(weights, float(args.low_weight)), weights)
    weights = torch_module.where(labels_um > args.high_threshold, torch_module.full_like(weights, float(args.high_weight)), weights)
    return torch_module.sum(weights * squared_error) / torch_module.clamp(torch_module.sum(weights), min=1e-8)


def train_one_epoch(
    torch_module: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
    label_mean: float,
    label_std: float,
    label_normalize: bool,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[Any] = []
    all_labels: List[Any] = []

    for batch in loader:
        oct_images, labels_um = get_batch_inputs(batch, device=device)
        targets = normalize(labels_um, label_mean, label_std, label_normalize)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(oct_images).squeeze(1)
        squared_error = (outputs - targets) ** 2
        loss = weighted_mse_loss(torch_module, squared_error, labels_um, args)
        if torch_module.isnan(loss):
            raise ValueError("NaN loss encountered during training.")
        loss.backward()
        optimizer.step()

        batch_size = int(labels_um.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_preds.append(denormalize(outputs.detach(), label_mean, label_std, label_normalize).cpu())
        all_labels.append(labels_um.detach().cpu())

    preds_um = torch_module.cat(all_preds)
    labels_um = torch_module.cat(all_labels)
    metrics = regression_metrics(torch_module, preds_um, labels_um)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def evaluate(
    torch_module: Any,
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    label_mean: float,
    label_std: float,
    label_normalize: bool,
    collect_predictions: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[Any] = []
    all_labels: List[Any] = []
    rows: List[Dict[str, Any]] = []

    with torch_module.no_grad():
        for batch in loader:
            oct_images, labels_um = get_batch_inputs(batch, device=device)
            targets = normalize(labels_um, label_mean, label_std, label_normalize)
            outputs = model(oct_images).squeeze(1)
            loss = criterion(outputs, targets)
            if torch_module.isnan(loss):
                raise ValueError("NaN loss encountered during evaluation.")

            preds_um = denormalize(outputs, label_mean, label_std, label_normalize)
            batch_size = int(labels_um.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_preds.append(preds_um.detach().cpu())
            all_labels.append(labels_um.detach().cpu())

            if collect_predictions:
                meta = batch["meta"]
                extras = meta.get("extras", [{} for _ in range(batch_size)])
                for index in range(batch_size):
                    pred = float(preds_um[index].detach().cpu().item())
                    label = float(labels_um[index].detach().cpu().item())
                    extra = extras[index] if index < len(extras) and isinstance(extras[index], dict) else {}
                    rows.append(
                        {
                            "sample_id": meta["sample_id"][index],
                            "patient_id": meta["patient_id"][index],
                            "eye_side": meta["eye_side"][index],
                            "split": meta["split"][index],
                            "vault_label_um": label,
                            "pred_vault_um": pred,
                            "abs_error_um": abs(pred - label),
                            "label_qc_flag": extra.get("label_qc_flag", ""),
                            "oct_path": meta["oct_path"][index],
                        }
                    )

    preds_um = torch_module.cat(all_preds)
    labels_um = torch_module.cat(all_labels)
    metrics = regression_metrics(torch_module, preds_um, labels_um)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics, pd.DataFrame(rows)


def range_metrics_from_predictions(
    predictions: pd.DataFrame,
    split: str,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
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
                    "split": split,
                    "vault_range": vault_range,
                    "n_samples": 0,
                    "mae_um": float("nan"),
                    "rmse_um": float("nan"),
                    "mean_signed_error_um": float("nan"),
                    "median_abs_error_um": float("nan"),
                    "overestimation_count": 0,
                    "underestimation_count": 0,
                }
            )
            continue
        rows.append(
            {
                "split": split,
                "vault_range": vault_range,
                "n_samples": int(len(sub)),
                "mae_um": float(sub["abs_error_um"].mean()),
                "rmse_um": float((sub["signed_error_um"] ** 2).mean() ** 0.5),
                "mean_signed_error_um": float(sub["signed_error_um"].mean()),
                "median_abs_error_um": float(sub["abs_error_um"].median()),
                "overestimation_count": int((sub["signed_error_um"] > 0).sum()),
                "underestimation_count": int((sub["signed_error_um"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def print_range_metrics(range_metrics: pd.DataFrame, split: str) -> None:
    sub = range_metrics[range_metrics["split"] == split]
    if sub.empty:
        return
    for _, row in sub.iterrows():
        if int(row["n_samples"]) == 0:
            print(f"{split} {row['vault_range']} MAE: n=0")
        else:
            print(
                f"{split} {row['vault_range']} MAE: "
                f"{row['mae_um']:.2f} um (n={int(row['n_samples'])})"
            )


def save_checkpoint(
    torch_module: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    epoch: int,
    best_val_mae: float,
    label_mean: float,
    label_std: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_mae": best_val_mae,
            "label_mean": label_mean,
            "label_std": label_std,
            "args": vars(args),
        },
        path,
    )


def write_log_row(path: Path, row: Dict[str, Any], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_start_summary(
    args: argparse.Namespace,
    manifest_df: pd.DataFrame,
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Any,
    label_stats: Dict[str, float],
    device: str,
) -> None:
    labels = pd.to_numeric(manifest_df["vault_label"], errors="coerce").dropna()
    print(f"Manifest path: {args.manifest}")
    print(f"Train/val/test samples: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")
    print(
        "Label stats all splits (um): "
        f"mean={labels.mean():.2f}, std={labels.std():.2f}, min={labels.min():.2f}, max={labels.max():.2f}"
    )
    print(
        "Train label normalization stats (um): "
        f"mean={label_stats['mean']:.2f}, std={label_stats['std']:.2f}, "
        f"min={label_stats['min']:.2f}, max={label_stats['max']:.2f}"
    )
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Label normalize: {args.label_normalize}")
    train_counts = compute_vault_range_counts(
        manifest_df,
        split="train",
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    print(f"Loss weight mode: {args.loss_weight_mode}")
    print(
        "Vault range thresholds/weights: "
        f"low < {args.low_threshold:.1f}, high > {args.high_threshold:.1f}; "
        f"weights low={args.low_weight:.2f}, medium={args.medium_weight:.2f}, high={args.high_weight:.2f}"
    )
    print(
        "Train vault range counts: "
        f"low={train_counts['low']}, medium={train_counts['medium']}, high={train_counts['high']}"
    )


def main() -> None:
    args = parse_args()
    args.manifest = str(resolve_project_path(args.manifest))

    torch, nn, DataLoader, models, transforms = require_torch_stack()
    set_seed(torch, args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)

    manifest_df = read_manifest(Path(args.manifest))
    label_stats = compute_label_stats(manifest_df)
    transform = build_transform(transforms, args.image_size)
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders(
        args=args,
        transform=transform,
        DataLoader=DataLoader,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Train, val, and test splits must all be non-empty.")

    model = build_model(models, nn, pretrained=args.pretrained, freeze_backbone=args.freeze_backbone).to(torch_device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    checkpoint_dir, log_dir, prediction_dir = run_output_dirs(args.run_name)
    report_dir = run_report_dir(args.run_name)
    latest_path = checkpoint_dir / "latest.pth"
    best_path = checkpoint_dir / "best.pth"
    log_path = log_dir / "train_log.csv"
    val_predictions_path = prediction_dir / "val_predictions.csv"
    test_predictions_path = prediction_dir / "test_predictions.csv"
    range_metrics_path = report_dir / "range_metrics.csv"
    for path in (latest_path, best_path, log_path, val_predictions_path, test_predictions_path, range_metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    print_start_summary(
        args=args,
        manifest_df=manifest_df,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        label_stats=label_stats,
        device=device,
    )

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
            label_normalize=args.label_normalize,
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
            label_normalize=args.label_normalize,
        )

        lr = optimizer.param_groups[0]["lr"]
        log_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae_um": train_metrics["mae"],
            "train_rmse_um": train_metrics["rmse"],
            "val_loss": val_metrics["loss"],
            "val_mae_um": val_metrics["mae"],
            "val_rmse_um": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "lr": lr,
            "loss_weight_mode": args.loss_weight_mode,
            "low_threshold": args.low_threshold,
            "high_threshold": args.high_threshold,
            "low_weight": args.low_weight,
            "medium_weight": args.medium_weight,
            "high_weight": args.high_weight,
            "train_low_count": compute_vault_range_counts(manifest_df, "train", args.low_threshold, args.high_threshold)["low"],
            "train_medium_count": compute_vault_range_counts(manifest_df, "train", args.low_threshold, args.high_threshold)["medium"],
            "train_high_count": compute_vault_range_counts(manifest_df, "train", args.low_threshold, args.high_threshold)["high"],
        }
        write_log_row(log_path, log_row, write_header=(epoch == 1))
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
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_mae={train_metrics['mae']:.2f}um "
            f"train_rmse={train_metrics['rmse']:.2f}um | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mae={val_metrics['mae']:.2f}um "
            f"val_rmse={val_metrics['rmse']:.2f}um "
            f"val_r2={val_metrics['r2']:.4f}"
        )

    checkpoint = torch.load(best_path, map_location=torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, val_predictions = evaluate(
        torch_module=torch,
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=torch_device,
        label_mean=label_stats["mean"],
        label_std=label_stats["std"],
        label_normalize=args.label_normalize,
        collect_predictions=True,
    )
    test_metrics, test_predictions = evaluate(
        torch_module=torch,
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=torch_device,
        label_mean=label_stats["mean"],
        label_std=label_stats["std"],
        label_normalize=args.label_normalize,
        collect_predictions=True,
    )
    val_predictions.to_csv(val_predictions_path, index=False, encoding="utf-8")
    test_predictions.to_csv(test_predictions_path, index=False, encoding="utf-8")
    val_range_metrics = range_metrics_from_predictions(
        val_predictions,
        split="val",
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    test_range_metrics = range_metrics_from_predictions(
        test_predictions,
        split="test",
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    range_metrics = pd.concat([val_range_metrics, test_range_metrics], ignore_index=True)
    range_metrics.to_csv(range_metrics_path, index=False, encoding="utf-8")

    print("Training complete")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val MAE: {best_val_mae:.2f} um")
    print(
        "Final test metrics: "
        f"MAE={test_metrics['mae']:.2f} um, "
        f"RMSE={test_metrics['rmse']:.2f} um, "
        f"R2={test_metrics['r2']:.4f}"
    )
    print(f"Overall val MAE: {val_metrics['mae']:.2f} um")
    print(f"Overall test MAE: {test_metrics['mae']:.2f} um")
    print_range_metrics(range_metrics, split="val")
    print_range_metrics(range_metrics, split="test")
    low_test = range_metrics[(range_metrics["split"] == "test") & (range_metrics["vault_range"] == "low")]
    if not low_test.empty and int(low_test.iloc[0]["n_samples"]) > 0:
        print(f"Low-vault test mean signed error: {low_test.iloc[0]['mean_signed_error_um']:.2f} um")
    print(f"Latest checkpoint: {latest_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Best checkpoint: {best_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Train log: {log_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Val predictions: {val_predictions_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Test predictions: {test_predictions_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Range metrics: {range_metrics_path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

"""Train AS-OCT image + true preop measurement concat fusion baseline.

This fusion baseline uses preoperative AS-OCT image and true preoperative
2DAnalysis measurements only. Postoperative 2DAnalysis measurements must not
be used as input features. UBM is not used.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts/checkpoints/fusion_baseline_batch_01_02"
LOG_DIR = PROJECT_ROOT / "artifacts/logs/fusion_baseline_batch_01_02"
PREDICTION_DIR = PROJECT_ROOT / "artifacts/predictions/fusion_baseline_batch_01_02"
FEATURE_COLUMNS = ["cct_mean_um", "acd_epi_mean_mm", "acd_endo_mean_mm", "clr_mean_um", "ata_mean_mm"]
REQUIRED_COLUMNS = [
    "global_sample_id",
    "sample_id",
    "batch_id",
    "global_patient_uid",
    "eye_side",
    "split",
    "oct_path",
    "vault_label",
    "label_qc_flag",
    "measurement_ready_status",
    *FEATURE_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AS-OCT + measurement fusion POD1 baseline.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/vault_as_oct_plus_preop_measurement_pod1_manifest_combined_ready.csv",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="", help="Default: cuda if available else cpu.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--label_normalize", dest="label_normalize", action="store_true", default=True)
    parser.add_argument("--no_label_normalize", dest="label_normalize", action="store_false")
    parser.add_argument("--measurement_hidden_dim", type=int, default=32)
    parser.add_argument("--fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--run_name", type=str, default="combined_fusion_ready_resnet18_measurement_concat_seed42_e30")
    return parser.parse_args()


def require_torch_stack() -> Tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except Exception as exc:
        raise RuntimeError(
            "torch/torchvision are required for fusion baseline training. "
            "Run this script in the environment where the fusion DataLoader smoke test passed."
        ) from exc
    return torch, nn, DataLoader, Dataset, (models, transforms)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def path_exists(path_text: object) -> bool:
    if pd.isna(path_text):
        return False
    text = str(path_text).strip()
    if not text or text.lower() == "nan":
        return False
    return resolve_project_path(text).exists()


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS).difference(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if df["global_sample_id"].duplicated().any():
        raise ValueError("Manifest has duplicated global_sample_id values.")
    if (df.groupby("global_patient_uid")["split"].nunique() > 1).any():
        raise ValueError("global_patient_uid crosses splits.")
    required_splits = {"train", "val", "test"}
    splits = set(df["split"].dropna().astype(str))
    if not required_splits.issubset(splits):
        raise ValueError(f"Manifest split must include train/val/test. Found: {sorted(splits)}")
    if (~df["oct_path"].map(path_exists)).any():
        missing_paths = df.loc[~df["oct_path"].map(path_exists), "oct_path"].head(5).tolist()
        raise FileNotFoundError(f"Some oct_path files do not exist: {missing_paths}")
    labels = pd.to_numeric(df["vault_label"], errors="coerce")
    if (labels.isna() | (labels <= 0)).any():
        raise ValueError("Manifest contains empty or non-positive vault_label values.")
    features = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError("Manifest contains missing measurement feature values.")
    return df


def compute_label_stats(df: pd.DataFrame) -> Dict[str, float]:
    labels = pd.to_numeric(df.loc[df["split"].eq("train"), "vault_label"], errors="coerce").dropna()
    if labels.empty:
        raise ValueError("Train split has no numeric vault_label values.")
    std = float(labels.std())
    if not math.isfinite(std) or std <= 0:
        std = 1.0
    return {"mean": float(labels.mean()), "std": std, "min": float(labels.min()), "max": float(labels.max())}


def compute_feature_stats(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    train_features = df.loc[df["split"].eq("train"), FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    mean = train_features.mean(axis=0).to_numpy(dtype=np.float32)
    std = train_features.std(axis=0).to_numpy(dtype=np.float32)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0).astype(np.float32)
    return mean, std


def build_transform(transforms_module: Any, image_size: int) -> Any:
    return transforms_module.Compose(
        [
            transforms_module.Resize((image_size, image_size)),
            transforms_module.ToTensor(),
        ]
    )


def run_output_dirs(run_name: str) -> Tuple[Path, Path, Path]:
    run_name = run_name.strip() or "combined_fusion_ready_resnet18_measurement_concat_seed42_e30"
    return CHECKPOINT_DIR / run_name, LOG_DIR / run_name, PREDICTION_DIR / run_name


def make_dataset_class(torch_module: Any, Dataset: Any):
    class FusionDataset(Dataset):
        def __init__(self, manifest_df: pd.DataFrame, split: str, transform: Any, feature_mean: np.ndarray, feature_std: np.ndarray):
            self.df = manifest_df[manifest_df["split"].astype(str).eq(split)].reset_index(drop=True).copy()
            self.transform = transform
            self.feature_mean = feature_mean.astype(np.float32)
            self.feature_std = feature_std.astype(np.float32)

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            row = self.df.iloc[index]
            with Image.open(resolve_project_path(row["oct_path"])) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                oct_image = self.transform(image)
            raw_features = pd.to_numeric(row[FEATURE_COLUMNS], errors="coerce").astype(float).to_numpy(dtype=np.float32)
            features = (raw_features - self.feature_mean) / self.feature_std
            if not np.isfinite(features).all():
                raise ValueError(f"NaN/Inf measurement features for {row['global_sample_id']}")
            return {
                "oct_image": oct_image,
                "measurement_features": torch_module.tensor(features, dtype=torch_module.float32),
                "vault_label": torch_module.tensor(float(row["vault_label"]), dtype=torch_module.float32),
                "meta": {
                    "global_sample_id": row["global_sample_id"],
                    "sample_id": row["sample_id"],
                    "batch_id": row["batch_id"],
                    "global_patient_uid": row["global_patient_uid"],
                    "eye_side": row["eye_side"],
                    "split": row["split"],
                    "label_qc_flag": row.get("label_qc_flag", ""),
                    "measurement_ready_status": row.get("measurement_ready_status", ""),
                    "oct_path": row["oct_path"],
                },
            }

    return FusionDataset


def fusion_collate(torch_module: Any, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "oct_images": torch_module.stack([item["oct_image"] for item in batch], dim=0),
        "measurement_features": torch_module.stack([item["measurement_features"] for item in batch], dim=0),
        "vault_labels": torch_module.stack([item["vault_label"] for item in batch], dim=0),
        "meta": [item["meta"] for item in batch],
    }


def build_dataloaders(
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
    collate_fn = lambda batch: fusion_collate(torch_module, batch)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_model(torch_module: Any, nn_module: Any, models_module: Any, args: argparse.Namespace) -> Any:
    class FusionRegressor(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            try:
                weights = models_module.ResNet18_Weights.IMAGENET1K_V1 if args.pretrained else None
                backbone = models_module.resnet18(weights=weights)
            except AttributeError:
                backbone = models_module.resnet18(pretrained=args.pretrained)
            image_dim = backbone.fc.in_features
            backbone.fc = nn_module.Identity()
            if args.freeze_backbone:
                for parameter in backbone.parameters():
                    parameter.requires_grad = False
            self.image_encoder = backbone
            self.measurement_encoder = nn_module.Sequential(
                nn_module.Linear(len(FEATURE_COLUMNS), args.measurement_hidden_dim),
                nn_module.ReLU(),
                nn_module.Dropout(0.1),
                nn_module.Linear(args.measurement_hidden_dim, args.measurement_hidden_dim),
                nn_module.ReLU(),
            )
            self.fusion_head = nn_module.Sequential(
                nn_module.Linear(image_dim + args.measurement_hidden_dim, args.fusion_hidden_dim),
                nn_module.ReLU(),
                nn_module.Dropout(0.2),
                nn_module.Linear(args.fusion_hidden_dim, 1),
            )

        def forward(self, oct_images: Any, measurement_features: Any) -> Any:
            image_features = self.image_encoder(oct_images)
            measurement_features = self.measurement_encoder(measurement_features)
            fused = torch_module.cat([image_features, measurement_features], dim=1)
            return self.fusion_head(fused).squeeze(1)

    return FusionRegressor()


def normalize(values: Any, mean: float, std: float, enabled: bool) -> Any:
    return (values - mean) / std if enabled else values


def denormalize(values: Any, mean: float, std: float, enabled: bool) -> Any:
    return values * std + mean if enabled else values


def regression_metrics(torch_module: Any, preds_um: Any, labels_um: Any) -> Dict[str, float]:
    errors = preds_um - labels_um
    mae = torch_module.mean(torch_module.abs(errors)).item()
    rmse = torch_module.sqrt(torch_module.mean(errors**2)).item()
    ss_tot = torch_module.sum((labels_um - torch_module.mean(labels_um)) ** 2)
    r2 = float("nan") if ss_tot.item() <= 0 else (1.0 - torch_module.sum(errors**2) / ss_tot).item()
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def batch_inputs(batch: Dict[str, Any], device: Any) -> Tuple[Any, Any, Any]:
    oct_images = batch["oct_images"].to(device)
    measurement_features = batch["measurement_features"].to(device)
    labels = batch["vault_labels"].to(device)
    if not (oct_images.isfinite().all() and measurement_features.isfinite().all() and labels.isfinite().all()):
        raise ValueError("Batch contains NaN or Inf.")
    return oct_images, measurement_features, labels


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
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    preds_all: list[Any] = []
    labels_all: list[Any] = []
    for batch in loader:
        oct_images, measurement_features, labels_um = batch_inputs(batch, device)
        targets = normalize(labels_um, label_mean, label_std, label_normalize)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(oct_images, measurement_features)
        loss = criterion(outputs, targets)
        if torch_module.isnan(loss):
            raise ValueError("NaN loss encountered during training.")
        loss.backward()
        optimizer.step()
        batch_size = int(labels_um.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        preds_all.append(denormalize(outputs.detach(), label_mean, label_std, label_normalize).cpu())
        labels_all.append(labels_um.detach().cpu())
    preds = torch_module.cat(preds_all)
    labels = torch_module.cat(labels_all)
    metrics = regression_metrics(torch_module, preds, labels)
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
    preds_all: list[Any] = []
    labels_all: list[Any] = []
    rows: list[dict[str, Any]] = []
    with torch_module.no_grad():
        for batch in loader:
            oct_images, measurement_features, labels_um = batch_inputs(batch, device)
            targets = normalize(labels_um, label_mean, label_std, label_normalize)
            outputs = model(oct_images, measurement_features)
            loss = criterion(outputs, targets)
            if torch_module.isnan(loss):
                raise ValueError("NaN loss encountered during evaluation.")
            preds_um = denormalize(outputs, label_mean, label_std, label_normalize)
            batch_size = int(labels_um.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            preds_all.append(preds_um.detach().cpu())
            labels_all.append(labels_um.detach().cpu())
            if collect_predictions:
                for index, meta in enumerate(batch["meta"]):
                    pred = float(preds_um[index].detach().cpu().item())
                    label = float(labels_um[index].detach().cpu().item())
                    rows.append(
                        {
                            "global_sample_id": meta["global_sample_id"],
                            "sample_id": meta["sample_id"],
                            "batch_id": meta["batch_id"],
                            "global_patient_uid": meta["global_patient_uid"],
                            "eye_side": meta["eye_side"],
                            "split": meta["split"],
                            "vault_label_um": label,
                            "pred_vault_um": pred,
                            "abs_error_um": abs(pred - label),
                            "signed_error_um": pred - label,
                            "label_qc_flag": meta["label_qc_flag"],
                            "measurement_ready_status": meta["measurement_ready_status"],
                            "oct_path": meta["oct_path"],
                        }
                    )
    preds = torch_module.cat(preds_all)
    labels = torch_module.cat(labels_all)
    metrics = regression_metrics(torch_module, preds, labels)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics, pd.DataFrame(rows)


def save_checkpoint(
    torch_module: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    epoch: int,
    best_val_mae: float,
    label_mean: float,
    label_std: float,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
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
            "measurement_feature_names": FEATURE_COLUMNS,
            "measurement_feature_mean": feature_mean.tolist(),
            "measurement_feature_std": feature_std.tolist(),
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
    df: pd.DataFrame,
    train_len: int,
    val_len: int,
    test_len: int,
    label_stats: Dict[str, float],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: str,
) -> None:
    labels = pd.to_numeric(df["vault_label"], errors="coerce")
    print(f"Manifest path: {args.manifest}")
    print(f"Train/val/test samples: {train_len} / {val_len} / {test_len}")
    print(f"Batch distribution: {df['batch_id'].value_counts(dropna=False).to_dict()}")
    print(f"measurement_ready_status distribution: {df['measurement_ready_status'].value_counts(dropna=False).to_dict()}")
    print(
        "Label stats all splits (um): "
        f"mean={labels.mean():.2f}, std={labels.std():.2f}, min={labels.min():.2f}, max={labels.max():.2f}"
    )
    print(
        "Train label stats (um): "
        f"mean={label_stats['mean']:.2f}, std={label_stats['std']:.2f}, "
        f"min={label_stats['min']:.2f}, max={label_stats['max']:.2f}"
    )
    print("Measurement feature train mean/std:")
    for name, mean, std in zip(FEATURE_COLUMNS, feature_mean, feature_std):
        print(f"  {name}: mean={float(mean):.4f}, std={float(std):.4f}")
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Label normalize: {args.label_normalize}")


def main() -> None:
    args = parse_args()
    args.manifest = str(resolve_project_path(args.manifest))
    torch, nn, DataLoader, Dataset, model_modules = require_torch_stack()
    models, transforms = model_modules
    set_seed(torch, args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)

    manifest_df = read_manifest(Path(args.manifest))
    label_stats = compute_label_stats(manifest_df)
    feature_mean, feature_std = compute_feature_stats(manifest_df)
    transform = build_transform(transforms, args.image_size)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        manifest_df, args, torch, Dataset, DataLoader, transform, feature_mean, feature_std
    )
    model = build_model(torch, nn, models, args).to(torch_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters available.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    checkpoint_dir, log_dir, prediction_dir = run_output_dirs(args.run_name)
    latest_path = checkpoint_dir / "latest.pth"
    best_path = checkpoint_dir / "best.pth"
    log_path = log_dir / "train_log.csv"
    val_predictions_path = prediction_dir / "val_predictions.csv"
    test_predictions_path = prediction_dir / "test_predictions.csv"
    for path in (latest_path, best_path, log_path, val_predictions_path, test_predictions_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    print_start_summary(
        args, manifest_df, len(train_ds), len(val_ds), len(test_ds), label_stats, feature_mean, feature_std, device
    )

    best_val_mae = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            torch, model, train_loader, optimizer, criterion, torch_device, label_stats["mean"], label_stats["std"], args.label_normalize
        )
        val_metrics, _ = evaluate(
            torch, model, val_loader, criterion, torch_device, label_stats["mean"], label_stats["std"], args.label_normalize
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
        }
        write_log_row(log_path, log_row, write_header=(epoch == 1))
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
            f"epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} "
            f"train_mae={train_metrics['mae']:.2f}um train_rmse={train_metrics['rmse']:.2f}um | "
            f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae']:.2f}um "
            f"val_rmse={val_metrics['rmse']:.2f}um val_r2={val_metrics['r2']:.4f}"
        )

    checkpoint = torch.load(best_path, map_location=torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, val_predictions = evaluate(
        torch, model, val_loader, criterion, torch_device, label_stats["mean"], label_stats["std"], args.label_normalize, collect_predictions=True
    )
    test_metrics, test_predictions = evaluate(
        torch, model, test_loader, criterion, torch_device, label_stats["mean"], label_stats["std"], args.label_normalize, collect_predictions=True
    )
    val_predictions.to_csv(val_predictions_path, index=False, encoding="utf-8")
    test_predictions.to_csv(test_predictions_path, index=False, encoding="utf-8")

    print("Training complete")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val MAE: {best_val_mae:.2f} um")
    print(f"Final test metrics: MAE={test_metrics['mae']:.2f} um, RMSE={test_metrics['rmse']:.2f} um, R2={test_metrics['r2']:.4f}")
    print(f"Latest checkpoint: {latest_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Best checkpoint: {best_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Train log: {log_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Val predictions: {val_predictions_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Test predictions: {test_predictions_path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

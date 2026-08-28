"""G2 reliability-aware soft gate pilot for frozen v5.2 matched experts.

This script prepares a deliberately small prediction-level fusion experiment.
It never retrains AS-OCT. Gate preprocessing and fitting use matched validation
eyes only; test labels are read only after the gate is frozen for evaluation.

The formal five-split variant uses AS-OCT view-wise prediction dispersion from
the frozen matched AS-OCT V0 model_seed=42 checkpoint for each outer split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="A NumPy version .* is required for this version of SciPy")
warnings.filterwarnings("ignore", message="X has feature names, but DecisionTreeRegressor was fitted without feature names")

from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


SEEDS = [42, 1001, 2002, 2026, 3407]
AS_OCT_MODEL_SEED = 42
FEATURES = ["cct_mean_um", "acd_epi_mean_mm", "acd_endo_mean_mm", "clr_mean_um", "ata_mean_mm"]
GATE_FEATURES = [
    "as_oct_pred_um",
    "measurement_pred_um",
    "disagreement_um",
    "as_oct_view_dispersion_imputed_um",
    "as_oct_single_view_flag",
    "measurement_tree_dispersion_um",
]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "v5_2_matched_fusion_audit" / "reliability_aware_gate_multiview_v1"
MATCHED_ROOT = ROOT / "artifacts" / "v5_2_matched_unimodal"
AUDIT_ROOT = ROOT / "artifacts" / "v5_2_matched_fusion_audit"
SPLIT_ROOT = ROOT / "data" / "splits" / "v5"
FUSION_MANIFEST = ROOT / "data" / "manifests" / "vault_as_oct_plus_measurement_pod1_manifest_v5_ready.csv"
LOW = 250.0
HIGH = 750.0


@dataclass
class PreflightResult:
    ok: bool
    lines: list[str]
    artifacts: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or preflight the G2 reliability-aware soft gate pilot.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Check paths, coverage, split alignment, and prerequisites only.")
    mode.add_argument("--run", action="store_true", help="Run the gate experiment from frozen predictions and checkpoints.")
    parser.add_argument("--outer-split-seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached/generated G2 outputs.")
    parser.add_argument("--device", default="", help="Device for optional AS-OCT inference-only cache generation.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def split_path(seed: int) -> Path:
    return SPLIT_ROOT / f"fusion_manifest_split_seed{seed}.csv"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def split_paths(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def oct_view_paths(row: pd.Series) -> list[str]:
    paths = split_paths(row.get("oct_paths", ""))
    if not paths:
        selected = str(row.get("oct_path", "") if pd.notna(row.get("oct_path", "")) else "").strip()
        paths = [selected] if selected else []
    return list(dict.fromkeys(paths))


def eye_norm(value: Any) -> str:
    return {"R": "OD", "L": "OS"}.get(str(value), str(value))


def vault_range(value: float) -> str:
    if value < LOW:
        return "low"
    if value <= HIGH:
        return "medium"
    return "high"


def mae(y_true: Any, y_pred: Any) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(p[mask] - y[mask])))


def std0(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan")
    return float(np.std(arr, ddof=0))


def standardize_prediction(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    if "prediction_um" not in out.columns and "pred_vault_um" in out.columns:
        out["prediction_um"] = out["pred_vault_um"]
    if "ground_truth_um" not in out.columns:
        if "vault_um" in out.columns:
            out["ground_truth_um"] = out["vault_um"]
        elif "vault_label_um" in out.columns:
            out["ground_truth_um"] = out["vault_label_um"]
    if "eye" not in out.columns and "eye_side" in out.columns:
        out["eye"] = out["eye_side"].map({"R": "OD", "L": "OS"}).fillna(out["eye_side"])
    if "global_sample_id" not in out.columns:
        out["global_sample_id"] = out["sample_id"].astype(str)
    required = {"split_seed", "sample_id", "patient_id", "eye", "ground_truth_um", "prediction_um", "split"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")
    out["eye"] = out["eye"].map(eye_norm)
    out["ground_truth_um"] = pd.to_numeric(out["ground_truth_um"], errors="coerce")
    out["prediction_um"] = pd.to_numeric(out["prediction_um"], errors="coerce")
    return out


def load_base_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meas = standardize_prediction(
        pd.read_csv(MATCHED_ROOT / "measurement" / "measurement_matched_validation_predictions.csv"), "measurement validation"
    )
    meas_test = standardize_prediction(
        pd.read_csv(MATCHED_ROOT / "measurement" / "measurement_matched_repeated_predictions.csv"), "measurement test"
    )
    asv = standardize_prediction(
        pd.read_csv(MATCHED_ROOT / "as_oct" / "as_oct_matched_validation_predictions.csv"), "AS-OCT validation"
    )
    ast = standardize_prediction(
        pd.read_csv(MATCHED_ROOT / "as_oct" / "as_oct_matched_test_predictions.csv"), "AS-OCT test"
    )
    return meas, meas_test, pd.concat([asv, ast], ignore_index=True)


def load_split(seed: int) -> pd.DataFrame:
    df = pd.read_csv(split_path(seed))
    df["eye"] = df["eye"].map(eye_norm)
    df["vault_um"] = pd.to_numeric(df["vault_um"], errors="coerce")
    return df


def duplicate_eye_count(df: pd.DataFrame) -> int:
    cols = ["global_sample_id"] if "global_sample_id" in df.columns else ["sample_id"]
    return int(df.duplicated(cols).sum())


def patient_leakage_count(df: pd.DataFrame) -> int:
    patient_col = "global_patient_uid" if "global_patient_uid" in df.columns else "patient_id"
    return int((df.groupby(patient_col)["split"].nunique() > 1).sum())


def as_oct_view_prediction_file(seed: int, split: str, out_dir: Path) -> Path:
    return out_dir / "cache" / f"as_oct_multiview_predictions_split_seed{seed}_model_seed42_{split}.csv"


def matched_as_oct_checkpoint(seed: int, model_seed: int = AS_OCT_MODEL_SEED) -> Path | None:
    direct = MATCHED_ROOT / "as_oct" / "checkpoints" / f"as_oct_matched_split_seed{seed}_model_seed{model_seed}" / "best.pth"
    if direct.exists():
        return direct
    table = ROOT / "artifacts" / "v5_as_oct_baselines" / "as_oct_checkpoints_v5.csv"
    if table.exists():
        ckpts = pd.read_csv(table)
        hit = ckpts[(ckpts["split_seed"].eq(seed)) & (ckpts["model_seed"].eq(model_seed))]
        if len(hit) == 1:
            p = ROOT / str(hit.iloc[0]["checkpoint"])
            if p.exists():
                return p
    return None


class OctViewDataset:
    def __init__(self, rows: list[dict[str, Any]], transform: Any, torch_module: Any) -> None:
        self.rows = rows
        self.transform = transform
        self.torch = torch_module

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        row = self.rows[index]
        with Image.open(resolve(row["oct_path"])).convert("RGB") as image:
            tensor = self.transform(image)
        return {
            "oct_image": tensor,
            "vault_label": self.torch.tensor(float(row["ground_truth_um"]), dtype=self.torch.float32),
            "meta": row,
        }


def collate_oct_views(batch: list[dict[str, Any]], torch_module: Any) -> dict[str, Any]:
    return {
        "oct_images": torch_module.stack([item["oct_image"] for item in batch], dim=0),
        "vault_labels": torch_module.stack([item["vault_label"] for item in batch]),
        "meta": [item["meta"] for item in batch],
    }


def infer_as_oct_view_predictions(seed: int, split: str, out_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    cache = as_oct_view_prediction_file(seed, split, out_dir)
    if cache.exists() and not args.force:
        return pd.read_csv(cache)
    ckpt_path = matched_as_oct_checkpoint(seed, AS_OCT_MODEL_SEED)
    if ckpt_path is None:
        raise FileNotFoundError(f"Missing frozen matched AS-OCT checkpoint for split_seed={seed}, model_seed=42")

    from train_as_oct_pod1_baseline import build_model, build_transform, require_torch_stack

    torch, nn, DataLoader, models, transforms = require_torch_stack()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(AS_OCT_MODEL_SEED)
    np.random.seed(AS_OCT_MODEL_SEED)
    torch.manual_seed(AS_OCT_MODEL_SEED)

    split_df = load_split(seed)
    target = split_df[split_df["split"].eq(split)].copy()
    image_rows: list[dict[str, Any]] = []
    for _, row in target.iterrows():
        paths = oct_view_paths(row)
        for view_index, path_text in enumerate(paths):
            image_rows.append(
                {
                    "split_seed": seed,
                    "model_seed": AS_OCT_MODEL_SEED,
                    "split": split,
                    "global_sample_id": row.get("global_sample_id", ""),
                    "sample_id": row["sample_id"],
                    "patient_id": row["patient_id"],
                    "global_patient_uid": row.get("global_patient_uid", row["patient_id"]),
                    "eye": eye_norm(row["eye"]),
                    "eye_side": row.get("eye_side", ""),
                    "ground_truth_um": float(row["vault_um"]),
                    "view_index": view_index,
                    "oct_path": path_text,
                    "is_frozen_protocol_selected_view": str(path_text) == str(row.get("oct_path", "")),
                }
            )

    checkpoint = torch.load(ckpt_path, map_location=device)
    train_labels = pd.to_numeric(split_df.loc[split_df["split"].eq("train"), "vault_label"], errors="coerce")
    label_mean = float(checkpoint.get("label_mean", train_labels.mean()))
    label_std = float(checkpoint.get("label_std", train_labels.std()))
    if not math.isfinite(label_std) or label_std <= 0:
        label_std = 1.0

    transform = build_transform(transforms, 224)
    dataset = OctViewDataset(image_rows, transform, torch)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_oct_views(batch, torch),
    )
    model = build_model(models, nn, pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["oct_images"].to(device)
            labels = batch["vault_labels"].to(device)
            raw = model(images).squeeze(1)
            preds = raw * label_std + label_mean
            for meta, pred, label in zip(batch["meta"], preds.detach().cpu().numpy(), labels.detach().cpu().numpy()):
                rows.append(
                    {
                        **meta,
                        "ground_truth_um": float(label),
                        "view_prediction_um": float(pred),
                        "checkpoint_path": rel(ckpt_path),
                    }
                )
    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def as_oct_view_dispersion(seed: int, split: str, out_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    cache = as_oct_view_prediction_file(seed, split, out_dir)
    if not cache.exists() and not args.run:
        raise RuntimeError(f"AS-OCT multi-view prediction cache is absent for split_seed={seed}, split={split}; --run will create it by inference-only.")
    views = pd.read_csv(cache) if cache.exists() and not args.force else infer_as_oct_view_predictions(seed, split, out_dir, args)
    if views.empty:
        raise RuntimeError(f"No AS-OCT view-wise predictions for split_seed={seed}, split={split}")
    summary = (
        views.groupby(["split_seed", "split", "sample_id"], as_index=False)
        .agg(
            as_oct_view_dispersion_raw_um=("view_prediction_um", lambda s: std0(pd.to_numeric(s, errors="coerce"))),
            n_as_oct_views=("view_prediction_um", "count"),
        )
    )
    return summary


def measurement_tree_dispersion(seed: int, split: str) -> pd.DataFrame:
    df = load_split(seed)
    train = df[df["split"].eq("train")].copy()
    target = df[df["split"].eq(split)].copy()
    model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42)
    model.fit(train[FEATURES], train["vault_um"].astype(float))
    target_x = target[FEATURES].to_numpy(dtype=float)
    tree_preds = np.vstack([tree.predict(target_x) for tree in model.estimators_])
    out = target[["split_seed", "split", "sample_id"]].copy()
    out["measurement_tree_dispersion_um"] = np.std(tree_preds, axis=0, ddof=0)
    return out


def build_gate_dataset(seed: int, split: str, out_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    meas_val, meas_test, as_all = load_base_predictions()
    meas = meas_val if split == "val" else meas_test
    meas = meas[(meas["split_seed"].eq(seed)) & (meas["split"].eq(split))]
    as_oct = as_all[(as_all["split_seed"].eq(seed)) & (as_all["split"].eq(split))]
    keys = ["split_seed", "split", "sample_id", "patient_id", "eye", "ground_truth_um"]
    merged = meas[keys + ["global_sample_id", "prediction_um"]].rename(columns={"prediction_um": "measurement_pred_um"}).merge(
        as_oct[keys + ["prediction_um"]].rename(columns={"prediction_um": "as_oct_pred_um"}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    split_df = load_split(seed)
    split_ids = split_df[split_df["split"].eq(split)][["sample_id", "global_patient_uid", "eye_side", "vault_range"]]
    merged = merged.merge(split_ids, on="sample_id", how="left", validate="one_to_one")
    merged["disagreement_um"] = (merged["as_oct_pred_um"] - merged["measurement_pred_um"]).abs()
    merged = merged.merge(as_oct_view_dispersion(seed, split, out_dir, args), on=["split_seed", "split", "sample_id"], how="left")
    merged = merged.merge(measurement_tree_dispersion(seed, split), on=["split_seed", "split", "sample_id"], how="left")
    merged["as_oct_single_view_flag"] = (merged["n_as_oct_views"].astype(float) == 1.0).astype(int)
    return merged


def apply_validation_median_dispersion_imputation(val: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    multiview = val.loc[val["n_as_oct_views"].astype(float) >= 2, "as_oct_view_dispersion_raw_um"]
    impute_value = float(pd.to_numeric(multiview, errors="coerce").median())
    if not math.isfinite(impute_value):
        raise RuntimeError("Validation multi-view median AS-OCT dispersion is not finite.")
    out_frames = []
    for frame in [val, test]:
        out = frame.copy()
        out["validation_dispersion_imputation_value_um"] = impute_value
        out["as_oct_view_dispersion_imputed_um"] = pd.to_numeric(out["as_oct_view_dispersion_raw_um"], errors="coerce").fillna(impute_value)
        out["as_oct_single_view_flag"] = (out["n_as_oct_views"].astype(float) == 1.0).astype(int)
        out_frames.append(out)
    return out_frames[0], out_frames[1], impute_value


def fit_scaler(x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_val, axis=0)
    scale = np.std(x_val, axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return mean, scale


def fit_gate(val: pd.DataFrame) -> dict[str, Any]:
    x = val[GATE_FEATURES].to_numpy(dtype=float)
    y = val["ground_truth_um"].to_numpy(dtype=float)
    as_pred = val["as_oct_pred_um"].to_numpy(dtype=float)
    meas_pred = val["measurement_pred_um"].to_numpy(dtype=float)
    mean, scale = fit_scaler(x)
    xs = (x - mean) / scale
    w = np.zeros(xs.shape[1], dtype=float)
    b = 0.0
    lr = 0.03
    l2 = 1e-4
    beta = 25.0
    n_iter = 2500
    for _ in range(n_iter):
        logits = np.clip(xs @ w + b, -35.0, 35.0)
        alpha = 1.0 / (1.0 + np.exp(-logits))
        pred = alpha * as_pred + (1.0 - alpha) * meas_pred
        err = pred - y
        abs_err = np.abs(err)
        dloss = np.where(abs_err < beta, err / beta, np.sign(err)) / len(y)
        dalpha = as_pred - meas_pred
        dlogit = dloss * dalpha * alpha * (1.0 - alpha)
        grad_w = xs.T @ dlogit + l2 * w
        grad_b = float(np.sum(dlogit))
        w -= lr * grad_w
        b -= lr * grad_b
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": w.tolist(),
        "bias": float(b),
        "optimizer": "fixed full-batch gradient descent",
        "loss": "Smooth-L1(beta=25 um) on fused vault prediction",
        "learning_rate": lr,
        "l2": l2,
        "iterations": n_iter,
    }


def apply_gate(df: pd.DataFrame, gate: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    x = out[GATE_FEATURES].to_numpy(dtype=float)
    mean = np.asarray(gate["feature_mean"], dtype=float)
    scale = np.asarray(gate["feature_scale"], dtype=float)
    w = np.asarray(gate["weights"], dtype=float)
    b = float(gate["bias"])
    logits = np.clip(((x - mean) / scale) @ w + b, -35.0, 35.0)
    out["alpha"] = 1.0 / (1.0 + np.exp(-logits))
    out["g2_pred_um"] = out["alpha"] * out["as_oct_pred_um"] + (1.0 - out["alpha"]) * out["measurement_pred_um"]
    return out


def fixed_additive_pred(df: pd.DataFrame, alpha: float) -> pd.Series:
    return alpha * df["as_oct_pred_um"] + (1.0 - alpha) * df["measurement_pred_um"]


def oracle_fraction(best_mae: float, g2_mae: float, oracle_mae: float) -> float:
    denom = best_mae - oracle_mae
    if not math.isfinite(denom) or abs(denom) <= 1e-12:
        return float("nan")
    return float((best_mae - g2_mae) / denom)


def run_experiment(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.glob("*.csv")) and not args.force:
        raise FileExistsError(f"{rel(out_dir)} already has outputs; use --force to overwrite G2 outputs.")
    pre = preflight(args, write_report=True)
    if not pre.ok:
        raise RuntimeError("PRECHECK FAILED; see preflight_report.txt")
    alpha = float(read_json(AUDIT_ROOT / "additive_fusion_frozen_alpha.json")["alpha"])
    per_split = []
    per_eye = []
    coeffs = []
    feature_summaries = []
    config = {
        "outer_split_seeds": args.outer_split_seeds,
        "gate_inputs": GATE_FEATURES,
        "as_oct_reliability": "view-wise prediction dispersion",
        "single_view_handling": "validation-median imputation + single-view indicator",
        "as_oct_reliability_proxy": "AS-OCT view-wise prediction dispersion / multi-view consistency reliability proxy",
        "as_oct_checkpoint_protocol": "Frozen matched AS-OCT V0 model_seed=42 checkpoint per outer split; inference-only over all available preoperative AS-OCT views.",
        "measurement_reliability_proxy": "RF tree prediction dispersion",
        "fixed_additive_alpha": alpha,
        "leakage_prevention": "Scaler and gate fit on matched validation only; test labels used only for final evaluation.",
    }
    write_json(out_dir / "experiment_config.json", config)
    for seed in args.outer_split_seeds:
        val = build_gate_dataset(seed, "val", out_dir, args)
        test = build_gate_dataset(seed, "test", out_dir, args)
        val, test, dispersion_imputation_value = apply_validation_median_dispersion_imputation(val, test)
        gate = fit_gate(val)
        test = apply_gate(test, gate)
        test["fixed_additive_pred_um"] = fixed_additive_pred(test, alpha)
        test["oracle_best_of_two_pred_um"] = np.where(
            (test["as_oct_pred_um"] - test["ground_truth_um"]).abs() <= (test["measurement_pred_um"] - test["ground_truth_um"]).abs(),
            test["as_oct_pred_um"],
            test["measurement_pred_um"],
        )
        test["measurement_abs_error_um"] = (test["measurement_pred_um"] - test["ground_truth_um"]).abs()
        test["as_oct_abs_error_um"] = (test["as_oct_pred_um"] - test["ground_truth_um"]).abs()
        test["fixed_additive_abs_error_um"] = (test["fixed_additive_pred_um"] - test["ground_truth_um"]).abs()
        test["g2_abs_error_um"] = (test["g2_pred_um"] - test["ground_truth_um"]).abs()
        test["oracle_best_of_two_abs_error_um"] = (test["oracle_best_of_two_pred_um"] - test["ground_truth_um"]).abs()
        measurement_mae = float(test["measurement_abs_error_um"].mean())
        as_mae = float(test["as_oct_abs_error_um"].mean())
        best_mae = min(measurement_mae, as_mae)
        fixed_mae = float(test["fixed_additive_abs_error_um"].mean())
        g2_mae = float(test["g2_abs_error_um"].mean())
        oracle_mae = float(test["oracle_best_of_two_abs_error_um"].mean())
        high = test[test["ground_truth_um"] > HIGH]
        multi = test[test["n_as_oct_views"].astype(float) >= 2]
        per_split.append(
            {
                "split_seed": seed,
                "n_val": int(len(val)),
                "n_test": int(len(test)),
                "validation_dispersion_imputation_value_um": dispersion_imputation_value,
                "measurement_rf_test_mae_um": measurement_mae,
                "as_oct_v0_test_mae_um": as_mae,
                "matched_best_unimodal_mae_um": best_mae,
                "fixed_additive_fusion_test_mae_um": fixed_mae,
                "g2_test_mae_um": g2_mae,
                "g2_delta_vs_matched_best_unimodal_um": g2_mae - best_mae,
                "g2_delta_vs_fixed_additive_fusion_um": g2_mae - fixed_mae,
                "measurement_rf_high_vault_mae_um": float(high["measurement_abs_error_um"].mean()) if len(high) else float("nan"),
                "as_oct_high_vault_mae_um": float(high["as_oct_abs_error_um"].mean()) if len(high) else float("nan"),
                "fixed_additive_high_vault_mae_um": float(high["fixed_additive_abs_error_um"].mean()) if len(high) else float("nan"),
                "g2_high_vault_mae_um": float(high["g2_abs_error_um"].mean()) if len(high) else float("nan"),
                "oracle_high_vault_mae_um": float(high["oracle_best_of_two_abs_error_um"].mean()) if len(high) else float("nan"),
                "oracle_best_of_two_mae_um": oracle_mae,
                "oracle_fraction_captured": oracle_fraction(best_mae, g2_mae, oracle_mae),
                "g2_beats_matched_best_unimodal": bool(g2_mae < best_mae),
                "g2_beats_fixed_additive": bool(g2_mae < fixed_mae),
                "alpha_mean": float(test["alpha"].mean()),
                "alpha_std": float(test["alpha"].std(ddof=0)),
                "alpha_median": float(test["alpha"].median()),
                "alpha_min": float(test["alpha"].min()),
                "alpha_max": float(test["alpha"].max()),
                "fraction_alpha_lt_0_05": float((test["alpha"] < 0.05).mean()),
                "fraction_alpha_gt_0_95": float((test["alpha"] > 0.95).mean()),
                "multiview_test_n": int(len(multi)),
                "multiview_measurement_rf_mae_um": float(multi["measurement_abs_error_um"].mean()) if len(multi) else float("nan"),
                "multiview_as_oct_mae_um": float(multi["as_oct_abs_error_um"].mean()) if len(multi) else float("nan"),
                "multiview_fixed_additive_mae_um": float(multi["fixed_additive_abs_error_um"].mean()) if len(multi) else float("nan"),
                "multiview_g2_mae_um": float(multi["g2_abs_error_um"].mean()) if len(multi) else float("nan"),
                "multiview_oracle_mae_um": float(multi["oracle_best_of_two_abs_error_um"].mean()) if len(multi) else float("nan"),
            }
        )
        for name, weight, mean, scale in zip(GATE_FEATURES, gate["weights"], gate["feature_mean"], gate["feature_scale"]):
            coeffs.append({"split_seed": seed, "feature": name, "coefficient": weight, "validation_scaler_mean": mean, "validation_scaler_scale": scale})
        coeffs.append({"split_seed": seed, "feature": "intercept", "coefficient": gate["bias"], "validation_scaler_mean": np.nan, "validation_scaler_scale": np.nan})
        for split_name, frame in [("val", val), ("test", test)]:
            for feature in [
                "as_oct_view_dispersion_raw_um",
                "as_oct_view_dispersion_imputed_um",
                "as_oct_single_view_flag",
                "measurement_tree_dispersion_um",
                "disagreement_um",
                "n_as_oct_views",
            ]:
                feature_summaries.append(
                    {
                        "split_seed": seed,
                        "split": split_name,
                        "feature": feature,
                        "n": int(frame[feature].notna().sum()),
                        "mean": float(frame[feature].mean()),
                        "std": float(frame[feature].std(ddof=0)),
                        "min": float(frame[feature].min()),
                        "max": float(frame[feature].max()),
                    }
                )
        per_eye.append(test)
    per_split_df = pd.DataFrame(per_split)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_split_df.to_csv(out_dir / "per_split_metrics.csv", index=False)
    pd.concat(per_eye, ignore_index=True).to_csv(out_dir / "per_eye_predictions.csv", index=False)
    pd.DataFrame(coeffs).to_csv(out_dir / "gate_coefficients.csv", index=False)
    pd.DataFrame(feature_summaries).to_csv(out_dir / "reliability_feature_summary.csv", index=False)
    aggregate = pd.DataFrame(
        [
            {
                "n_splits": int(len(per_split_df)),
                "g2_mae_mean_um": float(per_split_df["g2_test_mae_um"].mean()),
                "g2_mae_std_um": float(per_split_df["g2_test_mae_um"].std(ddof=1)) if len(per_split_df) > 1 else float("nan"),
                "wins_vs_matched_best_unimodal": int(per_split_df["g2_beats_matched_best_unimodal"].sum()),
                "wins_vs_fixed_additive": int(per_split_df["g2_beats_fixed_additive"].sum()),
                "mean_measurement_rf_high_vault_mae_um": float(per_split_df["measurement_rf_high_vault_mae_um"].mean()),
                "mean_as_oct_high_vault_mae_um": float(per_split_df["as_oct_high_vault_mae_um"].mean()),
                "mean_fixed_additive_high_vault_mae_um": float(per_split_df["fixed_additive_high_vault_mae_um"].mean()),
                "mean_g2_high_vault_mae_um": float(per_split_df["g2_high_vault_mae_um"].mean()),
                "mean_oracle_high_vault_mae_um": float(per_split_df["oracle_high_vault_mae_um"].mean()),
                "mean_oracle_fraction_captured": float(per_split_df["oracle_fraction_captured"].mean()),
                "mean_multiview_g2_mae_um": float(per_split_df["multiview_g2_mae_um"].mean()),
            }
        ]
    )
    aggregate.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    write_report(out_dir, per_split_df, aggregate)


def view_coverage_table(split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in split.iterrows():
        paths = oct_view_paths(row)
        missing = [p for p in paths if not resolve(p).exists()]
        rows.append(
            {
                "split_seed": int(row["split_seed"]),
                "split": row["split"],
                "global_sample_id": row.get("global_sample_id", row["sample_id"]),
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "global_patient_uid": row.get("global_patient_uid", row["patient_id"]),
                "eye": eye_norm(row["eye"]),
                "n_as_oct_views": len(paths),
                "missing_view_count": len(missing),
                "view_paths": ";".join(paths),
                "missing_view_paths": ";".join(missing),
            }
        )
    return pd.DataFrame(rows)


def fmt_ids(values: pd.Series) -> str:
    return "; ".join(values.astype(str).tolist())


def preflight(args: argparse.Namespace, write_report: bool = True) -> PreflightResult:
    out_dir = Path(args.out_dir)
    lines: list[str] = ["G2 Reliability-Aware Soft Gate multi-view preflight", ""]
    artifacts = {
        "fusion_manifest": rel(FUSION_MANIFEST),
        "split_dir": rel(SPLIT_ROOT),
        "matched_unimodal_root": rel(MATCHED_ROOT),
        "matched_audit_root": rel(AUDIT_ROOT),
    }
    failures: list[str] = []
    for seed in args.outer_split_seeds:
        if seed not in SEEDS:
            failures.append(f"Unexpected outer split seed requested: {seed}")
    required_paths = [
        FUSION_MANIFEST,
        MATCHED_ROOT / "measurement" / "measurement_frozen_configuration.json",
        MATCHED_ROOT / "measurement" / "measurement_matched_validation_predictions.csv",
        MATCHED_ROOT / "measurement" / "measurement_matched_repeated_predictions.csv",
        MATCHED_ROOT / "as_oct" / "as_oct_frozen_configuration.json",
        MATCHED_ROOT / "as_oct" / "as_oct_matched_validation_predictions.csv",
        MATCHED_ROOT / "as_oct" / "as_oct_matched_test_predictions.csv",
        AUDIT_ROOT / "matched_prediction_coverage_summary.csv",
        AUDIT_ROOT / "interaction_synergy_summary.csv",
        AUDIT_ROOT / "oracle_best_of_two_summary.csv",
        AUDIT_ROOT / "additive_fusion_frozen_alpha.json",
    ]
    for path in required_paths:
        if not path.exists():
            failures.append(f"Missing required artifact: {rel(path)}")
    if failures:
        lines.extend(f"FAIL: {x}" for x in failures)
        return finish_preflight(out_dir, False, lines, artifacts, write_report)

    meas_cfg = read_json(MATCHED_ROOT / "measurement" / "measurement_frozen_configuration.json")
    as_cfg = read_json(MATCHED_ROOT / "as_oct" / "as_oct_frozen_configuration.json")
    if meas_cfg.get("model") != "RandomForestRegressor" or meas_cfg.get("parameters", {}).get("n_estimators") != 500:
        failures.append(f"Frozen Measurement RF config unexpected: {meas_cfg}")
    if as_cfg.get("model") != "ResNet18" or as_cfg.get("model_seed") != 42:
        failures.append(f"Frozen AS-OCT V0 config unexpected: {as_cfg}")
    coverage = pd.read_csv(AUDIT_ROOT / "matched_prediction_coverage_summary.csv")
    cov = coverage[coverage["split_seed"].isin(args.outer_split_seeds)].copy()
    if not cov["strict_three_way_matched"].eq(56).all():
        failures.append(f"Strict three-way test coverage not 56/56 for selected splits: {cov.to_dict('records')}")
    total_cov = int(cov["strict_three_way_matched"].sum())
    lines.append(f"Strict matched test coverage for selected splits: {total_cov}/{56 * len(args.outer_split_seeds)}")

    interaction = pd.read_csv(AUDIT_ROOT / "interaction_synergy_summary.csv")
    add_wins = int((interaction["additive_gain_vs_best_unimodal"] < 0).sum())
    concat_wins = int((interaction["concat_gain_vs_best_unimodal"] < 0).sum())
    lines.append(f"Matched audit additive wins vs best unimodal: {add_wins}/5")
    lines.append(f"Matched audit concat wins vs best unimodal: {concat_wins}/5")
    if add_wins != 3:
        failures.append(f"Current matched audit additive wins are {add_wins}/5, expected formal sanity check 3/5.")
    if concat_wins != 0:
        failures.append(f"Current matched audit concat wins are {concat_wins}/5, expected formal sanity check 0/5.")

    meas_val, meas_test, as_all = load_base_predictions()
    for seed in args.outer_split_seeds:
        split = load_split(seed)
        dup = duplicate_eye_count(split)
        if dup:
            failures.append(f"Duplicate eye/global_sample_id in split_seed={seed}: {dup}")
        leakage = patient_leakage_count(split)
        if leakage:
            failures.append(f"Patient-level split leakage in split_seed={seed}: {leakage} patients")
        split_counts = split["split"].value_counts().to_dict()
        view_cov = view_coverage_table(split)
        n_views = view_cov["n_as_oct_views"].astype(float)
        one_view = view_cov[view_cov["n_as_oct_views"].eq(1)]
        missing_views = view_cov[view_cov["missing_view_count"].gt(0) | view_cov["n_as_oct_views"].eq(0)]
        val_one_view_count = int(((view_cov["split"].eq("val")) & view_cov["n_as_oct_views"].eq(1)).sum())
        test_one_view_count = int(((view_cov["split"].eq("test")) & view_cov["n_as_oct_views"].eq(1)).sum())
        val_multiview_count = int(((view_cov["split"].eq("val")) & view_cov["n_as_oct_views"].ge(2)).sum())
        test_multiview_count = int(((view_cov["split"].eq("test")) & view_cov["n_as_oct_views"].ge(2)).sum())
        lines.append(f"split_seed={seed}")
        lines.append(f"  train/val/test eye count: train={split_counts.get('train', 0)}, val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)}")
        lines.append(f"  AS-OCT views per eye: min={int(n_views.min())}, median={float(n_views.median()):.1f}, max={int(n_views.max())}")
        lines.append(f"  eyes with exactly one view: {len(one_view)}")
        lines.append(f"  eyes with missing views: {len(missing_views)}")
        lines.append(f"  duplicate eye check: {dup}")
        lines.append(f"  patient-level leakage check: {leakage}")
        lines.append(f"  val one-view count: {val_one_view_count}")
        lines.append(f"  test one-view count: {test_one_view_count}")
        lines.append(f"  val multi-view count: {val_multiview_count}")
        lines.append(f"  test multi-view count: {test_multiview_count}")
        if val_multiview_count == 0:
            failures.append(f"split_seed={seed} has no validation multi-view eyes for validation-median dispersion imputation.")
        for split_name in ["train", "val", "test"]:
            sub = view_cov[view_cov["split"].eq(split_name)]
            if not sub.empty:
                sv = sub["n_as_oct_views"].astype(float)
                lines.append(
                    f"  {split_name} view stats: n={len(sub)}, min={int(sv.min())}, "
                    f"median={float(sv.median()):.1f}, max={int(sv.max())}, "
                    f"one_view={int(sub['n_as_oct_views'].eq(1).sum())}, "
                    f"missing_view_eyes={int((sub['missing_view_count'].gt(0) | sub['n_as_oct_views'].eq(0)).sum())}"
                )
        if len(one_view):
            lines.append(
                "  WARNING: single-view eyes will use validation-median dispersion imputation "
                "plus as_oct_single_view_flag; raw dispersion remains NaN before imputation."
            )
        if len(missing_views):
            ids = missing_views["global_sample_id"] if "global_sample_id" in missing_views.columns else missing_views["sample_id"]
            lines.append(f"  missing-view eye IDs: {fmt_ids(ids)}")
            failures.append(f"split_seed={seed} has {len(missing_views)} eyes with missing AS-OCT views.")

        for split_name, expected in [("val", 56), ("test", 56)]:
            meas = meas_val if split_name == "val" else meas_test
            m = meas[(meas["split_seed"].eq(seed)) & (meas["split"].eq(split_name))]
            a = as_all[(as_all["split_seed"].eq(seed)) & (as_all["split"].eq(split_name))]
            lines.append(f"  matched Measurement/AS-OCT coverage {split_name}: measurement={len(m)}/{expected}, as_oct={len(a)}/{expected}")
            if len(m) != expected or len(a) != expected:
                failures.append(f"Prediction row count mismatch seed={seed} split={split_name}: measurement={len(m)}, as_oct={len(a)}")
            keys = ["split_seed", "split", "sample_id", "patient_id", "eye", "ground_truth_um"]
            base = m[keys + ["prediction_um"]].rename(columns={"prediction_um": "measurement_pred_um"}).merge(
                a[keys + ["prediction_um"]].rename(columns={"prediction_um": "as_oct_pred_um"}),
                on=keys,
                how="inner",
                validate="one_to_one",
            )
            if len(base) != expected:
                failures.append(f"Matched measurement/AS-OCT feature count mismatch seed={seed} split={split_name}: n={len(base)}")
            base["disagreement_um"] = (base["as_oct_pred_um"] - base["measurement_pred_um"]).abs()
            try:
                disp = measurement_tree_dispersion(seed, split_name)
                ok_disp = len(disp) == expected and np.isfinite(disp["measurement_tree_dispersion_um"].to_numpy(dtype=float)).all()
                lines.append(f"  RF tree dispersion availability {split_name}: {'available' if ok_disp else 'FAILED'}")
                if not ok_disp:
                    failures.append(f"RF tree dispersion not finite/complete seed={seed} split={split_name}")
            except Exception as exc:
                failures.append(f"RF tree dispersion prerequisite failed seed={seed} split={split_name}: {exc}")
            cache = as_oct_view_prediction_file(seed, split_name, out_dir)
            if cache.exists():
                lines.append(f"  AS-OCT view-wise prediction cache {split_name}: {rel(cache)}")
            else:
                lines.append(f"  AS-OCT view-wise prediction cache {split_name}: absent; --run will create inference-only cache if preflight passes")
        val_cache = as_oct_view_prediction_file(seed, "val", out_dir)
        if val_cache.exists():
            try:
                cache_args = argparse.Namespace(**vars(args))
                cache_args.run = True
                val_disp = as_oct_view_dispersion(seed, "val", out_dir, cache_args)
                median = float(pd.to_numeric(val_disp.loc[val_disp["n_as_oct_views"].ge(2), "as_oct_view_dispersion_raw_um"], errors="coerce").median())
                ok_median = math.isfinite(median)
                lines.append(
                    f"  validation median dispersion computable: {'yes' if ok_median else 'no'}"
                    f"{f' ({median:.6f} um)' if ok_median else ''}"
                )
                if not ok_median:
                    failures.append(f"split_seed={seed} validation median dispersion is not finite from existing cache.")
            except Exception as exc:
                failures.append(f"split_seed={seed} validation median dispersion cache check failed: {exc}")
        else:
            lines.append(
                "  validation median dispersion computable: yes after --run inference-only cache "
                f"(validation multi-view eyes={val_multiview_count}); numeric value not computed during preflight"
            )
        ckpt = matched_as_oct_checkpoint(seed, AS_OCT_MODEL_SEED)
        if ckpt is None:
            failures.append(f"Missing frozen matched AS-OCT checkpoint for split_seed={seed}, model_seed=42")
            lines.append("  Frozen AS-OCT checkpoint availability: MISSING")
        else:
            lines.append(f"  Frozen AS-OCT checkpoint availability: {rel(ckpt)}")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        failures.append(f"Output directory not writable: {rel(out_dir)} ({exc})")
    lines.extend(["", "Leakage guard: scaler and gate code consume validation labels only; test labels are evaluated after fitting."])
    if failures:
        lines.append("")
        lines.append("PRECHECK FAILED")
        lines.extend(f"FAIL: {x}" for x in failures)
    else:
        lines.append("")
        lines.append("PRECHECK PASSED")
    return finish_preflight(out_dir, not failures, lines, artifacts, write_report)


def finish_preflight(out_dir: Path, ok: bool, lines: list[str], artifacts: dict[str, str], write_report: bool) -> PreflightResult:
    if write_report:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preflight_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return PreflightResult(ok=ok, lines=lines, artifacts=artifacts)


def write_report(out_dir: Path, per_split: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    lines = [
        "# G2 Reliability-Aware Soft Gate Pilot",
        "",
        "## Hypothesis",
        "A single linear sigmoid gate using prediction dispersion/reliability proxy features may learn per-eye fusion weights for frozen AS-OCT V0 and frozen Measurement RF.",
        "",
        "## Data Source",
        f"- Matched unimodal root: `{rel(MATCHED_ROOT)}`",
        f"- Matched audit root: `{rel(AUDIT_ROOT)}`",
        "",
        "## Matched Cohort Protocol",
        "Each outer split uses matched validation eyes for scaler fitting and gate fitting, then evaluates once on matched test eyes.",
        "",
        "## Frozen Experts",
        "- AS-OCT V0: ResNet18 frozen checkpoints/predictions from the matched v5.2 artifacts.",
        "- Measurement RF: RandomForestRegressor with the frozen v5.2 feature set and parameters.",
        "",
        "## Gate Inputs",
        ", ".join(f"`{c}`" for c in GATE_FEATURES),
        "",
        "## Reliability Proxies",
        "AS-OCT reliability proxy is view-wise prediction dispersion from all available preoperative AS-OCT views using the frozen matched V0 model_seed=42 checkpoint for the current outer split. Measurement reliability proxy is RF tree prediction dispersion. Neither is treated as calibrated predictive uncertainty.",
        "",
        "View-wise dispersion was undefined for eyes with a single preoperative AS-OCT view. These values were median-imputed using multi-view eyes from the gate-training validation set only, with an explicit single-view indicator. No test information was used for imputation.",
        "",
        "## Gate Architecture",
        "`alpha = sigmoid(w^T z + b)` and `pred = alpha * AS-OCT + (1 - alpha) * Measurement`.",
        "",
        "## Leakage Prevention",
        "Feature scaling and gate fitting use validation rows only. Test labels are used only for final metric computation.",
        "",
        "## Per-Split Results",
        per_split.to_markdown(index=False),
        "",
        "## Aggregate Results",
        aggregate.to_markdown(index=False),
        "",
        "## Limitations",
        "This is a small pilot with fixed optimizer settings and no hyperparameter search. Results should be interpreted as diagnostic, not definitive.",
        "",
    ]
    (out_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outer_split_seeds = list(dict.fromkeys(args.outer_split_seeds))
    if args.preflight:
        preflight(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()

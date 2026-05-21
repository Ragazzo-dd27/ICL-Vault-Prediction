"""Prediction-level late fusion and residual correction analysis.

This script uses existing validation/test predictions only. It does not train a
new image model and does not modify any source predictions or manifests. Fusion
weights and residual correction are selected on the validation split, then
evaluated once on the test split.

The measurement inputs come from true preoperative 2DAnalysis measurements only.
Postoperative 2DAnalysis measurements must not be used as input features.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


AS_OCT_RUNS = [
    "combined_as_oct_strict_imagenet_seed42_e30",
    "combined_as_oct_strict_imagenet_seed2026_e30",
    "combined_as_oct_strict_imagenet_seed3407_e30",
]

FUSION_RUNS = [
    "combined_fusion_ready_concat_seed42_e30",
    "combined_fusion_ready_concat_seed2026_e30",
    "combined_fusion_ready_concat_seed3407_e30",
]

MEASUREMENT_MODEL_NAMES = [
    "linear_regression",
    "ridge_regression",
    "random_forest",
]

FEATURE_COLUMNS = [
    "cct_mean_um",
    "acd_epi_mean_mm",
    "acd_endo_mean_mm",
    "clr_mean_um",
    "ata_mean_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze late fusion and residual correction from existing predictions."
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/vault_as_oct_plus_preop_measurement_pod1_manifest_combined_ready.csv",
    )
    parser.add_argument(
        "--as_oct_pred_dir",
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01",
    )
    parser.add_argument(
        "--fusion_pred_dir",
        default="artifacts/predictions/fusion_baseline_batch_01_02",
    )
    parser.add_argument(
        "--measurement_report_dir",
        default="artifacts/reports/preop_measurement_baseline_batch_01",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/reports/combined_batch_01_02/late_fusion_analysis",
    )
    parser.add_argument("--weight_step", type=float, default=0.05)
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def ensure_global_sample_id(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    if "global_sample_id" in df.columns:
        df["global_sample_id"] = df["global_sample_id"].astype(str)
        return df
    if "batch_id" in df.columns and "sample_id" in df.columns:
        df["global_sample_id"] = df["batch_id"].astype(str) + "__" + df["sample_id"].astype(str)
        return df
    if "patient_id" in df.columns and "sample_id" in df.columns:
        batch_id = df["patient_id"].astype(str).str.extract(r"^(batch_\d+)__", expand=False)
        missing = batch_id.isna()
        if missing.any():
            warn(f"{source}: {int(missing.sum())} rows could not derive batch_id from patient_id.")
            batch_id = batch_id.fillna("")
        df["global_sample_id"] = np.where(
            batch_id.astype(str).str.len() > 0,
            batch_id.astype(str) + "__" + df["sample_id"].astype(str),
            df["sample_id"].astype(str),
        )
        return df
    raise ValueError(f"{source}: cannot construct global_sample_id.")


def read_prediction(path: Path, split: str, source: str) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"Missing prediction file: {path}")
        return None
    df = pd.read_csv(path)
    required = {"vault_label_um", "pred_vault_um"}
    missing = required - set(df.columns)
    if missing:
        warn(f"{path} missing columns {sorted(missing)}; skipping.")
        return None
    df = ensure_global_sample_id(df, source)
    df["prediction_split"] = split
    df["source_file"] = str(path)
    return df


def load_run_predictions(base_dir: Path, runs: list[str], method: str, split: str) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    loaded: list[str] = []
    for run_name in runs:
        path = base_dir / run_name / f"{split}_predictions.csv"
        df = read_prediction(path, split, run_name)
        if df is None:
            continue
        df["run_name"] = run_name
        df["method"] = method
        frames.append(df)
        loaded.append(str(path))
    if not frames:
        return pd.DataFrame(), loaded
    return pd.concat(frames, ignore_index=True), loaded


def load_measurement_predictions(report_dir: Path, split: str) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    loaded: list[str] = []
    for pred_dir in sorted(report_dir.glob("combined_measurement_ready_seed*/predictions")):
        run_name = pred_dir.parent.name
        for model_name in MEASUREMENT_MODEL_NAMES:
            path = pred_dir / f"{model_name}_{split}_predictions.csv"
            df = read_prediction(path, split, f"{run_name}/{model_name}")
            if df is None:
                continue
            df["run_name"] = run_name
            df["model_name"] = model_name
            df["method"] = "measurement"
            frames.append(df)
            loaded.append(str(path))
    if not frames:
        return pd.DataFrame(), loaded
    return pd.concat(frames, ignore_index=True), loaded


def ensemble_predictions(df: pd.DataFrame, pred_col_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("global_sample_id", as_index=False)
        .agg(
            vault_label_um=("vault_label_um", "first"),
            pred_mean_um=("pred_vault_um", "mean"),
            pred_std_um=("pred_vault_um", "std"),
            n_predictions=("pred_vault_um", "count"),
        )
        .rename(
            columns={
                "pred_mean_um": pred_col_name,
                "pred_std_um": pred_col_name.replace("_um", "_std_um"),
                "n_predictions": pred_col_name.replace("_um", "_n"),
            }
        )
    )
    return out


def metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    if len(y_true_arr) == 0:
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan}
    err = y_pred_arr - y_true_arr
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    r2 = math.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def align_split(manifest: pd.DataFrame, split: str, as_oct: pd.DataFrame, measurement: pd.DataFrame, fusion: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "global_sample_id",
        "sample_id",
        "batch_id",
        "global_patient_uid",
        "eye_side",
        "split",
        "vault_label",
        "label_qc_flag",
        "measurement_ready_status",
        *FEATURE_COLUMNS,
    ]
    meta = manifest[manifest["split"].astype(str).str.lower() == split][meta_cols].copy()
    meta = meta.rename(columns={"vault_label": "manifest_vault_label_um"})
    df = meta.merge(as_oct, on="global_sample_id", how="inner", suffixes=("", "_as_oct"))
    df = df.merge(measurement, on="global_sample_id", how="inner", suffixes=("", "_measurement"))
    df = df.merge(fusion, on="global_sample_id", how="inner", suffixes=("", "_fusion"))
    df["vault_label_um"] = df["manifest_vault_label_um"]
    return df


def search_late_fusion_weight(val_df: pd.DataFrame, step: float) -> tuple[float, pd.DataFrame]:
    weights = np.round(np.arange(0.0, 1.0 + step / 2, step), 4)
    rows = []
    for w in weights:
        pred = w * val_df["as_oct_pred_um"] + (1.0 - w) * val_df["measurement_pred_um"]
        m = metrics(val_df["vault_label_um"], pred)
        rows.append({"w_as_oct": w, "w_measurement": 1.0 - w, "val_mae_um": m["mae"], "val_rmse_um": m["rmse"], "val_r2": m["r2"]})
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["val_mae_um", "w_as_oct"], ascending=[True, False]).iloc[0]
    return float(best["w_as_oct"]), grid


def search_three_way_weights(val_df: pd.DataFrame, step: float) -> tuple[dict[str, float], pd.DataFrame]:
    weights = np.round(np.arange(0.0, 1.0 + step / 2, step), 4)
    rows = []
    for w1 in weights:
        for w2 in weights:
            w3 = round(1.0 - float(w1) - float(w2), 4)
            if w3 < -1e-9:
                continue
            if w3 < 0:
                w3 = 0.0
            pred = (
                w1 * val_df["as_oct_pred_um"]
                + w2 * val_df["measurement_pred_um"]
                + w3 * val_df["concat_fusion_pred_um"]
            )
            m = metrics(val_df["vault_label_um"], pred)
            rows.append(
                {
                    "w_as_oct": float(w1),
                    "w_measurement": float(w2),
                    "w_concat_fusion": float(w3),
                    "val_mae_um": m["mae"],
                    "val_rmse_um": m["rmse"],
                    "val_r2": m["r2"],
                }
            )
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["val_mae_um", "w_as_oct"], ascending=[True, False]).iloc[0]
    return {
        "w_as_oct": float(best["w_as_oct"]),
        "w_measurement": float(best["w_measurement"]),
        "w_concat_fusion": float(best["w_concat_fusion"]),
    }, grid


def fit_residual_model(val_df: pd.DataFrame) -> tuple[object, float]:
    feature_cols = ["measurement_pred_um", "as_oct_pred_um", *FEATURE_COLUMNS]
    x = val_df[feature_cols].to_numpy(dtype=float)
    y = (val_df["vault_label_um"] - val_df["as_oct_pred_um"]).to_numpy(dtype=float)
    alphas = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    model.fit(x, y)
    alpha = float(model.named_steps["ridgecv"].alpha_)
    return model, alpha


def add_residual_prediction(df: pd.DataFrame, model: object) -> pd.Series:
    feature_cols = ["measurement_pred_um", "as_oct_pred_um", *FEATURE_COLUMNS]
    residual = model.predict(df[feature_cols].to_numpy(dtype=float))
    return df["as_oct_pred_um"] + residual


def add_method_summary(rows: list[dict[str, object]], method_name: str, val_df: pd.DataFrame, test_df: pd.DataFrame, pred_col: str, params: str, notes: str) -> None:
    val_m = metrics(val_df["vault_label_um"], val_df[pred_col])
    test_m = metrics(test_df["vault_label_um"], test_df[pred_col])
    rows.append(
        {
            "method_name": method_name,
            "val_mae_um": val_m["mae"],
            "val_rmse_um": val_m["rmse"],
            "val_r2": val_m["r2"],
            "test_mae_um": test_m["mae"],
            "test_rmse_um": test_m["rmse"],
            "test_r2": test_m["r2"],
            "selected_weights_or_params": params,
            "notes": notes,
        }
    )


def plot_test_mae(summary: pd.DataFrame, out_path: Path) -> None:
    df = summary.sort_values("test_mae_um")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(np.arange(len(df)), df["test_mae_um"], color="#4c78a8")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["method_name"], rotation=30, ha="right")
    ax.set_ylabel("Test MAE (um)")
    ax.set_title("Late Fusion Test MAE Comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_weight_search(late_grid: pd.DataFrame, three_grid: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(late_grid["w_as_oct"], late_grid["val_mae_um"], marker="o")
    axes[0].set_xlabel("AS-OCT weight")
    axes[0].set_ylabel("Validation MAE (um)")
    axes[0].set_title("Two-Way Late Fusion Weight Search")
    axes[0].grid(alpha=0.25)

    best_by_as_oct = three_grid.groupby("w_as_oct", as_index=False)["val_mae_um"].min()
    axes[1].plot(best_by_as_oct["w_as_oct"], best_by_as_oct["val_mae_um"], marker="o", color="#54a24b")
    axes[1].set_xlabel("AS-OCT weight")
    axes[1].set_ylabel("Best validation MAE (um)")
    axes[1].set_title("Three-Way Fusion Search")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_residual_delta(test_df: pd.DataFrame, out_path: Path) -> None:
    df = test_df.copy()
    df["residual_delta_vs_as_oct"] = df["residual_fusion_abs_error_um"] - df["as_oct_abs_error_um"]
    df = df.sort_values("residual_delta_vs_as_oct")
    colors = np.where(df["residual_delta_vs_as_oct"] < 0, "#59a14f", "#e15759")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(np.arange(len(df)), df["residual_delta_vs_as_oct"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["global_sample_id"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Residual fusion abs error delta vs AS-OCT (um)")
    ax.set_title("Residual Correction Error Delta")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_best_method(test_df: pd.DataFrame, out_path: Path) -> None:
    counts = (
        test_df["best_late_fusion_method"]
        .value_counts()
        .reindex(["as_oct", "measurement", "concat_fusion", "late_fusion", "three_way_fusion", "residual_fusion"])
        .fillna(0)
        .astype(int)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(np.arange(len(counts)), counts.values, color="#f58518")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts.index, rotation=25, ha="right")
    ax.set_ylabel("Number of test samples")
    ax.set_title("Best Prediction-Level Method Per Sample")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_markdown(out_path: Path, summary: pd.DataFrame, best_w: float, best_three: dict[str, float], residual_alpha: float, val_n: int, test_n: int) -> None:
    best = summary.sort_values("test_mae_um").iloc[0]
    as_oct = summary[summary["method_name"] == "as_oct_ensemble"].iloc[0]
    late = summary[summary["method_name"] == "weighted_late_fusion"].iloc[0]
    residual = summary[summary["method_name"] == "residual_correction"].iloc[0]
    measurement = summary[summary["method_name"] == "measurement_ensemble"].iloc[0]
    concat = summary[summary["method_name"] == "concat_fusion_ensemble"].iloc[0]

    table_cols = [
        "method_name",
        "val_mae_um",
        "test_mae_um",
        "test_rmse_um",
        "test_r2",
        "selected_weights_or_params",
    ]
    table_df = summary[table_cols].copy()
    for col in ["val_mae_um", "test_mae_um", "test_rmse_um"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.2f}")
    table_df["test_r2"] = table_df["test_r2"].map(lambda x: f"{x:.3f}")
    table_lines = [
        "| " + " | ".join(table_cols) + " |",
        "| " + " | ".join(["---"] * len(table_cols)) + " |",
    ]
    for _, row in table_df.iterrows():
        table_lines.append("| " + " | ".join(str(row[col]) for col in table_cols) + " |")

    lines = [
        "# Late fusion and residual correction analysis",
        "",
        "## 分析目的",
        "",
        "本分析只基于已有 validation/test predictions，评估 prediction-level late fusion 与 residual correction 是否能更好利用 true preoperative measurement features。脚本没有训练新的深度图像模型，也没有修改任何 prediction、manifest 或训练代码。",
        "",
        f"- validation samples used for selection: {val_n}",
        f"- test samples used for final evaluation: {test_n}",
        "",
        "## Test 结果",
        "",
        "\n".join(table_lines),
        "",
        f"当前 test MAE 最低的方法是 **{best['method_name']}**，test MAE = {best['test_mae_um']:.2f} um。",
        "",
        "## 是否超过 AS-OCT ensemble",
        "",
        f"- AS-OCT ensemble test MAE: {as_oct['test_mae_um']:.2f} um",
        f"- measurement ensemble test MAE: {measurement['test_mae_um']:.2f} um",
        f"- concat fusion ensemble test MAE: {concat['test_mae_um']:.2f} um",
        f"- weighted late fusion test MAE: {late['test_mae_um']:.2f} um",
        f"- residual correction test MAE: {residual['test_mae_um']:.2f} um",
        "",
        f"Two-way weighted late fusion 在 validation 上选择的 AS-OCT 权重为 {best_w:.2f}，measurement 权重为 {1.0 - best_w:.2f}。",
        f"Three-way fusion 选择的权重为 AS-OCT {best_three['w_as_oct']:.2f}, measurement {best_three['w_measurement']:.2f}, concat fusion {best_three['w_concat_fusion']:.2f}。",
        f"Residual correction 使用 Ridge residual model，validation 内部选择 alpha = {residual_alpha:.4g}。",
        "",
        "如果最佳权重接近 1.0，说明 measurement prediction 对当前 split 的整体平均帮助有限；如果 residual correction 只改善少数样本但均值不降，说明 measurement features 具有局部互补性，但还没有形成稳定的全局增益。",
        "",
        "## 解释",
        "",
        "当前结果应被视作 pilot split 上的 prediction-level 诊断。measurement-only 已经显示局部互补信息，但简单加权和 residual correction 能否稳定超过 AS-OCT-only，需要看 validation 选择是否能泛化到 test。由于 val/test 样本都很小，少数样本会显著影响 MAE。",
        "",
        "## 下一步建议",
        "",
        "- 如果 late fusion 权重明显偏向 AS-OCT，可优先尝试更保守的 residual fusion，而不是加大 concat fusion head。",
        "- 如果 residual correction 在 test 上变差，应降低 residual model 自由度，或只在 AS-OCT 高不确定性样本上触发 correction。",
        "- 下一步可尝试固定 AS-OCT backbone、减小 fusion head、增加正则化，或基于 validation 误差学习 sample-level gating。",
        "- 暂不建议直接上复杂 cross-attention；当前更需要稳定、可解释的小模型融合验证。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = ensure_global_sample_id(pd.read_csv(args.manifest), "manifest")

    loaded_files: list[str] = []
    split_frames: dict[str, pd.DataFrame] = {}
    for split in ["val", "test"]:
        as_oct_raw, loaded = load_run_predictions(Path(args.as_oct_pred_dir), AS_OCT_RUNS, "as_oct", split)
        loaded_files.extend(loaded)
        measurement_raw, loaded = load_measurement_predictions(Path(args.measurement_report_dir), split)
        loaded_files.extend(loaded)
        fusion_raw, loaded = load_run_predictions(Path(args.fusion_pred_dir), FUSION_RUNS, "concat_fusion", split)
        loaded_files.extend(loaded)

        as_oct = ensemble_predictions(as_oct_raw, "as_oct_pred_um")
        measurement = ensemble_predictions(measurement_raw, "measurement_pred_um")
        fusion = ensemble_predictions(fusion_raw, "concat_fusion_pred_um")
        if as_oct.empty or measurement.empty or fusion.empty:
            raise RuntimeError(f"Missing one or more method families for split={split}.")
        aligned = align_split(manifest, split, as_oct, measurement, fusion)
        if aligned.empty:
            raise RuntimeError(f"No aligned samples for split={split}.")
        split_frames[split] = aligned

    val_df = split_frames["val"].copy()
    test_df = split_frames["test"].copy()

    best_w, late_grid = search_late_fusion_weight(val_df, args.weight_step)
    best_three, three_grid = search_three_way_weights(val_df, args.weight_step)
    residual_model, residual_alpha = fit_residual_model(val_df)

    for df in [val_df, test_df]:
        df["late_fusion_pred_um"] = best_w * df["as_oct_pred_um"] + (1.0 - best_w) * df["measurement_pred_um"]
        df["three_way_fusion_pred_um"] = (
            best_three["w_as_oct"] * df["as_oct_pred_um"]
            + best_three["w_measurement"] * df["measurement_pred_um"]
            + best_three["w_concat_fusion"] * df["concat_fusion_pred_um"]
        )
        df["residual_fusion_pred_um"] = add_residual_prediction(df, residual_model)

    summary_rows: list[dict[str, object]] = []
    add_method_summary(summary_rows, "as_oct_ensemble", val_df, test_df, "as_oct_pred_um", "mean of 3 AS-OCT seeds", "Seed ensemble baseline.")
    add_method_summary(summary_rows, "measurement_ensemble", val_df, test_df, "measurement_pred_um", "mean of linear/ridge/random_forest across seeds", "Prediction ensemble of selected measurement-only models.")
    add_method_summary(summary_rows, "concat_fusion_ensemble", val_df, test_df, "concat_fusion_pred_um", "mean of 3 concat fusion seeds", "Existing deep concat fusion seed ensemble.")
    add_method_summary(summary_rows, "weighted_late_fusion", val_df, test_df, "late_fusion_pred_um", f"w_as_oct={best_w:.2f}; w_measurement={1.0 - best_w:.2f}", "Weight selected by validation MAE.")
    add_method_summary(
        summary_rows,
        "three_way_weighted_fusion",
        val_df,
        test_df,
        "three_way_fusion_pred_um",
        f"w_as_oct={best_three['w_as_oct']:.2f}; w_measurement={best_three['w_measurement']:.2f}; w_concat_fusion={best_three['w_concat_fusion']:.2f}",
        "Three-way weights selected by validation MAE.",
    )
    add_method_summary(summary_rows, "residual_correction", val_df, test_df, "residual_fusion_pred_um", f"ridge_alpha={residual_alpha:.4g}", "Ridge residual model fit on validation residuals.")
    summary = pd.DataFrame(summary_rows)

    pred_cols = {
        "as_oct": "as_oct_pred_um",
        "measurement": "measurement_pred_um",
        "concat_fusion": "concat_fusion_pred_um",
        "late_fusion": "late_fusion_pred_um",
        "three_way_fusion": "three_way_fusion_pred_um",
        "residual_fusion": "residual_fusion_pred_um",
    }
    for method, col in pred_cols.items():
        test_df[f"{method}_abs_error_um"] = (test_df[col] - test_df["vault_label_um"]).abs()
    test_df["best_late_fusion_method"] = test_df[[f"{m}_abs_error_um" for m in pred_cols]].idxmin(axis=1)
    test_df["best_late_fusion_method"] = test_df["best_late_fusion_method"].str.replace("_abs_error_um", "", regex=False)

    pred_out_cols = [
        "global_sample_id",
        "vault_label_um",
        "as_oct_pred_um",
        "measurement_pred_um",
        "concat_fusion_pred_um",
        "late_fusion_pred_um",
        "three_way_fusion_pred_um",
        "residual_fusion_pred_um",
        *[f"{m}_abs_error_um" for m in pred_cols],
        "best_late_fusion_method",
    ]

    summary_path = out_dir / "late_fusion_summary.csv"
    preds_path = out_dir / "late_fusion_test_predictions.csv"
    md_path = out_dir / "late_fusion_analysis.md"
    late_grid_path = out_dir / "late_fusion_weight_search.csv"
    three_grid_path = out_dir / "three_way_weight_search.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8")
    test_df[pred_out_cols].to_csv(preds_path, index=False, encoding="utf-8")
    late_grid.to_csv(late_grid_path, index=False, encoding="utf-8")
    three_grid.to_csv(three_grid_path, index=False, encoding="utf-8")

    plot_test_mae(summary, figures_dir / "late_fusion_test_mae_comparison.png")
    plot_weight_search(late_grid, three_grid, figures_dir / "late_fusion_weight_search_val_mae.png")
    plot_residual_delta(test_df, figures_dir / "residual_correction_error_delta.png")
    plot_best_method(test_df, figures_dir / "per_sample_best_method_late_fusion.png")
    write_markdown(md_path, summary, best_w, best_three, residual_alpha, len(val_df), len(test_df))

    print(f"Loaded prediction files: {len(loaded_files)}")
    for path in loaded_files:
        print(f"  {path}")
    print(f"Aligned val samples: {len(val_df)}")
    print(f"Aligned test samples: {len(test_df)}")
    for method in ["as_oct_ensemble", "measurement_ensemble", "concat_fusion_ensemble", "weighted_late_fusion", "residual_correction"]:
        row = summary[summary["method_name"] == method].iloc[0]
        print(f"{method} test MAE: {row['test_mae_um']:.2f} um")
    print(f"Weighted late fusion best AS-OCT weight: {best_w:.2f}; test MAE: {summary[summary['method_name']=='weighted_late_fusion'].iloc[0]['test_mae_um']:.2f} um")
    print(f"Residual correction test MAE: {summary[summary['method_name']=='residual_correction'].iloc[0]['test_mae_um']:.2f} um")
    print("Output files:")
    for path in [summary_path, preds_path, late_grid_path, three_grid_path, md_path, figures_dir]:
        print(f"  {path}")


if __name__ == "__main__":
    main()

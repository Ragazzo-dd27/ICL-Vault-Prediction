"""Generate final combined v4 baseline comparison package.

Analysis only: this script reads completed primary/repeated outputs and writes
comparison tables, figures, manuscript table drafts, and an experiment freeze
record. It does not train models, alter splits, or modify previous result files.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "artifacts/reports/combined_batch_01_02_03_04"
OUT = BASE / "final_baseline_comparison"
DOCS = PROJECT / "docs/experiments"

MEAS_PRIMARY_DIR = BASE / "measurement_only_baseline_seed42"
MEAS_REP_DIR = BASE / "measurement_only_repeated_splits"
AS_PRIMARY_ENSEMBLE_DIR = BASE / "as_oct_only_ensemble_label_corrected_patient100_os_seed42_2026_3407"
AS_REP_DIR = BASE / "as_oct_only_repeated_splits_label_corrected_patient100_os"
FUSION_ENSEMBLE_DIR = BASE / "fusion_ensemble_seed42_2026_3407"
FUSION_REP_DIR = BASE / "fusion_repeated_splits_fixed_model_seed42"

SPLIT_SEEDS = [42, 1001, 2002, 2026, 3407]
MODEL_ORDER = ["Measurement RF", "Corrected AS-OCT", "Fusion"]
MODEL_COLORS = {
    "Measurement RF": "#1f77b4",
    "Corrected AS-OCT": "#d62728",
    "Fusion": "#2ca02c",
}
RANGE_ORDER = ["low", "medium", "high"]


def f(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def pm(mean: float, std: float) -> str:
    return f"{mean:.2f} +/- {std:.2f}"


def iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    frame = df.copy()
    for col in frame.columns:
        frame[col] = frame[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(frame.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in frame.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def prediction_range_ratio(predictions: pd.DataFrame) -> float:
    label_col = "vault_label_um" if "vault_label_um" in predictions.columns else "y_true_um"
    pred_col = "pred_vault_um" if "pred_vault_um" in predictions.columns else "y_pred_um"
    labels = pd.to_numeric(predictions[label_col], errors="coerce")
    preds = pd.to_numeric(predictions[pred_col], errors="coerce")
    label_range = float(labels.max() - labels.min())
    pred_range = float(preds.max() - preds.min())
    return pred_range / label_range if label_range > 0 else float("nan")


def load_primary_comparison() -> pd.DataFrame:
    meas_overall_path = MEAS_PRIMARY_DIR / "measurement_only_v4_overall_metrics.csv"
    meas_pred_path = MEAS_PRIMARY_DIR / "measurement_only_v4_predictions.csv"
    as_overall_path = AS_PRIMARY_ENSEMBLE_DIR / "as_oct_v4_corrected_ensemble_overall_metrics.csv"
    as_pred_path = AS_PRIMARY_ENSEMBLE_DIR / "as_oct_v4_corrected_ensemble_predictions.csv"
    fusion_overall_path = FUSION_ENSEMBLE_DIR / "fusion_v4_ensemble_overall_metrics.csv"
    fusion_pred_path = FUSION_ENSEMBLE_DIR / "fusion_v4_ensemble_predictions.csv"
    for path in [meas_overall_path, meas_pred_path, as_overall_path, as_pred_path, fusion_overall_path, fusion_pred_path]:
        require_file(path)

    meas_overall = pd.read_csv(meas_overall_path)
    meas_rf = meas_overall[meas_overall["model"].eq("Random Forest Regressor")].iloc[0]
    meas_preds = pd.read_csv(meas_pred_path)
    meas_rf_preds = meas_preds[meas_preds["model"].eq("Random Forest Regressor")]

    as_overall = pd.read_csv(as_overall_path).iloc[0]
    fusion_overall = pd.read_csv(fusion_overall_path).iloc[0]

    rows = [
        {
            "model": "Measurement RF",
            "input": "Preoperative measurements",
            "evaluation_variant": "primary seed42 Random Forest",
            "n_train": int(meas_rf["n_train"]),
            "n_val": int(meas_rf["n_val"]),
            "n_test": int(meas_rf["n_test"]),
            "MAE": float(meas_rf["MAE"]),
            "RMSE": float(meas_rf["RMSE"]),
            "R2": float(meas_rf["R2"]),
            "mean_signed_error": float(meas_rf["mean_signed_error"]),
            "prediction_range_ratio": prediction_range_ratio(meas_rf_preds),
        },
        {
            "model": "Corrected AS-OCT",
            "input": "AS-OCT image",
            "evaluation_variant": "primary corrected 3-seed ensemble",
            "n_train": 241,
            "n_val": 51,
            "n_test": int(as_overall["n_samples"]),
            "MAE": float(as_overall["MAE"]),
            "RMSE": float(as_overall["RMSE"]),
            "R2": float(as_overall["R2"]),
            "mean_signed_error": float(as_overall["mean_signed_error"]),
            "prediction_range_ratio": float(as_overall["prediction_range_label_range_ratio"]),
        },
        {
            "model": "Fusion",
            "input": "AS-OCT image + preoperative measurements",
            "evaluation_variant": "primary 3-seed ensemble",
            "n_train": 227,
            "n_val": 48,
            "n_test": int(fusion_overall["n_samples"]),
            "MAE": float(fusion_overall["mae_um"]),
            "RMSE": float(fusion_overall["rmse_um"]),
            "R2": float(fusion_overall["r2"]),
            "mean_signed_error": float(fusion_overall["mean_signed_error_um"]),
            "prediction_range_ratio": float(fusion_overall["prediction_range_label_range_ratio"]),
        },
    ]
    return pd.DataFrame(rows)


def load_repeated_model_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [
        MEAS_REP_DIR / "measurement_repeated_split_overall_metrics.csv",
        MEAS_REP_DIR / "measurement_repeated_split_range_metrics.csv",
        MEAS_REP_DIR / "measurement_repeated_split_predictions.csv",
        AS_REP_DIR / "corrected_as_oct_repeated_split_overall_metrics.csv",
        AS_REP_DIR / "corrected_as_oct_repeated_split_range_metrics.csv",
        FUSION_REP_DIR / "fusion_repeated_split_overall_metrics.csv",
        FUSION_REP_DIR / "fusion_repeated_split_range_metrics.csv",
    ]
    for path in paths:
        require_file(path)

    meas_overall = pd.read_csv(paths[0])
    meas_range = pd.read_csv(paths[1])
    meas_pred = pd.read_csv(paths[2])
    as_overall = pd.read_csv(paths[3])
    as_range = pd.read_csv(paths[4])
    fusion_overall = pd.read_csv(paths[5])
    fusion_range = pd.read_csv(paths[6])

    meas_rf = meas_overall[meas_overall["model"].eq("Random Forest Regressor")].copy()
    meas_rf_pred = meas_pred[meas_pred["model"].eq("Random Forest Regressor")].copy()
    meas_ratios = (
        meas_rf_pred.groupby("split_seed")
        .apply(prediction_range_ratio)
        .rename("prediction_range_ratio")
        .reset_index()
    )
    meas_rf = meas_rf.merge(meas_ratios, on="split_seed", how="left")

    rows = []
    for _, row in meas_rf.iterrows():
        rows.append(
            {
                "split_seed": int(row["split_seed"]),
                "model": "Measurement RF",
                "MAE": float(row["MAE"]),
                "RMSE": float(row["RMSE"]),
                "R2": float(row["R2"]),
                "mean_signed_error": float(row["mean_signed_error"]),
                "prediction_range_ratio": float(row["prediction_range_ratio"]),
            }
        )
    for _, row in as_overall.iterrows():
        rows.append(
            {
                "split_seed": int(row["split_seed"]),
                "model": "Corrected AS-OCT",
                "MAE": float(row["test_mae"]),
                "RMSE": float(row["test_rmse"]),
                "R2": float(row["test_r2"]),
                "mean_signed_error": float(row["mean_signed_error"]),
                "prediction_range_ratio": float(row["prediction_range_label_range_ratio"]),
            }
        )
    for _, row in fusion_overall.iterrows():
        rows.append(
            {
                "split_seed": int(row["split_seed"]),
                "model": "Fusion",
                "MAE": float(row["test_mae"]),
                "RMSE": float(row["test_rmse"]),
                "R2": float(row["test_r2"]),
                "mean_signed_error": float(row["mean_signed_error"]),
                "prediction_range_ratio": float(row["prediction_range_label_range_ratio"]),
            }
        )
    repeated = pd.DataFrame(rows)

    range_frames = []
    range_frames.append(
        meas_range[meas_range["model"].eq("Random Forest Regressor")]
        .assign(model="Measurement RF")
        .rename(columns={"absolute_error_contribution_percentage": "absolute_error_contribution_pct"})
    )
    range_frames.append(
        as_range.assign(model="Corrected AS-OCT").rename(
            columns={"absolute_error_contribution_percentage": "absolute_error_contribution_pct"}
        )
    )
    range_frames.append(fusion_range.assign(model="Fusion"))
    ranges = pd.concat(range_frames, ignore_index=True)
    range_wide_rows = []
    for (split_seed, model), sub in ranges.groupby(["split_seed", "model"]):
        item = {"split_seed": int(split_seed), "model": model}
        for name in RANGE_ORDER:
            r = sub[sub["vault_range"].eq(name)].iloc[0]
            item[f"{name}_MAE"] = float(r["MAE"])
            item[f"{name}_signed_error"] = float(r["mean_signed_error"])
            item[f"{name}_n"] = int(r["n"])
            item[f"{name}_overestimation_proportion"] = float(r["overestimation_count"]) / float(r["n"]) if r["n"] else float("nan")
            item[f"{name}_underestimation_proportion"] = float(r["underestimation_count"]) / float(r["n"]) if r["n"] else float("nan")
        range_wide_rows.append(item)
    range_wide = pd.DataFrame(range_wide_rows)
    repeated = repeated.merge(range_wide, on=["split_seed", "model"], how="left")
    repeated["model"] = pd.Categorical(repeated["model"], categories=MODEL_ORDER, ordered=True)
    repeated = repeated.sort_values(["split_seed", "model"]).reset_index(drop=True)
    return repeated, ranges, meas_rf_pred


def aggregate_repeated(repeated: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        sub = repeated[repeated["model"].astype(str).eq(model)]
        range_sub = ranges[ranges["model"].astype(str).eq(model)]
        item = {
            "model": model,
            "MAE_mean": float(sub["MAE"].mean()),
            "MAE_std": float(sub["MAE"].std(ddof=1)),
            "MAE_mean_std": pm(sub["MAE"].mean(), sub["MAE"].std(ddof=1)),
            "MAE_median": float(sub["MAE"].median()),
            "MAE_IQR": iqr(sub["MAE"]),
            "MAE_min": float(sub["MAE"].min()),
            "MAE_max": float(sub["MAE"].max()),
            "RMSE_mean": float(sub["RMSE"].mean()),
            "RMSE_std": float(sub["RMSE"].std(ddof=1)),
            "R2_mean": float(sub["R2"].mean()),
            "R2_std": float(sub["R2"].std(ddof=1)),
            "signed_error_mean": float(sub["mean_signed_error"].mean()),
            "signed_error_std": float(sub["mean_signed_error"].std(ddof=1)),
            "prediction_range_ratio_mean": float(sub["prediction_range_ratio"].mean()),
            "prediction_range_ratio_std": float(sub["prediction_range_ratio"].std(ddof=1)),
            "low_overestimation_proportion": float(sub["low_overestimation_proportion"].mean()),
            "high_underestimation_proportion": float(sub["high_underestimation_proportion"].mean()),
        }
        for name in RANGE_ORDER:
            r = range_sub[range_sub["vault_range"].eq(name)]
            item[f"{name}_MAE_mean"] = float(r["MAE"].mean())
            item[f"{name}_MAE_std"] = float(r["MAE"].std(ddof=1))
            item[f"{name}_MAE_mean_std"] = pm(r["MAE"].mean(), r["MAE"].std(ddof=1))
            item[f"{name}_signed_error_mean"] = float(r["mean_signed_error"].mean())
            item[f"{name}_signed_error_std"] = float(r["mean_signed_error"].std(ddof=1))
        rows.append(item)
    return pd.DataFrame(rows)


def paired_comparison(repeated: pd.DataFrame) -> pd.DataFrame:
    wide = repeated.pivot(index="split_seed", columns="model", values="MAE").reset_index()
    comparisons = [
        ("AS-OCT vs RF", "Corrected AS-OCT", "Measurement RF"),
        ("Fusion vs RF", "Fusion", "Measurement RF"),
        ("Fusion vs AS-OCT", "Fusion", "Corrected AS-OCT"),
    ]
    rows = []
    for comparison, a, b in comparisons:
        deltas = wide[a] - wide[b]
        a_wins = int((deltas < 0).sum())
        b_wins = int((deltas > 0).sum())
        ties = int((deltas == 0).sum())
        for split_seed, delta in zip(wide["split_seed"], deltas):
            rows.append(
                {
                    "comparison": comparison,
                    "split_seed": int(split_seed),
                    "model_a": a,
                    "model_b": b,
                    "delta_MAE": float(delta),
                    "winner": a if delta < 0 else (b if delta > 0 else "tie"),
                    "model_a_win_count": a_wins,
                    "model_b_win_count": b_wins,
                    "tie_count": ties,
                    "paired_delta_mean": float(deltas.mean()),
                    "paired_delta_std": float(deltas.std(ddof=1)),
                    "note": "Descriptive paired comparison across five splits; no significance claim.",
                }
            )
    return pd.DataFrame(rows)


def save_csvs(primary: pd.DataFrame, repeated: pd.DataFrame, aggregate: pd.DataFrame, paired: pd.DataFrame) -> None:
    primary.to_csv(OUT / "primary_baseline_comparison.csv", index=False, encoding="utf-8")
    repeated.to_csv(OUT / "repeated_split_model_comparison.csv", index=False, encoding="utf-8")
    aggregate.to_csv(OUT / "repeated_split_aggregate_comparison.csv", index=False, encoding="utf-8")
    paired.to_csv(OUT / "paired_model_comparison.csv", index=False, encoding="utf-8")


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_figures(repeated: pd.DataFrame, aggregate: pd.DataFrame, ranges: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Figure A
    source_a = repeated[["split_seed", "model", "MAE"]].copy()
    source_a.to_csv(OUT / "figure_repeated_split_mae_comparison_source_data.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for model in MODEL_ORDER:
        sub = source_a[source_a["model"].astype(str).eq(model)].sort_values("split_seed")
        ax.plot(sub["split_seed"].astype(str), sub["MAE"], marker="o", label=model, color=MODEL_COLORS[model])
    ax.set_xlabel("Split seed")
    ax.set_ylabel("Test MAE (um)")
    ax.set_title("Repeated split MAE")
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, "figure_repeated_split_mae_comparison")

    # Figure B
    source_b = repeated[["model", "split_seed", "MAE"]].copy()
    source_b.to_csv(OUT / "figure_aggregate_mae_stability_source_data.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    x = np.arange(len(MODEL_ORDER))
    means = [source_b[source_b["model"].astype(str).eq(model)]["MAE"].mean() for model in MODEL_ORDER]
    stds = [source_b[source_b["model"].astype(str).eq(model)]["MAE"].std(ddof=1) for model in MODEL_ORDER]
    ax.errorbar(x, means, yerr=stds, fmt="o", color="black", capsize=3, label="Mean ± SD")
    rng = np.random.default_rng(7)
    for i, model in enumerate(MODEL_ORDER):
        vals = source_b[source_b["model"].astype(str).eq(model)]["MAE"].to_numpy()
        jitter = rng.normal(0, 0.035, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=MODEL_COLORS[model], s=18, alpha=0.85)
    ax.set_xticks(x, MODEL_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Test MAE (um)")
    ax.set_title("Aggregate MAE stability")
    ax.set_ylim(bottom=0)
    style_axes(ax)
    save_figure(fig, "figure_aggregate_mae_stability")

    # Figure C
    source_c = ranges[["model", "split_seed", "vault_range", "MAE"]].copy()
    source_c.to_csv(OUT / "figure_range_specific_mae_source_data.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    width = 0.22
    x = np.arange(len(RANGE_ORDER))
    for offset, model in zip([-width, 0, width], MODEL_ORDER):
        sub = source_c[source_c["model"].astype(str).eq(model)]
        means = [sub[sub["vault_range"].eq(r)]["MAE"].mean() for r in RANGE_ORDER]
        stds = [sub[sub["vault_range"].eq(r)]["MAE"].std(ddof=1) for r in RANGE_ORDER]
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=2, label=model, color=MODEL_COLORS[model], alpha=0.85)
    ax.set_xticks(x, ["Low", "Medium", "High"])
    ax.set_xlabel("Vault range")
    ax.set_ylabel("Range MAE (um)")
    ax.set_title("Range-specific error")
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, "figure_range_specific_mae")

    # Figure D
    source_d = repeated[["model", "split_seed", "prediction_range_ratio"]].copy()
    source_d.to_csv(OUT / "figure_prediction_range_compression_source_data.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for i, model in enumerate(MODEL_ORDER):
        sub = source_d[source_d["model"].astype(str).eq(model)]
        mean = sub["prediction_range_ratio"].mean()
        std = sub["prediction_range_ratio"].std(ddof=1)
        ax.errorbar(i, mean, yerr=std, fmt="o", color="black", capsize=3)
        ax.scatter(np.full(len(sub), i) + np.linspace(-0.06, 0.06, len(sub)), sub["prediction_range_ratio"], color=MODEL_COLORS[model], s=18)
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Prediction range / label range")
    ax.set_title("Prediction range compression")
    ax.set_ylim(bottom=0)
    style_axes(ax)
    save_figure(fig, "figure_prediction_range_compression")

    # Figure E
    bias_rows = []
    for model in MODEL_ORDER:
        sub = repeated[repeated["model"].astype(str).eq(model)]
        bias_rows.append(
            {
                "model": model,
                "bias_metric": "Low-vault overestimation",
                "proportion": float(sub["low_overestimation_proportion"].mean()),
                "std": float(sub["low_overestimation_proportion"].std(ddof=1)),
            }
        )
        bias_rows.append(
            {
                "model": model,
                "bias_metric": "High-vault underestimation",
                "proportion": float(sub["high_underestimation_proportion"].mean()),
                "std": float(sub["high_underestimation_proportion"].std(ddof=1)),
            }
        )
    source_e = pd.DataFrame(bias_rows)
    source_e.to_csv(OUT / "figure_range_bias_direction_source_data.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    x = np.arange(len(MODEL_ORDER))
    width = 0.32
    low_vals = source_e[source_e["bias_metric"].eq("Low-vault overestimation")]["proportion"].to_numpy() * 100
    low_std = source_e[source_e["bias_metric"].eq("Low-vault overestimation")]["std"].to_numpy() * 100
    high_vals = source_e[source_e["bias_metric"].eq("High-vault underestimation")]["proportion"].to_numpy() * 100
    high_std = source_e[source_e["bias_metric"].eq("High-vault underestimation")]["std"].to_numpy() * 100
    ax.bar(x - width / 2, low_vals, width=width, yerr=low_std, capsize=2, color="#9467bd", label="Low overestimation")
    ax.bar(x + width / 2, high_vals, width=width, yerr=high_std, capsize=2, color="#8c564b", label="High underestimation")
    ax.set_xticks(x, MODEL_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Proportion (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Range bias direction")
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, "figure_range_bias_direction")


def latex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("%", "\\%")


def write_latex_tables(primary: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Primary split baseline performance.}",
        "\\label{tab:primary_baselines}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & MAE & RMSE & $R^2$ & Range ratio \\\\",
        "\\midrule",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"{latex_escape(row['model'])} & {row['MAE']:.2f} & {row['RMSE']:.2f} & {row['R2']:.2f} & {row['prediction_range_ratio']:.2f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\footnotesize Primary results report the seed42 split; AS-OCT and fusion primary rows use three-seed ensembles, whereas measurement-only reports the selected Random Forest baseline.",
        "\\end{table}",
        "",
    ]
    (OUT / "paper_table_primary_results.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Repeated patient-level split stability with fixed model seed 42.}",
        "\\label{tab:repeated_stability}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & MAE mean $\\pm$ SD & MAE range & RMSE mean $\\pm$ SD & Range ratio \\\\",
        "\\midrule",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"{latex_escape(row['model'])} & {row['MAE_mean']:.2f} $\\pm$ {row['MAE_std']:.2f} & "
            f"{row['MAE_min']:.2f}--{row['MAE_max']:.2f} & {row['RMSE_mean']:.2f} $\\pm$ {row['RMSE_std']:.2f} & "
            f"{row['prediction_range_ratio_mean']:.2f} $\\pm$ {row['prediction_range_ratio_std']:.2f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\footnotesize Repeated evaluation uses five patient-level split seeds (42, 1001, 2002, 2026, 3407) and fixed model seed 42 for AS-OCT and fusion.",
        "\\end{table}",
        "",
    ]
    (OUT / "paper_table_repeated_stability.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Range-specific MAE across repeated patient-level splits.}",
        "\\label{tab:range_metrics}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model & Low MAE & Medium MAE & High MAE \\\\",
        "\\midrule",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"{latex_escape(row['model'])} & {row['low_MAE_mean']:.2f} $\\pm$ {row['low_MAE_std']:.2f} & "
            f"{row['medium_MAE_mean']:.2f} $\\pm$ {row['medium_MAE_std']:.2f} & "
            f"{row['high_MAE_mean']:.2f} $\\pm$ {row['high_MAE_std']:.2f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\footnotesize Vault ranges are low ($<500$~um), medium (500--800~um), and high ($>800$~um). Values are mean $\\pm$ sample SD across repeated splits.",
        "\\end{table}",
        "",
    ]
    (OUT / "paper_table_range_metrics.tex").write_text("\n".join(lines), encoding="utf-8")


def write_report(primary: pd.DataFrame, repeated: pd.DataFrame, aggregate: pd.DataFrame, paired: pd.DataFrame) -> None:
    best_avg = aggregate.sort_values("MAE_mean").iloc[0]["model"]
    most_stable = aggregate.sort_values("MAE_std").iloc[0]["model"]
    as_vs_rf = paired[paired["comparison"].eq("AS-OCT vs RF")]
    fusion_vs_rf = paired[paired["comparison"].eq("Fusion vs RF")]
    fusion_vs_as = paired[paired["comparison"].eq("Fusion vs AS-OCT")]
    lines = [
        "# Final Baseline Comparison Report",
        "",
        "## 1. Cohort and Evaluation Protocol",
        "- Three completed baselines are frozen: measurement-only Random Forest, corrected AS-OCT-only, and AS-OCT + measurement concat fusion.",
        "- Primary results are reported separately from repeated split stability results.",
        "- Repeated evaluation uses patient-level split seeds 42, 1001, 2002, 2026, and 3407. AS-OCT and fusion repeated results use fixed model_seed42.",
        "- Primary AS-OCT and fusion ensemble results are not averaged together with repeated fixed-seed results.",
        "",
        "## 2. Primary Split Results",
        md_table(primary[["model", "evaluation_variant", "n_test", "MAE", "RMSE", "R2", "mean_signed_error", "prediction_range_ratio"]].round(3)),
        "",
        "## 3. Repeated Split Stability",
        md_table(
            aggregate[
                [
                    "model",
                    "MAE_mean_std",
                    "MAE_median",
                    "MAE_IQR",
                    "MAE_min",
                    "MAE_max",
                    "RMSE_mean",
                    "RMSE_std",
                    "prediction_range_ratio_mean",
                    "prediction_range_ratio_std",
                ]
            ].round(3)
        ),
        "",
        f"- Best average repeated performance: {best_avg}.",
        f"- Most stable repeated MAE: {most_stable}.",
        "",
        "## 4. Paired Model Comparisons",
        md_table(
            paired.groupby("comparison")
            .agg(
                delta_mean=("delta_MAE", "mean"),
                delta_std=("delta_MAE", lambda x: x.std(ddof=1)),
                model_a=("model_a", "first"),
                model_b=("model_b", "first"),
                model_a_wins=("model_a_win_count", "first"),
                model_b_wins=("model_b_win_count", "first"),
            )
            .reset_index()
            .round(3)
        ),
        "- These are descriptive paired comparisons across five splits; no significance claim is made.",
        "",
        "## 5. Range-Specific Error",
        md_table(
            aggregate[
                [
                    "model",
                    "low_MAE_mean_std",
                    "medium_MAE_mean_std",
                    "high_MAE_mean_std",
                    "low_overestimation_proportion",
                    "high_underestimation_proportion",
                ]
            ].round(3)
        ),
        "- High-vault cases remain the largest range-specific error source for all three model families.",
        "",
        "## 6. Regression-to-the-Mean / Range Compression",
        "- All three models show prediction range compression, with prediction range / label range ratios well below 1.0.",
        "- Low-vault overestimation and high-vault underestimation are consistent across model families.",
        "",
        "## 7. patient_100 OS Label Correction",
        "- The original AS-OCT label 7901 um for `batch_03__patient_100_OS_20240517` was manually verified as a transcription error and corrected to 701 um.",
        "- This correction affects AS-OCT-only results because the sample belongs to the AS-OCT cohort.",
        "- Measurement-only and fusion cohorts/results are unaffected because this OS sample is not in the fusion/measurement-ready cohorts.",
        "- Superseded 7901-label AS-OCT outputs are excluded from all formal tables in this package.",
        "",
        "## 8. Main Scientific Findings",
        "- Measurement-only RF currently provides the best average repeated-split performance.",
        "- Corrected AS-OCT-only is competitive but does not consistently outperform RF.",
        "- Simple concatenation fusion does not provide stable incremental benefit over measurement-only RF.",
        "- Fusion is not consistently better than corrected AS-OCT-only across the five repeated splits.",
        "- Image information is not completely invalid: AS-OCT and fusion are competitive on some splits, but the image models do not provide robust average superiority under this protocol.",
        "",
        "## 9. Limitations",
        "- The repeated split analysis has only five patient-level splits and should be interpreted descriptively.",
        "- High-vault sample counts remain small, so high-range metrics are informative but unstable.",
        "- The fusion architecture is intentionally simple and may not exhaust all possible multimodal approaches.",
        "",
        "## 10. Recommended Manuscript Narrative",
        "Measurement-only RF achieved the strongest average repeated-split performance. AS-OCT-only and simple concat fusion were competitive but did not consistently improve over measurement-only modeling. Across all models, low-vault overestimation, high-vault underestimation, and prediction range compression persisted, indicating that the dominant limitation is not merely modality choice but also range-dependent regression behavior.",
        "",
        "## 11. Recommended Supervisor Update Narrative",
        "The v4 baselines are now frozen after correcting the patient_100 OS label error and rerunning the affected AS-OCT analyses. Measurement-only RF remains the strongest and most stable baseline. Corrected AS-OCT is close but wins only one of five paired splits against RF. Fusion wins one of five against RF and three of five against corrected AS-OCT, so simple fusion does not justify a stronger claim yet.",
        "",
        "## 12. Whether Further Experiments Are Necessary",
        "- No further primary baseline experiments are necessary before reporting these findings.",
        "- Complex fusion, weighted loss, lower learning rate, or calibration should be reserved as secondary experiments only if requested by a supervisor or reviewer.",
        "- The current results are comparative evidence, not a clinical-readiness claim.",
        "",
        "## Direct Answers",
        f"- Which model has best average performance? {best_avg}.",
        f"- Which model is most stable? {most_stable} by repeated MAE sample SD.",
        f"- Does AS-OCT stably outperform measurement-only? No; AS-OCT wins {int(as_vs_rf['model_a_win_count'].iloc[0])}/5 against RF.",
        f"- Does fusion bring stable gain? No; fusion wins {int(fusion_vs_rf['model_a_win_count'].iloc[0])}/5 against RF and {int(fusion_vs_as['model_a_win_count'].iloc[0])}/5 against corrected AS-OCT.",
        "- Is image information completely ineffective? No, but it is not robustly superior under this baseline protocol.",
        "- Is primary seed42 representative? Yes for the broad repeated-split range, but primary and repeated protocols are reported separately.",
        "- Are low/high vault biases stable? Yes; low-vault overestimation and high-vault underestimation persist across models.",
    ]
    (OUT / "final_baseline_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_freeze_record(primary: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    lines = [
        "# Baseline Experiments Frozen After V4 Repeated Evaluation",
        "",
        f"- Freeze date: generated from completed combined v4 outputs.",
        "- Data version: combined batch_01 + batch_02 + batch_03 + batch_04 v4 manifests and repeated patient-level splits.",
        "- Label correction: patient_100 OS AS-OCT label corrected from 7901 um to 701 um after manual source verification.",
        "- Frozen baselines: measurement-only RF, corrected AS-OCT-only, and AS-OCT + measurement concat fusion.",
        "- Primary protocol: seed42 patient-level split; AS-OCT and fusion primary ensembles use three model seeds.",
        "- Repeated protocol: split seeds 42, 1001, 2002, 2026, 3407; AS-OCT and fusion use fixed model_seed42.",
        "",
        "## Final Result Paths",
        f"- Final comparison package: `{OUT.relative_to(PROJECT).as_posix()}`",
        f"- Measurement-only repeated: `{MEAS_REP_DIR.relative_to(PROJECT).as_posix()}`",
        f"- Corrected AS-OCT repeated: `{AS_REP_DIR.relative_to(PROJECT).as_posix()}`",
        f"- Fusion repeated: `{FUSION_REP_DIR.relative_to(PROJECT).as_posix()}`",
        "",
        "## Frozen Primary Results",
        md_table(primary[["model", "MAE", "RMSE", "R2", "prediction_range_ratio"]].round(2)),
        "",
        "## Frozen Repeated Results",
        md_table(aggregate[["model", "MAE_mean_std", "MAE_min", "MAE_max", "prediction_range_ratio_mean"]].round(3)),
        "",
        "## Decision",
        "- No further tuning will be performed around the current primary/repeated baseline results.",
        "- No additional calibration, weighted loss, lower learning rate, complex fusion, or repeated split generation is part of the frozen primary analysis.",
        "- Secondary experiments may be added only if requested by a supervisor or reviewer.",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "baseline_experiments_frozen_after_v4_repeated_evaluation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def qc_formal_outputs() -> pd.DataFrame:
    checks = []
    required = {
        "measurement_primary_overall": MEAS_PRIMARY_DIR / "measurement_only_v4_overall_metrics.csv",
        "measurement_primary_range": MEAS_PRIMARY_DIR / "measurement_only_v4_range_metrics.csv",
        "measurement_primary_predictions": MEAS_PRIMARY_DIR / "measurement_only_v4_predictions.csv",
        "measurement_primary_summary": MEAS_PRIMARY_DIR / "measurement_only_v4_summary.md",
        "measurement_repeated_overall": MEAS_REP_DIR / "measurement_repeated_split_overall_metrics.csv",
        "measurement_repeated_range": MEAS_REP_DIR / "measurement_repeated_split_range_metrics.csv",
        "measurement_repeated_predictions": MEAS_REP_DIR / "measurement_repeated_split_predictions.csv",
        "measurement_repeated_summary": MEAS_REP_DIR / "measurement_repeated_split_summary.md",
        "as_oct_primary_overall": AS_PRIMARY_ENSEMBLE_DIR / "as_oct_v4_corrected_ensemble_overall_metrics.csv",
        "as_oct_primary_range": AS_PRIMARY_ENSEMBLE_DIR / "as_oct_v4_corrected_ensemble_range_metrics.csv",
        "as_oct_primary_predictions": AS_PRIMARY_ENSEMBLE_DIR / "as_oct_v4_corrected_ensemble_predictions.csv",
        "as_oct_primary_summary": AS_PRIMARY_ENSEMBLE_DIR / "as_oct_v4_corrected_ensemble_summary.md",
        "as_oct_repeated_overall": AS_REP_DIR / "corrected_as_oct_repeated_split_overall_metrics.csv",
        "as_oct_repeated_range": AS_REP_DIR / "corrected_as_oct_repeated_split_range_metrics.csv",
        "as_oct_repeated_predictions": AS_REP_DIR / "corrected_as_oct_repeated_split_predictions.csv",
        "as_oct_repeated_summary": AS_REP_DIR / "corrected_as_oct_repeated_split_summary.md",
        "fusion_primary_overall": FUSION_ENSEMBLE_DIR / "fusion_v4_ensemble_overall_metrics.csv",
        "fusion_primary_range": FUSION_ENSEMBLE_DIR / "fusion_v4_ensemble_range_metrics.csv",
        "fusion_primary_predictions": FUSION_ENSEMBLE_DIR / "fusion_v4_ensemble_predictions.csv",
        "fusion_primary_summary": FUSION_ENSEMBLE_DIR / "fusion_v4_ensemble_summary.md",
        "fusion_repeated_overall": FUSION_REP_DIR / "fusion_repeated_split_overall_metrics.csv",
        "fusion_repeated_range": FUSION_REP_DIR / "fusion_repeated_split_range_metrics.csv",
        "fusion_repeated_predictions": FUSION_REP_DIR / "fusion_repeated_split_predictions.csv",
        "fusion_repeated_summary": FUSION_REP_DIR / "fusion_repeated_split_summary.md",
    }
    for name, path in required.items():
        checks.append({"check": name, "path": path.relative_to(PROJECT).as_posix(), "exists": path.exists()})
        require_file(path)
    for name, path in [
        ("measurement_repeated_splits", MEAS_REP_DIR / "measurement_repeated_split_overall_metrics.csv"),
        ("as_oct_repeated_splits", AS_REP_DIR / "corrected_as_oct_repeated_split_overall_metrics.csv"),
        ("fusion_repeated_splits", FUSION_REP_DIR / "fusion_repeated_split_overall_metrics.csv"),
    ]:
        df = pd.read_csv(path)
        seeds = sorted(int(x) for x in df["split_seed"].unique())
        checks.append({"check": name, "path": ",".join(map(str, seeds)), "exists": seeds == SPLIT_SEEDS})
        if seeds != SPLIT_SEEDS:
            raise ValueError(f"{name} seeds mismatch: {seeds}")
    return pd.DataFrame(checks)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qc = qc_formal_outputs()
    primary = load_primary_comparison()
    repeated, ranges, _ = load_repeated_model_comparison()
    aggregate = aggregate_repeated(repeated, ranges)
    paired = paired_comparison(repeated)

    qc.to_csv(OUT / "final_baseline_qc_checks.csv", index=False, encoding="utf-8")
    save_csvs(primary, repeated, aggregate, paired)
    make_figures(repeated, aggregate, ranges)
    write_latex_tables(primary, aggregate)
    write_report(primary, repeated, aggregate, paired)
    write_freeze_record(primary, aggregate)
    print(f"Wrote final package to {OUT.relative_to(PROJECT).as_posix()}")
    print(f"Best repeated MAE: {aggregate.sort_values('MAE_mean').iloc[0]['model']}")


if __name__ == "__main__":
    main()

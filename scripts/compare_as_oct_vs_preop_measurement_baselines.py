"""Compare Batch 01 AS-OCT-only and preop measurement-only POD1 baselines.

This script only reads existing summary files and writes comparison reports.
It does not modify training outputs, manifests, checkpoints, or source data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_AS_OCT_GROUP = Path("artifacts/reports/as_oct_pod1_baseline_batch_01/group_summary.csv")
DEFAULT_AS_OCT_RUNS = Path("artifacts/reports/as_oct_pod1_baseline_batch_01/summary.csv")
DEFAULT_MEAS_GROUP = Path(
    "artifacts/reports/preop_measurement_baseline_batch_01/summary/group_summary_by_dataset_and_model.csv"
)
DEFAULT_MEAS_BEST = Path(
    "artifacts/reports/preop_measurement_baseline_batch_01/summary/best_val_selected_summary.csv"
)
DEFAULT_MEAS_ALL = Path(
    "artifacts/reports/preop_measurement_baseline_batch_01/summary/summary_all_runs.csv"
)
DEFAULT_OUT_DIR = Path("artifacts/reports/baseline_comparison_batch_01")


AS_OCT_ALLOWED_CLEAN_RUNS = {
    "as_oct_pod1_clean_resnet18_random_e30",
    "as_oct_pod1_clean_resnet18_random_seed2026_e30",
    "as_oct_pod1_clean_resnet18_random_seed3407_e30",
    "as_oct_pod1_clean_resnet18_imagenet_e30",
    "as_oct_pod1_clean_resnet18_imagenet_seed2026_e30",
    "as_oct_pod1_clean_resnet18_imagenet_seed3407_e30",
    "as_oct_pod1_clean_resnet18_imagenet_freeze_e30",
}

AS_OCT_FULL_SENSITIVITY_RUN = "as_oct_pod1_full_resnet18_imagenet_seed42_e30"

MEAS_MODEL_NAMES = {
    "linear_regression": "linear",
    "ridge_regression": "ridge",
    "mlp_regressor": "mlp",
    "random_forest": "random_forest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AS-OCT-only and preop measurement-only Batch 01 POD1 baselines."
    )
    parser.add_argument("--as_oct_group", type=Path, default=DEFAULT_AS_OCT_GROUP)
    parser.add_argument("--as_oct_runs", type=Path, default=DEFAULT_AS_OCT_RUNS)
    parser.add_argument("--measurement_group", type=Path, default=DEFAULT_MEAS_GROUP)
    parser.add_argument("--measurement_best", type=Path, default=DEFAULT_MEAS_BEST)
    parser.add_argument("--measurement_all", type=Path, default=DEFAULT_MEAS_ALL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_csv(path: Path, label: str, warnings: List[str]) -> Optional[pd.DataFrame]:
    if not path.exists():
        warnings.append(f"WARNING: missing {label}: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive for local files
        warnings.append(f"WARNING: failed to read {label}: {path} ({exc})")
        return None
    print(f"Loaded {label}: {path} ({len(df)} rows)")
    return df


def finite_std(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(result):
        return 0.0
    return result


def classify_as_oct_run(run_name: str, experiment_family: str) -> Optional[Dict[str, str]]:
    """Classify AS-OCT runs while keeping full-manifest sensitivity separate."""
    if run_name == AS_OCT_FULL_SENSITIVITY_RUN:
        return {
            "baseline_family": "as_oct_full_imagenet_finetune_sensitivity",
            "dataset_subset": "full_sensitivity",
            "model_name": "imagenet_finetune",
            "notes": "full manifest sensitivity experiment; not included in clean AS-OCT baseline comparison.",
        }
    if run_name not in AS_OCT_ALLOWED_CLEAN_RUNS:
        return None
    if experiment_family == "random_init":
        return {
            "baseline_family": "as_oct_clean_random_init",
            "dataset_subset": "clean_strict",
            "model_name": "random_init",
            "notes": "Clean AS-OCT-only ResNet18 random initialization.",
        }
    if experiment_family == "imagenet_finetune":
        return {
            "baseline_family": "as_oct_clean_imagenet_finetune",
            "dataset_subset": "clean_strict",
            "model_name": "imagenet_finetune",
            "notes": "Clean AS-OCT-only ResNet18 ImageNet pretrained full fine-tuning.",
        }
    if experiment_family == "imagenet_freeze":
        return {
            "baseline_family": "as_oct_clean_imagenet_freeze",
            "dataset_subset": "clean_strict",
            "model_name": "imagenet_freeze",
            "notes": "Clean AS-OCT-only ResNet18 ImageNet pretrained frozen backbone.",
        }
    return None


def build_as_oct_comparison(as_oct_runs: Optional[pd.DataFrame]) -> pd.DataFrame:
    if as_oct_runs is None or as_oct_runs.empty:
        return pd.DataFrame()
    run_rows: List[Dict[str, object]] = []
    for _, row in as_oct_runs.iterrows():
        run_name = str(row["run_name"])
        family = str(row["experiment_family"])
        info = classify_as_oct_run(run_name, family)
        if info is None:
            continue
        run_rows.append(
            {
                **info,
                "best_val_mae_um": row["best_val_mae_um"],
                "test_mae_um": row["test_mae_um"],
                "test_rmse_um": row["test_rmse_um"],
                "test_r2": row["test_r2"],
            }
        )
    run_df = pd.DataFrame(run_rows)
    if run_df.empty:
        return run_df

    rows: List[Dict[str, object]] = []
    for keys, group in run_df.groupby(["baseline_family", "dataset_subset", "model_name", "notes"], dropna=False):
        baseline_family, dataset_subset, model_name, notes = keys
        rows.append(
            {
                "baseline_family": baseline_family,
                "input_modality": "AS-OCT image",
                "dataset_subset": dataset_subset,
                "model_name": model_name,
                "n_runs": len(group),
                "test_mae_mean": group["test_mae_um"].mean(),
                "test_mae_std": group["test_mae_um"].std(ddof=1),
                "test_rmse_mean": group["test_rmse_um"].mean(),
                "test_rmse_std": group["test_rmse_um"].std(ddof=1),
                "test_r2_mean": group["test_r2"].mean(),
                "test_r2_std": group["test_r2"].std(ddof=1),
                "val_mae_mean": group["best_val_mae_um"].mean(),
                "val_mae_std": group["best_val_mae_um"].std(ddof=1),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def build_measurement_comparison(measurement_group: Optional[pd.DataFrame]) -> pd.DataFrame:
    if measurement_group is None or measurement_group.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for _, row in measurement_group.iterrows():
        dataset_type = str(row["dataset_type"])
        model_name_raw = str(row["model_name"])
        short_model = MEAS_MODEL_NAMES.get(model_name_raw, model_name_raw)
        rows.append(
            {
                "baseline_family": f"preop_measurement_{dataset_type}_{short_model}",
                "input_modality": "preoperative CASIA2 2DAnalysis measurements",
                "dataset_subset": dataset_type,
                "model_name": short_model,
                "n_runs": int(row["n_runs"]),
                "test_mae_mean": row["test_mae_mean"],
                "test_mae_std": row["test_mae_std"],
                "test_rmse_mean": row["test_rmse_mean"],
                "test_rmse_std": row["test_rmse_std"],
                "test_r2_mean": row["test_r2_mean"],
                "test_r2_std": row["test_r2_std"],
                "val_mae_mean": row["val_mae_mean"],
                "val_mae_std": row["val_mae_std"],
                "notes": (
                    "Preoperative measurement-only baseline; postoperative "
                    "2DAnalysis measurements are excluded from input features."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_best_val_comparison(
    as_oct_runs: Optional[pd.DataFrame], measurement_best: Optional[pd.DataFrame]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if as_oct_runs is not None and not as_oct_runs.empty:
        for _, row in as_oct_runs.iterrows():
            run_name = str(row["run_name"])
            info = classify_as_oct_run(run_name, str(row["experiment_family"]))
            if info is None:
                continue
            rows.append(
                {
                    "baseline_type": "as_oct_only",
                    "run_name": run_name,
                    "dataset_subset": info["dataset_subset"],
                    "selected_model_by_val": info["model_name"],
                    "seed": row.get("seed", ""),
                    "selected_val_mae_um": row["best_val_mae_um"],
                    "corresponding_test_mae_um": row["test_mae_um"],
                    "corresponding_test_rmse_um": row["test_rmse_um"],
                    "corresponding_test_r2": row["test_r2"],
                    "notes": info["notes"],
                }
            )
    if measurement_best is not None and not measurement_best.empty:
        for _, row in measurement_best.iterrows():
            rows.append(
                {
                    "baseline_type": "preop_measurement_only",
                    "run_name": row["run_name"],
                    "dataset_subset": row["dataset_type"],
                    "selected_model_by_val": row["selected_model_by_val"],
                    "seed": row["seed"],
                    "selected_val_mae_um": row["selected_val_mae_um"],
                    "corresponding_test_mae_um": row["corresponding_test_mae_um"],
                    "corresponding_test_rmse_um": row["corresponding_test_rmse_um"],
                    "corresponding_test_r2": row["corresponding_test_r2"],
                    "notes": "Best model selected by validation MAE within this run.",
                }
            )
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_无可用数据。_"
    view = df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: fmt(x))
    view = view.fillna("").astype(str)
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in view.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_markdown(
    path: Path,
    comparison_df: pd.DataFrame,
    best_df: pd.DataFrame,
    measurement_group: Optional[pd.DataFrame],
    warnings: List[str],
) -> None:
    as_oct_rows = comparison_df[comparison_df["input_modality"] == "AS-OCT image"]
    clean_as_oct_rows = as_oct_rows[as_oct_rows["dataset_subset"] == "clean_strict"]
    sensitivity_rows = as_oct_rows[as_oct_rows["dataset_subset"] == "full_sensitivity"]
    meas_rows = comparison_df[comparison_df["input_modality"] != "AS-OCT image"]
    ready_rows = meas_rows[meas_rows["dataset_subset"] == "ready"]
    strict_rows = meas_rows[meas_rows["dataset_subset"] == "strict"]

    official_df = comparison_df[comparison_df["dataset_subset"] != "full_sensitivity"]
    best = official_df.sort_values("test_mae_mean", na_position="last").head(1)
    best_line = ""
    if not best.empty:
        row = best.iloc[0]
        best_line = (
            f"当前正式 clean comparison 中 family-level 平均 test MAE 最低的是 `{row['baseline_family']}`，"
            f"test MAE mean = {fmt(row['test_mae_mean'])} um。"
        )

    lines = [
        "# Batch 01 AS-OCT-only 与术前 measurement-only baseline 对比",
        "",
        "## 数据与输入区别",
        "",
        (
            "本报告统一比较两条已完成的 Batch 01 POD1 vault regression pilot baseline。"
            "AS-OCT-only baseline 使用术前 AS-OCT raw image 作为输入；"
            "preop measurement-only baseline 使用真正术前 CASIA2 2DAnalysis 中人工核对后的 "
            "CCT、ACD Epi、ACD Endo、CLR、ATA 等结构化测量值作为输入。"
        ),
        "",
        (
            "两条路线均使用 POD1 manually verified vault mean 作为标签，并沿用 patient-level split。"
            "measurement-only 路线明确排除了术后 2DAnalysis measurement 作为输入特征，以避免信息泄漏。"
        ),
        "",
        "## AS-OCT clean 与 full sensitivity 的处理",
        "",
        (
            "clean AS-OCT-only baseline 只包含 clean strict manifest 上的 7 个 run："
            "3 个 random init、3 个 ImageNet fine-tune 和 1 个 ImageNet freeze。"
            "`as_oct_pod1_full_resnet18_imagenet_seed42_e30` 是 full manifest sensitivity experiment，"
            "不计入 clean AS-OCT baseline family。"
        ),
        "",
        (
            "full sensitivity row 仍保留在 summary 表中，用于展示 label QC / clean filtering 的敏感性；"
            "正式与 measurement-only 比较时应优先看 clean AS-OCT-only family。"
        ),
        "",
        "## Family-level 汇总",
        "",
        markdown_table(
            comparison_df,
            [
                "baseline_family",
                "n_runs",
                "val_mae_mean",
                "val_mae_std",
                "test_mae_mean",
                "test_mae_std",
                "test_rmse_mean",
                "test_r2_mean",
            ],
        ),
        "",
        "## Clean AS-OCT-only 结果",
        "",
        markdown_table(
            clean_as_oct_rows,
            ["baseline_family", "n_runs", "val_mae_mean", "test_mae_mean", "test_rmse_mean", "test_r2_mean"],
        ),
        "",
        (
            "Clean AS-OCT-only 结果显示，ImageNet fine-tune 在 validation MAE 上整体优于 random init 和 "
            "freeze backbone；但在当前很小的 test set 上，各 run 的 test MAE 仍有明显波动。"
        ),
        "",
        "## Full manifest sensitivity",
        "",
        markdown_table(
            sensitivity_rows,
            ["baseline_family", "n_runs", "val_mae_mean", "test_mae_mean", "test_rmse_mean", "test_r2_mean"],
        ),
        "",
        "## Preop Measurement-only 结果",
        "",
        "### Ready subset",
        "",
        markdown_table(
            ready_rows,
            ["baseline_family", "n_runs", "val_mae_mean", "test_mae_mean", "test_rmse_mean", "test_r2_mean"],
        ),
        "",
        "### Strict subset",
        "",
        markdown_table(
            strict_rows,
            ["baseline_family", "n_runs", "val_mae_mean", "test_mae_mean", "test_rmse_mean", "test_r2_mean"],
        ),
        "",
        (
            "Linear/Ridge 的结果较稳定，且保留较好的可解释性。MLP 在部分 setting 中 test MAE 较低，"
            "但需要注意此前训练中存在 convergence warning 风险，因此当前不宜将其作为稳健结论。"
            "Random Forest 在 validation 上可被选中，但 test 表现存在明显 val-test gap，提示可能受小样本和超参数选择影响。"
        ),
        "",
        "## 当前主要结论",
        "",
        (
            "在当前 batch_01 pilot split 中，preoperative measurement-only baseline shows stronger test "
            "performance than AS-OCT-only baseline, but further validation is needed due to the small test set."
        ),
        "",
        best_line,
        "",
        (
            "这说明术前 measurement features 具有较强预测价值，但当前 test set 很小，结果不能过度解释，"
            "也不能写成 measurement-only 一定优于 AS-OCT-only。更合理的下一步是构建 AS-OCT + measurement "
            "fusion baseline，检查图像信息与结构化术前测量是否互补。"
        ),
        "",
        "## Validation-selected run-level 对比",
        "",
        markdown_table(
            best_df,
            [
                "baseline_type",
                "run_name",
                "dataset_subset",
                "selected_model_by_val",
                "selected_val_mae_um",
                "corresponding_test_mae_um",
            ],
        ),
    ]

    if measurement_group is not None and not measurement_group.empty:
        lines.extend(
            [
                "",
                "## 术前 measurement-only 备注",
                "",
                (
                    "ready subset 包含 measurement_ready 与 measurement_ready_with_confirmed_outlier；"
                    "strict subset 仅包含 measurement_ready。ready/strict 的差异可用于评估已确认 outlier "
                    "样本对模型稳定性的影响。"
                ),
            ]
        )
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in warnings]])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_bar(
    df: pd.DataFrame,
    path: Path,
    y_col: str,
    err_col: str,
    title: str,
    ylabel: str,
    max_label_len: int = 28,
) -> None:
    if df.empty:
        return
    ordered = df.sort_values(y_col, na_position="last")
    labels = [
        label if len(label) <= max_label_len else label[: max_label_len - 3] + "..."
        for label in ordered["baseline_family"].astype(str)
    ]
    x = np.arange(len(ordered))
    y = ordered[y_col].astype(float).to_numpy()
    yerr = np.array([finite_std(v) for v in ordered[err_col]])

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.7), 5))
    ax.bar(x, y, yerr=yerr, capsize=4, color="#4C78A8", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_as_oct_vs_measurement(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    view = df[df["dataset_subset"] != "full_sensitivity"].copy()
    view["modality_group"] = np.where(
        view["input_modality"].eq("AS-OCT image"), "AS-OCT-only", "Preop measurement-only"
    )
    grouped = (
        view.groupby("modality_group", as_index=False)
        .agg(test_mae_mean=("test_mae_mean", "mean"), test_mae_std=("test_mae_mean", "std"))
        .sort_values("test_mae_mean")
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(grouped))
    ax.bar(
        x,
        grouped["test_mae_mean"],
        yerr=grouped["test_mae_std"].fillna(0),
        capsize=4,
        color=["#59A14F", "#4C78A8"][: len(grouped)],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["modality_group"], rotation=10, ha="right")
    ax.set_ylabel("Mean Test MAE (um)")
    ax.set_title("AS-OCT-only vs Preop Measurement-only")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_ready_vs_strict(df: pd.DataFrame, path: Path) -> None:
    meas = df[df["input_modality"] != "AS-OCT image"].copy()
    if meas.empty:
        return
    pivot = meas.pivot_table(index="model_name", columns="dataset_subset", values="test_mae_mean", aggfunc="mean")
    pivot = pivot.reindex(["linear", "ridge", "mlp", "random_forest"]).dropna(how="all")
    x = np.arange(len(pivot))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if "ready" in pivot:
        ax.bar(x - width / 2, pivot["ready"], width, label="ready", color="#4C78A8", edgecolor="black", linewidth=0.6)
    if "strict" in pivot:
        ax.bar(x + width / 2, pivot["strict"], width, label="strict", color="#F58518", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Mean Test MAE (um)")
    ax.set_title("Ready vs Strict Measurement Baselines")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_figures(df: pd.DataFrame, figures_dir: Path) -> List[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figures_dir / "test_mae_comparison_bar.png",
        figures_dir / "val_mae_comparison_bar.png",
        figures_dir / "as_oct_vs_measurement_test_mae.png",
        figures_dir / "ready_vs_strict_measurement_test_mae.png",
    ]
    plot_bar(df, paths[0], "test_mae_mean", "test_mae_std", "Baseline Test MAE Comparison", "Mean Test MAE (um)")
    plot_bar(df, paths[1], "val_mae_mean", "val_mae_std", "Baseline Validation MAE Comparison", "Mean Validation MAE (um)")
    plot_as_oct_vs_measurement(df, paths[2])
    plot_ready_vs_strict(df, paths[3])
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    warnings: List[str] = []

    as_oct_group = load_csv(args.as_oct_group, "AS-OCT group summary", warnings)
    as_oct_runs = load_csv(args.as_oct_runs, "AS-OCT run summary", warnings)
    measurement_group = load_csv(args.measurement_group, "measurement group summary", warnings)
    measurement_best = load_csv(args.measurement_best, "measurement best-val summary", warnings)
    measurement_all = load_csv(args.measurement_all, "measurement all-runs summary", warnings)
    if measurement_all is None:
        warnings.append("WARNING: measurement all-runs summary unavailable; comparison uses grouped data only.")

    if as_oct_group is not None:
        warnings.append(
            "NOTE: AS-OCT group_summary.csv was loaded for traceability, but AS-OCT families are re-aggregated "
            "from run-level summary.csv so the full sensitivity run stays separate."
        )

    comparison_df = pd.concat(
        [build_as_oct_comparison(as_oct_runs), build_measurement_comparison(measurement_group)],
        ignore_index=True,
    )
    comparison_df = comparison_df.sort_values(["input_modality", "dataset_subset", "test_mae_mean"]).reset_index(
        drop=True
    )
    best_df = build_best_val_comparison(as_oct_runs, measurement_best)

    comparison_path = args.output_dir / "baseline_comparison_summary.csv"
    best_path = args.output_dir / "best_val_selected_comparison.csv"
    md_path = args.output_dir / "baseline_comparison.md"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")
    best_df.to_csv(best_path, index=False, encoding="utf-8")
    write_markdown(md_path, comparison_df, best_df, measurement_group, warnings)
    figure_paths = write_figures(comparison_df, figures_dir)

    print(f"Compared baseline families: {len(comparison_df)}")
    clean_random = comparison_df[comparison_df["baseline_family"] == "as_oct_clean_random_init"]
    clean_finetune = comparison_df[comparison_df["baseline_family"] == "as_oct_clean_imagenet_finetune"]
    clean_freeze = comparison_df[comparison_df["baseline_family"] == "as_oct_clean_imagenet_freeze"]
    full_sensitivity = comparison_df[
        comparison_df["baseline_family"] == "as_oct_full_imagenet_finetune_sensitivity"
    ]
    measurement_families = comparison_df[comparison_df["input_modality"] != "AS-OCT image"]
    ready_families = measurement_families[measurement_families["dataset_subset"] == "ready"]
    strict_families = measurement_families[measurement_families["dataset_subset"] == "strict"]
    print(f"Clean AS-OCT random n_runs: {int(clean_random['n_runs'].iloc[0]) if not clean_random.empty else 0}")
    print(
        "Clean AS-OCT imagenet_finetune n_runs: "
        f"{int(clean_finetune['n_runs'].iloc[0]) if not clean_finetune.empty else 0}"
    )
    print(f"Clean AS-OCT imagenet_freeze n_runs: {int(clean_freeze['n_runs'].iloc[0]) if not clean_freeze.empty else 0}")
    print(f"Full sensitivity n_runs: {int(full_sensitivity['n_runs'].iloc[0]) if not full_sensitivity.empty else 0}")
    print(
        "Measurement ready/strict families: "
        f"ready={len(ready_families)}, strict={len(strict_families)}"
    )
    if not comparison_df.empty:
        official_df = comparison_df[comparison_df["dataset_subset"] != "full_sensitivity"]
        best_test = official_df.sort_values("test_mae_mean").iloc[0]
        print(
            "Lowest test MAE baseline in formal clean comparison: "
            f"{best_test['baseline_family']} ({float(best_test['test_mae_mean']):.2f} um)"
        )
        stable = official_df.copy()
        stable["test_mae_std_filled"] = stable["test_mae_std"].map(finite_std)
        stable = stable.sort_values(["test_mae_std_filled", "test_mae_mean"]).iloc[0]
        print(
            "Most stable baseline by test MAE std: "
            f"{stable['baseline_family']} (std {finite_std(stable['test_mae_std']):.2f} um)"
        )

    for warning in warnings:
        print(warning)
    print(f"Comparison summary: {comparison_path}")
    print(f"Best-val selected comparison: {best_path}")
    print(f"Markdown summary: {md_path}")
    print("Figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

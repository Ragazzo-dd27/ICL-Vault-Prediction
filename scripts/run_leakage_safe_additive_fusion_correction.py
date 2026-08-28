# -*- coding: utf-8 -*-
"""Leakage-safe formal correction for validation-tuned additive fusion.

This script uses existing matched validation/test predictions only. It does not
train models, run CNN inference, rerun RF, rerun G2, alter split assignments, or
search a new alpha protocol. For each outer split, alpha is selected only from
that split's validation rows and then evaluated once on that split's test rows.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MATCHED_ROOT = ROOT / "artifacts" / "v5_2_matched_unimodal"
AUDIT_ROOT = ROOT / "artifacts" / "v5_2_matched_fusion_audit"
OUT = AUDIT_ROOT / "additive_fusion_leakage_safe_formal"
G2_CORRECTED = ROOT / "artifacts" / "reports" / "v5_2_matched_fusion_audit" / "reliability_aware_gate_multiview_v1_formal_corrected"
FORMAL_REPORT = ROOT / "docs" / "experiments" / "additive_fusion_leakage_safe_formal_report.md"
PAPER_READINESS = ROOT / "docs" / "experiments" / "validation_tuned_additive_fusion_paper_readiness_audit_leakage_safe.md"
SEEDS = [42, 1001, 2002, 2026, 3407]
GRID = [round(i * 0.05, 2) for i in range(21)]
TIE_TOLERANCE_UM = 0.5
TOL = 1e-8


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing source artifact: {rel(path)}")
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing source artifact: {rel(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(label: str, actual: float, expected: float, tol: float = TOL) -> None:
    if not (math.isfinite(float(actual)) and math.isfinite(float(expected))):
        raise RuntimeError(f"{label} non-finite: actual={actual}, expected={expected}")
    if abs(float(actual) - float(expected)) > tol:
        raise RuntimeError(f"{label} mismatch: actual={actual}, expected={expected}")


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def standard_prediction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "prediction_um" not in out.columns and "pred_vault_um" in out.columns:
        out["prediction_um"] = out["pred_vault_um"]
    if "ground_truth_um" not in out.columns:
        if "vault_um" in out.columns:
            out["ground_truth_um"] = out["vault_um"]
        elif "vault_label_um" in out.columns:
            out["ground_truth_um"] = out["vault_label_um"]
    for col in ["split_seed", "split", "sample_id", "patient_id", "global_patient_uid", "eye", "ground_truth_um", "prediction_um"]:
        if col not in out.columns:
            raise RuntimeError(f"Prediction artifact missing column: {col}")
    out["ground_truth_um"] = pd.to_numeric(out["ground_truth_um"], errors="coerce")
    out["prediction_um"] = pd.to_numeric(out["prediction_um"], errors="coerce")
    return out


def load_sources() -> dict[str, Any]:
    alpha_resolution = read_json(AUDIT_ROOT / "alpha_semantics_resolution.json")
    leakage_audit = read_json(AUDIT_ROOT / "additive_fusion_cross_split_leakage_audit.json")
    frozen_alpha = read_json(AUDIT_ROOT / "additive_fusion_frozen_alpha.json")
    return {
        "alpha_resolution": alpha_resolution,
        "leakage_audit": leakage_audit,
        "frozen_alpha": frozen_alpha,
        "measurement_val": standard_prediction(read_csv(MATCHED_ROOT / "measurement" / "measurement_matched_validation_predictions.csv")),
        "as_oct_val": standard_prediction(read_csv(MATCHED_ROOT / "as_oct" / "as_oct_matched_validation_predictions.csv")),
        "test": read_csv(AUDIT_ROOT / "matched_three_way_predictions.csv"),
        "g2_per_eye": read_csv(G2_CORRECTED / "per_eye_predictions.csv"),
        "g2_per_split": read_csv(G2_CORRECTED / "per_split_metrics.csv"),
        "historical_resolved_aggregate": read_csv(AUDIT_ROOT / "additive_fusion_paper_readiness_audit_resolved" / "table_additive_aggregate.csv")
        if (AUDIT_ROOT / "additive_fusion_paper_readiness_audit_resolved" / "table_additive_aggregate.csv").exists()
        else pd.DataFrame(),
    }


def verify_source_protocol() -> None:
    script = (ROOT / "scripts" / "audit_v5_2_matched_fusion_interaction.py").read_text(encoding="utf-8")
    required_snippets = [
        "pred = alpha * m + (1 - alpha) * a",
        'search["pooled_validation_MAE"] <= best + 0.5',
        'sort_values("alpha", ascending=False)',
    ]
    missing = [s for s in required_snippets if s not in script]
    if missing:
        raise RuntimeError(f"Original alpha source/tie rule could not be verified: missing {missing}")


def build_validation_dataset(src: dict[str, Any]) -> pd.DataFrame:
    keys = ["split_seed", "split", "sample_id", "patient_id", "global_patient_uid", "eye", "ground_truth_um"]
    val = src["measurement_val"][keys + ["prediction_um"]].rename(columns={"prediction_um": "measurement_pred_um"}).merge(
        src["as_oct_val"][keys + ["prediction_um"]].rename(columns={"prediction_um": "as_oct_pred_um"}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    val["eye_key"] = val["global_patient_uid"].astype(str) + "__" + val["eye"].astype(str)
    return val


def build_test_dataset(src: dict[str, Any]) -> pd.DataFrame:
    test = src["test"].copy()
    test = test[test["usable_for_audit"].astype(bool)].copy()
    for col in ["ground_truth_um", "measurement_pred_um", "as_oct_pred_um", "fusion_pred_um"]:
        test[col] = pd.to_numeric(test[col], errors="coerce")
    g2 = src["g2_per_eye"][["split_seed", "sample_id", "g2_pred_um", "g2_abs_error_um"]].copy()
    test = test.merge(g2, on=["split_seed", "sample_id"], how="left", validate="one_to_one")
    test["eye_key"] = test["global_patient_uid"].astype(str) + "__" + test["eye"].astype(str)
    return test


def select_alpha_for_split(val: pd.DataFrame, seed: int) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    split_val = val[val["split_seed"].eq(seed)].copy()
    if split_val.empty:
        raise RuntimeError(f"No validation rows for seed {seed}")
    y = split_val["ground_truth_um"].to_numpy(float)
    m = split_val["measurement_pred_um"].to_numpy(float)
    a = split_val["as_oct_pred_um"].to_numpy(float)
    rows = []
    for alpha in GRID:
        pred = alpha * m + (1.0 - alpha) * a
        rows.append({"split_seed": seed, "alpha": alpha, "validation_mae": float(np.abs(pred - y).mean()), "n_val": int(len(split_val))})
    grid_df = pd.DataFrame(rows)
    best = float(grid_df["validation_mae"].min())
    best_alpha = float(grid_df.sort_values(["validation_mae", "alpha"], ascending=[True, True]).iloc[0]["alpha"])
    eligible = grid_df[grid_df["validation_mae"] <= best + TIE_TOLERANCE_UM].copy()
    selected = float(eligible.sort_values("alpha", ascending=False).iloc[0]["alpha"])
    selected_mae = float(grid_df.loc[np.isclose(grid_df["alpha"], selected), "validation_mae"].iloc[0])
    info = {
        "split_seed": seed,
        "n_val": int(len(split_val)),
        "selected_alpha_measurement": selected,
        "selected_alpha_as_oct": 1.0 - selected,
        "best_validation_mae": best,
        "best_validation_alpha": best_alpha,
        "selected_validation_mae": selected_mae,
        "tie_rule_affected_selected_alpha": bool(abs(selected - best_alpha) > 1e-12),
        "n_alphas_within_tie_tolerance": int(len(eligible)),
    }
    return selected, grid_df, info


def apply_additive(test: pd.DataFrame, alpha_rows: pd.DataFrame) -> pd.DataFrame:
    out = test.copy()
    alpha_map = alpha_rows.set_index("split_seed")["selected_alpha_measurement"].to_dict()
    out["selected_alpha_measurement"] = out["split_seed"].map(alpha_map).astype(float)
    out["selected_alpha_as_oct"] = 1.0 - out["selected_alpha_measurement"]
    out["additive_pred_um"] = out["selected_alpha_measurement"] * out["measurement_pred_um"] + out["selected_alpha_as_oct"] * out["as_oct_pred_um"]
    out["measurement_abs_error_um"] = (out["measurement_pred_um"] - out["ground_truth_um"]).abs()
    out["as_oct_abs_error_um"] = (out["as_oct_pred_um"] - out["ground_truth_um"]).abs()
    out["concat_abs_error_um"] = (out["fusion_pred_um"] - out["ground_truth_um"]).abs()
    out["additive_abs_error_um"] = (out["additive_pred_um"] - out["ground_truth_um"]).abs()
    out["oracle_best_of_two_abs_error_um"] = np.minimum(out["measurement_abs_error_um"], out["as_oct_abs_error_um"])
    out["best_unimodal_abs_error_um"] = out["oracle_best_of_two_abs_error_um"]
    out["delta_additive_vs_per_eye_best_unimodal_um"] = out["additive_abs_error_um"] - out["best_unimodal_abs_error_um"]
    return out


def metric_mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if len(df) else float("nan")


def per_split_metrics(per_eye: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        g = per_eye[per_eye["split_seed"].eq(seed)]
        measurement = metric_mean(g, "measurement_abs_error_um")
        as_oct = metric_mean(g, "as_oct_abs_error_um")
        best = min(measurement, as_oct)
        additive = metric_mean(g, "additive_abs_error_um")
        concat = metric_mean(g, "concat_abs_error_um")
        oracle = metric_mean(g, "oracle_best_of_two_abs_error_um")
        g2 = metric_mean(g, "g2_abs_error_um")
        denom = best - oracle
        rows.append(
            {
                "split_seed": seed,
                "n_test": int(len(g)),
                "measurement_rf_test_mae_um": measurement,
                "as_oct_v0_test_mae_um": as_oct,
                "matched_best_unimodal_mae_um": best,
                "leakage_safe_additive_test_mae_um": additive,
                "concat_test_mae_um": concat,
                "oracle_best_of_two_mae_um": oracle,
                "g2_test_mae_um": g2,
                "additive_delta_vs_measurement_um": additive - measurement,
                "additive_delta_vs_as_oct_um": additive - as_oct,
                "additive_delta_vs_matched_best_unimodal_um": additive - best,
                "additive_delta_vs_concat_um": additive - concat,
                "additive_delta_vs_g2_um": additive - g2,
                "additive_beats_measurement": bool(additive < measurement),
                "additive_beats_as_oct": bool(additive < as_oct),
                "additive_beats_matched_best_unimodal": bool(additive < best),
                "additive_beats_concat": bool(additive < concat),
                "additive_beats_g2": bool(additive < g2),
                "oracle_fraction_captured": float((best - additive) / denom) if abs(denom) > 1e-12 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def high_vault_metrics(per_eye: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        g = per_eye[(per_eye["split_seed"].eq(seed)) & (per_eye["vault_range"].astype(str).str.lower().eq("high"))]
        measurement = metric_mean(g, "measurement_abs_error_um")
        as_oct = metric_mean(g, "as_oct_abs_error_um")
        rows.append(
            {
                "split_seed": seed,
                "n_high": int(len(g)),
                "measurement_high_vault_mae_um": measurement,
                "as_oct_high_vault_mae_um": as_oct,
                "matched_best_unimodal_high_vault_mae_um": min(measurement, as_oct),
                "leakage_safe_additive_high_vault_mae_um": metric_mean(g, "additive_abs_error_um"),
                "concat_high_vault_mae_um": metric_mean(g, "concat_abs_error_um"),
                "g2_high_vault_mae_um": metric_mean(g, "g2_abs_error_um"),
                "oracle_high_vault_mae_um": metric_mean(g, "oracle_best_of_two_abs_error_um"),
            }
        )
    df = pd.DataFrame(rows)
    mean = {"split_seed": "mean", "n_high": int(df["n_high"].sum())}
    for col in df.columns:
        if col not in {"split_seed", "n_high"}:
            mean[col] = float(df[col].mean())
    return pd.concat([df, pd.DataFrame([mean])], ignore_index=True)


def range_metrics(per_eye: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = {
        "leakage_safe_additive": ("additive_pred_um", "additive_abs_error_um"),
        "measurement_rf": ("measurement_pred_um", "measurement_abs_error_um"),
        "as_oct_v0": ("as_oct_pred_um", "as_oct_abs_error_um"),
        "concat": ("fusion_pred_um", "concat_abs_error_um"),
        "g2": ("g2_pred_um", "g2_abs_error_um"),
    }
    for seed in SEEDS:
        g = per_eye[per_eye["split_seed"].eq(seed)]
        target_range = float(g["ground_truth_um"].max() - g["ground_truth_um"].min())
        target_sd = float(g["ground_truth_um"].std(ddof=1))
        for method, (pred_col, abs_col) in methods.items():
            pred_range = float(g[pred_col].max() - g[pred_col].min())
            pred_sd = float(g[pred_col].std(ddof=1))
            row = {
                "split_seed": seed,
                "method": method,
                "n": int(len(g)),
                "target_range_um": target_range,
                "prediction_range_um": pred_range,
                "prediction_range_ratio": pred_range / target_range if target_range > 0 else float("nan"),
                "target_sd_um": target_sd,
                "prediction_sd_um": pred_sd,
                "prediction_sd_to_target_sd": pred_sd / target_sd if target_sd > 0 else float("nan"),
            }
            for group in ["low", "medium", "high"]:
                sg = g[g["vault_range"].astype(str).str.lower().eq(group)]
                row[f"{group}_n"] = int(len(sg))
                row[f"{group}_signed_error_um"] = float((sg[pred_col] - sg["ground_truth_um"]).mean()) if len(sg) else float("nan")
                row[f"{group}_mae_um"] = float(sg[abs_col].mean()) if len(sg) else float("nan")
            rows.append(row)
    df = pd.DataFrame(rows)
    mean = df.groupby("method", as_index=False).agg(
        split_seed=("split_seed", lambda _: "mean"),
        n=("n", "sum"),
        target_range_um=("target_range_um", "mean"),
        prediction_range_um=("prediction_range_um", "mean"),
        prediction_range_ratio=("prediction_range_ratio", "mean"),
        target_sd_um=("target_sd_um", "mean"),
        prediction_sd_um=("prediction_sd_um", "mean"),
        prediction_sd_to_target_sd=("prediction_sd_to_target_sd", "mean"),
        low_n=("low_n", "sum"),
        low_signed_error_um=("low_signed_error_um", "mean"),
        low_mae_um=("low_mae_um", "mean"),
        medium_n=("medium_n", "sum"),
        medium_signed_error_um=("medium_signed_error_um", "mean"),
        medium_mae_um=("medium_mae_um", "mean"),
        high_n=("high_n", "sum"),
        high_signed_error_um=("high_signed_error_um", "mean"),
        high_mae_um=("high_mae_um", "mean"),
    )
    return pd.concat([df, mean[df.columns]], ignore_index=True)


def oracle_headroom(per_split: pd.DataFrame) -> pd.DataFrame:
    rows = per_split[
        [
            "split_seed",
            "matched_best_unimodal_mae_um",
            "leakage_safe_additive_test_mae_um",
            "oracle_best_of_two_mae_um",
            "oracle_fraction_captured",
        ]
    ].copy()
    vals = rows["oracle_fraction_captured"].dropna()
    summary = pd.DataFrame(
        [
            {
                "split_seed": "summary",
                "matched_best_unimodal_mae_um": float("nan"),
                "leakage_safe_additive_test_mae_um": float("nan"),
                "oracle_best_of_two_mae_um": float("nan"),
                "oracle_fraction_captured": float(vals.mean()),
                "oracle_fraction_median": float(vals.median()),
                "oracle_fraction_min": float(vals.min()),
                "oracle_fraction_max": float(vals.max()),
            }
        ]
    )
    return pd.concat([rows, summary], ignore_index=True)


def aggregate_metrics(per_split: pd.DataFrame, alpha: pd.DataFrame, high: pd.DataFrame, range_df: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    high_mean = high[high["split_seed"].astype(str).eq("mean")].iloc[0]
    add_range = range_df[(range_df["split_seed"].astype(str).eq("mean")) & (range_df["method"].eq("leakage_safe_additive"))].iloc[0]
    oracle_summary = oracle[oracle["split_seed"].astype(str).eq("summary")].iloc[0]
    return pd.DataFrame(
        [
            {
                "n_splits": int(len(per_split)),
                "additive_mae_mean_um": float(per_split["leakage_safe_additive_test_mae_um"].mean()),
                "additive_mae_std_um": float(per_split["leakage_safe_additive_test_mae_um"].std(ddof=1)),
                "wins_vs_measurement": int(per_split["additive_beats_measurement"].sum()),
                "wins_vs_as_oct": int(per_split["additive_beats_as_oct"].sum()),
                "wins_vs_matched_best_unimodal": int(per_split["additive_beats_matched_best_unimodal"].sum()),
                "wins_vs_concat": int(per_split["additive_beats_concat"].sum()),
                "wins_vs_g2": int(per_split["additive_beats_g2"].sum()),
                "alpha_mean": float(alpha["selected_alpha_measurement"].mean()),
                "alpha_std": float(alpha["selected_alpha_measurement"].std(ddof=1)),
                "alpha_median": float(alpha["selected_alpha_measurement"].median()),
                "alpha_min": float(alpha["selected_alpha_measurement"].min()),
                "alpha_max": float(alpha["selected_alpha_measurement"].max()),
                "mean_high_vault_additive_mae_um": float(high_mean["leakage_safe_additive_high_vault_mae_um"]),
                "mean_prediction_range_ratio": float(add_range["prediction_range_ratio"]),
                "mean_prediction_sd_to_target_sd": float(add_range["prediction_sd_to_target_sd"]),
                "mean_oracle_fraction_captured": float(oracle_summary["oracle_fraction_captured"]),
                "median_oracle_fraction_captured": float(oracle_summary["oracle_fraction_median"]),
                "min_oracle_fraction_captured": float(oracle_summary["oracle_fraction_min"]),
                "max_oracle_fraction_captured": float(oracle_summary["oracle_fraction_max"]),
            }
        ]
    )


def per_eye_diagnostic(per_eye: pd.DataFrame) -> pd.DataFrame:
    d = per_eye["delta_additive_vs_per_eye_best_unimodal_um"]
    return pd.DataFrame(
        [
            {
                "pooled_repeated_test_n": int(len(per_eye)),
                "unique_eyes": int(per_eye["global_sample_id"].nunique()),
                "mean_delta_um": float(d.mean()),
                "median_delta_um": float(d.median()),
                "fraction_improved": float((d < 0).mean()),
                "fraction_worsened": float((d > 0).mean()),
                "fraction_abs_delta_lt_10um": float((d.abs() < 10).mean()),
                "fraction_improvement_gt_25um": float((d < -25).mean()),
                "fraction_worsening_gt_25um": float((d > 25).mean()),
            }
        ]
    )


def paper_decision(per_split: pd.DataFrame, high: pd.DataFrame) -> tuple[str, str, str]:
    wins = int(per_split["additive_beats_matched_best_unimodal"].sum())
    mean_delta = float(per_split["additive_delta_vs_matched_best_unimodal_um"].mean())
    high_mean = high[high["split_seed"].astype(str).eq("mean")].iloc[0]
    high_delta = float(high_mean["leakage_safe_additive_high_vault_mae_um"] - high_mean["matched_best_unimodal_high_vault_mae_um"])
    if wins >= 4 and mean_delta <= -3.0 and high_delta <= 0:
        return "A", "PAPER-READY CORE METHOD", "central proposed method"
    if wins >= 3:
        return "B", "PAPER-READY SUPPORTING METHOD, NOT STRONG ENOUGH AS CORE NOVELTY", "primary simple fusion baseline / supporting fusion result"
    return "C", "SUPPORTING BASELINE ONLY", "supporting baseline only"


def qc_checks(val: pd.DataFrame, test: pd.DataFrame, per_eye: pd.DataFrame, alpha: pd.DataFrame, per_split: pd.DataFrame, aggregate: pd.DataFrame) -> dict[str, Any]:
    if sorted(per_split["split_seed"].astype(int).tolist()) != SEEDS:
        raise RuntimeError("Expected exactly five formal seeds")
    if len(alpha) != 5 or alpha["split_seed"].nunique() != 5:
        raise RuntimeError("Expected exactly one selected alpha per split")
    for seed in SEEDS:
        v = val[val["split_seed"].eq(seed)]
        t = test[test["split_seed"].eq(seed)]
        if len(v) != 56 or len(t) != 56:
            raise RuntimeError(f"Unexpected row counts for seed {seed}: val={len(v)}, test={len(t)}")
        if set(v["global_patient_uid"]) & set(t["global_patient_uid"]):
            raise RuntimeError(f"Same-split val/test patient overlap in seed {seed}")
        if set(v["eye_key"]) & set(t["eye_key"]):
            raise RuntimeError(f"Same-split val/test eye overlap in seed {seed}")
        if t["global_sample_id"].duplicated().any():
            raise RuntimeError(f"Duplicate test global_sample_id in seed {seed}")
    if per_eye["g2_pred_um"].isna().any() or per_eye["g2_abs_error_um"].isna().any():
        raise RuntimeError("Missing corrected G2 prediction/error after alignment")
    expected = per_eye["selected_alpha_measurement"] * per_eye["measurement_pred_um"] + per_eye["selected_alpha_as_oct"] * per_eye["as_oct_pred_um"]
    if not np.allclose(expected, per_eye["additive_pred_um"], atol=1e-10, rtol=0):
        raise RuntimeError("Additive predictions do not reproduce formula")
    for _, row in per_split.iterrows():
        g = per_eye[per_eye["split_seed"].eq(row["split_seed"])]
        assert_close(f"seed {row['split_seed']} additive MAE", g["additive_abs_error_um"].mean(), row["leakage_safe_additive_test_mae_um"])
    assert_close("aggregate additive mean", per_split["leakage_safe_additive_test_mae_um"].mean(), aggregate.iloc[0]["additive_mae_mean_um"])
    return {
        "exactly_5_outer_seeds": True,
        "one_selected_alpha_per_split": True,
        "validation_selection_per_split_only": True,
        "same_split_val_test_patient_overlap": 0,
        "same_split_val_test_eye_overlap": 0,
        "test_labels_used_for_alpha_selection": False,
        "matched_test_predictions_aligned": True,
        "duplicate_global_sample_id_within_split": 0,
        "additive_predictions_reproduce_formula": True,
        "test_maes_recompute_from_per_eye_errors": True,
        "aggregate_metrics_recompute_from_split_metrics": True,
        "pooled_validation_selection_in_final_protocol": False,
    }


def write_config(alpha: pd.DataFrame, qc: dict[str, Any], decision_code: str, decision_text: str) -> None:
    config = {
        "fusion_type": "validation-tuned additive late fusion",
        "alpha_semantics": "measurement_weight",
        "alpha_scope": "per_outer_split_validation_only",
        "alpha_grid": "0.00:0.05:1.00",
        "alpha_grid_values": GRID,
        "selection_metric": "MAE",
        "tie_rule": "Among alpha values whose validation MAE is within 0.5 um of the best validation MAE, choose the largest alpha.",
        "formula": "pred_additive = alpha * pred_measurement + (1 - alpha) * pred_as_oct",
        "test_labels_used_for_alpha_selection": False,
        "pooled_cross_split_validation": False,
        "protocol_status": "leakage_safe",
        "formal_split_seeds": SEEDS,
        "selected_alpha_by_split": {str(int(r.split_seed)): float(r.selected_alpha_measurement) for r in alpha.itertuples()},
        "qc": qc,
        "paper_readiness_decision_code": decision_code,
        "paper_readiness_decision": decision_text,
    }
    (OUT / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_manifest(src: dict[str, Any], qc: dict[str, Any], decision_code: str, decision_text: str, generated: list[Path]) -> None:
    path = OUT / "formal_correction_manifest.json"
    generated_all = [*generated, path]
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git(["rev-parse", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "formal_correction_type": "leakage_safe_per_outer_split_validation_alpha",
        "source_resolution_reports": [
            "docs/experiments/additive_fusion_alpha_semantics_resolution.md",
            "docs/experiments/additive_fusion_cross_split_leakage_audit.md",
            "artifacts/v5_2_matched_fusion_audit/alpha_semantics_resolution.json",
            "artifacts/v5_2_matched_fusion_audit/additive_fusion_cross_split_leakage_audit.json",
        ],
        "source_prediction_artifacts": [
            "artifacts/v5_2_matched_unimodal/measurement/measurement_matched_validation_predictions.csv",
            "artifacts/v5_2_matched_unimodal/as_oct/as_oct_matched_validation_predictions.csv",
            "artifacts/v5_2_matched_fusion_audit/matched_three_way_predictions.csv",
            "artifacts/reports/v5_2_matched_fusion_audit/reliability_aware_gate_multiview_v1_formal_corrected/per_eye_predictions.csv",
        ],
        "alpha_semantics": "measurement_weight",
        "alpha_scope": "per_outer_split_validation_only",
        "protocol_status": "leakage_safe",
        "qc_status": "PASSED",
        "qc": qc,
        "generated_files": [rel(p) for p in generated_all],
        "paper_readiness_decision_code": decision_code,
        "paper_readiness_decision": decision_text,
        "no_model_training": True,
        "no_inference": True,
        "no_g2_rerun": True,
        "patient_identifiers_in_manifest": False,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def historical_result(src: dict[str, Any]) -> dict[str, Any]:
    agg = src["historical_resolved_aggregate"]
    if agg.empty:
        return {
            "mae": 167.90690384312433,
            "std": 24.559625903323486,
            "wins": 3,
            "source": "fallback from resolved audit record",
        }
    row = agg[agg["Method"].eq("Corrected additive fusion")].iloc[0]
    return {
        "mae": float(row["Repeated MAE mean"]),
        "std": float(row["Repeated MAE SD"]),
        "wins": int(float(row["Wins vs matched best unimodal"])),
        "source": "artifacts/v5_2_matched_fusion_audit/additive_fusion_paper_readiness_audit_resolved/table_additive_aggregate.csv",
    }


def write_reports(
    src: dict[str, Any],
    alpha: pd.DataFrame,
    per_split: pd.DataFrame,
    aggregate: pd.DataFrame,
    high: pd.DataFrame,
    range_df: pd.DataFrame,
    oracle: pd.DataFrame,
    per_eye_diag: pd.DataFrame,
    decision_code: str,
    decision_text: str,
    manuscript_role: str,
) -> None:
    hist = historical_result(src)
    agg = aggregate.iloc[0]
    high_mean = high[high["split_seed"].astype(str).eq("mean")].iloc[0]
    range_add = range_df[(range_df["split_seed"].astype(str).eq("mean")) & (range_df["method"].eq("leakage_safe_additive"))].iloc[0]
    oracle_summary = oracle[oracle["split_seed"].astype(str).eq("summary")].iloc[0]
    alpha_view = alpha[
        [
            "split_seed",
            "n_val",
            "n_test",
            "selected_alpha_measurement",
            "selected_alpha_as_oct",
            "best_validation_mae",
            "selected_validation_mae",
            "tie_rule_affected_selected_alpha",
        ]
    ]
    report = f"""# Leakage-Safe Validation-Tuned Additive Fusion

## 1. Why protocol correction was required

The previous additive fusion result used one pooled global alpha selected from matched validation predictions across all repeated outer split seeds. A cross-split role audit found that every reported outer test split contained patients and eyes that had contributed labels to the pooled validation objective in another split. The previous global-alpha result is therefore retained as a historical diagnostic, not as the final unbiased repeated-test formal result.

## 2. Previous pooled-alpha contamination

Historical pooled-global alpha diagnostic:

- alpha = 0.35, Measurement weight
- MAE = {hist['mae']:.2f} +/- {hist['std']:.2f} um
- wins vs matched best unimodal = {hist['wins']}/5
- label: NOT VALID AS UNBIASED REPEATED-TEST FORMAL RESULT

## 3. Corrected per-outer-split protocol

For each formal outer split, alpha was selected using only that split's validation predictions and labels. The original grid, formula, selection metric, and tie rule were preserved.

Formula:

`pred_additive = alpha * Measurement + (1 - alpha) * AS-OCT`

Grid: `0.00, 0.05, ..., 1.00`

Tie rule: among alpha values whose validation MAE is within 0.5 um of the best validation MAE, choose the largest alpha.

## 4. Alpha selection

{md_table(alpha_view, 4)}

Alpha summary:

- mean = {agg['alpha_mean']:.3f}
- SD = {agg['alpha_std']:.3f}
- median = {agg['alpha_median']:.3f}
- min/max = {agg['alpha_min']:.3f}/{agg['alpha_max']:.3f}

## 5. Leakage prevention

Same-split validation/test patient overlap was zero for all five splits. Same-split validation/test eye overlap was zero for all five splits. Test labels were not used for alpha selection. No pooled cross-split validation selection remains in the final protocol.

## 6. Repeated held-out test performance

{md_table(per_split[['split_seed','n_test','measurement_rf_test_mae_um','as_oct_v0_test_mae_um','matched_best_unimodal_mae_um','leakage_safe_additive_test_mae_um','concat_test_mae_um','g2_test_mae_um','oracle_best_of_two_mae_um']], 3)}

Final leakage-safe additive repeated MAE:

`{agg['additive_mae_mean_um']:.2f} +/- {agg['additive_mae_std_um']:.2f} um`

## 7. Comparison with unimodal and concat

{md_table(per_split[['split_seed','additive_delta_vs_measurement_um','additive_delta_vs_as_oct_um','additive_delta_vs_matched_best_unimodal_um','additive_delta_vs_concat_um','additive_delta_vs_g2_um']], 3)}

Wins:

- vs Measurement RF: {int(agg['wins_vs_measurement'])}/5
- vs AS-OCT V0: {int(agg['wins_vs_as_oct'])}/5
- vs matched best unimodal: {int(agg['wins_vs_matched_best_unimodal'])}/5
- vs concat: {int(agg['wins_vs_concat'])}/5
- vs G2: {int(agg['wins_vs_g2'])}/5

## 8. High-Vault performance

{md_table(high, 3)}

Leakage-safe additive High-Vault MAE was `{high_mean['leakage_safe_additive_high_vault_mae_um']:.2f} um`.

## 9. Range compression

{md_table(range_df[range_df['split_seed'].astype(str).eq('mean')][['method','target_range_um','prediction_range_um','prediction_range_ratio','prediction_sd_to_target_sd','low_signed_error_um','medium_signed_error_um','high_signed_error_um']], 3)}

Leakage-safe additive prediction range ratio was `{range_add['prediction_range_ratio']:.3f}`. Range compression persists.

## 10. Oracle headroom

{md_table(oracle, 3)}

Mean oracle fraction captured was `{oracle_summary['oracle_fraction_captured']:.3f}`.

## 11. Per-eye diagnostic

{md_table(per_eye_diag, 3)}

The pooled repeated-test diagnostic may include the same physical eye in different outer test splits. This is a repeated-split robustness diagnostic, not an independent-eye cohort.

## 12. Comparison with historical global-alpha diagnostic

| Protocol | MAE mean | MAE SD | Wins vs matched best unimodal | Formal status |
| --- | ---: | ---: | ---: | --- |
| Historical pooled-global alpha=0.35 | {hist['mae']:.2f} | {hist['std']:.2f} | {hist['wins']}/5 | NOT VALID AS UNBIASED REPEATED-TEST FORMAL RESULT |
| Leakage-safe per-split validation alpha | {agg['additive_mae_mean_um']:.2f} | {agg['additive_mae_std_um']:.2f} | {int(agg['wins_vs_matched_best_unimodal'])}/5 | Final corrected formal result |

Differences between these rows reflect protocol correction, not model retraining or inference changes.

## 13. Implication for manuscript

The previous paper-readiness conclusion must be updated using the leakage-safe formal result. The corrected result should replace the pooled-global alpha diagnostic wherever an unbiased repeated-test additive comparator is required.

G2-only metrics remain unchanged, and G2 remains NO-GO unless its predefined decision criteria are changed for independent reasons.

## 14. Final formal additive result

Final leakage-safe additive fusion result:

`{agg['additive_mae_mean_um']:.2f} +/- {agg['additive_mae_std_um']:.2f} um`

Paper-readiness decision:

{decision_code}. {decision_text}

Recommended manuscript role:

{manuscript_role}
"""
    FORMAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FORMAL_REPORT.write_text(report, encoding="utf-8")

    paper = f"""# Validation-Tuned Additive Fusion Paper-Readiness Audit Leakage-Safe

## 1. Basis for re-evaluation

This paper-readiness conclusion supersedes the previous pooled-global alpha additive conclusion for formal repeated-test reporting. The previous `167.91 +/- 24.56 um` result is retained as a historical diagnostic only because cross-split role contamination was detected in the pooled global alpha selection.

## 2. Leakage-safe formal result

The corrected protocol selects alpha independently within each outer split validation set, using the original grid, formula, validation MAE metric, and tie rule. It then evaluates that alpha once on the corresponding held-out test split.

Final leakage-safe additive MAE:

`{agg['additive_mae_mean_um']:.2f} +/- {agg['additive_mae_std_um']:.2f} um`

Wins:

- vs matched best unimodal: {int(agg['wins_vs_matched_best_unimodal'])}/5
- vs concat: {int(agg['wins_vs_concat'])}/5
- vs G2: {int(agg['wins_vs_g2'])}/5

## 3. High-Vault and range behavior

High-Vault MAE:

`{high_mean['leakage_safe_additive_high_vault_mae_um']:.2f} um`

Prediction range ratio:

`{range_add['prediction_range_ratio']:.3f}`

The leakage-safe additive result should be interpreted with its High-Vault and range-compression limitations intact.

## 4. Oracle and per-eye diagnostics

Mean oracle fraction captured:

`{oracle_summary['oracle_fraction_captured']:.3f}`

Per-eye diagnostic:

{md_table(per_eye_diag, 3)}

## 5. Decision

{decision_code}. {decision_text}

Recommended manuscript role:

{manuscript_role}

This decision is derived from the leakage-safe outer-test result only. It does not reuse the previous pooled-global alpha paper-readiness decision.

## 6. G2 comparison

G2 was not rerun. Existing corrected G2 predictions/results were used. G2-only metrics and the G2 NO-GO decision remain unchanged. The additive-vs-G2 comparison is updated only through the corrected additive protocol.
"""
    PAPER_READINESS.parent.mkdir(parents=True, exist_ok=True)
    PAPER_READINESS.write_text(paper, encoding="utf-8")


def main() -> None:
    try:
        verify_source_protocol()
        src = load_sources()
        if src["alpha_resolution"].get("resolution_status") != "RESOLVED":
            raise RuntimeError("Alpha semantics are not resolved")
        if src["leakage_audit"].get("audit_status") != "CROSS-SPLIT_ROLE_CONTAMINATION_DETECTED":
            raise RuntimeError("Cross-split leakage audit status is not the expected contamination finding")
        val = build_validation_dataset(src)
        test = build_test_dataset(src)
        alpha_infos = []
        grid_frames = []
        for seed in SEEDS:
            _, grid_df, info = select_alpha_for_split(val, seed)
            alpha_infos.append(info)
            grid_frames.append(grid_df)
        alpha_df = pd.DataFrame(alpha_infos)
        grid_all = pd.concat(grid_frames, ignore_index=True)
        per_eye = apply_additive(test, alpha_df)
        per_split = per_split_metrics(per_eye)
        alpha_df = alpha_df.merge(per_split[["split_seed", "n_test"]], on="split_seed", how="left", validate="one_to_one")
        high = high_vault_metrics(per_eye)
        range_df = range_metrics(per_eye)
        oracle = oracle_headroom(per_split)
        aggregate = aggregate_metrics(per_split, alpha_df, high, range_df, oracle)
        per_eye_diag = per_eye_diagnostic(per_eye)
        decision_code, decision_text, manuscript_role = paper_decision(per_split, high)
        qc = qc_checks(val, test, per_eye, alpha_df, per_split, aggregate)

        OUT.mkdir(parents=True, exist_ok=True)
        outputs = {
            "per_split_alpha_selection.csv": alpha_df,
            "per_split_test_metrics.csv": per_split,
            "aggregate_metrics.csv": aggregate,
            "per_eye_test_predictions.csv": per_eye,
            "validation_alpha_grid.csv": grid_all,
            "high_vault_metrics.csv": high,
            "range_metrics.csv": range_df,
            "oracle_headroom.csv": oracle,
        }
        generated: list[Path] = []
        for name, df in outputs.items():
            path = OUT / name
            df.to_csv(path, index=False)
            generated.append(path)
        write_config(alpha_df, qc, decision_code, decision_text)
        generated.append(OUT / "experiment_config.json")
        write_reports(src, alpha_df, per_split, aggregate, high, range_df, oracle, per_eye_diag, decision_code, decision_text, manuscript_role)
        generated.extend([FORMAL_REPORT, PAPER_READINESS])
        (OUT / "experiment_report.md").write_text(FORMAL_REPORT.read_text(encoding="utf-8"), encoding="utf-8")
        generated.append(OUT / "experiment_report.md")
        write_manifest(src, qc, decision_code, decision_text, generated)

        agg = aggregate.iloc[0]
        high_mean = high[high["split_seed"].astype(str).eq("mean")].iloc[0]
        range_add = range_df[(range_df["split_seed"].astype(str).eq("mean")) & (range_df["method"].eq("leakage_safe_additive"))].iloc[0]
        oracle_summary = oracle[oracle["split_seed"].astype(str).eq("summary")].iloc[0]
        hist = historical_result(src)

        print("LEAKAGE-SAFE ADDITIVE FUSION FORMAL CORRECTION")
        print("")
        print("QC:")
        print("PASSED")
        print("")
        print("Per-split selected Measurement alpha:")
        for row in alpha_df.itertuples():
            print(f"seed{int(row.split_seed)}: {row.selected_alpha_measurement:.2f}")
        print("")
        print("Formal repeated additive MAE:")
        print(f"{agg['additive_mae_mean_um']:.6f} +/- {agg['additive_mae_std_um']:.6f} um")
        print("")
        print("Wins vs matched best unimodal:")
        print(f"{int(agg['wins_vs_matched_best_unimodal'])}/5")
        print("")
        print("Wins vs concat:")
        print(f"{int(agg['wins_vs_concat'])}/5")
        print("")
        print("Wins vs G2:")
        print(f"{int(agg['wins_vs_g2'])}/5")
        print("")
        print("High-Vault MAE:")
        print(f"{high_mean['leakage_safe_additive_high_vault_mae_um']:.6f}")
        print("")
        print("Prediction range ratio:")
        print(f"{range_add['prediction_range_ratio']:.6f}")
        print("")
        print("Mean oracle fraction captured:")
        print(f"{oracle_summary['oracle_fraction_captured']:.6f}")
        print("")
        print("Historical contaminated global-alpha MAE:")
        print(f"{hist['mae']:.2f} +/- {hist['std']:.2f} um (diagnostic only; NOT VALID AS UNBIASED REPEATED-TEST FORMAL RESULT)")
        print("")
        print("Paper-readiness decision:")
        print(decision_code)
        print("")
        print("Recommended manuscript role:")
        print(manuscript_role)
    except Exception as exc:
        print("LEAKAGE-SAFE ADDITIVE CORRECTION QC FAILED")
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()

"""Build repeated patient-level splits for combined batch_01 + batch_02.

This script only generates split CSVs and distribution reports for repeated
split stability evaluation. It does not modify source manifests, predictions,
checkpoints, or training code.

The repeated splits are patient-level based on global_patient_uid. Optional
patient_052 forced-test splits are stress/sensitivity splits only; they do not
replace the main split and are not a reason to remove patient_052.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [42, 2026, 3407, 1001, 2002]
DEFAULT_FORCED_TEST_SEEDS = [52052, 52053]
SPLIT_ORDER = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate repeated patient-level splits for the combined AS-OCT strict manifest."
    )
    parser.add_argument(
        "--manifest",
        default="",
        help=(
            "Input manifest. If omitted, the script auto-detects the combined "
            "batch_01 + batch_02 AS-OCT-only strict manifest."
        ),
    )
    parser.add_argument("--split_dir", default="data/splits", help="Directory for per-seed split CSV files.")
    parser.add_argument(
        "--report_dir",
        default="artifacts/reports/combined_batch_01_02/repeated_patient_split_stability",
        help="Directory for repeated split distribution reports.",
    )
    parser.add_argument("--seeds", default="42,2026,3407,1001,2002", help="Comma-separated standard split seeds.")
    parser.add_argument(
        "--forced_test_seeds",
        default="52052,52053",
        help="Comma-separated seeds for patient_052 forced-test stress splits.",
    )
    parser.add_argument(
        "--include_patient052_forced_test",
        action="store_true",
        help="Also generate patient_052 forced-test stress splits.",
    )
    parser.add_argument(
        "--only_forced_test",
        action="store_true",
        help="Generate only patient_052 forced-test stress splits and reuse existing standard split CSVs for reports.",
    )
    parser.add_argument(
        "--overwrite_standard",
        action="store_true",
        help="Allow rewriting existing standard repeated split CSVs. Default preserves existing standard split files.",
    )
    parser.add_argument("--forced_patient_token", default="patient_052")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--low_threshold", type=float, default=500.0)
    parser.add_argument("--high_threshold", type=float, default=800.0)
    parser.add_argument("--max_attempts", type=int, default=1000)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def auto_find_manifest(user_manifest: str) -> Path:
    if user_manifest:
        path = resolve_path(user_manifest)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return path

    preferred = PROJECT_ROOT / "data/manifests/vault_as_oct_only_pod1_manifest_combined_strict.csv"
    if preferred.exists():
        return preferred

    candidates = sorted((PROJECT_ROOT / "data/manifests").glob("*as_oct*combined*strict*.csv"))
    if not candidates:
        raise FileNotFoundError("Could not auto-detect combined AS-OCT strict manifest under data/manifests.")
    return candidates[0]


def parse_seeds(text: str, default: List[int]) -> List[int]:
    seeds = []
    for item in text.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    return seeds or default


def find_first_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of the expected columns exists: {list(candidates)}")
    return None


def assign_vault_range(value: float, low_threshold: float, high_threshold: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < low_threshold:
        return "low"
    if value <= high_threshold:
        return "medium"
    return "high"


def prepare_manifest(df: pd.DataFrame, low_threshold: float, high_threshold: float) -> pd.DataFrame:
    out = df.copy()
    vault_col = find_first_column(out, ["vault_um", "vault_label", "vault_label_um", "pod1_vault_mean_um", "label"])
    out["vault_um"] = pd.to_numeric(out[vault_col], errors="coerce")
    if out["vault_um"].isna().any():
        missing = int(out["vault_um"].isna().sum())
        raise ValueError(f"Found {missing} rows with missing vault label in {vault_col}.")

    if "global_patient_uid" not in out.columns:
        if {"batch_id", "patient_uid"}.issubset(out.columns):
            out["global_patient_uid"] = out["batch_id"].astype(str) + "__" + out["patient_uid"].astype(str)
        else:
            patient_col = find_first_column(out, ["patient_id", "patient_uid"])
            out["global_patient_uid"] = out[patient_col].astype(str)

    if "global_sample_id" not in out.columns:
        if {"batch_id", "sample_id"}.issubset(out.columns):
            out["global_sample_id"] = out["batch_id"].astype(str) + "__" + out["sample_id"].astype(str)
        else:
            out["global_sample_id"] = out["sample_id"].astype(str)

    if "patient_id" not in out.columns:
        out["patient_id"] = out["global_patient_uid"]
    if "patient_uid" not in out.columns:
        out["patient_uid"] = out["patient_id"]
    if "eye" not in out.columns:
        out["eye"] = out["eye_side"] if "eye_side" in out.columns else ""

    out["vault_range"] = out["vault_um"].map(lambda x: assign_vault_range(x, low_threshold, high_threshold))
    return out


def target_patient_counts(n_patients: int, ratios: Dict[str, float]) -> Dict[str, int]:
    n_train = int(round(n_patients * ratios["train"]))
    n_val = int(round(n_patients * ratios["val"]))
    if n_train + n_val > n_patients:
        n_val = max(0, n_patients - n_train)
    return {"train": n_train, "val": n_val, "test": n_patients - n_train - n_val}


def score_assignment(sample_df: pd.DataFrame, split_map: Dict[str, str], ratios: Dict[str, float]) -> float:
    assigned = sample_df.copy()
    assigned["split"] = assigned["global_patient_uid"].map(split_map)
    score = 0.0
    range_table = pd.crosstab(assigned["split"], assigned["vault_range"]).reindex(
        index=SPLIT_ORDER, columns=["low", "medium", "high"], fill_value=0
    )
    for split in SPLIT_ORDER:
        for vault_range in ["low", "medium", "high"]:
            if int(range_table.loc[split, vault_range]) == 0:
                score += 1000.0

    test_low = int(range_table.loc["test", "low"])
    test_high = int(range_table.loc["test", "high"])
    if test_low < 2:
        score += (2 - test_low) * 250.0
    if test_high < 1:
        score += 500.0

    total_samples = len(assigned)
    sample_counts = assigned["split"].value_counts().reindex(SPLIT_ORDER, fill_value=0)
    for split in SPLIT_ORDER:
        score += abs(sample_counts[split] / total_samples - ratios[split]) * 100.0

    return score


def build_one_split(
    sample_df: pd.DataFrame,
    seed: int,
    ratios: Dict[str, float],
    max_attempts: int,
    forced_test_patients: List[str] | None = None,
) -> Tuple[pd.DataFrame, float]:
    patients = sorted(sample_df["global_patient_uid"].unique())
    forced_test_patients = sorted(set(forced_test_patients or []))
    unknown_forced = sorted(set(forced_test_patients) - set(patients))
    if unknown_forced:
        raise ValueError(f"Forced test patients are not in manifest: {unknown_forced}")

    counts = target_patient_counts(len(patients), ratios)
    if len(forced_test_patients) > counts["test"]:
        raise ValueError("Forced test patient count exceeds target test patient count.")

    remaining_patients = [patient for patient in patients if patient not in forced_test_patients]
    remaining_counts = {
        "train": counts["train"],
        "val": counts["val"],
        "test": counts["test"] - len(forced_test_patients),
    }
    best_map: Dict[str, str] | None = None
    best_score = float("inf")
    rng = np.random.default_rng(seed)

    for _ in range(max_attempts):
        shuffled = np.array(remaining_patients, dtype=object)
        rng.shuffle(shuffled)
        split_map: Dict[str, str] = {patient: "test" for patient in forced_test_patients}
        start = 0
        for split in SPLIT_ORDER:
            stop = start + remaining_counts[split]
            for patient in shuffled[start:stop]:
                split_map[str(patient)] = split
            start = stop
        score = score_assignment(sample_df, split_map, ratios)
        if score < best_score:
            best_score = score
            best_map = split_map
        if score < 1e-9:
            break

    if best_map is None:
        raise RuntimeError(f"Could not build split for seed {seed}.")

    out = sample_df.copy()
    out["split"] = out["global_patient_uid"].map(best_map)
    out = out.sort_values(["split", "global_patient_uid", "eye", "global_sample_id"], kind="stable").reset_index(drop=True)
    return out, best_score


def find_forced_patients(df: pd.DataFrame, token: str) -> List[str]:
    searchable = pd.Series(False, index=df.index)
    for col in ["global_patient_uid", "patient_id", "patient_uid", "global_sample_id", "sample_id"]:
        if col in df.columns:
            searchable = searchable | df[col].astype(str).str.contains(token, case=False, na=False)
    patients = sorted(df.loc[searchable, "global_patient_uid"].unique())
    if not patients:
        raise ValueError(f"Could not find forced-test patient token: {token}")
    return patients


def split_distribution(seed: int, split_df: pd.DataFrame, split_type: str) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        group = split_df[split_df["split"] == split]
        rows.append(
            {
                "split_type": split_type,
                "split_seed": seed,
                "split": split,
                "n_samples": len(group),
                "n_patients": group["global_patient_uid"].nunique(),
                "n_low": int((group["vault_range"] == "low").sum()),
                "n_medium": int((group["vault_range"] == "medium").sum()),
                "n_high": int((group["vault_range"] == "high").sum()),
                "mean_vault": float(group["vault_um"].mean()) if len(group) else np.nan,
                "std_vault": float(group["vault_um"].std(ddof=1)) if len(group) > 1 else np.nan,
                "min_vault": float(group["vault_um"].min()) if len(group) else np.nan,
                "max_vault": float(group["vault_um"].max()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def patient052_rows(seed: int, split_df: pd.DataFrame, split_type: str, token: str = "patient_052") -> pd.DataFrame:
    searchable = pd.Series(False, index=split_df.index)
    for col in ["global_patient_uid", "patient_uid", "patient_id", "global_sample_id", "sample_id"]:
        if col in split_df.columns:
            searchable = searchable | split_df[col].astype(str).str.contains(token, case=False, na=False)
    cols = ["sample_id", "patient_id", "patient_uid", "global_patient_uid", "eye", "vault_um", "split"]
    cols = [col for col in cols if col in split_df.columns]
    out = split_df.loc[searchable, cols].copy()
    out.insert(0, "split_seed", seed)
    out.insert(0, "split_type", split_type)
    return out


def output_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "global_sample_id",
        "sample_id",
        "patient_id",
        "patient_uid",
        "global_patient_uid",
        "batch_id",
        "eye",
        "eye_side",
        "vault_um",
        "vault_range",
        "split",
        "label_qc_flag",
        "oct_path",
    ]
    return [col for col in preferred if col in df.columns]


def patient_cross_split(split_df: pd.DataFrame) -> bool:
    return bool((split_df.groupby("global_patient_uid")["split"].nunique() > 1).any())


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


def warn_for_distribution(distribution: pd.DataFrame) -> List[str]:
    warnings = []
    for _, row in distribution[distribution["split"] == "test"].iterrows():
        label = f"{row['split_type']} seed {int(row['split_seed'])}"
        if int(row["n_low"]) < 2:
            warnings.append(f"{label}: test split low-vault samples are few (n_low={int(row['n_low'])}).")
        if int(row["n_high"]) < 1:
            warnings.append(f"{label}: test split has no high-vault samples.")
    return warnings


def write_summary(
    path: Path,
    manifest_path: Path,
    prepared: pd.DataFrame,
    all_distribution: pd.DataFrame,
    patient052: pd.DataFrame,
    warnings: List[str],
    split_paths: Dict[Tuple[str, int], Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Repeated patient-level split generation summary",
        "",
        "本步骤为 combined batch_01 + batch_02 cohort 生成 repeated patient-level splits，用于后续 repeated split stability evaluation。",
        "本步骤只生成 split 和分布统计，不训练模型，不修改原始 manifest、prediction 或 checkpoint，也不覆盖既有主 split。",
        "",
        f"- 输入 manifest: `{manifest_path.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- 总样本数: {len(prepared)}",
        f"- 总 patient 数: {prepared['global_patient_uid'].nunique()}",
        "- split 规则: patient-level based on `global_patient_uid`; train/val/test target ratio approximately 70/15/15.",
        "- vault range: low < 500 um; medium 500-800 um; high > 800 um.",
        "- patient_052 未被特殊排除；forced-test stress split 只是将其固定放入 test 以评估稳定性。",
        "",
        "## Standard Repeated Splits",
        "",
    ]
    for (split_type, seed), split_path in sorted(split_paths.items(), key=lambda item: (item[0][0], item[0][1])):
        if split_type == "standard_repeated":
            lines.append(f"- seed {seed}: `{split_path.relative_to(PROJECT_ROOT).as_posix()}`")

    lines.extend(["", "## Patient 052 Forced-Test Stress Splits", ""])
    forced_paths = [(seed, path_) for (split_type, seed), path_ in split_paths.items() if split_type == "patient052_forced_test"]
    if forced_paths:
        for seed, split_path in sorted(forced_paths):
            lines.append(f"- seed {seed}: `{split_path.relative_to(PROJECT_ROOT).as_posix()}`")
    else:
        lines.append("- 本次未生成 forced-test stress split。")

    lines.extend(["", "## Train / Val / Test Sample Counts", ""])
    counts = all_distribution.pivot_table(
        index=["split_type", "split_seed"], columns="split", values="n_samples", aggfunc="first"
    ).reset_index()
    counts = counts[["split_type", "split_seed"] + SPLIT_ORDER]
    lines.extend(md_table(counts))

    lines.extend(["## Vault Range Distribution", ""])
    display_cols = [
        "split_type",
        "split_seed",
        "split",
        "n_samples",
        "n_patients",
        "n_low",
        "n_medium",
        "n_high",
        "mean_vault",
        "std_vault",
    ]
    lines.extend(md_table(all_distribution.sort_values(["split_type", "split_seed", "split"]), display_cols))

    lines.extend(["## Patient 052 Split Location", ""])
    if patient052.empty:
        lines.append("未在当前 manifest 中找到 patient_052。")
        lines.append("")
    else:
        cols = ["split_type", "split_seed", "sample_id", "patient_id", "eye", "vault_um", "split"]
        cols = [col for col in cols if col in patient052.columns]
        lines.extend(md_table(patient052.sort_values(["split_type", "split_seed", "eye"]), cols))

    lines.extend(["## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- 未发现 test split 中 low/high 样本覆盖不足的 warning。")

    lines.extend(
        [
            "",
            "## Stress Split Interpretation",
            "",
            "- `patient052_forced_test` split 只用于 stress / sensitivity evaluation，不替代原始主 split。",
            "- 这类 split 不作为删除 patient_052 的依据。",
            "- patient_052 已被医生确认 POD1 vault 标签、AS-OCT 图像、眼别、日期和 visit 对齐无误，因此必须作为真实有效的模型失败病例保留。",
            "- 所有 repeated splits 均保留全部样本；同一个 `global_patient_uid` 不会跨 train/val/test。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def read_existing_split(path: Path, manifest_prepared: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "global_sample_id" in df.columns:
        split_cols = ["global_sample_id", "split"]
        merged = manifest_prepared.drop(columns=["split"], errors="ignore").merge(df[split_cols], on="global_sample_id", how="inner")
    else:
        merged = manifest_prepared.copy()
        key_cols = ["sample_id", "split"]
        merged = merged.drop(columns=["split"], errors="ignore").merge(df[key_cols], on="sample_id", how="inner")
    if len(merged) != len(manifest_prepared):
        raise ValueError(f"Existing split {path} has {len(merged)} matched rows, expected {len(manifest_prepared)}.")
    return merged


def write_split(path: Path, split_df: pd.DataFrame, overwrite: bool = True) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df[output_columns(split_df)].to_csv(path, index=False, encoding="utf-8")
    return True


def print_distribution(seed: int, split_type: str, score: float | None, dist: pd.DataFrame) -> None:
    score_text = "" if score is None else f" score={score:.3f}"
    print(f"\n{split_type} seed {seed}{score_text}")
    print(dist[["split", "n_samples", "n_patients", "n_low", "n_medium", "n_high"]].to_string(index=False))


def main() -> None:
    args = parse_args()
    standard_seeds = parse_seeds(args.seeds, DEFAULT_SEEDS)
    forced_seeds = parse_seeds(args.forced_test_seeds, DEFAULT_FORCED_TEST_SEEDS)
    run_standard = not args.only_forced_test
    run_forced = args.include_patient052_forced_test or args.only_forced_test

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    ratio_sum = sum(ratios.values())
    if not np.isclose(ratio_sum, 1.0):
        ratios = {key: value / ratio_sum for key, value in ratios.items()}

    manifest_path = auto_find_manifest(args.manifest)
    split_dir = resolve_path(args.split_dir)
    report_dir = resolve_path(args.report_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    prepared = prepare_manifest(manifest, args.low_threshold, args.high_threshold)

    all_distributions: List[pd.DataFrame] = []
    all_patient052: List[pd.DataFrame] = []
    split_paths: Dict[Tuple[str, int], Path] = {}

    print(f"Input manifest: {manifest_path}")
    print(f"Total samples: {len(prepared)}")
    print(f"Total patients: {prepared['global_patient_uid'].nunique()}")

    if run_standard:
        for seed in standard_seeds:
            split_path = split_dir / f"repeated_patient_split_seed{seed}.csv"
            split_type = "standard_repeated"
            if split_path.exists() and not args.overwrite_standard:
                split_df = read_existing_split(split_path, prepared)
                score = None
                print(f"Preserved existing standard split: {split_path}")
            else:
                split_df, score = build_one_split(prepared, seed, ratios, args.max_attempts)
                write_split(split_path, split_df, overwrite=True)

            if patient_cross_split(split_df):
                raise RuntimeError(f"Patient leakage detected for standard split seed {seed}.")
            dist = split_distribution(seed, split_df, split_type)
            all_distributions.append(dist)
            all_patient052.append(patient052_rows(seed, split_df, split_type, args.forced_patient_token))
            split_paths[(split_type, seed)] = split_path
            print_distribution(seed, split_type, score, dist)
    else:
        for seed in standard_seeds:
            split_path = split_dir / f"repeated_patient_split_seed{seed}.csv"
            if split_path.exists():
                split_type = "standard_repeated"
                split_df = read_existing_split(split_path, prepared)
                if patient_cross_split(split_df):
                    raise RuntimeError(f"Patient leakage detected for existing standard split seed {seed}.")
                dist = split_distribution(seed, split_df, split_type)
                all_distributions.append(dist)
                all_patient052.append(patient052_rows(seed, split_df, split_type, args.forced_patient_token))
                split_paths[(split_type, seed)] = split_path

    if run_forced:
        forced_patients = find_forced_patients(prepared, args.forced_patient_token)
        print(f"Forced-test patients: {forced_patients}")
        for seed in forced_seeds:
            split_type = "patient052_forced_test"
            split_df, score = build_one_split(
                prepared,
                seed,
                ratios,
                args.max_attempts,
                forced_test_patients=forced_patients,
            )
            if patient_cross_split(split_df):
                raise RuntimeError(f"Patient leakage detected for forced-test split seed {seed}.")
            forced_rows = patient052_rows(seed, split_df, split_type, args.forced_patient_token)
            if forced_rows.empty or not (forced_rows["split"] == "test").all():
                raise RuntimeError(f"{args.forced_patient_token} was not fully assigned to test for seed {seed}.")

            split_path = split_dir / f"repeated_patient_split_patient052test_seed{seed}.csv"
            write_split(split_path, split_df, overwrite=True)
            dist = split_distribution(seed, split_df, split_type)
            all_distributions.append(dist)
            all_patient052.append(forced_rows)
            split_paths[(split_type, seed)] = split_path
            print_distribution(seed, split_type, score, dist)

    if not all_distributions:
        raise RuntimeError("No split distributions were generated or loaded.")

    distribution_df = pd.concat(all_distributions, ignore_index=True)
    patient052_df = pd.concat(all_patient052, ignore_index=True) if all_patient052 else pd.DataFrame()
    warnings = warn_for_distribution(distribution_df)

    distribution_path = report_dir / "split_distribution.csv"
    patient052_path = report_dir / "patient052_split_location.csv"
    summary_path = report_dir / "repeated_split_generation_summary.md"
    distribution_df.to_csv(distribution_path, index=False, encoding="utf-8")
    patient052_df.to_csv(patient052_path, index=False, encoding="utf-8")
    write_summary(summary_path, manifest_path, prepared, distribution_df, patient052_df, warnings, split_paths)

    print("\nPatient_052 split locations:")
    if patient052_df.empty:
        print("patient_052 not found.")
    else:
        cols = [col for col in ["split_type", "split_seed", "sample_id", "patient_id", "eye", "vault_um", "split"] if col in patient052_df.columns]
        print(patient052_df[cols].to_string(index=False))

    print("\nOutput files:")
    for path in split_paths.values():
        print(path)
    print(distribution_path)
    print(patient052_path)
    print(summary_path)


if __name__ == "__main__":
    main()

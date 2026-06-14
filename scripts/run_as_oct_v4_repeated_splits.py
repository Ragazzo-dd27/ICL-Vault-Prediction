"""Run combined v4 AS-OCT-only repeated splits sequentially.

This runner does not create splits and does not run fusion/measurement models.
It calls scripts/train_as_oct_v4_clean.py for each requested split seed using a
fixed model seed. split_seed controls the patient assignment manifest;
model_seed controls model initialization and training randomness.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data/splits/combined_batch_01_02_03_04_repeated"
PRIMARY_MANIFEST = PROJECT_ROOT / (
    "data/manifests/"
    "vault_as_oct_only_pod1_manifest_combined_batch_01_02_03_04_strict_split_seed42.csv"
)
BASE_REPORT_DIR = PROJECT_ROOT / "artifacts/reports/combined_batch_01_02_03_04/as_oct_only_repeated_splits"
BASE_CHECKPOINT_DIR = PROJECT_ROOT / "artifacts/checkpoints/combined_batch_01_02_03_04/as_oct_only_repeated_splits"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/train_as_oct_v4_clean.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AS-OCT-only v4 repeated splits sequentially.")
    parser.add_argument("--split_seeds", default="1001,2002,2026,3407")
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--skip_completed", action="store_true")
    return parser.parse_args()


def parse_seed_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"global_sample_id", "global_patient_uid", "split", "vault_label", "oct_path"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def split_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {split: int((df["split"] == split).sum()) for split in ["train", "val", "test"]}


def patient_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {
        split: int(df.loc[df["split"] == split, "global_patient_uid"].nunique())
        for split in ["train", "val", "test"]
    }


def validate_manifest(df: pd.DataFrame, manifest_path: Path) -> None:
    leakage = int((df.groupby("global_patient_uid")["split"].nunique() > 1).sum())
    if leakage != 0:
        raise ValueError(f"{manifest_path} has patient leakage: {leakage}")
    duplicated = int(df["global_sample_id"].duplicated().sum())
    if duplicated != 0:
        raise ValueError(f"{manifest_path} has duplicate global_sample_id: {duplicated}")
    missing_images = []
    for value in df["oct_path"].fillna(""):
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            missing_images.append(str(value))
            if len(missing_images) >= 5:
                break
    if missing_images:
        raise FileNotFoundError(f"{manifest_path} missing image examples: {missing_images}")


def check_seed42_consistency() -> None:
    repeated = read_manifest(SPLIT_DIR / "as_oct_manifest_split_seed42.csv")
    primary = read_manifest(PRIMARY_MANIFEST)
    cols = ["global_sample_id", "global_patient_uid", "split", "vault_label", "oct_path"]
    repeated_view = repeated[cols].sort_values("global_sample_id").reset_index(drop=True)
    primary_view = primary[cols].sort_values("global_sample_id").reset_index(drop=True)
    if len(repeated_view) != len(primary_view):
        raise ValueError(f"seed42 row count mismatch: repeated={len(repeated_view)}, primary={len(primary_view)}")
    if repeated_view["global_sample_id"].tolist() != primary_view["global_sample_id"].tolist():
        raise ValueError("seed42 global_sample_id order/set mismatch.")
    mismatches = []
    for col in ["global_patient_uid", "split", "oct_path"]:
        if not repeated_view[col].astype(str).equals(primary_view[col].astype(str)):
            mismatches.append(col)
    label_diff = (
        pd.to_numeric(repeated_view["vault_label"], errors="coerce")
        - pd.to_numeric(primary_view["vault_label"], errors="coerce")
    ).abs().max()
    if not pd.notna(label_diff) or float(label_diff) > 1e-9:
        mismatches.append("vault_label")
    if mismatches:
        raise ValueError(f"seed42 repeated manifest does not match primary manifest for: {mismatches}")
    print("seed42 repeated manifest matches the primary seed42 manifest exactly.", flush=True)


def output_paths(split_seed: int, model_seed: int) -> Dict[str, Path]:
    report_dir = BASE_REPORT_DIR / f"split_seed{split_seed}_model_seed{model_seed}"
    checkpoint_dir = BASE_CHECKPOINT_DIR / f"split_seed{split_seed}_model_seed{model_seed}"
    prefix = f"as_oct_v4_seed{model_seed}"
    return {
        "report_dir": report_dir,
        "checkpoint_dir": checkpoint_dir,
        "best": checkpoint_dir / "best.pth",
        "latest": checkpoint_dir / "latest.pth",
        "overall": report_dir / f"{prefix}_overall_metrics.csv",
        "range": report_dir / f"{prefix}_range_metrics.csv",
        "predictions": report_dir / f"{prefix}_predictions.csv",
        "log": report_dir / f"{prefix}_training_log.csv",
        "summary": report_dir / f"{prefix}_summary.md",
    }


def completed(paths: Dict[str, Path], expected_test_rows: int) -> bool:
    required = ["best", "latest", "overall", "range", "predictions", "log", "summary"]
    if not all(paths[key].exists() for key in required):
        return False
    try:
        predictions = pd.read_csv(paths["predictions"])
    except Exception:
        return False
    return len(predictions) == expected_test_rows


def run_one(split_seed: int, args: argparse.Namespace) -> None:
    manifest = SPLIT_DIR / f"as_oct_manifest_split_seed{split_seed}.csv"
    df = read_manifest(manifest)
    validate_manifest(df, manifest)
    counts = split_counts(df)
    pcounts = patient_counts(df)
    paths = output_paths(split_seed, args.model_seed)

    print("", flush=True)
    print(f"=== AS-OCT repeated split start: split_seed={split_seed}, model_seed={args.model_seed} ===", flush=True)
    print(f"manifest: {relative(manifest)}", flush=True)
    print(f"train/val/test eyes: {counts['train']} / {counts['val']} / {counts['test']}", flush=True)
    print(f"train/val/test patients: {pcounts['train']} / {pcounts['val']} / {pcounts['test']}", flush=True)
    print(f"report_dir: {relative(paths['report_dir'])}", flush=True)
    print(f"checkpoint_dir: {relative(paths['checkpoint_dir'])}", flush=True)

    if args.skip_completed and completed(paths, counts["test"]):
        print(f"skip_completed: split_seed={split_seed} already complete.", flush=True)
        return

    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--manifest",
        str(manifest),
        "--report_dir",
        str(paths["report_dir"]),
        "--checkpoint_dir",
        str(paths["checkpoint_dir"]),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.model_seed),
        "--num_workers",
        str(args.num_workers),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"split_seed={split_seed} failed with exit code {result.returncode}")
    if not completed(paths, counts["test"]):
        raise RuntimeError(f"split_seed={split_seed} finished but required outputs are incomplete.")
    print(f"=== AS-OCT repeated split complete: split_seed={split_seed} ===", flush=True)


def main() -> None:
    args = parse_args()
    if args.model_seed != 42:
        print(f"WARNING: requested model_seed={args.model_seed}; planned protocol fixes model_seed=42.", flush=True)
    check_seed42_consistency()
    for split_seed in parse_seed_list(args.split_seeds):
        try:
            run_one(split_seed, args)
        except Exception as exc:
            print(f"FAILED split_seed={split_seed}: {exc}", flush=True)
            raise


if __name__ == "__main__":
    main()

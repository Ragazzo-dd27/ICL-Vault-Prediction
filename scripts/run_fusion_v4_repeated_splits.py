"""Sequential runner for combined v4 fusion repeated patient-level splits.

This runner does not implement model logic. It invokes train_fusion_v4_clean.py
with one split at a time so a single GPU is used sequentially.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT / "data/splits/combined_batch_01_02_03_04_repeated"
REPORT_ROOT = PROJECT / "artifacts/reports/combined_batch_01_02_03_04/fusion_repeated_splits_fixed_model_seed42"
CKPT_ROOT = PROJECT / "artifacts/checkpoints/combined_batch_01_02_03_04/fusion_repeated_splits_fixed_model_seed42"
TRAIN_SCRIPT = PROJECT / "scripts/train_fusion_v4_clean.py"
REQUIRED_REPORT_FILES = [
    "fusion_v4_seed{seed}_overall_metrics.csv",
    "fusion_v4_seed{seed}_range_metrics.csv",
    "fusion_v4_seed{seed}_predictions.csv",
    "fusion_v4_seed{seed}_training_log.csv",
    "fusion_v4_seed{seed}_summary.md",
]


def parse_split_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fusion repeated split training sequentially.")
    parser.add_argument("--split_seeds", default="1001,2002,2026,3407")
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def split_counts(manifest: Path) -> tuple[dict[str, int], dict[str, int]]:
    df = pd.read_csv(manifest)
    eyes = {split: int((df["split"].astype(str) == split).sum()) for split in ["train", "val", "test"]}
    patients = {
        split: int(df.loc[df["split"].astype(str).eq(split), "global_patient_uid"].nunique())
        for split in ["train", "val", "test"]
    }
    return eyes, patients


def complete(report_dir: Path, checkpoint_dir: Path, model_seed: int, expected_test: int) -> bool:
    files = [report_dir / pattern.format(seed=model_seed) for pattern in REQUIRED_REPORT_FILES]
    files.extend([checkpoint_dir / "best.pth", checkpoint_dir / "latest.pth"])
    if not all(path.exists() for path in files):
        return False
    predictions = report_dir / f"fusion_v4_seed{model_seed}_predictions.csv"
    try:
        return len(pd.read_csv(predictions)) == expected_test
    except Exception:
        return False


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    split_seeds = parse_split_seeds(args.split_seeds)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = REPORT_ROOT / "fusion_repeated_runner_status.txt"
    log_path = REPORT_ROOT / "fusion_repeated_runner.log"
    append_log(log_path, f"[{datetime.now().isoformat(timespec='seconds')}] runner start")
    status_path.write_text("RUNNING\n", encoding="utf-8")

    for split_seed in split_seeds:
        manifest = SPLIT_DIR / f"fusion_manifest_split_seed{split_seed}.csv"
        if not manifest.exists():
            status_path.write_text(f"FAILED split_seed={split_seed}: missing manifest\n", encoding="utf-8")
            raise FileNotFoundError(manifest)
        report_dir = REPORT_ROOT / f"split_seed{split_seed}_model_seed{args.model_seed}"
        checkpoint_dir = CKPT_ROOT / f"split_seed{split_seed}_model_seed{args.model_seed}"
        eyes, patients = split_counts(manifest)
        print("=" * 80, flush=True)
        print(f"split_seed: {split_seed}", flush=True)
        print(f"model_seed: {args.model_seed}", flush=True)
        print(f"manifest: {manifest}", flush=True)
        print(f"train/val/test eyes: {eyes['train']} / {eyes['val']} / {eyes['test']}", flush=True)
        print(f"train/val/test patients: {patients['train']} / {patients['val']} / {patients['test']}", flush=True)
        print(f"report_dir: {report_dir}", flush=True)
        print(f"checkpoint_dir: {checkpoint_dir}", flush=True)

        if complete(report_dir, checkpoint_dir, args.model_seed, eyes["test"]):
            if args.overwrite:
                print("Existing complete run found; --overwrite set, training will run and overwrite files.", flush=True)
            else:
                print("Existing complete run found; skipping because --overwrite is not set.", flush=True)
                append_log(log_path, f"SKIP split_seed={split_seed} complete")
                continue
        elif report_dir.exists() or checkpoint_dir.exists():
            if not args.overwrite:
                message = (
                    f"Refusing to overwrite incomplete existing output for split_seed={split_seed}. "
                    "Use --overwrite only after manually confirming the directory is disposable."
                )
                status_path.write_text(f"FAILED split_seed={split_seed}: {message}\n", encoding="utf-8")
                raise RuntimeError(message)

        cmd = [
            args.python,
            "-u",
            str(TRAIN_SCRIPT),
            "--manifest",
            str(manifest),
            "--report_dir",
            str(report_dir),
            "--checkpoint_dir",
            str(checkpoint_dir),
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
        append_log(log_path, "RUN " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=PROJECT)
        if result.returncode != 0:
            status_path.write_text(f"FAILED split_seed={split_seed} returncode={result.returncode}\n", encoding="utf-8")
            raise RuntimeError(f"Training failed for split_seed={split_seed} with return code {result.returncode}")
        if not complete(report_dir, checkpoint_dir, args.model_seed, eyes["test"]):
            status_path.write_text(f"FAILED split_seed={split_seed}: missing/incomplete outputs after training\n", encoding="utf-8")
            raise RuntimeError(f"Run completed but output validation failed for split_seed={split_seed}")
        append_log(log_path, f"DONE split_seed={split_seed}")

    status_path.write_text("ALL_COMPLETE\n", encoding="utf-8")
    append_log(log_path, f"[{datetime.now().isoformat(timespec='seconds')}] runner complete")
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

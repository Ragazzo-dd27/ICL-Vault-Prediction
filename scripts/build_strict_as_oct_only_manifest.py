"""Build strict AS-OCT-only POD1 manifests with UBM loading disabled.

These outputs are for an AS-OCT-only POD1 baseline. UBM metadata may remain in
free-form columns if present, but the fields consumed by VaultDataset are set
so UBM is intentionally not loaded as a model input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRICT_NOTE = "Strict AS-OCT-only manifest; UBM intentionally disabled for this baseline."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict AS-OCT-only POD1 manifests.")
    parser.add_argument(
        "--full_in",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_full.csv",
        help="Input full AS-OCT-only POD1 manifest.",
    )
    parser.add_argument(
        "--clean_in",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean.csv",
        help="Input clean AS-OCT-only POD1 manifest.",
    )
    parser.add_argument(
        "--full_out",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_full_strict.csv",
        help="Output strict full AS-OCT-only POD1 manifest.",
    )
    parser.add_argument(
        "--clean_out",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean_strict.csv",
        help="Output strict clean AS-OCT-only POD1 manifest.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def append_strict_note(value: object) -> str:
    if pd.isna(value):
        return STRICT_NOTE
    text = str(value).strip()
    if not text:
        return STRICT_NOTE
    if STRICT_NOTE in text:
        return text
    return f"{text} | {STRICT_NOTE}"


def build_strict_manifest(df: pd.DataFrame) -> pd.DataFrame:
    strict = df.copy()

    strict["has_ubm"] = False
    strict["ubm_path"] = ""
    strict["ubm_alignment_status"] = "not_used_in_as_oct_only_baseline"
    strict["device_ubm"] = ""
    strict["notes"] = strict.get("notes", pd.Series([""] * len(strict))).map(append_strict_note)

    return strict


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def print_stats(full_strict: pd.DataFrame, clean_strict: pd.DataFrame) -> None:
    labels = pd.to_numeric(clean_strict["vault_label"], errors="coerce").dropna()
    print(f"full_strict rows: {len(full_strict)}")
    print(f"clean_strict rows: {len(clean_strict)}")
    print("has_ubm distribution:")
    for value, count in clean_strict["has_ubm"].value_counts(dropna=False).sort_index().items():
        print(f"  {value}: {count}")
    ubm_path_nonempty = int(clean_strict["ubm_path"].fillna("").astype(str).str.strip().ne("").sum())
    print(f"clean_strict nonempty ubm_path rows: {ubm_path_nonempty}")
    print("clean_strict split distribution:")
    for split, count in clean_strict["split"].value_counts().sort_index().items():
        print(f"  {split}: {count}")
    if labels.empty:
        print("clean_strict vault_label stats: no numeric labels")
    else:
        print(
            "clean_strict vault_label stats: "
            f"mean={labels.mean():.2f}, "
            f"std={labels.std():.2f}, "
            f"min={labels.min():.2f}, "
            f"max={labels.max():.2f}"
        )


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    full_in = resolve_project_path(args.full_in)
    clean_in = resolve_project_path(args.clean_in)
    full_out = resolve_project_path(args.full_out)
    clean_out = resolve_project_path(args.clean_out)

    full_strict = build_strict_manifest(pd.read_csv(full_in))
    clean_strict = build_strict_manifest(pd.read_csv(clean_in))

    write_csv(full_strict, full_out)
    write_csv(clean_strict, clean_out)
    print_stats(full_strict=full_strict, clean_strict=clean_strict)
    print(f"full_strict output: {relative_path(full_out)}")
    print(f"clean_strict output: {relative_path(clean_out)}")


if __name__ == "__main__":
    main()

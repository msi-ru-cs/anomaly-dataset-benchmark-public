#!/usr/bin/env python
"""
Prepare subgroup directories from any dataset `series` folder (NAB, MS, Exa).

This script autodetects the dataset name by reading the folder immediately
after `runs/` in the given path. For example:

    F:/.../runs/nab/.../series  -> dataset_name = "nab"
    F:/.../runs/ms/.../series   -> dataset_name = "ms"

Creates:

    working_data/subgroups/<dataset_name>/<prefix>/...

Usage:

  python prep_subgroups.py --series_dir "path/to/.../runs/nab/.../series"
"""

import argparse
import os
import shutil
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--series_dir",
        required=True,
        help="Folder containing all series subfolders.",
    )
    ap.add_argument(
        "--mode",
        choices=["copy", "link"],
        default="copy",
        help="copy = copy folders (default), link = create symlinks.",
    )
    return ap.parse_args()


def extract_dataset_name(series_dir: Path):
    """
    Extract dataset name from path:
    ... / runs / <dataset_name> / <run_id> / series

    Works for NAB, MS, Exa, etc.
    """
    parts = series_dir.parts
    dataset = None

    for i, p in enumerate(parts):
        if p.lower() == "runs" and i + 1 < len(parts):
            dataset = parts[i + 1]
            break

    return dataset or "unknown"


def main():
    args = parse_args()
    series_dir = Path(args.series_dir).resolve()

    if not series_dir.exists() or not series_dir.is_dir():
        raise SystemExit(f"[ERR] Not a directory: {series_dir}")

    # --- Detect dataset name (nab, ms, exa, etc.) ---
    dataset_name = extract_dataset_name(series_dir)
    print(f"[INFO] Detected dataset name: {dataset_name}")

    # --- Create working_data/subgroups/<dataset_name> ---
    root = Path.cwd()
    out_root = root / "working_data" / "subgroups" / dataset_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] series_dir: {series_dir}")
    print(f"[INFO] output:     {out_root}")
    print(f"[INFO] mode:       {args.mode}")

    prefixes = set()

    # --- Process each child folder inside `series` ---
    for child in sorted(series_dir.iterdir()):
        if not child.is_dir():
            continue

        name = child.name

        # Prefix rule: <prefix>__xxxxx.csv  (NAB style)
        if "__" in name:
            prefix = name.split("__", 1)[0]
        else:
            # fallback for ms/exa etc
            prefix = name.split("_", 1)[0]

        prefixes.add(prefix)

        group_dir = out_root / prefix
        group_dir.mkdir(parents=True, exist_ok=True)

        target = group_dir / name

        if target.exists():
            print(f"[SKIP] Exists already: {target}")
            continue

        if args.mode == "copy":
            print(f"[COPY] {child} -> {target}")
            shutil.copytree(child, target)
        else:
            print(f"[LINK] {child} -> {target}")
            os.symlink(child, target, target_is_directory=True)

    print("\n[INFO] Subgroups created:")
    for p in sorted(prefixes):
        print(f"  {p} -> working_data/subgroups/{dataset_name}/{p}")

    print("\n[DONE] Now point your sweep YAML `series_dir` to:")
    print(f"  working_data/subgroups/{dataset_name}/<prefix>/")


if __name__ == "__main__":
    main()



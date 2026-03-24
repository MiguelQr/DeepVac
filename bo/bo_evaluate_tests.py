#!/usr/bin/env python3
"""Compute per-run MSE from OPC samples and update the run-level table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bo_common import append_rows_csv


REQUIRED_SAMPLE_COLS = [
    "run_id",
    "timestamp",
    "kp",
    "ki",
    "kd",
    "temp",
    "temp_ref",
]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-csv", default="history/bo_samples.csv")
    ap.add_argument("--runs-csv", default="history/bo_runs.csv")
    ap.add_argument("--replace-runs", action="store_true", help="Overwrite runs-csv with recomputed values")
    ap.add_argument("--top", type=int, default=10)
    return ap


def main() -> None:
    args = build_parser().parse_args()

    sample_path = Path(args.samples_csv)
    if not sample_path.exists() or sample_path.stat().st_size == 0:
        raise FileNotFoundError(f"Samples file not found or empty: {args.samples_csv}")

    samples = pd.read_csv(sample_path)
    missing = [c for c in REQUIRED_SAMPLE_COLS if c not in samples.columns]
    if missing:
        raise ValueError(f"Samples file missing required columns: {missing}")

    samples["sq_error"] = (samples["temp_ref"].astype(float) - samples["temp"].astype(float)) ** 2

    grouped = samples.groupby("run_id", as_index=False).agg(
        start_ts=("timestamp", "min"),
        end_ts=("timestamp", "max"),
        num_samples=("timestamp", "count"),
        kp=("kp", "first"),
        ki=("ki", "first"),
        kd=("kd", "first"),
        mse=("sq_error", "mean"),
    )
    grouped["duration_s"] = grouped["end_ts"] - grouped["start_ts"]

    out_cols = ["run_id", "start_ts", "end_ts", "duration_s", "num_samples", "kp", "ki", "kd", "mse"]
    grouped = grouped[out_cols].sort_values("mse", ascending=True).reset_index(drop=True)

    out_path = Path(args.runs_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.replace_runs:
        grouped.to_csv(out_path, index=False)
    else:
        existing = pd.DataFrame()
        if out_path.exists() and out_path.stat().st_size > 0:
            existing = pd.read_csv(out_path)

        if existing.empty:
            grouped.to_csv(out_path, index=False)
        else:
            merged = pd.concat([existing, grouped], ignore_index=True)
            merged = merged.sort_values("start_ts").drop_duplicates(subset=["run_id"], keep="last")
            merged.to_csv(out_path, index=False)

    print(f"Wrote run evaluations to: {args.runs_csv}")
    print("Top runs by MSE:")

    top = grouped.head(max(args.top, 1))
    for i, row in top.iterrows():
        print(
            f"{i + 1:2d}. run_id={row['run_id']} mse={row['mse']:.6f} "
            f"kp={row['kp']:.4f} ki={row['ki']:.4f} kd={row['kd']:.4f}"
        )


if __name__ == "__main__":
    main()

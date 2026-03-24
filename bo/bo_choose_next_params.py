#!/usr/bin/env python3
"""Fit a Gaussian Process on all run-level MSE summaries and suggest next PID coefficients."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from bo_common import fit_gp_model, save_json, suggest_next_params


def parse_bounds(text: str) -> Tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid bounds format: {text}. Use low,high")
    lo = float(parts[0])
    hi = float(parts[1])
    if lo >= hi:
        raise ValueError(f"Invalid bounds {text}: low must be < high")
    return lo, hi


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # Root folder that contains per-run folders and/or a top-level bo_runs.csv
    ap.add_argument("--history-root", default="history")

    # Search bounds for candidate generation
    ap.add_argument("--kp-bounds", default="6,25")
    ap.add_argument("--ki-bounds", default="200,1000")
    ap.add_argument("--kd-bounds", default="5,20")

    # BO / active learning settings
    ap.add_argument("--n-candidates", type=int, default=7000)
    ap.add_argument("--xi", type=float, default=0.01, help="Expected Improvement exploration parameter")
    ap.add_argument("--seed", type=int, default=0)

    # Outputs
    ap.add_argument("--merged-runs-out", default="history/bo_all_runs_merged.csv")
    ap.add_argument("--model-out", default="history/bo_gp_model.pkl")
    ap.add_argument("--next-out", default="history/bo_next_params.json")

    return ap


def load_all_run_summaries(history_root: str) -> pd.DataFrame:
    """Load all bo_runs.csv files under history_root recursively.

    Expected columns in each file:
      run_id, kp, ki, kd, mse
    Additional columns are preserved if present.
    """
    root = Path(history_root)
    if not root.exists():
        raise FileNotFoundError(f"History root does not exist: {history_root}")

    csv_paths: List[Path] = sorted(root.rglob("bo_runs.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No bo_runs.csv files found under: {history_root}")

    frames: List[pd.DataFrame] = []
    required_cols = {"run_id", "kp", "ki", "kd", "mse"}

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable file: {path} ({exc})")
            continue

        if df.empty:
            print(f"[WARN] Skipping empty file: {path}")
            continue

        missing = required_cols - set(df.columns)
        if missing:
            print(f"[WARN] Skipping {path}: missing required columns {sorted(missing)}")
            continue

        df = df.copy()
        df["source_csv"] = str(path)
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid bo_runs.csv files were loaded.")

    merged = pd.concat(frames, ignore_index=True)

    # Keep only rows with valid numeric training columns
    for col in ["kp", "ki", "kd", "mse"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(subset=["run_id", "kp", "ki", "kd", "mse"]).copy()

    # If status exists, keep completed rows only
    if "status" in merged.columns:
        merged = merged[merged["status"].astype(str).str.lower() == "completed"].copy()

    # Deduplicate by run_id, keeping the last occurrence
    merged = merged.drop_duplicates(subset=["run_id"], keep="last").reset_index(drop=True)

    if merged.empty:
        raise RuntimeError("After cleaning, no valid training runs remain.")

    return merged


def main() -> None:
    args = build_parser().parse_args()

    bounds: Dict[str, Tuple[float, float]] = {
        "kp": parse_bounds(args.kp_bounds),
        "ki": parse_bounds(args.ki_bounds),
        "kd": parse_bounds(args.kd_bounds),
    }

    runs = load_all_run_summaries(args.history_root)

    # Save merged table for inspection / reproducibility
    Path(args.merged_runs_out).parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.merged_runs_out, index=False)

    if len(runs) < 3:
        raise RuntimeError(
            f"Need at least 3 completed runs to fit the GP reliably. Found only {len(runs)}."
        )

    model = fit_gp_model(runs)

    suggestion = suggest_next_params(
        model=model,
        bounds=bounds,
        n_candidates=args.n_candidates,
        xi=args.xi,
        random_state=args.seed,
    )

    # Best observed run for reference
    best_idx = runs["mse"].astype(float).idxmin()
    best_row = runs.loc[best_idx]

    payload = {
        "suggested": {
            "kp": float(suggestion["kp"]),
            "ki": float(suggestion["ki"]),
            "kd": float(suggestion["kd"]),
        },
        "prediction": {
            "mse_mean": float(suggestion["pred_mse"]),
            "mse_std": float(suggestion["pred_std"]),
            "expected_improvement": float(suggestion["expected_improvement"]),
            "best_observed_mse": float(suggestion["incumbent_mse"]),
        },
        "best_observed_run": {
            "run_id": str(best_row["run_id"]),
            "kp": float(best_row["kp"]),
            "ki": float(best_row["ki"]),
            "kd": float(best_row["kd"]),
            "mse": float(best_row["mse"]),
        },
        "meta": {
            "n_training_runs": int(model["n_samples"]),
            "n_loaded_runs": int(len(runs)),
            "history_root": str(Path(args.history_root).resolve()),
            "merged_runs_csv": str(Path(args.merged_runs_out).resolve()),
            "acquisition": "expected_improvement",
            "active_learning_rule": "maximize_expected_improvement",
            "xi": float(args.xi),
            "n_candidates": int(args.n_candidates),
            "seed": int(args.seed),
            "bounds": {
                "kp": [bounds["kp"][0], bounds["kp"][1]],
                "ki": [bounds["ki"][0], bounds["ki"][1]],
                "kd": [bounds["kd"][0], bounds["kd"][1]],
            },
        },
    }

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as fh:
        pickle.dump(model, fh)

    save_json(args.next_out, payload)

    print(f"Loaded {len(runs)} run summaries from: {args.history_root}")
    print(f"Saved merged runs table to: {args.merged_runs_out}")
    print()
    print("Best observed run so far:")
    print(f"  run_id={best_row['run_id']}")
    print(f"  kp={float(best_row['kp']):.6f}")
    print(f"  ki={float(best_row['ki']):.6f}")
    print(f"  kd={float(best_row['kd']):.6f}")
    print(f"  mse={float(best_row['mse']):.6f}")
    print()
    print("Suggested next PID gains (Bayesian optimization / active learning):")
    print(f"  kp={float(suggestion['kp']):.6f}")
    print(f"  ki={float(suggestion['ki']):.6f}")
    print(f"  kd={float(suggestion['kd']):.6f}")
    print()
    print(f"Predicted MSE mean={float(suggestion['pred_mse']):.6f}")
    print(f"Predicted MSE std={float(suggestion['pred_std']):.6f}")
    print(f"Expected improvement={float(suggestion['expected_improvement']):.6f}")
    print()
    print(f"Saved model to: {args.model_out}")
    print(f"Saved suggestion to: {args.next_out}")


if __name__ == "__main__":
    main()
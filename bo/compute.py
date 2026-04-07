#!/usr/bin/env python3
"""Fit a Gaussian Process on all summaries and suggest next PID coefficients"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from bo_common import fit_gp_model, save_json, suggest_next_params, parse_bounds, compute_run_cost, append_mae_column


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--kp-bounds", default="6,50")
    ap.add_argument("--ki-bounds", default="200,1000")
    ap.add_argument("--kd-bounds", default="5,20")

    ap.add_argument("--n-candidates", type=int, default=7000)
    ap.add_argument("--xi", type=float, default=0.01, help="Expected Improvement exploration parameter")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--history-root", default="history")
    ap.add_argument("--params-history", default="history/bo_params_history.json")

    ap.add_argument("--merged-runs-out", default="history/bo_all_runs.csv")
    ap.add_argument("--model-out", default="history/bo_gp_model.pkl")
    ap.add_argument("--next-out", default="history/bo_next_params.json")
    ap.add_argument("--history-csv", default="history/bo_training_progress.csv")
    ap.add_argument("--suggestions-dir", default="history/suggestions")

    return ap


def write_params_history(path: str, entry: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([entry], indent=2), encoding="utf-8")


def append_row_csv(path: str, row: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists() or p.stat().st_size == 0
    with p.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_all_run_summaries(
    history_root: str,
    near_temp: float = 2.0,
    overshoot_weight: float = 10.0,
    time_below_weight: float = 0.02,
    rewrite_missing_cost: bool = True,
) -> pd.DataFrame:
    """
    Load all bo_runs.csv files under history_root
    """
    root = Path(history_root)
    if not root.exists():
        raise FileNotFoundError(f"History root does not exist: {history_root}")

    csv_paths: List[Path] = sorted(root.rglob("bo_runs.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No bo_runs.csv files found")

    frames: List[pd.DataFrame] = []
    base_required = {"run_id", "kp", "ki", "kd"}

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable file: {path} ({exc})")
            continue

        if df.empty:
            print(f"[WARN] Skipping empty file: {path}")
            continue

        missing_base = base_required - set(df.columns)
        if missing_base:
            print(f"[WARN] Skipping {path}: missing required columns {sorted(missing_base)}")
            continue

        df = df.copy()

        if "cost" not in df.columns:
            path = path.with_name("bo_samples.csv")

            try:
                samples_df = pd.read_csv(path)
            except Exception as exc:
                print(f"[WARN] Skipping {path}: could not read file {path} ({exc})")
                continue

            try:
                samples_df = append_mae_column(samples_df)
                metrics = compute_run_cost(
                    samples_df,
                    near_temp=near_temp,
                    overshoot_weight=overshoot_weight,
                    time_below_weight=time_below_weight,
                )
            except Exception as exc:
                print(f"[WARN] Skipping {path}: failed to compute cost from {path} ({exc})")
                continue

            df["cost"] = float(metrics["cost"])
            df["tail_mae"] = metrics["tail_mae"]
            df["negative_overshoot"] = metrics["negative_overshoot"]
            df["time_below_0"] = metrics["time_below_0"]
            df["min_temp"] = metrics["min_temp"]
            df["tail_start_index"] = metrics["tail_start_index"]

            if rewrite_missing_cost:
                try:
                    df.to_csv(path, index=False)
                    print(f"[INFO] Added cost to: {path}")
                except Exception as exc:
                    print(f"[WARN] Could not rewrite {path} with cost column ({exc})")

        df["source_csv"] = str(path)
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid bo_runs.csv files were loaded.")

    merged = pd.concat(frames, ignore_index=True)

    for col in ["kp", "ki", "kd", "cost"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(subset=["run_id", "kp", "ki", "kd", "cost"]).copy()

    if "status" in merged.columns:
        merged = merged[merged["status"].astype(str).str.lower() == "completed"].copy()

    merged = merged.drop_duplicates(subset=["run_id"], keep="last").reset_index(drop=True)

    return merged

def main() -> None:
    args = build_parser().parse_args()

    bounds: Dict[str, Tuple[float, float]] = {
        "kp": parse_bounds(args.kp_bounds),
        "ki": parse_bounds(args.ki_bounds),
        "kd": parse_bounds(args.kd_bounds),
    }

    runs = load_all_run_summaries(args.history_root)

    Path(args.merged_runs_out).parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.merged_runs_out, index=False)

    if not len(runs):
        raise RuntimeError(
            f"No runs found"
        )

    runs_for_fit = runs.copy()
    runs_for_fit["mse"] = runs_for_fit["cost"].astype(float)

    model = fit_gp_model(runs_for_fit)

    suggestion = suggest_next_params(
        model=model,
        bounds=bounds,
        n_candidates=args.n_candidates,
        xi=args.xi,
        random_state=args.seed,
    )

    best_idx = runs["cost"].astype(float).idxmin()
    best_row = runs.loc[best_idx]

    payload = {
        "suggested": {
            "kp": float(suggestion["kp"]),
            "ki": float(suggestion["ki"]),
            "kd": float(suggestion["kd"]),
        },
        "prediction": {
            "cost_mean": float(suggestion["pred_mse"]),
            "cost_std": float(suggestion["pred_std"]),
            "expected_improvement": float(suggestion["expected_improvement"]),
            "best_observed_cost": float(best_row["cost"]),
        },
        "best_observed_run": {
            "run_id": str(best_row["run_id"]),
            "kp": float(best_row["kp"]),
            "ki": float(best_row["ki"]),
            "kd": float(best_row["kd"]),
            "cost": float(best_row["cost"]),
            "tail_mae": None if pd.isna(best_row.get("tail_mae")) else float(best_row["tail_mae"]),
            "negative_overshoot": None if pd.isna(best_row.get("negative_overshoot")) else float(best_row["negative_overshoot"]),
            "time_below_0": None if pd.isna(best_row.get("time_below_0")) else float(best_row["time_below_0"]),
            "min_temp": None if pd.isna(best_row.get("min_temp")) else float(best_row["min_temp"]),
        },
        "meta": {
            "run_id": "",
            "n_training_runs": int(model["n_samples"]),
            "n_loaded_runs": int(len(runs)),
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

    generated_at = int(time.time())
    iteration_snapshot = {
        "generated_at": generated_at,
        **payload,
    }

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as fh:
        pickle.dump(model, fh)

    save_json(args.next_out, payload)

    snapshot_dir = Path(args.suggestions_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_name = f"bo_next_params_{generated_at}.json"
    snapshot_path = snapshot_dir / snapshot_name
    save_json(str(snapshot_path), iteration_snapshot)

    params_entry = {
        "timestamp": int(time.time()),
        "run_id": "",
        "kp": "",
        "ki": "",
        "kd": "",
        "mse": "",
        "next_suggested": payload["suggested"],
        "prediction": payload["prediction"],
        "meta": {
            "run_id": "",
            "n_training_runs": int(model["n_samples"]),
            "n_loaded_runs": int(len(runs)),
            "acquisition": "expected_improvement",
            "active_learning_rule": "maximize_expected_improvement",
            "xi": float(args.xi),
            "n_candidates": int(args.n_candidates),
            "seed": int(args.seed),
            "test_duration": "",
            "dt": "",
            "progress_every": "",
        },
    }

    write_params_history(args.params_history, params_entry)

    progress_row = {
        "generated_at": generated_at,
        "n_training_runs": int(model["n_samples"]),
        "n_loaded_runs": int(len(runs)),
        "best_run_id": str(best_row["run_id"]),
        "best_observed_cost": float(best_row["cost"]),
        "suggested_kp": float(suggestion["kp"]),
        "suggested_ki": float(suggestion["ki"]),
        "suggested_kd": float(suggestion["kd"]),
        "pred_cost_mean": float(suggestion["pred_mse"]),
        "pred_cost_std": float(suggestion["pred_std"]),
        "expected_improvement": float(suggestion["expected_improvement"]),
        "next_out": str(Path(args.next_out).resolve()),
        "snapshot_json": str(snapshot_path.resolve()),
    }
    append_row_csv(args.history_csv, progress_row)

    print(f"Loaded {len(runs)} run summaries from: {args.history_root}")
    print(f"Saved runs table to: {args.merged_runs_out}")
    print()
    print("Best observed run so far:")
    print(f"  run_id={best_row['run_id']}")
    print(f"  kp={float(best_row['kp']):.6f}")
    print(f"  ki={float(best_row['ki']):.6f}")
    print(f"  kd={float(best_row['kd']):.6f}")
    print(f"  cost={float(best_row['cost']):.6f}")
    if "tail_mae" in best_row:
        print(f"  tail_mae={best_row['tail_mae']}")
    if "negative_overshoot" in best_row:
        print(f"  negative_overshoot={best_row['negative_overshoot']}")
    if "time_below_0" in best_row:
        print(f"  time_below_0={best_row['time_below_0']}")
    print()
    print("Suggested next PID gains:")
    print(f"  kp={float(suggestion['kp']):.6f}")
    print(f"  ki={float(suggestion['ki']):.6f}")
    print(f"  kd={float(suggestion['kd']):.6f}")
    print()
    print(f"Predicted cost mean={float(suggestion['pred_mse']):.6f}")
    print(f"Predicted cost std={float(suggestion['pred_std']):.6f}")
    print(f"Expected improvement={float(suggestion['expected_improvement']):.6f}")
    print()
    print(f"Saved model to: {args.model_out}")
    print(f"Saved suggestion to: {args.next_out}")
    print(f"Saved suggestion snapshot to: {snapshot_path}")
    print(f"Rewrote params history to: {args.params_history}")
    print(f"Appended training progress to: {args.history_csv}")


if __name__ == "__main__":
    main()
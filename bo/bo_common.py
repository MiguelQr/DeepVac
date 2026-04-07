#!/usr/bin/env python3
"""Shared utilities for OPC-based Bayesian optimization of PID gains."""

from __future__ import annotations

import csv
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from opcua import Client
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


@dataclass
class OPCNodeMap:
    temp: str
    temp_ref: str
    temp_raw: str
    temp_u: str
    temp_u_p: str
    temp_u_i: str
    temp_u_d: str


def timestamp_now() -> float:
    return time.time()


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def read_opc_snapshot(client: Client, nodes: OPCNodeMap) -> Dict[str, float]:
    return {
        "temp": float(client.get_node(nodes.temp).get_value()),
        "temp_ref": float(client.get_node(nodes.temp_ref).get_value()),
        "temp_raw": float(client.get_node(nodes.temp_raw).get_value()),
        "temp_u": float(client.get_node(nodes.temp_u).get_value()),
        "temp_u_p": float(client.get_node(nodes.temp_u_p).get_value()),
        "temp_u_i": float(client.get_node(nodes.temp_u_i).get_value()),
        "temp_u_d": float(client.get_node(nodes.temp_u_d).get_value()),
    }


def run_opc_test(
    endpoint: str,
    nodes: OPCNodeMap,
    kp: float,
    ki: float,
    kd: float,
    duration_s: float,
    dt_s: float,
    run_id: Optional[str] = None,
    progress_every_s: float = 30.0,
    verbose: bool = True,
    read_retries: int = 2,
    read_retry_delay_s: float = 0.25,
    max_consecutive_failures: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if dt_s <= 0:
        raise ValueError("dt_s must be > 0")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if progress_every_s < 0:
        raise ValueError("progress_every_s must be >= 0")
    if read_retries < 0:
        raise ValueError("read_retries must be >= 0")
    if read_retry_delay_s < 0:
        raise ValueError("read_retry_delay_s must be >= 0")
    if max_consecutive_failures <= 0:
        raise ValueError("max_consecutive_failures must be > 0")

    run_id = run_id or make_run_id()
    client = Client(endpoint)
    connected = False
    rows = []
    t0 = timestamp_now()
    next_progress_ts = t0 + progress_every_s if progress_every_s > 0 else float("inf")
    sq_error_sum = 0.0
    consecutive_failures = 0

    def read_with_retries() -> Dict[str, float]:
        last_exc: Optional[Exception] = None
        for attempt in range(read_retries + 1):
            try:
                return read_opc_snapshot(client, nodes)
            except Exception as exc:
                last_exc = exc
                if attempt < read_retries and read_retry_delay_s > 0:
                    time.sleep(read_retry_delay_s)
        assert last_exc is not None
        raise last_exc

    try:
        try:
            client.connect()
            connected = True
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to OPC endpoint '{endpoint}'. "
                "Verify server is running, reachable, and listening on this port."
            ) from exc

        if verbose:
            print(
                f"[run {run_id}] connected, sampling every {dt_s:.3f}s for {duration_s:.1f}s"
            )

        while True:
            now = timestamp_now()
            if now - t0 >= duration_s:
                break

            try:
                snap = read_with_retries()
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                if verbose:
                    elapsed = now - t0
                    print(
                        f"[run {run_id}] read timeout/error at {elapsed:.1f}s "
                        f"(consecutive_failures={consecutive_failures}): {exc}"
                    )

                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        f"Too many consecutive OPC read failures ({consecutive_failures}). "
                        "Stopping run to avoid logging invalid data."
                    ) from exc

                time.sleep(dt_s)
                continue

            sq_error = float((snap["temp_ref"] - snap["temp"]) ** 2)
            sq_error_sum += sq_error
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp": now,
                    "kp": float(kp),
                    "ki": float(ki),
                    "kd": float(kd),
                    **snap,
                    "sq_error": sq_error,
                }
            )

            if verbose and now >= next_progress_ts:
                elapsed = now - t0
                remaining = max(duration_s - elapsed, 0.0)
                n = len(rows)
                running_mse = sq_error_sum / n
                print(
                    f"[run {run_id}] samples={n} elapsed={elapsed:.1f}s "
                    f"remaining={remaining:.1f}s temp={snap['temp']:.3f} "
                    f"temp_ref={snap['temp_ref']:.3f} mse={running_mse:.6f}"
                )
                next_progress_ts += progress_every_s

            time.sleep(dt_s)
    finally:
        if connected:
            try:
                client.disconnect()
            except Exception:
                # Ignore shutdown errors to preserve the original failure context.
                pass

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No OPC samples were collected. Check connectivity and test timing.")

    mse = float(df["sq_error"].mean())
    summary = {
        "run_id": run_id,
        "start_ts": float(df["timestamp"].iloc[0]),
        "end_ts": float(df["timestamp"].iloc[-1]),
        "duration_s": float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]),
        "num_samples": int(len(df)),
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),
        "mse": mse,
    }
    return df, summary


def append_rows_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def history_run_file(run_id: str, filename: str, folder_name: str = "history") -> str:
    return str(Path(folder_name) / run_id / Path(filename).name)


def load_runs_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)

    # Fallback for run-scoped storage under history/<run_id>/<filename>.
    target_name = p.name
    tables = []
    for match in Path("history").glob(f"*/{target_name}"):
        if match.is_file() and match.stat().st_size > 0:
            tables.append(pd.read_csv(match))
    if tables:
        return pd.concat(tables, ignore_index=True)

    return pd.DataFrame(columns=["run_id", "kp", "ki", "kd", "mse"])


def fit_gp_model(runs_df: pd.DataFrame) -> Dict[str, object]:
    required = ["kp", "ki", "kd", "mse"]
    missing = [c for c in required if c not in runs_df.columns]
    if missing:
        raise ValueError(f"Missing columns in runs table: {missing}")

    clean = runs_df.dropna(subset=required).copy()
    if len(clean) < 3:
        raise ValueError("At least 3 completed runs are required to fit a GP model")

    X = clean[["kp", "ki", "kd"]].to_numpy(dtype=float)
    y = clean["mse"].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0, 1.0], nu=2.5) + WhiteKernel(
        noise_level=1e-4,
        noise_level_bounds=(1e-8, 1e-1),
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )
    gp.fit(Xs, y)

    return {
        "scaler": scaler,
        "gp": gp,
        "n_samples": int(len(clean)),
        "best_mse": float(np.min(y)),
    }


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = y_best - mu - xi
    z = improvement / sigma_safe
    ei = improvement * _normal_cdf(z) + sigma_safe * _normal_pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def suggest_next_params(
    model: Dict[str, object],
    bounds: Dict[str, Tuple[float, float]],
    n_candidates: int = 5000,
    xi: float = 0.01,
    random_state: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(random_state)

    kp_lo, kp_hi = bounds["kp"]
    ki_lo, ki_hi = bounds["ki"]
    kd_lo, kd_hi = bounds["kd"]

    candidates = np.column_stack(
        [
            rng.uniform(kp_lo, kp_hi, size=n_candidates),
            rng.uniform(ki_lo, ki_hi, size=n_candidates),
            rng.uniform(kd_lo, kd_hi, size=n_candidates),
        ]
    )

    scaler: StandardScaler = model["scaler"]
    gp: GaussianProcessRegressor = model["gp"]

    Xs = scaler.transform(candidates)
    mu, std = gp.predict(Xs, return_std=True)

    y_best = float(model["best_mse"])
    ei = expected_improvement(mu=mu, sigma=std, y_best=y_best, xi=xi)
    idx = int(np.argmax(ei))

    return {
        "kp": float(candidates[idx, 0]),
        "ki": float(candidates[idx, 1]),
        "kd": float(candidates[idx, 2]),
        "pred_mse": float(mu[idx]),
        "pred_std": float(std[idx]),
        "expected_improvement": float(ei[idx]),
        "incumbent_mse": y_best,
    }


def save_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

from typing import Tuple


def parse_bounds(text: str) -> Tuple[float, float]:
    
    if not isinstance(text, str):
        raise ValueError(f"Bounds must be a string, got: {type(text)}")

    parts = [p.strip() for p in text.split(",")]

    if len(parts) != 2:
        raise ValueError(f"Invalid bounds format: '{text}'. Expected 'low,high'")

    try:
        lo = float(parts[0])
        hi = float(parts[1])
    except ValueError:
        raise ValueError(f"Bounds must be numeric: '{text}'")

    if lo >= hi:
        raise ValueError(f"Invalid bounds '{text}': low must be < high")

    return lo, hi

def append_mae_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with a per-row absolute error column named 'mae'.

    Required columns:
        - temp
        - temp_ref
    """
    required = {"temp", "temp_ref"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"append_mae_column missing required columns: {sorted(missing)}")

    out = df.copy()
    out["mae"] = (out["temp_ref"].astype(float) - out["temp"].astype(float)).abs()
    return out


def compute_run_cost(
    df: pd.DataFrame,
    near_temp: float = 5.0,
    overshoot_weight: float = 10.0,
    time_below_weight: float = 0.02,
) -> dict:
    """
    Compute a near-target cost for cooling to 0 C.

    cost = tail_mae + overshoot_weight * negative_overshoot^2
                        + time_below_weight * time_below_0

    """
    required = {"timestamp", "temp", "temp_ref"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_run_cost missing required columns: {sorted(missing)}")

    work = df.sort_values("timestamp").reset_index(drop=True).copy()
    work["temp"] = work["temp"].astype(float)
    work["temp_ref"] = work["temp_ref"].astype(float)
    work["timestamp"] = work["timestamp"].astype(float)

    near_idx = work.index[work["temp"] <= near_temp]
    if len(near_idx) == 0:
        return {
            "cost": 1e9,
            "tail_mae": None,
            "negative_overshoot": None,
            "time_below_0": None,
            "min_temp": None,
            "tail_start_index": None,
            "reason": "never_reached_near_target",
        }

    start = int(near_idx[0])
    tail = work.iloc[start:].copy()

    err = tail["temp_ref"] - tail["temp"]
    tail_mae = float(np.mean(np.abs(err)))

    min_temp = float(tail["temp"].min())
    negative_overshoot = float(max(0.0, -min_temp))

    below0 = tail["temp"] < 0.0
    dt = tail["timestamp"].diff().fillna(0.0).to_numpy(dtype=float)
    time_below_0 = float(np.sum(dt[below0.to_numpy()]))

    cost = float(
        tail_mae
        + overshoot_weight * (negative_overshoot ** 2)
        + time_below_weight * time_below_0
    )

    return {
        "cost": cost,
        "tail_mae": tail_mae,
        "negative_overshoot": negative_overshoot,
        "time_below_0": time_below_0,
        "min_temp": min_temp,
        "tail_start_index": start,
    }

def compute_run_cost_band(
    df,
    entry_band=2.0,
    settle_band=0.5,
    overshoot_weight=10.0,
    wrong_side_weight=0.02,
):

    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    temp = df["temp"].astype(float).to_numpy()
    temp_ref = df["temp_ref"].astype(float).to_numpy()
    t = df["timestamp"].astype(float).to_numpy()

    target = float(temp_ref[0])
    start_temp = float(temp[0])

    # 1 for cooling, -1 for heating
    direction = 1.0 if start_temp > target else -1.0

    abs_err = np.abs(temp - target)

    idx = np.where(abs_err <= entry_band)[0]
    
    if len(idx) == 0:
        return {
            "cost": 1e9,
            "tail_mae": None,
            "overshoot": None,
            "time_on_wrong_side": None,
            "max_wrong_side_dev": None,
            "settle_fraction": None,
            "tail_start_index": None,
            "target": target,
            "start_temp": start_temp,
            "direction": direction,
            "reason": "never_reached_entry_band",
        }

    start_idx = int(idx[0])
    tail_temp = temp[start_idx:]
    tail_time = t[start_idx:]

    tail_mae = float(np.mean(np.abs(tail_temp - target)))

    dev = tail_temp - target

    if direction > 0:
        wrong_dev = np.maximum(0.0, -dev)
    else:
        wrong_dev = np.maximum(0.0, dev)

    overshoot = float(np.max(wrong_dev)) if len(wrong_dev) else 0.0

    dt = np.diff(tail_time, prepend=tail_time[0])
    time_on_wrong_side = float(np.sum(dt[wrong_dev > 0]))

    cost = (
        tail_mae
        + overshoot_weight * (overshoot ** 2)
        + wrong_side_weight * time_on_wrong_side
    )

    within_settle = np.abs(tail_temp - target) <= settle_band
    settle_fraction = float(np.mean(within_settle)) if len(tail_temp) else 0.0

    return {
        "cost": cost,
        "tail_mae": tail_mae,
        "overshoot": overshoot,
        "time_on_wrong_side": time_on_wrong_side,
        "max_wrong_side_dev": overshoot,
        "settle_fraction": settle_fraction,
        "tail_start_index": start_idx,
        "target": target,
        "start_temp": start_temp,
        "direction": direction,
    }
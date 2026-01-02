#!/usr/bin/env python3
"""
Decision-tree-based real-time PID gain adjuster (sets individual Kp, Ki, Kd).

- Trains 3 small DecisionTreeRegressors to predict bounded gain multipliers:
    dKp_mult, dKi_mult, dKd_mult
- At runtime, apply:
    Kp <- clamp( smooth(Kp * (1 + dKp)), Kp_min, Kp_max )
    Ki <- clamp( smooth(Ki * (1 + dKi)), Ki_min, Ki_max )
    Kd <- clamp( smooth(Kd * (1 + dKd)), Kd_min, Kd_max )

Compute future performance:
- If future bias remains high -> increase Ki
- If future jitter high -> decrease Kp and/or increase Kd
- If future MAE high + slow response -> increase Kp

USAGE:
  Train:
    python decisiontree.py train --csv20 readings/20dec.csv --csv21 readings/21dec.csv --out dt_pid_adjuster.pkl

  Print rules:
    python decisiontree.py print-rules --model dt_pid_adjuster.pkl

  Simulate advice on a run (offline):
    python decisiontree.py simulate --model dt_pid_adjuster.pkl --csv readings/20dec.csv --kp0 8 --ki0 800 --kd0 10 --out advice_20dec.csv
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class BuildConfig:
    window_s: int = 60          # past window for features
    horizon_s: int = 60         # future horizon for labels
    step_lookback_s: int = 10   # detect setpoint changes
    min_change_sp: float = 0.5  # step threshold (degC)
    sample_hz: float = 1.0      # assume 1Hz logs

@dataclass(frozen=True)
class CostWeights:
    w_mae: float = 1.0
    w_jitter: float = 0.25
    w_bias: float = 0.25

@dataclass(frozen=True)
class GainBounds:
    kp: Tuple[float, float]
    ki: Tuple[float, float]
    kd: Tuple[float, float]

@dataclass(frozen=True)
class DeltaBounds:
    # bounds on multiplier deltas, e.g. [-0.15, +0.15] means +/-15% per update
    dkp: Tuple[float, float] = (-0.15, 0.15)
    dki: Tuple[float, float] = (-0.20, 0.20)
    dkd: Tuple[float, float] = (-0.20, 0.20)

@dataclass(frozen=True)
class LabelThresholds:
    # thresholds that trigger target updates, in degC units
    bias_hi: float = 2.0
    jitter_hi: float = 3.0
    mae_hi: float = 3.0
    slow_slope: float = 0.02    # |dT/dt| small -> slow response


# -----------------------------
# Loading & preprocessing
# -----------------------------

def load_run(csv_path: str, run_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", decimal=",", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    colmap = {
        "Дата/время": "timestamp",
        "Температура в объёме, °C": "temp",
        "Уставка по температуре в объёме, °C": "sp",
        "Управление по температуре (П), %": "p_term",
        "Управление по температуре (И), %": "i_term",
        "Управление по температуре (Д), %": "d_term",
        "Давление нагнетания 404 комп., атм": "p404",
        "Давление нагнетания 23 комп., атм": "p23",
    }
    missing = [k for k in colmap if k not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns: {missing}\nFound: {list(df.columns)}")

    df = df.rename(columns=colmap)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df["run"] = run_name

    # Derived signals
    df["e"] = df["sp"] - df["temp"]
    df["de"] = df["e"].diff().fillna(0.0)
    df["dtemp"] = df["temp"].diff().fillna(0.0)

    return df


# -----------------------------
# Feature engineering
# -----------------------------

def rolling_features(df: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    W = int(cfg.window_s * cfg.sample_hz)
    L = int(cfg.step_lookback_s * cfg.sample_hz)

    out = df.copy()

    # Error stats
    out["e_abs_mean_W"] = out["e"].abs().rolling(W, min_periods=W).mean()
    out["e_mean_W"]     = out["e"].rolling(W, min_periods=W).mean()
    out["e_std_W"]      = out["e"].rolling(W, min_periods=W).std()

    out["de_mean_W"]    = out["de"].rolling(W, min_periods=W).mean()
    out["de_std_W"]     = out["de"].rolling(W, min_periods=W).std()

    out["temp_slope_W"] = out["dtemp"].rolling(W, min_periods=W).mean()

    # Controller activity (terms are in %, but still useful as features)
    for c in ["p_term", "i_term", "d_term"]:
        out[f"{c}_mean_W"] = out[c].rolling(W, min_periods=W).mean()
        out[f"{c}_std_W"]  = out[c].rolling(W, min_periods=W).std()

    # Pressures
    for c in ["p404", "p23"]:
        out[f"{c}_mean_W"] = out[c].rolling(W, min_periods=W).mean()
        out[f"{c}_std_W"]  = out[c].rolling(W, min_periods=W).std()

    # Setpoint step
    out["sp_step"] = out["sp"] - out["sp"].shift(L)
    out["is_step"] = (out["sp_step"].abs() >= cfg.min_change_sp).astype(float)

    return out


def future_quality(df: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    """
    Compute future MAE/jitter/bias over horizon H using e(t+1..t+H).
    """
    H = int(cfg.horizon_s * cfg.sample_hz)
    e_f = df["e"].shift(-1)

    out = pd.DataFrame(index=df.index)
    out["f_mae"]    = e_f.abs().rolling(H, min_periods=H).mean()
    out["f_jitter"] = e_f.rolling(H, min_periods=H).std()
    out["f_bias"]   = e_f.rolling(H, min_periods=H).mean().abs()
    return out


# -----------------------------
# Target labeling for ΔKp, ΔKi, ΔKd (multiplier deltas)
# -----------------------------

def clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)

def label_deltas(
    feat_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    cfg: BuildConfig,
    db: DeltaBounds,
    thr: LabelThresholds
) -> pd.DataFrame:
    """
    Produce supervised targets dkp_mult, dki_mult, dkd_mult in bounded ranges.

    Heuristic intent:
      - High future bias => increase Ki
      - High future jitter => reduce Kp and/or increase Kd
      - High future MAE + slow response => increase Kp
      - During/just after SP step => do NOT adapt (targets 0)
    """
    # Start with zeros
    n = len(feat_df)
    dkp = np.zeros(n, dtype=float)
    dki = np.zeros(n, dtype=float)
    dkd = np.zeros(n, dtype=float)

    f_mae = fut_df["f_mae"].to_numpy()
    f_jit = fut_df["f_jitter"].to_numpy()
    f_bias = fut_df["f_bias"].to_numpy()

    e_mean = feat_df["e_mean_W"].to_numpy()         # current bias sign info
    e_std  = feat_df["e_std_W"].to_numpy()
    temp_slope = feat_df["temp_slope_W"].to_numpy()
    is_step = feat_df["is_step"].to_numpy()

    # 1) Bias correction via Ki
    bias_hi = f_bias > thr.bias_hi
    # If mean error is positive (SP > T), need more heating -> Ki up (still just a magnitude)
    dki[bias_hi] = 0.10

    # 2) Jitter suppression via Kp/Kd
    jitter_hi = f_jit > thr.jitter_hi
    dkp[jitter_hi] = -0.08
    dkd[jitter_hi] = +0.08

    # 3) Slow response + high MAE => Kp up (only if jitter is not already high)
    slow = np.abs(temp_slope) < thr.slow_slope
    mae_hi = f_mae > thr.mae_hi
    improve_speed = slow & mae_hi & (~jitter_hi)
    dkp[improve_speed] += +0.08

    # 4) If jitter is low but MAE high and bias high, also allow a small Ki increase
    dki[(~jitter_hi) & mae_hi & bias_hi] += +0.05

    # 5) Freeze adaptation around steps (avoid destabilizing during transients)
    dkp[is_step > 0.5] = 0.0
    dki[is_step > 0.5] = 0.0
    dkd[is_step > 0.5] = 0.0

    # Clip to bounds
    dkp = clip(dkp, db.dkp[0], db.dkp[1])
    dki = clip(dki, db.dki[0], db.dki[1])
    dkd = clip(dkd, db.dkd[0], db.dkd[1])

    return pd.DataFrame({"dkp_mult": dkp, "dki_mult": dki, "dkd_mult": dkd})


def build_training_table(df: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    feat = rolling_features(df, cfg)
    fut = future_quality(feat, cfg)

    # Build labels
    y = label_deltas(feat, fut, cfg, DeltaBounds(), LabelThresholds())

    # Keep rows where everything is valid
    cols_x = [
        "sp", "temp", "e",
        "e_abs_mean_W", "e_mean_W", "e_std_W",
        "de_mean_W", "de_std_W", "temp_slope_W",
        "p_term_mean_W", "p_term_std_W",
        "i_term_mean_W", "i_term_std_W",
        "d_term_mean_W", "d_term_std_W",
        "p404_mean_W", "p404_std_W",
        "p23_mean_W", "p23_std_W",
        "sp_step", "is_step",
        "run", "timestamp",
    ]
    data = pd.concat([feat[cols_x], y], axis=1).dropna().reset_index(drop=True)
    return data


# -----------------------------
# Model bundle
# -----------------------------

@dataclass(frozen=True)
class AdjusterBundle:
    cfg: BuildConfig
    feature_cols: List[str]
    tree_kp: DecisionTreeRegressor
    tree_ki: DecisionTreeRegressor
    tree_kd: DecisionTreeRegressor
    delta_bounds: DeltaBounds
    gain_bounds: GainBounds
    # runtime smoothing / guardrails
    ema_alpha: float = 0.2            # 0..1 (higher = faster updates)
    min_seconds_between_updates: int = 5
    freeze_on_step: bool = True


def train_tree(X: np.ndarray, y: np.ndarray, max_depth: int, min_leaf: int) -> DecisionTreeRegressor:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    m = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=0)
    m.fit(Xtr, ytr)
    print(f"  R^2: {m.score(Xte, yte):.3f}")
    return m


# -----------------------------
# Runtime advisor
# -----------------------------

def clamp_gain(val: float, lo: float, hi: float) -> float:
    return float(np.clip(val, lo, hi))

def compute_latest_features(window_df: pd.DataFrame, cfg: BuildConfig, feature_cols: List[str]) -> Optional[pd.Series]:
    df = window_df.copy().reset_index(drop=True)
    df["e"] = df["sp"] - df["temp"]
    df["de"] = df["e"].diff().fillna(0.0)
    df["dtemp"] = df["temp"].diff().fillna(0.0)
    feat = rolling_features(df, cfg).iloc[-1]
    if feat[feature_cols].isna().any():
        return None
    return feat

def predict_deltas(bundle: AdjusterBundle, feat_row: pd.Series) -> Tuple[float, float, float]:
    X = feat_row[bundle.feature_cols].to_numpy().reshape(1, -1)
    dkp = float(bundle.tree_kp.predict(X)[0])
    dki = float(bundle.tree_ki.predict(X)[0])
    dkd = float(bundle.tree_kd.predict(X)[0])

    # clip to delta bounds
    dkp = float(np.clip(dkp, bundle.delta_bounds.dkp[0], bundle.delta_bounds.dkp[1]))
    dki = float(np.clip(dki, bundle.delta_bounds.dki[0], bundle.delta_bounds.dki[1]))
    dkd = float(np.clip(dkd, bundle.delta_bounds.dkd[0], bundle.delta_bounds.dkd[1]))
    return dkp, dki, dkd

def apply_deltas(
    bundle: AdjusterBundle,
    kp: float, ki: float, kd: float,
    dkp: float, dki: float, dkd: float
) -> Tuple[float, float, float]:
    kp_new = kp * (1.0 + dkp)
    ki_new = ki * (1.0 + dki)
    kd_new = kd * (1.0 + dkd)

    kp_new = clamp_gain(kp_new, bundle.gain_bounds.kp[0], bundle.gain_bounds.kp[1])
    ki_new = clamp_gain(ki_new, bundle.gain_bounds.ki[0], bundle.gain_bounds.ki[1])
    kd_new = clamp_gain(kd_new, bundle.gain_bounds.kd[0], bundle.gain_bounds.kd[1])
    return kp_new, ki_new, kd_new


# -----------------------------
# CLI
# -----------------------------

def cmd_train(args: argparse.Namespace) -> None:
    cfg = BuildConfig(window_s=args.window_s, horizon_s=args.horizon_s)

    df20 = load_run(args.csv20, "20dec")
    df21 = load_run(args.csv21, "21dec")
    df = pd.concat([df20, df21], ignore_index=True)

    data = build_training_table(df, cfg)
    print(f"Training rows: {len(data):,}")

    feature_cols = [c for c in data.columns if c not in ["dkp_mult", "dki_mult", "dkd_mult", "run", "timestamp"]]
    X = data[feature_cols].to_numpy()

    print("Training tree for dKp_mult:")
    tree_kp = train_tree(X, data["dkp_mult"].to_numpy(), args.max_depth, args.min_leaf)
    print("Training tree for dKi_mult:")
    tree_ki = train_tree(X, data["dki_mult"].to_numpy(), args.max_depth, args.min_leaf)
    print("Training tree for dKd_mult:")
    tree_kd = train_tree(X, data["dkd_mult"].to_numpy(), args.max_depth, args.min_leaf)

    # Gain bounds must be set from your safe engineering limits.
    # Provide them on CLI (recommended). Defaults are conservative placeholders.
    gb = GainBounds(
        kp=(args.kp_min, args.kp_max),
        ki=(args.ki_min, args.ki_max),
        kd=(args.kd_min, args.kd_max),
    )

    bundle = AdjusterBundle(
        cfg=cfg,
        feature_cols=feature_cols,
        tree_kp=tree_kp,
        tree_ki=tree_ki,
        tree_kd=tree_kd,
        delta_bounds=DeltaBounds(),
        gain_bounds=gb,
        ema_alpha=args.ema_alpha,
        min_seconds_between_updates=args.min_update_s,
        freeze_on_step=True,
    )

    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Saved adjuster bundle: {args.out}")
    print("Next: python decisiontree.py print-rules --model dt_pid_adjuster.pkl")


def cmd_print_rules(args: argparse.Namespace) -> None:
    with open(args.model, "rb") as f:
        bundle: AdjusterBundle = pickle.load(f)

    print("\n=== Tree rules for dKp_mult ===")
    print(export_text(bundle.tree_kp, feature_names=bundle.feature_cols))
    print("\n=== Tree rules for dKi_mult ===")
    print(export_text(bundle.tree_ki, feature_names=bundle.feature_cols))
    print("\n=== Tree rules for dKd_mult ===")
    print(export_text(bundle.tree_kd, feature_names=bundle.feature_cols))

    print("\nDelta bounds:", bundle.delta_bounds)
    print("Gain bounds:", bundle.gain_bounds)


def cmd_simulate(args: argparse.Namespace) -> None:
    with open(args.model, "rb") as f:
        bundle: AdjusterBundle = pickle.load(f)

    df = load_run(args.csv, "sim")
    cfg = bundle.cfg

    # current gains
    kp, ki, kd = float(args.kp0), float(args.ki0), float(args.kd0)
    kp_f, ki_f, kd_f = kp, ki, kd  # filtered (EMA)

    W = int(cfg.window_s * cfg.sample_hz)
    last_update_idx = -10**9

    out_rows = []
    for idx in range(len(df)):
        # Need enough history for rolling features
        if idx < W:
            continue

        window = df.iloc[idx - W + 1: idx + 1][["sp", "temp", "p_term", "i_term", "d_term", "p404", "p23"]].copy()

        # compute features
        feat_row = compute_latest_features(window, cfg, bundle.feature_cols)
        if feat_row is None:
            continue

        # freeze on step
        if bundle.freeze_on_step and float(feat_row["is_step"]) > 0.5:
            dkp = dki = dkd = 0.0
        else:
            # update rate limit
            if (idx - last_update_idx) < int(bundle.min_seconds_between_updates * cfg.sample_hz):
                dkp = dki = dkd = 0.0
            else:
                dkp, dki, dkd = predict_deltas(bundle, feat_row)
                last_update_idx = idx

        kp_new, ki_new, kd_new = apply_deltas(bundle, kp_f, ki_f, kd_f, dkp, dki, dkd)

        # EMA smoothing
        a = bundle.ema_alpha
        kp_f = (1 - a) * kp_f + a * kp_new
        ki_f = (1 - a) * ki_f + a * ki_new
        kd_f = (1 - a) * kd_f + a * kd_new

        out_rows.append({
            "timestamp": df.loc[idx, "timestamp"],
            "sp": df.loc[idx, "sp"],
            "temp": df.loc[idx, "temp"],
            "e": df.loc[idx, "sp"] - df.loc[idx, "temp"],
            "dkp_mult": dkp, "dki_mult": dki, "dkd_mult": dkd,
            "Kp_suggested": kp_f, "Ki_suggested": ki_f, "Kd_suggested": kd_f,
        })

    out = pd.DataFrame(out_rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote simulation advice: {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("train")
    s.add_argument("--csv20", required=True)
    s.add_argument("--csv21", required=True)
    s.add_argument("--out", default="dt_pid_adjuster.pkl")
    s.add_argument("--window-s", type=int, default=60)
    s.add_argument("--horizon-s", type=int, default=60)
    s.add_argument("--max-depth", type=int, default=4)
    s.add_argument("--min-leaf", type=int, default=200)

    # safe gain bounds (YOU should set these)
    s.add_argument("--kp-min", type=float, default=2.0)
    s.add_argument("--kp-max", type=float, default=25.0)
    s.add_argument("--ki-min", type=float, default=50.0)
    s.add_argument("--ki-max", type=float, default=2000.0)
    s.add_argument("--kd-min", type=float, default=1.0)
    s.add_argument("--kd-max", type=float, default=150.0)

    # runtime smoothing/rate limiting
    s.add_argument("--ema-alpha", type=float, default=0.2)
    s.add_argument("--min-update-s", type=int, default=5)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("print-rules")
    s.add_argument("--model", required=True)
    s.set_defaults(func=cmd_print_rules)

    s = sub.add_parser("simulate")
    s.add_argument("--model", required=True)
    s.add_argument("--csv", required=True)
    s.add_argument("--kp0", type=float, required=True)
    s.add_argument("--ki0", type=float, required=True)
    s.add_argument("--kd0", type=float, required=True)
    s.add_argument("--out", default="advice.csv")
    s.set_defaults(func=cmd_simulate)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

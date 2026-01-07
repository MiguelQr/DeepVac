#!/usr/bin/env python3
"""
Decision-tree-based PID gain adjuster (sets individual Kp, Ki, Kd).

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
    python decisiontree.py train --csv20 readings/20dec.csv --csv21 readings/21dec.csv --out pid.pkl

  Print rules:
    python decisiontree.py print-rules --model pid.pkl

  Simulate offline:
    python decisiontree.py simulate --model pid.pkl --csv readings/20dec.csv --kp0 8 --ki0 800 --kd0 10 --out advice_20dec.csv
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split


# ----------------------------
# Settings
# ----------------------------

# Feature window and label horizon. Logs are 1 Hz, so seconds = samples.
WINDOW_S = 30
HORIZON_S = 30

# Detect setpoint step: compare to value 10 seconds ago, and if change >= 0.5°C => step
STEP_LOOKBACK = 5
MIN_STEP = 0.5

# Delta limits per update (multipliers)
DKP_MIN, DKP_MAX = -0.15, 0.15
DKI_MIN, DKI_MAX = -0.20, 0.20
DKD_MIN, DKD_MAX = -0.20, 0.20

# Coefficient gain bounds (safety)
KP_MIN, KP_MAX = 2.0, 25.0
KI_MIN, KI_MAX = 50.0, 2000.0
KD_MIN, KD_MAX = 1.0, 150.0

# Label thresholds (degC)
BIAS_HI = 2.0
JITTER_HI = 3.0
MAE_HI = 3.0
SLOW_SLOPE = 0.02  # degC per sample (since sample_hz=1)

# Runtime behavior for offline simulate
EMA_ALPHA = 0.2
MIN_UPDATE_S = 5
FREEZE_ON_STEP = True


# ----------------------------
# Helpers
# ----------------------------

def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def clamp_gain(x, lo, hi):
    return float(np.clip(x, lo, hi))


# ----------------------------
# Load CSV 
# ----------------------------

def load_run(csv_path, run_name):
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
        print(f"{csv_path}: missing columns: {missing}\nFound: {list(df.columns)}")

    df = df.rename(columns=colmap)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df["run"] = run_name

    df["e"] = df["sp"] - df["temp"] # error
    df["de"] = df["e"].diff().fillna(0.0) # derivative of error
    df["dtemp"] = df["temp"].diff().fillna(0.0) # derivative of temperature

    return df


# ----------------------------
# Feature engineering
# ----------------------------

def add_rolling_features(df):
    """
    Adds rolling window features based on last WINDOW_S samples.
    Sample rate ~ 1 Hz.
    """
    W = int(WINDOW_S)
    L = int(STEP_LOOKBACK)

    out = df.copy()

    out["e_abs_mean_W"] = out["e"].abs().rolling(W, min_periods=W).mean()
    out["e_mean_W"] = out["e"].rolling(W, min_periods=W).mean()
    out["e_std_W"] = out["e"].rolling(W, min_periods=W).std()

    out["de_mean_W"] = out["de"].rolling(W, min_periods=W).mean()
    out["de_std_W"] = out["de"].rolling(W, min_periods=W).std()

    out["temp_slope_W"] = out["dtemp"].rolling(W, min_periods=W).mean()

    # Control terms
    for c in ["p_term", "i_term", "d_term"]:
        out[f"{c}_mean_W"] = out[c].rolling(W, min_periods=W).mean()
        out[f"{c}_std_W"] = out[c].rolling(W, min_periods=W).std()

    # Pressures
    for c in ["p404", "p23"]:
        out[f"{c}_mean_W"] = out[c].rolling(W, min_periods=W).mean()
        out[f"{c}_std_W"] = out[c].rolling(W, min_periods=W).std()

    # Setpoint step detector
    out["sp_step"] = out["sp"] - out["sp"].shift(L) # setpoint change over last L samples
    out["is_step"] = (out["sp_step"].abs() >= MIN_STEP).astype(float) # binary flag

    return out


def compute_future(df):
    """
    Computes future MAE / jitter / bias over the next HORIZON_S samples.
    (Uses e(t+1..t+H), implemented by shifting by -1 then rolling forward.)
    """
    H = int(HORIZON_S)
    e_f = df["e"].shift(-1) # shift error from next row to current

    fut = pd.DataFrame(index=df.index)
    fut["f_mae"] = e_f.abs().rolling(H, min_periods=H).mean()
    fut["f_jitter"] = e_f.rolling(H, min_periods=H).std()
    fut["f_bias"] = e_f.rolling(H, min_periods=H).mean().abs()
    return fut


# ----------------------------
# Label targets
# ----------------------------

def build_delta_targets(feat_df, fut_df):
    """
    Create supervised targets for 3 deltas:
      dkp_mult, dki_mult, dkd_mult

    IMPORTANT:
    These are not "true optimal deltas". They are *rule-based labels*:
      - If future bias is high -> increase Ki
      - If future jitter is high -> reduce Kp and increase Kd
      - If future MAE high and response is slow -> increase Kp
      - If setpoint just changed -> do nothing (freeze)
    """
    n = len(feat_df)
    dkp = np.zeros(n)
    dki = np.zeros(n)
    dkd = np.zeros(n)

    f_mae = fut_df["f_mae"].to_numpy()
    f_jit = fut_df["f_jitter"].to_numpy()
    f_bias = fut_df["f_bias"].to_numpy()

    temp_slope = feat_df["temp_slope_W"].to_numpy()
    is_step = feat_df["is_step"].to_numpy()

    bias_hi = f_bias > BIAS_HI
    jitter_hi = f_jit > JITTER_HI
    mae_hi = f_mae > MAE_HI
    slow = np.abs(temp_slope) < SLOW_SLOPE

    # -------------------------------------------------------------------

    # TODO: improve rules with expert knowledge or more data

    # Bias correction via Ki
    dki[bias_hi] = 0.10

    # Jitter suppression via Kp/Kd
    dkp[jitter_hi] = -0.08
    dkd[jitter_hi] = +0.08

    # If slow + high MAE (and not jittery), try Kp up
    improve_speed = slow & mae_hi & (~jitter_hi)
    dkp[improve_speed] += 0.08

    # Additional small Ki bump when bias+MAE high but not jittery
    dki[(~jitter_hi) & mae_hi & bias_hi] += 0.05

    # -------------------------------------------------------------------

    # Don't adjust on setpoint steps
    dkp[is_step > 0.5] = 0.0
    dki[is_step > 0.5] = 0.0
    dkd[is_step > 0.5] = 0.0

    dkp = clip(dkp, DKP_MIN, DKP_MAX)
    dki = clip(dki, DKI_MIN, DKI_MAX)
    dkd = clip(dkd, DKD_MIN, DKD_MAX)

    return pd.DataFrame({"dkp_mult": dkp, "dki_mult": dki, "dkd_mult": dkd})


def build_training_table(df_all):
    feat = add_rolling_features(df_all)
    fut = compute_future(feat)
    y = build_delta_targets(feat, fut)

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


# ----------------------------
# Train / Save model
# ----------------------------

def train_one_tree(X, y, max_depth, min_leaf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    m = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=0)
    m.fit(X_train, y_train)
    r2 = m.score(X_test, y_test)
    return m, r2


def cmd_train(args):
    df20 = load_run(args.csv20, "20dec")
    df21 = load_run(args.csv21, "21dec")
    df_all = pd.concat([df20, df21], ignore_index=True)

    data = build_training_table(df_all)
    print(f"Training rows: {len(data):,}")

    feature_cols = [c for c in data.columns if c not in ["dkp_mult", "dki_mult", "dkd_mult", "run", "timestamp"]]
    X = data[feature_cols].to_numpy()

    print("Training tree for dKp_mult...")
    tree_kp, r2_kp = train_one_tree(X, data["dkp_mult"].to_numpy(), args.max_depth, args.min_leaf)
    print(f"  R^2: {r2_kp:.3f}")

    print("Training tree for dKi_mult...")
    tree_ki, r2_ki = train_one_tree(X, data["dki_mult"].to_numpy(), args.max_depth, args.min_leaf)
    print(f"  R^2: {r2_ki:.3f}")

    print("Training tree for dKd_mult...")
    tree_kd, r2_kd = train_one_tree(X, data["dkd_mult"].to_numpy(), args.max_depth, args.min_leaf)
    print(f"  R^2: {r2_kd:.3f}")

    model = {
        "feature_cols": feature_cols,
        "tree_kp": tree_kp,
        "tree_ki": tree_ki,
        "tree_kd": tree_kd,

        "window_s": WINDOW_S,
        "step_lookback_s": STEP_LOOKBACK,
        "min_step": MIN_STEP,

        "delta_bounds": {"dkp": (DKP_MIN, DKP_MAX), "dki": (DKI_MIN, DKI_MAX), "dkd": (DKD_MIN, DKD_MAX)},
        "gain_bounds": {"kp": (KP_MIN, KP_MAX), "ki": (KI_MIN, KI_MAX), "kd": (KD_MIN, KD_MAX)},

        "ema_alpha": EMA_ALPHA,
        "min_update_s": MIN_UPDATE_S,
        "freeze_on_step": FREEZE_ON_STEP,
    }

    with open(args.out, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to: {args.out}")


def print_rules(args):
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    feats = model["feature_cols"]

    print("\n=== Tree rules for dKp_mult ===")
    print(export_text(model["tree_kp"], feature_names=feats))

    print("\n=== Tree rules for dKi_mult ===")
    print(export_text(model["tree_ki"], feature_names=feats))

    print("\n=== Tree rules for dKd_mult ===")
    print(export_text(model["tree_kd"], feature_names=feats))

    print("\nDelta bounds:", model["delta_bounds"])
    print("Gain bounds:", model["gain_bounds"])


# ----------------------------
# Offline simulate
# ----------------------------

def predict_deltas(model, feat_row):
    feats = model["feature_cols"]
    X = feat_row[feats].to_numpy().reshape(1, -1)

    dkp = float(model["tree_kp"].predict(X)[0])
    dki = float(model["tree_ki"].predict(X)[0])
    dkd = float(model["tree_kd"].predict(X)[0])

    dkp = float(np.clip(dkp, *model["delta_bounds"]["dkp"]))
    dki = float(np.clip(dki, *model["delta_bounds"]["dki"]))
    dkd = float(np.clip(dkd, *model["delta_bounds"]["dkd"]))
    return dkp, dki, dkd


def apply_deltas(model, kp, ki, kd, dkp, dki, dkd):
    kp2 = kp * (1.0 + dkp)
    ki2 = ki * (1.0 + dki)
    kd2 = kd * (1.0 + dkd)

    kp2 = clamp_gain(kp2, *model["gain_bounds"]["kp"])
    ki2 = clamp_gain(ki2, *model["gain_bounds"]["ki"])
    kd2 = clamp_gain(kd2, *model["gain_bounds"]["kd"])
    return kp2, ki2, kd2


def simulate(args):
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    df = load_run(args.csv, "sim")

    # current gains
    kp = float(args.kp0)
    ki = float(args.ki0)
    kd = float(args.kd0)

    # smoothed gains
    kp_f, ki_f, kd_f = kp, ki, kd

    W = int(model["window_s"])
    min_update = int(model["min_update_s"]) # update every s seconds
    last_update_idx = -10**9

    out_rows = []

    base_cols = ["sp", "temp", "p_term", "i_term", "d_term", "p404", "p23"]

    for i in range(len(df)):
        if i < W:
            continue

        window = df.iloc[i - W + 1 : i + 1][base_cols].copy()

        tmp = window.copy()
        tmp["e"] = tmp["sp"] - tmp["temp"]
        tmp["de"] = tmp["e"].diff().fillna(0.0)
        tmp["dtemp"] = tmp["temp"].diff().fillna(0.0)

        tmp = add_rolling_features(tmp)
        feat_row = tmp.iloc[-1]

        # skip if features incomplete
        if feat_row[model["feature_cols"]].isna().any():
            continue

        # freeze on step
        if model["freeze_on_step"] and float(feat_row.get("is_step", 0.0)) > 0.5:
            dkp = dki = dkd = 0.0
        else:
            # update every min_update seconds
            if (i - last_update_idx) < min_update:
                dkp = dki = dkd = 0.0
            else:
                dkp, dki, dkd = predict_deltas(model, feat_row)
                last_update_idx = i

        kp_new, ki_new, kd_new = apply_deltas(model, kp_f, ki_f, kd_f, dkp, dki, dkd)

        # EMA smoothing
        a = float(model["ema_alpha"])
        kp_f = (1 - a) * kp_f + a * kp_new
        ki_f = (1 - a) * ki_f + a * ki_new
        kd_f = (1 - a) * kd_f + a * kd_new

        out_rows.append({
            "timestamp": df.loc[i, "timestamp"],
            "sp": df.loc[i, "sp"],
            "temp": df.loc[i, "temp"],
            "e": df.loc[i, "sp"] - df.loc[i, "temp"],
            "dkp_mult": dkp,
            "dki_mult": dki,
            "dkd_mult": dkd,
            "Kp_suggested": kp_f,
            "Ki_suggested": ki_f,
            "Kd_suggested": kd_f,
        })

    out = pd.DataFrame(out_rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote to: {args.out}")


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--csv20", required=True)
    t.add_argument("--csv21", required=True)
    t.add_argument("--out", default="pid_advisor.pkl")
    t.add_argument("--max-depth", type=int, default=4)
    t.add_argument("--min-leaf", type=int, default=200)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("print-rules")
    pr.add_argument("--model", required=True)
    pr.set_defaults(func=print_rules)

    s = sub.add_parser("simulate")
    s.add_argument("--model", required=True)
    s.add_argument("--csv", required=True)
    s.add_argument("--kp0", type=float, required=True)
    s.add_argument("--ki0", type=float, required=True)
    s.add_argument("--kd0", type=float, required=True)
    s.add_argument("--out", default="advice.csv")
    s.set_defaults(func=simulate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
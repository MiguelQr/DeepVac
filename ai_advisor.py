#!/usr/bin/env python3
"""
ai_advisor.py

Real-time OPC UA loop that:
- reads chamber telemetry
- computes rolling-window features
- uses dt_pid_adjuster.pkl to suggest bounded Kp/Ki/Kd updates
- optionally writes new gains back via OPC UA
- logs all readings and actions to JSONL

Prereqs:
  pip install opcua pandas numpy scikit-learn

Run:
  python ai_advisor.py \
    --endpoint opc.tcp://192.168.88.166:12345 \
    --model pid_advisor.pkl \
    --duration 300 \
    --dt 0.5 \
    --write

Model was trained on:
  sp, temp, p_term, i_term, d_term, p404, p23
In the OPC:
  temp, temp_raw, pressure, con, con_p, con_i, con_d
Map:
  sp        -> (provide node id)
  temp      -> temp
  p_term    -> con_p   (or your P contribution)
  i_term    -> con_i
  d_term    -> con_d
  p404/p23  -> use pressure as p404 and set p23=0.0 unless you have both
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import deque
from dataclasses import asdict
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from opcua import Client


# -----------------------------
# Load AdjusterBundle (from dt_pid_adjuster.py)
# -----------------------------
# We rely on the bundle having:
#   cfg.window_s, cfg.sample_hz, feature_cols,
#   delta_bounds, gain_bounds,
#   tree_kp/tree_ki/tree_kd
# and helper logic re-implemented here.

def clamp(val: float, lo: float, hi: float) -> float:
    return float(np.clip(val, lo, hi))

def rolling_features_runtime(df: pd.DataFrame, window_s: int, step_lookback_s: int, min_change_sp: float, sample_hz: float) -> pd.Series:
    """
    Compute the same last-row features as training.
    df must contain columns: sp,temp,p_term,i_term,d_term,p404,p23
    """
    W = int(window_s * sample_hz)
    L = int(step_lookback_s * sample_hz)

    df = df.copy()
    df["e"] = df["sp"] - df["temp"]
    df["de"] = df["e"].diff().fillna(0.0)
    df["dtemp"] = df["temp"].diff().fillna(0.0)

    # rolling
    df["e_abs_mean_W"] = df["e"].abs().rolling(W, min_periods=W).mean()
    df["e_mean_W"]     = df["e"].rolling(W, min_periods=W).mean()
    df["e_std_W"]      = df["e"].rolling(W, min_periods=W).std()

    df["de_mean_W"]    = df["de"].rolling(W, min_periods=W).mean()
    df["de_std_W"]     = df["de"].rolling(W, min_periods=W).std()

    df["temp_slope_W"] = df["dtemp"].rolling(W, min_periods=W).mean()

    for c in ["p_term", "i_term", "d_term"]:
        df[f"{c}_mean_W"] = df[c].rolling(W, min_periods=W).mean()
        df[f"{c}_std_W"]  = df[c].rolling(W, min_periods=W).std()

    for c in ["p404", "p23"]:
        df[f"{c}_mean_W"] = df[c].rolling(W, min_periods=W).mean()
        df[f"{c}_std_W"]  = df[c].rolling(W, min_periods=W).std()

    df["sp_step"] = df["sp"] - df["sp"].shift(L)
    df["is_step"] = (df["sp_step"].abs() >= min_change_sp).astype(float)

    return df.iloc[-1]


def predict_deltas(bundle: Any, feat_row: pd.Series) -> Tuple[float, float, float]:
    X = feat_row[bundle.feature_cols].to_numpy().reshape(1, -1)
    dkp = float(bundle.tree_kp.predict(X)[0])
    dki = float(bundle.tree_ki.predict(X)[0])
    dkd = float(bundle.tree_kd.predict(X)[0])

    dkp = float(np.clip(dkp, bundle.delta_bounds.dkp[0], bundle.delta_bounds.dkp[1]))
    dki = float(np.clip(dki, bundle.delta_bounds.dki[0], bundle.delta_bounds.dki[1]))
    dkd = float(np.clip(dkd, bundle.delta_bounds.dkd[0], bundle.delta_bounds.dkd[1]))
    return dkp, dki, dkd


def apply_deltas(bundle: Any, kp: float, ki: float, kd: float, dkp: float, dki: float, dkd: float) -> Tuple[float, float, float]:
    kp_new = kp * (1.0 + dkp)
    ki_new = ki * (1.0 + dki)
    kd_new = kd * (1.0 + dkd)

    kp_new = clamp(kp_new, bundle.gain_bounds.kp[0], bundle.gain_bounds.kp[1])
    ki_new = clamp(ki_new, bundle.gain_bounds.ki[0], bundle.gain_bounds.ki[1])
    kd_new = clamp(kd_new, bundle.gain_bounds.kd[0], bundle.gain_bounds.kd[1])
    return kp_new, ki_new, kd_new


# -----------------------------
# OPC UA
# -----------------------------

@dataclass
class NodeMap:
    temp: str
    sp: str
    p_term: str
    i_term: str
    d_term: str
    pressure: str
    kp: str
    ki: str
    kd: str


def read_nodes(client: Client, nodes: NodeMap) -> Dict[str, float]:
    """
    Read required nodes from OPC UA.
    """
    out = {}
    out["temp"] = float(client.get_node(nodes.temp).get_value())
    out["sp"] = float(client.get_node(nodes.sp).get_value())
    out["p_term"] = float(client.get_node(nodes.p_term).get_value())
    out["i_term"] = float(client.get_node(nodes.i_term).get_value())
    out["d_term"] = float(client.get_node(nodes.d_term).get_value())
    pressure = float(client.get_node(nodes.pressure).get_value())
    out["p404"] = pressure
    out["p23"] = 0.0  # if you have a second pressure channel, map it here
    return out


def read_gains(client: Client, nodes: NodeMap) -> Tuple[float, float, float]:
    kp = float(client.get_node(nodes.kp).get_value())
    ki = float(client.get_node(nodes.ki).get_value())
    kd = float(client.get_node(nodes.kd).get_value())
    return kp, ki, kd


def write_gains(client: Client, nodes: NodeMap, kp: float, ki: float, kd: float) -> None:
    client.get_node(nodes.kp).set_value(kp)
    client.get_node(nodes.ki).set_value(ki)
    client.get_node(nodes.kd).set_value(kd)


# -----------------------------
# Main loop
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--write", action="store_true", help="Actually write Kp/Ki/Kd back to OPC.")
    ap.add_argument("--log", default="opcua_ai_log.jsonl")

    # Safety / update cadence
    ap.add_argument("--update-every", type=float, default=5.0, help="Seconds between gain updates.")
    ap.add_argument("--ema-alpha", type=float, default=0.2, help="Extra EMA smoothing on suggested gains.")
    ap.add_argument("--freeze-on-step", action="store_true", default=True)

    # Node ids
    ap.add_argument("--node-temp", default="ns=2;s=Testa chamber.temp")
    ap.add_argument("--node-sp", required=True, help="Node id for temperature setpoint (SP).")
    ap.add_argument("--node-p", default="ns=2;s=Testa chamber.temp_u_p")
    ap.add_argument("--node-i", default="ns=2;s=Testa chamber.temp_u_i")
    ap.add_argument("--node-d", default="ns=2;s=Testa chamber.temp_u_d")
    ap.add_argument("--node-pressure", default="ns=2;s=Testa chamber.pres")

    ap.add_argument("--node-kp", required=True, help="Node id for Kp gain (write target).")
    ap.add_argument("--node-ki", required=True, help="Node id for Ki gain (write target).")
    ap.add_argument("--node-kd", required=True, help="Node id for Kd gain (write target).")

    args = ap.parse_args()

    # Load model bundle
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    # Override/align smoothing with runtime args (optional)
    bundle.ema_alpha = float(args.ema_alpha)
    bundle.freeze_on_step = bool(args.freeze_on_step)

    nodes = NodeMap(
        temp=args.node_temp,
        sp=args.node_sp,
        p_term=args.node_p,
        i_term=args.node_i,
        d_term=args.node_d,
        pressure=args.node_pressure,
        kp=args.node_kp,
        ki=args.node_ki,
        kd=args.node_kd,
    )

    client = Client(args.endpoint)
    client.connect()
    print(f"Connected to {args.endpoint}")

    # Rolling buffer sized to the model window
    W = int(bundle.cfg.window_s * bundle.cfg.sample_hz)
    buf: Deque[Dict[str, float]] = deque(maxlen=W)

    # Current gains from OPC
    kp_cur, ki_cur, kd_cur = read_gains(client, nodes)
    kp_f, ki_f, kd_f = kp_cur, ki_cur, kd_cur

    last_update_t = 0.0
    start = time.time()

    with open(args.log, "w", encoding="utf-8") as logf:
        while True:
            now = time.time()
            if now - start >= args.duration:
                break

            # Read telemetry
            sample = read_nodes(client, nodes)
            sample["timestamp"] = now
            buf.append(sample)

            # Not enough history yet
            if len(buf) < W:
                time.sleep(args.dt)
                continue

            # Build window df for feature computation
            window_df = pd.DataFrame(list(buf), columns=["sp", "temp", "p_term", "i_term", "d_term", "p404", "p23", "timestamp"])

            feat = rolling_features_runtime(
                window_df[["sp", "temp", "p_term", "i_term", "d_term", "p404", "p23"]],
                window_s=bundle.cfg.window_s,
                step_lookback_s=bundle.cfg.step_lookback_s,
                min_change_sp=bundle.cfg.min_change_sp,
                sample_hz=bundle.cfg.sample_hz,
            )

            # If model features not ready (NaNs)
            if feat[bundle.feature_cols].isna().any():
                time.sleep(args.dt)
                continue

            # Rate limit updates
            do_update = (now - last_update_t) >= args.update_every

            # Freeze during SP step
            if bundle.freeze_on_step and float(feat.get("is_step", 0.0)) > 0.5:
                dkp = dki = dkd = 0.0
                do_update = False
            else:
                dkp, dki, dkd = predict_deltas(bundle, feat)

            kp_sug, ki_sug, kd_sug = apply_deltas(bundle, kp_f, ki_f, kd_f, dkp, dki, dkd)

            # EMA smoothing
            a = float(bundle.ema_alpha)
            kp_f = (1 - a) * kp_f + a * kp_sug
            ki_f = (1 - a) * ki_f + a * ki_sug
            kd_f = (1 - a) * kd_f + a * kd_sug

            # Write gains (optional)
            wrote = False
            if args.write and do_update:
                write_gains(client, nodes, kp_f, ki_f, kd_f)
                last_update_t = now
                wrote = True

            # Log record
            rec = {
                "t": now,
                "sp": sample["sp"],
                "temp": sample["temp"],
                "e": sample["sp"] - sample["temp"],
                "p_term": sample["p_term"],
                "i_term": sample["i_term"],
                "d_term": sample["d_term"],
                "pressure": sample["p404"],
                "features": {k: float(feat[k]) for k in bundle.feature_cols},
                "dkp": dkp, "dki": dki, "dkd": dkd,
                "kp": kp_f, "ki": ki_f, "kd": kd_f,
                "wrote": wrote,
            }
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logf.flush()

            # Console
            print(
                f"SP={sample['sp']:.2f} T={sample['temp']:.2f} e={sample['sp']-sample['temp']:+.2f} | "
                f"dP={dkp:+.3f} dI={dki:+.3f} dD={dkd:+.3f} -> "
                f"Kp={kp_f:.2f} Ki={ki_f:.2f} Kd={kd_f:.2f} {'[WROTE]' if wrote else ''}"
            )

            time.sleep(args.dt)

    client.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    main()

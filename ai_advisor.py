#!/usr/bin/env python3
"""
ai_advisor.py

Real-time OPC UA loop that:
- reads chamber telemetry
- computes rolling-window features
- uses pid.pkl to suggest bounded Kp/Ki/Kd updates
- optionally writes new gains back via OPC UA
- logs all readings and actions to JSONL

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
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import numpy as np
import pandas as pd
from opcua import Client


# -----------------------------
# Helpers
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


# -----------------------------
# Feature computation 
# -----------------------------

def rolling_features_runtime(
    window_df: pd.DataFrame,
    window_s: float,
    step_lookback_s: float,
    min_step: float,
    sample_hz: float,
) -> pd.Series:
    """
    Compute last-row features from a rolling window.

    window_df must contain columns:
      sp, temp, p_term, i_term, d_term, p404, p23

    window_s, step_lookback_s, min_step are taken from the saved model.
    sample_hz is derived from your loop dt (sample_hz = 1/dt).
    """
    W = int(round(window_s * sample_hz))
    L = int(round(step_lookback_s * sample_hz))
    L = max(L, 1)

    df = window_df.copy()

    # base signals
    df["e"] = df["sp"] - df["temp"]                 # error
    df["de"] = df["e"].diff().fillna(0.0)           # derivative of error
    df["dtemp"] = df["temp"].diff().fillna(0.0)     # derivative of temperature

    # rolling stats over last W samples
    df["e_abs_mean_W"] = df["e"].abs().rolling(W, min_periods=W).mean()
    df["e_mean_W"] = df["e"].rolling(W, min_periods=W).mean()
    df["e_std_W"] = df["e"].rolling(W, min_periods=W).std()

    df["de_mean_W"] = df["de"].rolling(W, min_periods=W).mean()
    df["de_std_W"] = df["de"].rolling(W, min_periods=W).std()

    df["temp_slope_W"] = df["dtemp"].rolling(W, min_periods=W).mean()

    for c in ["p_term", "i_term", "d_term"]:
        df[f"{c}_mean_W"] = df[c].rolling(W, min_periods=W).mean()
        df[f"{c}_std_W"] = df[c].rolling(W, min_periods=W).std()

    for c in ["p404", "p23"]:
        df[f"{c}_mean_W"] = df[c].rolling(W, min_periods=W).mean()
        df[f"{c}_std_W"] = df[c].rolling(W, min_periods=W).std()

    # setpoint step detector
    df["sp_step"] = df["sp"] - df["sp"].shift(L)
    df["is_step"] = (df["sp_step"].abs() >= float(min_step)).astype(float)

    return df.iloc[-1]


def predict_deltas(model: dict, feat_row: pd.Series) -> Tuple[float, float, float]:
    """
    Predict (dkp_mult, dki_mult, dkd_mult) using the 3 decision trees in the model dict.
    """
    feats = model["feature_cols"]
    X = feat_row[feats].to_numpy().reshape(1, -1)

    dkp = float(model["tree_kp"].predict(X)[0])
    dki = float(model["tree_ki"].predict(X)[0])
    dkd = float(model["tree_kd"].predict(X)[0])

    dkp = float(np.clip(dkp, *model["delta_bounds"]["dkp"]))
    dki = float(np.clip(dki, *model["delta_bounds"]["dki"]))
    dkd = float(np.clip(dkd, *model["delta_bounds"]["dkd"]))
    return dkp, dki, dkd


def apply_deltas(model: dict, kp: float, ki: float, kd: float, dkp: float, dki: float, dkd: float) -> Tuple[float, float, float]:
    """
    Apply deltas as multiplicative updates and clamp to safe bounds.
    """
    kp2 = kp * (1.0 + dkp)
    ki2 = ki * (1.0 + dki)
    kd2 = kd * (1.0 + dkd)

    kp2 = clamp(kp2, *model["gain_bounds"]["kp"])
    ki2 = clamp(ki2, *model["gain_bounds"]["ki"])
    kd2 = clamp(kd2, *model["gain_bounds"]["kd"])
    return kp2, ki2, kd2


# -----------------------------
# OPC UA mapping
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


def opc_read(client: Client, nodes: NodeMap) -> Dict[str, float]:
    """
    Read telemetry nodes.
    p404 and p23: training had two pressures; if you only have one, we map:
      p404 = pressure, p23 = 0.0
    """
    temp = float(client.get_node(nodes.temp).get_value())
    sp = float(client.get_node(nodes.sp).get_value())
    p_term = float(client.get_node(nodes.p_term).get_value())
    i_term = float(client.get_node(nodes.i_term).get_value())
    d_term = float(client.get_node(nodes.d_term).get_value())
    pressure = float(client.get_node(nodes.pressure).get_value())

    return {
        "temp": temp,
        "sp": sp,
        "p_term": p_term,
        "i_term": i_term,
        "d_term": d_term,
        "p404": pressure,
        "p23": 0.0,
    }


def opc_read_gains(client: Client, nodes: NodeMap) -> Tuple[float, float, float]:
    kp = float(client.get_node(nodes.kp).get_value())
    ki = float(client.get_node(nodes.ki).get_value())
    kd = float(client.get_node(nodes.kd).get_value())
    return kp, ki, kd


def opc_write_gains(client: Client, nodes: NodeMap, kp: float, ki: float, kd: float) -> None:
    client.get_node(nodes.kp).set_value(kp)
    client.get_node(nodes.ki).set_value(ki)
    client.get_node(nodes.kd).set_value(kd)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--write", action="store_true", help="Actually write Kp/Ki/Kd to OPC UA.")
    ap.add_argument("--log", default="opcua_ai_log.jsonl")

    ap.add_argument("--update-every", type=float, default=5.0, help="Seconds between writes (rate limit).")
    ap.add_argument("--ema-alpha", type=float, default=0.2, help="EMA smoothing on suggested gains.")
    ap.add_argument("--freeze-on-step", action="store_true", default=True, help="Do not update during SP steps.")

    ap.add_argument("--node-temp", default="ns=2;s=Testa chamber.temp")
    ap.add_argument("--node-sp", required=True, help="Node id for SP (temperature setpoint).")
    ap.add_argument("--node-p", default="ns=2;s=Testa chamber.temp_u_p")
    ap.add_argument("--node-i", default="ns=2;s=Testa chamber.temp_u_i")
    ap.add_argument("--node-d", default="ns=2;s=Testa chamber.temp_u_d")
    ap.add_argument("--node-pressure", default="ns=2;s=Testa chamber.pres")

    ap.add_argument("--node-kp", required=True, help="Node id for writable Kp gain.")
    ap.add_argument("--node-ki", required=True, help="Node id for writable Ki gain.")
    ap.add_argument("--node-kd", required=True, help="Node id for writable Kd gain.")

    args = ap.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    model["ema_alpha"] = float(args.ema_alpha)
    model["freeze_on_step"] = bool(args.freeze_on_step)

    # Sampling
    if args.dt <= 0:
        raise ValueError("--dt must be > 0")
    sample_hz = 1.0 / float(args.dt)

    # Rolling buffer length in samples
    window_s = float(model["window_s"])
    W = int(round(window_s * sample_hz))

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

    # Rolling telemetry buffer
    buf: Deque[Dict[str, float]] = deque(maxlen=W)

    kp_f, ki_f, kd_f = opc_read_gains(client, nodes)

    last_write_t = 0.0
    start = time.time()

    with open(args.log, "w", encoding="utf-8") as logf:
        while True:
            now = time.time()
            if now - start >= args.duration:
                break

            # ---- 1) Read telemetry
            sample = opc_read(client, nodes)
            sample["timestamp"] = now
            buf.append(sample)

            # ---- 2) Wait until buffer has enough history
            if len(buf) < W:
                time.sleep(args.dt)
                continue

            # ---- 3) Build feature row from rolling window
            win_df = pd.DataFrame(list(buf), columns=["sp", "temp", "p_term", "i_term", "d_term", "p404", "p23", "timestamp"])
            feat = rolling_features_runtime(
                win_df[["sp", "temp", "p_term", "i_term", "d_term", "p404", "p23"]],
                window_s=float(model["window_s"]),
                step_lookback_s=float(model.get("step_lookback_s", 5.0)),
                min_step=float(model.get("min_step", 0.5)),
                sample_hz=sample_hz,
            )

            # skip incomplete rows
            if feat[model["feature_cols"]].isna().any():
                time.sleep(args.dt)
                continue

            # ---- 4) Update gains every N seconds
            do_write = (now - last_write_t) >= float(args.update_every)

            # Freeze during SP steps
            if model["freeze_on_step"] and float(feat.get("is_step", 0.0)) > 0.5:
                dkp = dki = dkd = 0.0
                do_write = False
            else:
                dkp, dki, dkd = predict_deltas(model, feat)

            # ---- 5) Apply deltas + EMA smoothing
            kp_sug, ki_sug, kd_sug = apply_deltas(model, kp_f, ki_f, kd_f, dkp, dki, dkd)

            a = float(model["ema_alpha"])
            kp_f = (1 - a) * kp_f + a * kp_sug
            ki_f = (1 - a) * ki_f + a * ki_sug
            kd_f = (1 - a) * kd_f + a * kd_sug

            # ---- 6) Write
            wrote = False
            if args.write and do_write:
                opc_write_gains(client, nodes, kp_f, ki_f, kd_f)
                last_write_t = now
                wrote = True

            # ---- 7) Log
            rec = {
                "t": now,
                "sp": sample["sp"],
                "temp": sample["temp"],
                "e": sample["sp"] - sample["temp"],
                "p_term": sample["p_term"],
                "i_term": sample["i_term"],
                "d_term": sample["d_term"],
                "pressure": sample["p404"],
                "features": {k: float(feat[k]) for k in model["feature_cols"]},
                "dkp": dkp, "dki": dki, "dkd": dkd,
                "kp": kp_f, "ki": ki_f, "kd": kd_f,
                "wrote": wrote,
            }
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logf.flush()

            print(
                f"SP={sample['sp']:.2f} T={sample['temp']:.2f} e={sample['sp']-sample['temp']:+.2f} | "
                f"dKp={dkp:+.3f} dKi={dki:+.3f} dKd={dkd:+.3f} -> "
                f"Kp={kp_f:.3f} Ki={ki_f:.3f} Kd={kd_f:.3f} {'[WROTE]' if wrote else ''}"
            )

            time.sleep(args.dt)

    client.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    main()
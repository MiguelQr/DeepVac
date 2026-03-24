#!/usr/bin/env python3
"""
Run:
  python ai_advisor.py \
    --endpoint opc.tcp://192.168.88.144:12345 \
    --model pid.pkl \
    --duration 600 \
    --dt 1.0 \
    --kp0 10 --ki0 600 --kd0 6 \
    --telemetry-csv telemetry.csv \
    --advice-csv advice.csv \
    --log opc_log.jsonl

python ai_advisor.py --endpoint opc.tcp://192.168.88.144:12345 --model pid.pkl --duration 1200 --dt 1.0 --kp0 15 --ki0 1000 --kd0 6 --telemetry-csv telemetry.csv --advice-csv advice.csv --log opc_log.jsonl

Model was trained on:
  sp, temp, p_term, i_term, d_term

In the OPC:
  temp_ref, temp, temp_raw, temp_u_p, temp_u_i, temp_u_d

Map:
  sp        -> temp_ref
  temp      -> temp
  p_term    -> temp_u_p
  i_term    -> temp_u_i
  d_term    -> temp_u_d

This runtime script:
  - DOES NOT write gains to OPC UA.
  - Only logs telemetry + suggested gains.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from opcua import Client


# -----------------------------
# Helpers
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def fmt_dt(ts: float) -> str:
    # DD.MM.YY HH:MM:SS (local time)
    return time.strftime("%d.%m.%y %H:%M:%S", time.localtime(ts))


def fmt_num(x: float, ndigits: int = 2) -> str:
    s = f"{x:.{ndigits}f}"
    return s.replace(".", ",")


# -----------------------------
# CSV Writers
# -----------------------------

class TelemetryCSVWriter:
    """
    Writes telemetry rows
    """
    HEADER = [
        "Date/Time",
        "Temperature",
        "Setpoint Temperature °C",
        "Temperature Control (P) %",
        "Temperature Control (I) %",
        "Temperature Control (D) %",
    ]

    def __init__(self, path: str) -> None:
        self.path = path
        self.fh = open(path, "a", encoding="utf-8", newline="")
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.fh.tell() == 0:
            self.fh.write(";".join(self.HEADER) + "\n")
            self.fh.flush()

    def write(self, ts: float, temp: float, sp: float, p_term: float, i_term: float, d_term: float) -> None:
        row = [
            fmt_dt(ts),
            fmt_num(temp, 2),
            fmt_num(sp, 2),
            fmt_num(p_term, 2),
            fmt_num(i_term, 2),
            fmt_num(d_term, 2),
        ]
        self.fh.write(";".join(row) + "\n")
        self.fh.flush()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass


class AdviceCSVWriter:
    """
    Writes suggested gains + deltas
    """
    HEADER = [
        "Date/Time",
        "dkp_mult",
        "dki_mult",
        "dkd_mult",
        "Kp_suggested",
        "Ki_suggested",
        "Kd_suggested",
        "is_step",
        "updated",
    ]

    def __init__(self, path: str) -> None:
        self.path = path
        self.fh = open(path, "a", encoding="utf-8", newline="")
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.fh.tell() == 0:
            self.fh.write(";".join(self.HEADER) + "\n")
            self.fh.flush()

    def write(
        self,
        ts: float,
        dkp: float,
        dki: float,
        dkd: float,
        kp: float,
        ki: float,
        kd: float,
        is_step: float,
        updated: bool,
    ) -> None:
        row = [
            fmt_dt(ts),
            fmt_num(dkp, 4),
            fmt_num(dki, 4),
            fmt_num(dkd, 4),
            fmt_num(kp, 4),
            fmt_num(ki, 4),
            fmt_num(kd, 4),
            fmt_num(is_step, 0),
            "1" if updated else "0",
        ]
        self.fh.write(";".join(row) + "\n")
        self.fh.flush()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass


# -----------------------------
# Feature computation (must match training)
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

    window_df must contain:
      sp, temp, p_term, i_term, d_term
    """
    W = int(round(window_s * sample_hz))
    L = int(round(step_lookback_s * sample_hz))
    L = max(L, 1)

    df = window_df.copy()

    df["e"] = df["sp"] - df["temp"]
    df["de"] = df["e"].diff().fillna(0.0)
    df["dtemp"] = df["temp"].diff().fillna(0.0)

    df["e_abs_mean_W"] = df["e"].abs().rolling(W, min_periods=W).mean()
    df["e_mean_W"] = df["e"].rolling(W, min_periods=W).mean()
    df["e_std_W"] = df["e"].rolling(W, min_periods=W).std()

    df["de_mean_W"] = df["de"].rolling(W, min_periods=W).mean()
    df["de_std_W"] = df["de"].rolling(W, min_periods=W).std()

    df["temp_slope_W"] = df["dtemp"].rolling(W, min_periods=W).mean()

    for c in ["p_term", "i_term", "d_term"]:
        df[f"{c}_mean_W"] = df[c].rolling(W, min_periods=W).mean()
        df[f"{c}_std_W"] = df[c].rolling(W, min_periods=W).std()

    df["sp_step"] = df["sp"] - df["sp"].shift(L)
    df["is_step"] = (df["sp_step"].abs() >= float(min_step)).astype(float)

    return df.iloc[-1]


def assert_model_features_present(model: dict, feat_row: pd.Series) -> None:
    missing = [c for c in model["feature_cols"] if c not in feat_row.index]
    if missing:
        raise RuntimeError(
            "Model expects features that are not computed online.\n"
            f"Missing: {missing}\n"
            "This usually means your pid.pkl was trained with different feature columns.\n"
            "Fix by retraining or aligning feature computation."
        )


def predict_deltas(model: dict, feat_row: pd.Series) -> Tuple[float, float, float]:
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
    kp2 = kp * (1.0 + dkp)
    ki2 = ki * (1.0 + dki)
    kd2 = kd * (1.0 + dkd)

    kp2 = clamp(kp2, *model["gain_bounds"]["kp"])
    ki2 = clamp(ki2, *model["gain_bounds"]["ki"])
    kd2 = clamp(kd2, *model["gain_bounds"]["kd"])
    return kp2, ki2, kd2


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


def opc_read(client: Client, nodes: NodeMap) -> Dict[str, float]:
    temp = float(client.get_node(nodes.temp).get_value())
    sp = float(client.get_node(nodes.sp).get_value())
    p_term = float(client.get_node(nodes.p_term).get_value())
    i_term = float(client.get_node(nodes.i_term).get_value())
    d_term = float(client.get_node(nodes.d_term).get_value())
    return {"temp": temp, "sp": sp, "p_term": p_term, "i_term": i_term, "d_term": d_term}


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=1.0)

    # Initial gains (required for meaningful suggestions)
    ap.add_argument("--kp0", type=float, required=True)
    ap.add_argument("--ki0", type=float, required=True)
    ap.add_argument("--kd0", type=float, required=True)

    # Outputs
    ap.add_argument("--telemetry-csv", default="telemetry.csv")
    ap.add_argument("--advice-csv", default="advice.csv")
    ap.add_argument("--log", default="opc_log.jsonl")

    # Node IDs
    ap.add_argument("--node-temp", default="ns=2;s=Testa chamber.temp")
    ap.add_argument("--node-sp", default="ns=2;s=Testa chamber.temp_ref")
    ap.add_argument("--node-p", default="ns=2;s=Testa chamber.temp_u_p")
    ap.add_argument("--node-i", default="ns=2;s=Testa chamber.temp_u_i")
    ap.add_argument("--node-d", default="ns=2;s=Testa chamber.temp_u_d")

    args = ap.parse_args()

    if args.dt <= 0:
        raise ValueError("--dt must be > 0")
    sample_hz = 1.0 / float(args.dt)

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Use model runtime parameters (same as decisiontree.py)
    ema_alpha = float(model.get("ema_alpha", 0.2))
    min_update_s = float(model.get("min_update_s", 5))
    freeze_on_step = bool(model.get("freeze_on_step", True))

    window_s = float(model["window_s"])
    W = int(round(window_s * sample_hz))
    if W < 2:
        raise ValueError("Computed rolling window length W < 2; check dt and model['window_s'].")

    nodes = NodeMap(
        temp=args.node_temp,
        sp=args.node_sp,
        p_term=args.node_p,
        i_term=args.node_i,
        d_term=args.node_d,
    )

    client = Client(args.endpoint)
    client.connect()
    print(f"Connected to {args.endpoint}")

    telem_writer = TelemetryCSVWriter(args.telemetry_csv)
    advice_writer = AdviceCSVWriter(args.advice_csv)

    # Rolling telemetry buffer
    buf: Deque[Dict[str, float]] = deque(maxlen=W)

    # Internal gain state (start from provided baseline gains)
    kp_f, ki_f, kd_f = float(args.kp0), float(args.ki0), float(args.kd0)

    last_update_t = -1e18
    start = time.time()

    try:
        with open(args.log, "w", encoding="utf-8") as logf:
            while True:
                now = time.time()
                if now - start >= args.duration:
                    break

                # 1) Read online telemetry
                sample = opc_read(client, nodes)
                sample["timestamp"] = now
                buf.append(sample)

                # 1b) Write telemetry CSV each sample
                telem_writer.write(
                    ts=now,
                    temp=sample["temp"],
                    sp=sample["sp"],
                    p_term=sample["p_term"],
                    i_term=sample["i_term"],
                    d_term=sample["d_term"],
                )

                # 2) Wait for buffer
                if len(buf) < W:
                    time.sleep(args.dt)
                    continue

                # 3) Compute features
                win_df = pd.DataFrame(list(buf), columns=["sp", "temp", "p_term", "i_term", "d_term", "timestamp"])
                feat = rolling_features_runtime(
                    win_df[["sp", "temp", "p_term", "i_term", "d_term"]],
                    window_s=float(model["window_s"]),
                    step_lookback_s=float(model.get("step_lookback_s", 5.0)),
                    min_step=float(model.get("min_step", 0.5)),
                    sample_hz=sample_hz,
                )

                # Ensure model features exist
                assert_model_features_present(model, feat)

                # Skip if NaNs
                if feat[model["feature_cols"]].isna().any():
                    time.sleep(args.dt)
                    continue

                is_step = float(feat.get("is_step", 0.0))

                # 4) Determine whether we are allowed to update now
                allow_update = (now - last_update_t) >= min_update_s
                updated = False

                # Freeze on step
                if freeze_on_step and is_step > 0.5:
                    dkp = dki = dkd = 0.0
                else:
                    if allow_update:
                        dkp, dki, dkd = predict_deltas(model, feat)
                        last_update_t = now
                        updated = True
                    else:
                        dkp = dki = dkd = 0.0

                # 5) Apply deltas -> proposed new gains, then EMA smooth
                kp_new, ki_new, kd_new = apply_deltas(model, kp_f, ki_f, kd_f, dkp, dki, dkd)

                a = ema_alpha
                kp_f = (1 - a) * kp_f + a * kp_new
                ki_f = (1 - a) * ki_f + a * ki_new
                kd_f = (1 - a) * kd_f + a * kd_new

                # 6) Advice CSV
                advice_writer.write(
                    ts=now,
                    dkp=dkp,
                    dki=dki,
                    dkd=dkd,
                    kp=kp_f,
                    ki=ki_f,
                    kd=kd_f,
                    is_step=is_step,
                    updated=updated,
                )

                # 7) JSONL debug log
                rec = {
                    "t": now,
                    "sp": sample["sp"],
                    "temp": sample["temp"],
                    "e": sample["sp"] - sample["temp"],
                    "p_term": sample["p_term"],
                    "i_term": sample["i_term"],
                    "d_term": sample["d_term"],
                    "features": {k: float(feat[k]) for k in model["feature_cols"]},
                    "is_step": is_step,
                    "dkp_mult": dkp,
                    "dki_mult": dki,
                    "dkd_mult": dkd,
                    "Kp_suggested": kp_f,
                    "Ki_suggested": ki_f,
                    "Kd_suggested": kd_f,
                    "updated": updated,
                }
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logf.flush()

                print(
                    f"SP={sample['sp']:.2f} T={sample['temp']:.2f} e={sample['sp']-sample['temp']:+.2f} | "
                    f"step={int(is_step>0.5)} upd={int(updated)} | "
                    f"dKp={dkp:+.3f} dKi={dki:+.3f} dKd={dkd:+.3f} -> "
                    f"Kp={kp_f:.3f} Ki={ki_f:.3f} Kd={kd_f:.3f}"
                )

                time.sleep(args.dt)

    finally:
        telem_writer.close()
        advice_writer.close()
        client.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()

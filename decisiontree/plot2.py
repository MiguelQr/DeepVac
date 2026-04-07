import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot temperature vs time for readings CSV with multiple setpoints.")
    ap.add_argument("--csv", default="readings/20.csv", help="Path to readings CSV")
    ap.add_argument("--out", default="plots/readings_temp_vs_setpoints.png", help="Output image path")
    ap.add_argument("--show", action="store_true", help="Show interactive plot window")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=";", decimal=",", encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]

    time_col = "timestamp"
    temp_col = "temperature"
    sp_col = "setpoint"

    required = [time_col, temp_col, sp_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df[time_col] = pd.to_datetime(df[time_col], format="%d.%m.%y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    t_min = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds() / 60.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_min, df[temp_col], label="Temperature", linewidth=1.8)
    ax.step(t_min, df[sp_col], where="post", linestyle="--", linewidth=1.6, label="Setpoint")

    # Mark setpoint changes and annotate value
    sp_change = df[sp_col].ne(df[sp_col].shift(1)).fillna(True)
    change_idx = df.index[sp_change]
    for i in change_idx:
        x = float(t_min.iloc[i])
        y = float(df[sp_col].iloc[i])
        ax.axvline(x=x, color="gray", alpha=0.18, linewidth=0.8)
        ax.text(x, y, f"{y:.1f}°C", fontsize=8, va="bottom", ha="left", alpha=0.85)

    ax.set_title("Temperature vs Time")
    ax.set_xlabel("Time since start (min)")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
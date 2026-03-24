import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot telemetry temperature vs time.")
    ap.add_argument("--csv", default="history/1/telemetry.csv", help="Path to telemetry CSV")
    ap.add_argument("--out", default="plots/telemetry_temp_vs_time.png", help="Output image path")
    ap.add_argument("--show", action="store_true", help="Show interactive plot window")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=";", decimal=",", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    time_col = "Date/Time"
    temp_col = "Temperature"
    sp_col = "Setpoint Temperature °C"

    df[time_col] = pd.to_datetime(df[time_col], format="%d.%m.%y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    elapsed_minutes = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds() / 60.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(elapsed_minutes, df[temp_col], label="Temperature", linewidth=1.8)
    ax.plot(elapsed_minutes, df[sp_col], "--", label="Setpoint", linewidth=1.5)

    # Add setpoint info in title
    sp_unique = sorted(df[sp_col].dropna().unique().tolist())
    sp_text = ", ".join(f"{v:.2f}" for v in sp_unique[:6])
    if len(sp_unique) > 6:
        sp_text += ", ..."
    ax.set_title(f"Temperature vs Time")

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
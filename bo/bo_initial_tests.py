#!/usr/bin/env python3
"""Initial OPC tests with manually provided PID gains and log data."""

from __future__ import annotations

import argparse

from bo_common import (
    OPCNodeMap,
    append_rows_csv,
    history_run_file,
    run_opc_test,
)


ENDPOINT = "opc.tcp://192.168.88.144:12345"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--kp", type=float, required=True, help="Manually applied kp value")
    ap.add_argument("--ki", type=float, required=True, help="Manually applied ki value")
    ap.add_argument("--kd", type=float, required=True, help="Manually applied kd value")
    ap.add_argument("--test-duration", type=float, default=900.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--progress-every", type=float, default=60.0, help="Print progress every N seconds (0 disables)")

    ap.add_argument("--samples-csv", default="history/bo_samples.csv")
    ap.add_argument("--runs-csv", default="history/bo_runs.csv")

    ap.add_argument("--node-temp", default="ns=2;s=Testa chamber.temp")
    ap.add_argument("--node-temp_ref", default="ns=2;s=Testa chamber.temp_ref")
    ap.add_argument("--node-temp-raw", default="ns=2;s=Testa chamber.temp_raw")
    ap.add_argument("--node-temp-u", default="ns=2;s=Testa chamber.temp_u")
    ap.add_argument("--node-p", default="ns=2;s=Testa chamber.temp_u_p")
    ap.add_argument("--node-i", default="ns=2;s=Testa chamber.temp_u_i")
    ap.add_argument("--node-d", default="ns=2;s=Testa chamber.temp_u_d")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    kp = float(args.kp)
    ki = float(args.ki)
    kd = float(args.kd)

    nodes = OPCNodeMap(
        temp=args.node_temp,
        temp_ref=args.node_temp_ref,
        temp_raw=args.node_temp_raw,
        temp_u=args.node_temp_u,
        temp_u_p=args.node_p,
        temp_u_i=args.node_i,
        temp_u_d=args.node_d,
    )

    print(f"Testing kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")

    df_samples, run_summary = run_opc_test(
        endpoint=ENDPOINT,
        nodes=nodes,
        kp=kp,
        ki=ki,
        kd=kd,
        duration_s=args.test_duration,
        dt_s=args.dt,
        progress_every_s=args.progress_every,
        verbose=True,
    )

    run_id = str(run_summary["run_id"])
    samples_out = history_run_file(run_id, args.samples_csv)
    runs_out = history_run_file(run_id, args.runs_csv)

    append_rows_csv(samples_out, df_samples.to_dict(orient="records"))
    append_rows_csv(runs_out, [run_summary])

    print(
        f"  run_id={run_summary['run_id']} samples={run_summary['num_samples']} mse={run_summary['mse']:.6f}"
    )
    print(f"  samples_csv={samples_out}")
    print(f"  runs_csv={runs_out}")


if __name__ == "__main__":
    main()

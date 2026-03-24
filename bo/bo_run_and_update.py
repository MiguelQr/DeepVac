#!/usr/bin/env python3
"""Run one OPC test with manually provided PID gains, append data, and update GP model."""

from __future__ import annotations

import argparse
import pickle
from typing import Dict

from bo_common import (
    OPCNodeMap,
    append_rows_csv,
    fit_gp_model,
    history_run_file,
    load_runs_table,
    run_opc_test,
    save_json,
)


ENDPOINT = "opc.tcp://192.168.88.144:12345"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--test-duration", type=float, default=180.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--progress-every", type=float, default=30.0, help="Print progress every N seconds (0 disables)")

    ap.add_argument("--samples-csv", default="history/bo_samples.csv")
    ap.add_argument("--runs-csv", default="history/bo_runs.csv")
    ap.add_argument("--model-out", default="history/bo_gp_model.pkl")
    ap.add_argument("--result-out", default="history/bo_last_result.json")

    ap.add_argument("--kp", type=float, required=True, help="Manually applied kp value")
    ap.add_argument("--ki", type=float, required=True, help="Manually applied ki value")
    ap.add_argument("--kd", type=float, required=True, help="Manually applied kd value")

    ap.add_argument("--node-temp", default="ns=2;s=Testa chamber.temp")
    ap.add_argument("--node-temp_ref", default="ns=2;s=Testa chamber.temp_ref")
    ap.add_argument("--node-temp-raw", default="ns=2;s=Testa chamber.temp_raw")
    ap.add_argument("--node-temp-u", default="ns=2;s=Testa chamber.temp_u")
    ap.add_argument("--node-p", default="ns=2;s=Testa chamber.temp_u_p")
    ap.add_argument("--node-i", default="ns=2;s=Testa chamber.temp_u_i")
    ap.add_argument("--node-d", default="ns=2;s=Testa chamber.temp_u_d")

    return ap


def main() -> None:
    args = build_parser().parse_args()
    params: Dict[str, float] = {"kp": args.kp, "ki": args.ki, "kd": args.kd}

    nodes = OPCNodeMap(
        temp=args.node_temp,
        temp_ref=args.node_temp_ref,
        temp_raw=args.node_temp_raw,
        temp_u=args.node_temp_u,
        temp_u_p=args.node_p,
        temp_u_i=args.node_i,
        temp_u_d=args.node_d,
    )

    df_samples, run_summary = run_opc_test(
        endpoint=ENDPOINT,
        nodes=nodes,
        kp=params["kp"],
        ki=params["ki"],
        kd=params["kd"],
        duration_s=args.test_duration,
        dt_s=args.dt,
        progress_every_s=args.progress_every,
        verbose=True,
    )

    run_id = str(run_summary["run_id"])
    samples_out = history_run_file(run_id, args.samples_csv)
    runs_out = history_run_file(run_id, args.runs_csv)
    model_out = history_run_file(run_id, args.model_out)
    result_out = history_run_file(run_id, args.result_out)

    append_rows_csv(samples_out, df_samples.to_dict(orient="records"))
    append_rows_csv(runs_out, [run_summary])

    # load_runs_table will aggregate history/<run_id>/bo_runs.csv when needed.
    runs = load_runs_table(args.runs_csv)
    model = fit_gp_model(runs)
    with open(model_out, "wb") as fh:
        pickle.dump(model, fh)

    save_json(
        result_out,
        {
            "run_summary": run_summary,
            "used_params": params,
            "objective": "MSE(setpoint - temp)",
            "files": {
                "samples_csv": samples_out,
                "runs_csv": runs_out,
                "model_out": model_out,
            },
        },
    )

    print("Run completed.")
    print(f"run_id={run_summary['run_id']} mse={run_summary['mse']:.6f}")
    print(f"kp={run_summary['kp']:.6f} ki={run_summary['ki']:.6f} kd={run_summary['kd']:.6f}")
    print(f"samples_csv={samples_out}")
    print(f"runs_csv={runs_out}")
    print(f"Updated model: {model_out}")
    print(f"Saved run result: {result_out}")


if __name__ == "__main__":
    main()

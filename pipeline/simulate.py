"""Simulated production: replay test engines towards the API, cycle by cycle.

Start the API with a prediction log first, then, in another terminal:
    python -m pipeline.simulate --scenario normal
Scenarios: normal (test FD001), sensor_offset (test FD001, two sensors drift during the second
half), new_fault_mode (test FD003, a fault mode the model never saw). Test FD001 engines 1 to 50
are kept aside as the reference period of the drift report, so FD001 scenarios replay engines
51 to 100. The true remaining life of every cycle sent goes to monitoring/ground_truth.jsonl,
for the error tracking.
"""

import argparse
import json
import os
import time

import httpx

from pipeline.monitor import load_reference
from turbofan.config import load_params
from turbofan.data import load_raw, load_test_rul
from turbofan.monitoring import MONITORING_DIR
from turbofan.simulate import apply_sensor_offset, iter_engine_cycles, true_rul

SCENARIOS = ["normal", "sensor_offset", "new_fault_mode"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=SCENARIOS, default="normal")
    parser.add_argument("--engines", type=int, help="replay only the first N engines")
    parser.add_argument("--step", type=int, default=1, help="send one cycle in N, to go faster")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://127.0.0.1:8000"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params()
    settings = params["monitoring"]
    if args.scenario == "new_fault_mode":
        subset = settings["new_fault_subset"]
        test = load_raw(subset, "test")
    else:
        # engines 1 to reference_engines are the reference period: only the others are replayed
        subset = "FD001"
        test = load_raw(subset, "test")
        test = test[test["unit_id"] > settings["reference_engines"]]
    if args.engines:
        test = test[test["unit_id"].isin(test["unit_id"].unique()[: args.engines])]
    truth = true_rul(test, load_test_rul(subset))
    if args.scenario == "sensor_offset":
        # the offset is measured in standard deviations of the reference, like the drift
        reference_std = load_reference(settings)[settings["offset_sensors"]].std()
        test = apply_sensor_offset(
            test,
            settings["offset_sensors"],
            reference_std,
            settings["offset_std"],
            settings["offset_start"],
        )

    MONITORING_DIR.mkdir(exist_ok=True)
    truth_path = MONITORING_DIR / "ground_truth.jsonl"
    lifetimes = test.groupby("unit_id").size()  # number of cycles of each engine
    print(f"scenario {args.scenario}: {len(lifetimes)} engines of test {subset} -> {args.api_url}")
    start = time.perf_counter()
    n_sent = 0
    with (
        httpx.Client(base_url=args.api_url, timeout=10) as client,
        truth_path.open("a", encoding="utf-8", newline="\n") as truth_file,
    ):
        client.get("/health").raise_for_status()  # fails at once if the API is not running
        for unit_id, history in iter_engine_cycles(test, args.step):
            cycles = history.drop(columns="unit_id").to_dict(orient="records")
            response = client.post("/predict", json={"unit_id": unit_id, "cycles": cycles})
            response.raise_for_status()
            # in real life this truth is only known after the failure; the replay knows it now
            last = history.index[-1]
            record = {
                "scenario": args.scenario,
                "unit_id": unit_id,
                "cycle": int(history.loc[last, "cycle"]),
                "rul_true": int(truth[last]),
            }
            truth_file.write(json.dumps(record) + "\n")
            n_sent += 1
            if len(history) == lifetimes[unit_id] and unit_id % 10 == 0:
                print(f"  engine {unit_id} done, {n_sent} predictions so far")
    seconds = time.perf_counter() - start
    print(f"sent {n_sent} predictions in {seconds:.0f} s; truth appended to {truth_path}")


if __name__ == "__main__":
    main()

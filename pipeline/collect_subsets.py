"""Gather the scores of the chain on the four subsets into reports/subsets.json.

FD001 is the workspace (the DVC pipeline). The other subsets are DVC experiments: the same
chain, up to evaluate, with other parameters, run in a temporary copy of the project, e.g.
    dvc exp run --temp -n fd002-regimes -S data.subset=FD002 -S regimes.normalize=true evaluate
DVC keeps each experiment's parameters and metrics; `dvc exp show --json` lists them all.
"""

import json
import subprocess
import sys

from turbofan.config import REPORTS_DIR

MODELS = ["baseline", "ridge", "random_forest", "xgboost", "lstm", "cnn"]


def summary(name: str, data: dict) -> dict:
    """The parameters and scores of one run of the chain (workspace or experiment)."""
    params = data["params"]["params.yaml"]["data"]
    # DVC writes the metric paths with the separator of the machine: \ on Windows
    metrics = {
        path.replace("\\", "/"): content["data"] for path, content in data["metrics"].items()
    }
    val = metrics["reports/metrics/val_classic.json"] | metrics["reports/metrics/val_deep.json"]
    test = metrics["reports/metrics/test.json"]
    return {
        "name": name,
        "subset": params["data"]["subset"],
        "normalize_by_regime": params["regimes"]["normalize"],
        "n_train_engines": metrics["reports/prepare.json"]["parts"]["train"]["n_engines"],
        "n_test_engines": metrics["reports/prepare.json"]["parts"]["test"]["n_engines"],
        "models": {
            model: {
                "val_rmse": val[model]["rmse"],
                "test_rmse": test[model]["rmse"],
                "test_mae": test[model]["mae"],
                "test_nasa_score": test[model]["nasa_score"],
                "coverage": test[model]["coverage"],
            }
            for model in MODELS
        },
    }


def main() -> None:
    shown = subprocess.run(
        # --all-commits: the experiments stay attached to the commit they started from
        [sys.executable, "-m", "dvc", "exp", "show", "--json", "--all-commits"],
        capture_output=True,
        text=True,
        check=True,
    )
    entries = json.loads(shown.stdout)
    runs = [summary("pipeline", entries[0]["data"])]  # the workspace: FD001
    for commit in entries[1:]:
        for experiment in commit.get("experiments") or []:
            # an experiment still running shows half-written files: wait until it has finished
            if experiment["executor"] is not None:
                print(f"skipped {experiment['name']}: {experiment['executor']['state']}")
                continue
            runs.append(summary(experiment["name"], experiment["revs"][0]["data"]))
    # FD001 first, then by subset, the run without regime scaling before the one with it
    runs.sort(key=lambda run: (run["subset"], run["normalize_by_regime"]))
    path = REPORTS_DIR / "subsets.json"
    path.write_text(json.dumps(runs, indent=2), encoding="utf-8", newline="\n")
    for run in runs:
        lstm = run["models"]["lstm"]
        print(
            f"{run['name']:<15} {run['subset']} regimes={run['normalize_by_regime']!s:<5} "
            f"lstm test rmse {lstm['test_rmse']:.2f}"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

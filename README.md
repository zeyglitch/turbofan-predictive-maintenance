# turbofan-predictive-maintenance

[![ci](https://github.com/zeyglitch/turbofan-predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/zeyglitch/turbofan-predictive-maintenance/actions/workflows/ci.yml)

Predicting how many flight cycles a turbofan engine has left before failure, using NASA's simulated C-MAPSS data.

## Installation

Python 3.13 is required. The commands below are for Windows PowerShell.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

On Linux or macOS, create the environment with `python3.13 -m venv .venv` and activate it with `source .venv/bin/activate`.

PyTorch is installed in its processor-only version, which is enough here. On Windows that is the version pip finds by default. On Linux, add `--extra-index-url https://download.pytorch.org/whl/cpu` to the `pip install` command to get the 200 MB processor build instead of the 514 MB default one. For a graphics card, install torch first with the command given on pytorch.org.

On Windows, keep the project folder close to the drive root (for example `C:\Projects\...`): JupyterLab installs files with very long paths, and pip fails with a long path error if the folder path is longer than about 100 characters.

## Data

The data is the NASA C-MAPSS "Turbofan Engine Degradation Simulation" dataset (Saxena et al., 2008). It is simulated, not measured on real engines. Download it with:

```powershell
python -m pipeline.download
```

The archive (12 MB) is checked against a SHA-256 fingerprint, then the text files are written to `data/raw/`. Running the command again does nothing if the files are already there.

The exploration notebook, `notebooks/01_exploration.ipynb`, shows what the files contain. To open it:

```powershell
jupyter lab notebooks/01_exploration.ipynb
```

## Preparing the data

```powershell
python -m pipeline.prepare
```

This adds the target (remaining useful life, capped at 125 cycles), splits the 100 training engines into train, validation and calibration sets (70 / 15 / 15 engines, never splitting an engine), scales the sensors with statistics learnt on the training engines only, and writes `data/processed/*.parquet`, `models/scaler.joblib` and `reports/prepare.json`.

## First models

```powershell
python -m pipeline.featurize
python -m pipeline.train_classic
```

`featurize` adds, for each sensor, its mean and standard deviation over the last 30 cycles of the same engine, and writes `data/features/*.parquet`. `train_classic` trains the models, prints their scores on the validation engines and writes `reports/metrics/val_classic.json`.

## Experiment tracking

`train_classic` trains four models (constant baseline, Ridge, random forest, XGBoost) and records each run in MLflow: settings, validation scores, a predicted-against-true plot and the model itself. Everything stays on this machine (`mlflow.db` and `mlartifacts/`). To browse the runs:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

then open http://127.0.0.1:5000 and choose "Model training". A small grid of XGBoost settings, outside the main chain, is run with:

```powershell
python -m pipeline.grid xgboost
```

## Neural network

```powershell
python -m pipeline.train_deep
```

`train_deep` cuts each engine into windows of 50 consecutive cycles and trains an LSTM (PyTorch) on them, with early stopping on the validation engines. Every epoch is recorded in MLflow, which draws the learning curves. It writes `models/lstm.pt`, `reports/training_curves.csv` and `reports/metrics/val_deep.json`. On a laptop processor, training takes about three minutes. A small grid (hidden size, then window length) is run with `python -m pipeline.grid lstm`.

`train_deep` also trains a 1D convolutional network (CNN) on the same windows.

## Evaluation and champion model

```powershell
python -m pipeline.evaluate
python -m pipeline.register
```

`evaluate` scores the six models on the 100 test engines (the last cycle of each engine, compared with `RUL_FD001.txt`). It also builds a 90 % prediction interval for each model from its errors on the calibration engines (split conformal prediction), then measures how often the true value falls inside it. It writes `reports/metrics/test.json` and `reports/predictions_test.parquet`. The test set is used only here, never to choose or tune a model.

`register` picks the model with the lowest validation RMSE, registers it in MLflow as `turbofan-rul` with the alias `champion`, and writes `models/champion/`: the model, the scaler and `metadata.json`. This folder is everything needed to predict, without MLflow.

## Explainability

```powershell
python -m pipeline.explain
```

`explain` shows which inputs the models rely on, on the validation engines. For XGBoost, it computes SHAP values on 2,000 rows: global importance, a beeswarm plot, and the explanation of one prediction. For the LSTM, it computes permutation importance: how much the validation RMSE rises when one sensor is shuffled between windows. Tables and figures are written to `reports/explain/`, with `sensor_ranking.csv` comparing the two methods sensor by sensor.

## Prediction API

```powershell
uvicorn api.main:app --port 8000
```

The API loads `models/champion/` once, at start-up, and answers on three routes:

| Route | Answer |
|---|---|
| `GET /health` | `{"status": "ok", "model_loaded": true}` |
| `GET /model` | the champion's type, registry version, date, sensors and scores |
| `POST /predict` | the history of one engine → remaining cycles, 90 % interval, alert |

The interactive documentation is at http://127.0.0.1:8000/docs: each route has a "Try it out" button. Example call from PowerShell, with a two-cycle history:

```powershell
$cycle = @{setting_1 = -0.0006; setting_2 = 0.0004; setting_3 = 100.0; s_1 = 518.67; s_2 = 642.58; s_3 = 1581.22; s_4 = 1398.91; s_5 = 14.62; s_6 = 21.61; s_7 = 554.42; s_8 = 2388.08; s_9 = 9056.4; s_10 = 1.3; s_11 = 47.23; s_12 = 521.79; s_13 = 2388.06; s_14 = 8130.11; s_15 = 8.4024; s_16 = 0.03; s_17 = 393.0; s_18 = 2388.0; s_19 = 100.0; s_20 = 38.81; s_21 = 23.3552}
$body = @{unit_id = 1; cycles = @(($cycle + @{cycle = 1}), ($cycle + @{cycle = 2}))} | ConvertTo-Json -Depth 3
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType application/json -Body $body
```

A malformed request (empty history, missing sensor, text instead of a number) gets a 422 answer saying which field is wrong. Two environment variables set the files used: `MODEL_DIR`, the bundle folder (`models/champion` by default), and `PREDICTION_LOG`, a file where each prediction is appended as one JSON line. Without `PREDICTION_LOG` nothing is logged; to keep a log, start the API with:

```powershell
$env:PREDICTION_LOG = "monitoring/predictions.jsonl"
uvicorn api.main:app --port 8000
```

## Dashboard

```powershell
python -m streamlit run dashboard/app.py
```

then open http://localhost:8501. The dashboard reads the files written by the pipeline (`data/`, `reports/`, `dvc.yaml`) and trains nothing; a page whose files are missing says which command to run. Its pages:

- **Overview**: key figures, the pipeline drawn from `dvc.yaml`, when each stage last wrote its outputs.
- **Data explorer**: the raw files of FD001 to FD004, lifetimes, which sensors are kept and why, correlations, the 21 sensors of one engine.
- **Preprocessing**: four before/after views (scaling, capped target, engine split, rolling window).
- **Models**: the six models' scores, the networks' learning curves, predictions against truth, errors and the alert's confusion matrix.
- **Engine view**: one test engine replayed cycle by cycle, with the prediction and its interval at each cycle, asked to the API. Start the API first (`uvicorn api.main:app --port 8000`); the page says so if it does not answer. `API_URL` changes the API's address.
- **Explainability**: SHAP for XGBoost, permutation importance for the LSTM, one prediction explained.
- **Live monitoring**: the predictions logged by the API, data drift sensor by sensor and the real error over time, refreshed every 5 seconds while a simulation runs (see below).
- **Retraining**: each run of the retraining flow, the champion against the challenger on the test engines, and the decision (see below).
- **Glossary**: the project's terms, explained simply, with a search box.

`python -m` matters: it puts the project folder on Python's import path, so the pages find `turbofan/` and `dashboard/`.

## Run with Docker

The API, the dashboard and the MLflow interface can run together in containers, with Docker Desktop started:

```powershell
docker compose up --build
```

| Service | Address | Image |
|---|---|---|
| `api` | http://localhost:8000/docs | `Dockerfile.api`: FastAPI and processor-only torch (`requirements-api.txt`) |
| `dashboard` | http://localhost:8501 | `Dockerfile.dashboard`: Streamlit, no torch (`requirements-dashboard.txt`) |
| `mlflow` | http://localhost:5000 | the official `ghcr.io/mlflow/mlflow:v3.18.0` |

Training stays on the machine: run `dvc repro` (or `dvc pull`) first. The containers use its results through mounted folders: `models/champion` (read-only) for the API, `data/`, `reports/` and `models/` (read-only) for the dashboard, `mlflow.db` and `mlartifacts/` for MLflow. The API appends every prediction to `monitoring/predictions.jsonl`. The dashboard reaches the API by its service name, `http://api:8000`. Stop everything with Ctrl+C, or `docker compose down`.

MLflow recorded each run's files under the absolute path of the machine that trained (`file:///C:/Projects/turbofan-predictive-maintenance/mlartifacts`). The compose file mounts `mlartifacts/` at that same path inside the container. If the project lives elsewhere, set `MLFLOW_ARTIFACTS_PATH` to that path, written the same way (`/D:/work/turbofan-predictive-maintenance/mlartifacts`), before `docker compose up`.

## Simulated production and drift monitoring

Once the model is in service, are the sensors it receives still like the ones it knows? A simulation replays test engines towards the API, cycle by cycle, as if they were flying, then a drift report compares what the API received with a normal period of service.

Start the API with a prediction log (see above, or use `docker compose up`), then, in another terminal:

```powershell
python -m pipeline.simulate --scenario normal
python -m pipeline.monitor
```

| Scenario | Engines replayed | What happens |
|---|---|---|
| `normal` | test FD001, engines 51 to 100 | nothing unusual |
| `sensor_offset` | the same engines | from halfway through the replay, `s_11` and `s_4` drift slowly, up to 1.5 standard deviations |
| `new_fault_mode` | test FD003, 100 engines | a second fault mode, never seen in training |

`--engines N` replays only the first N engines and `--step N` sends one cycle in N, to go faster. Every prediction is appended to `monitoring/predictions.jsonl` by the API, and the true remaining life of every cycle sent to `monitoring/ground_truth.jsonl`; empty `monitoring/` before switching scenarios.

`monitor` compares each of the 14 sensors with the reference period, test FD001 engines 1 to 50: the Wasserstein distance between the two distributions, in standard deviations of the reference. A sensor drifts when this distance is 0.1 or more. The reference is a period of service and not the training engines: training engines are followed until failure, engines in service are mostly seen before it, so a normal fleet would already look drifted. It writes `reports/monitoring/drift.json` and `drift.png`. Results with the full replay:

| Scenario | Predictions | Drifted sensors | Share |
|---|---|---|---|
| `normal` | 6,757 | none (largest distance 0.083) | 0.00 |
| `sensor_offset` | 6,757 | `s_4` (0.449), `s_11` (0.452) | 0.14 |
| `new_fault_mode` | 16,596 | 13 of 14, all but `s_14` | 0.93 |

The drift is measured on everything logged so far: with few engines, it raises false alarms (replaying engines 51 to 95 without any offset already flags 9 sensors). Read it on many engines.

When the truth has been written, `monitor` also measures the real error: the RMSE against the true remaining life, capped at 125, on every cycle sent, and on a rolling window of the last 2,000 predictions (about 15 engines):

| Scenario | RMSE | First half | Second half | Rolling RMSE, lowest / highest |
|---|---|---|---|---|
| `normal` | 14.94 | 14.72 | 15.14 | 13.0 / 17.2 |
| `sensor_offset` | 23.08 | 14.72 | 29.14 | 13.9 / 35.7 |
| `new_fault_mode` | 22.80 | 20.94 | 24.53 | 12.0 / 30.9 |

This RMSE covers every cycle, not only the last one of each engine, so it is not comparable with the test RMSE of the evaluation. The **Live monitoring** page of the dashboard shows the same figures while the predictions arrive.

The test is written with scipy rather than Evidently: Evidently 0.7.23 requires plotly < 6 and does not import with pandas 3.1.

## Retraining with Prefect

When the drift is high, a flow retrains the model on more data and promotes it only if a written rule says so. After a simulation:

```powershell
python -m flows.retrain_flow
```

Prefect runs six tasks in order and records each run:

| Task | What it does |
|---|---|
| `check_drift` | the drift report on `monitoring/predictions.jsonl`; below a drift share of 0.3, the flow stops here |
| `train_challenger` | the same model with the same settings as the champion, trained on FD001 + FD003 (engines of FD003 renumbered 1001 to 1100, each subset split 70/15/15), registered in MLflow with the alias `challenger` |
| `evaluate_both` | both models on the 200 test engines of FD001 and FD003, through `predict_engine`, the function behind the API |
| `decide` | promote only if the RMSE on the 200 engines drops by at least 2 % **and** the RMSE on FD001 rises by less than 10 % (`params.yaml`, section `retraining`) |
| `promote` | the challenger is saved in `models/challenger/`; if promoted, it replaces `models/champion/` and the alias `champion` moves to its version |
| `write_report` | `reports/retraining/<date>.json`: trigger, drift, scores, decision and reason |

Without a Prefect server, the flow starts a temporary one. Its database stays in `.prefect/` in the project folder, and no usage statistics are sent (`flows/__init__.py`). To browse the runs, start the server in another terminal, then open http://127.0.0.1:4200:

```powershell
$env:PREFECT_HOME = ".prefect"; $env:PREFECT_SERVER_ANALYTICS_ENABLED = "false"
prefect server start
```

Results after a full `new_fault_mode` simulation (drift share 0.93), RMSE at the last cycle of each test engine:

| Model | FD001 (100 engines) | FD003 (100 engines) | All 200 |
|---|---|---|---|
| Champion (LSTM v5, trained on FD001) | 14.85 | 46.25 | 34.34 |
| Challenger (LSTM v6, trained on FD001 + FD003) | 14.67 | 13.23 | 13.97 |

The challenger was promoted: 59.3 % better on the 200 engines, 1.2 % better on FD001. Replaying the simulations with it, the RMSE on every cycle went from 22.80 to 10.40 on `new_fault_mode` and from 14.94 to 12.19 on `normal`. A second run of the flow trained an identical challenger (same data, same seed): improvement 0.0 %, champion kept.

Here the test engines decide the promotion, which keeps the demonstration simple; a real project would keep a separate set of engines for this decision. The drift reference stays the FD001 period: after a promotion, the FD003 engines still look drifted.

The flow replaces `models/champion/` without DVC, so `dvc status` then reports it as modified, and two tests that compare the served model with the pipeline's evaluation are skipped. To go back to the pipeline's champion:

```powershell
dvc checkout models/champion --force
```

and move the alias back in MLflow (`MlflowClient().set_registered_model_alias("turbofan-rul", "champion", "5")`). The API loads the champion at start-up: restart it after a promotion.
## Reproducible pipeline

The chain `download → prepare → featurize → train_classic`, with `train_deep` after `prepare`, then `evaluate → register` and `explain`, is declared in `dvc.yaml`. With the environment activated (the stages call `python`), one command rebuilds whatever is out of date:

```powershell
dvc repro
```

DVC reruns only the stages whose code, parameters (`params.yaml`) or input data changed, and records what was used in `dvc.lock`. Useful commands:

```powershell
dvc dag             # the graph of stages
dvc status          # what is out of date, and why
dvc metrics show    # scores in reports/
dvc metrics diff    # scores compared with the last commit
dvc push            # copy data and models to the DVC remote
dvc pull            # get them back on another machine or after a clean
```

Data and models are not stored in git: DVC keeps them in `.dvc/cache` and in a local remote, `../.dvc-storage/turbofan` (a folder next to the project).

## Checks

```powershell
ruff check .
pytest
```

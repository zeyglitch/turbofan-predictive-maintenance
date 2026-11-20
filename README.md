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

## Reproducible pipeline

The chain `download → prepare → featurize → train_classic`, with `train_deep` after `prepare`, is declared in `dvc.yaml`. With the environment activated (the stages call `python`), one command rebuilds whatever is out of date:

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

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

`featurize` adds, for each sensor, its mean and standard deviation over the last 30 cycles of the same engine, and writes `data/features/*.parquet`. `train_classic` trains a constant baseline and a Ridge regression, prints their scores on the validation engines and writes `reports/metrics/val_classic.json`.

## Checks

```powershell
ruff check .
pytest
```

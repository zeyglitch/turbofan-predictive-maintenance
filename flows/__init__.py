"""Prefect flows. This file runs before any flow module, so before prefect is imported.

PREFECT_HOME: Prefect's database stays in the project folder (.prefect/, ignored by git) instead
of the user's home folder. DO_NOT_TRACK and PREFECT_SERVER_ANALYTICS_ENABLED: no usage
statistics sent to Prefect's (or MLflow's) developers. MLFLOW_DISABLE_AGENT_HINT: no MLflow
message about AI coding assistants at every run. setdefault: a value already set wins.
"""

import os

from turbofan.config import ROOT

os.environ.setdefault("PREFECT_HOME", str(ROOT / ".prefect"))
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

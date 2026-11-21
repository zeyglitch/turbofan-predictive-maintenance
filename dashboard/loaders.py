"""Everything the dashboard reads, in one place, with Streamlit's cache.

The dashboard computes no model: it reads the files written by the pipeline (data/, reports/,
dvc.yaml). DASHBOARD_ROOT changes the folder it reads from (the project folder by default);
the tests point it to a folder of small toy files.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from turbofan.config import ROOT, load_params
from turbofan.data import load_raw


def root() -> Path:
    return Path(os.environ.get("DASHBOARD_ROOT", ROOT))


def path(*parts: str) -> Path:
    """A file of the project, for example path("reports", "champion.json")."""
    return root().joinpath(*parts)


def require(files: list[Path], command: str) -> None:
    """Stop the page with a help message if a file is missing, and say how to create it."""
    missing = [file for file in files if not file.exists()]
    if missing:
        names = ", ".join(f"`{file.relative_to(root()).as_posix()}`" for file in missing)
        st.info(
            f"Missing: {names}. Run `{command}` in the project folder, then reload this page.",
            title="Nothing to show yet",
            icon=":material/info:",
        )
        st.stop()


def params() -> dict:
    return load_params(path("params.yaml"))


# Each reader takes the file's modification time as a second argument: it is part of the cache
# key, so a file rewritten by `dvc repro` is read again, and an unchanged one comes from memory.


def read_json(file: Path) -> dict:
    return _read_json(str(file), file.stat().st_mtime)


@st.cache_data
def _read_json(file: str, mtime: float) -> dict:
    return json.loads(Path(file).read_text(encoding="utf-8"))


def read_parquet(file: Path) -> pd.DataFrame:
    return _read_parquet(str(file), file.stat().st_mtime)


@st.cache_data
def _read_parquet(file: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(file)


def read_csv(file: Path) -> pd.DataFrame:
    return _read_csv(str(file), file.stat().st_mtime)


@st.cache_data
def _read_csv(file: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(file)


def read_raw(subset: str, part: str) -> pd.DataFrame:
    """A raw NASA file (train or test), with its 26 named columns."""
    file = path("data", "raw", f"{part}_{subset}.txt")
    return _read_raw(subset, part, str(file.parent), file.stat().st_mtime)


@st.cache_data
def _read_raw(subset: str, part: str, raw_dir: str, mtime: float) -> pd.DataFrame:
    return load_raw(subset, part, Path(raw_dir))


def pipeline_stages() -> dict[str, dict]:
    """For each stage of dvc.yaml: the files it reads and the files it writes."""
    stages = yaml.safe_load(path("dvc.yaml").read_text(encoding="utf-8"))["stages"]
    return {
        name: {
            "deps": stage.get("deps", []),
            # an output is either a plain path or {path: options}
            "outs": [
                next(iter(out)) if isinstance(out, dict) else out
                for out in stage.get("outs", []) + stage.get("metrics", [])
            ],
        }
        for name, stage in stages.items()
    }


def pipeline_edges(stages: dict[str, dict]) -> list[tuple[str, str]]:
    """(a, b) when stage b reads a file, or a file inside a folder, that stage a writes."""
    edges = []
    for name, stage in stages.items():
        for dep in stage["deps"]:
            for other, other_stage in stages.items():
                if any(dep == out or dep.startswith(out + "/") for out in other_stage["outs"]):
                    edges.append((other, name))
    return sorted(set(edges))


def drop_shortcuts(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove a -> c when a longer path a -> b -> ... -> c already exists, to lighten the diagram.

    The order of the stages does not change: c still runs after a.
    """

    def reachable(start: str, skip: tuple[str, str]) -> set[str]:
        seen, todo = set(), [start]
        while todo:
            node = todo.pop()
            for a, b in edges:
                if a == node and (a, b) != skip and b not in seen:
                    seen.add(b)
                    todo.append(b)
        return seen

    return [(a, c) for a, c in edges if c not in reachable(a, skip=(a, c))]


def stage_status(stages: dict[str, dict]) -> pd.DataFrame:
    """For each stage: its outputs, and when they were last written (None if one is missing)."""
    rows = []
    for name, stage in stages.items():
        outs = [path(out) for out in stage["outs"]]
        done = all(out.exists() for out in outs)
        rows.append(
            {
                "stage": name,
                "outputs": ", ".join(stage["outs"]),
                "last_written": (
                    datetime.fromtimestamp(max(map(last_modified, outs))) if done else None
                ),
            }
        )
    return pd.DataFrame(rows)


def last_modified(file: Path) -> float:
    """Modification time of a file, or of the most recent file inside a folder."""
    if file.is_dir():
        return max((f.stat().st_mtime for f in file.rglob("*") if f.is_file()), default=0.0)
    return file.stat().st_mtime

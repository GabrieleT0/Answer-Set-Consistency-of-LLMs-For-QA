from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def default_results_dir() -> Path:
    output_dir = REPO_ROOT / "output"
    if (output_dir / "summary.csv").exists() or (output_dir / "summary_idk_worst.csv").exists():
        return output_dir
    return REPO_ROOT / "data" / "evaluation_results"


def default_charts_dir() -> Path:
    return REPO_ROOT / "charts"


def load_llm_names(root_dir: str | Path | None = None) -> list[str]:
    base = Path(root_dir) if root_dir else REPO_ROOT
    with open(base / "data" / "llm_info.json", "r", encoding="utf-8") as f:
        return list(json.load(f).keys())


def load_llm_info(root_dir: str | Path | None = None) -> dict:
    base = Path(root_dir) if root_dir else REPO_ROOT
    with open(base / "data" / "llm_info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def read_summary(folder: str | Path | None = None, time: str | None = None) -> pd.DataFrame:
    folder_path = Path(folder) if folder else default_results_dir()
    candidates = []
    if time:
        candidates.append(folder_path / f"summary_{time}.csv")
    candidates.extend([
        folder_path / "summary.csv",
        folder_path / "summary_idk_worst.csv",
        folder_path / "summary_filtered_idk.csv",
    ])
    return _read_first_existing(candidates, "summary CSV")


def read_pvalue_matrices(folder: str | Path | None = None, time: str | None = None) -> pd.DataFrame:
    folder_path = Path(folder) if folder else default_results_dir()
    candidates = []
    if time:
        candidates.append(folder_path / f"p_value_matrices_{time}.csv")
    candidates.append(folder_path / "p_value_matrices.csv")
    return _read_first_existing(candidates, "p-value matrix CSV")


def _read_first_existing(candidates: list[Path], label: str) -> pd.DataFrame:
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    formatted = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n  - {formatted}")

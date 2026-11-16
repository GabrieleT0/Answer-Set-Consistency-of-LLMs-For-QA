"""Write one overall-summary CSV for each evaluation action."""

import json
from pathlib import Path

import pandas as pd

from src.evaluation.paths import REPO_ROOT


def split(df, prefix, actions=None, llm_info=None, folder=None):
    if actions is None:
        actions = ["zero-shot", "classification", "fixing"]
    output_dir = Path(folder) if folder is not None else REPO_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if llm_info is None:
        llm_path = REPO_ROOT / "data" / "llm_info.json"
        with llm_path.open("r", encoding="utf-8") as f:
            llm_info = json.load(f)
    name_id_map = {key: value["ID"] for key, value in llm_info.items()}

    df = df[df["dataset"] == "overall"]

    for act in actions:
        df_act = df[df["action"] == act].copy()
        df_act["ID"] = df_act["llm"].map(name_id_map)
        if df_act["ID"].isna().any():
            missing = sorted(df_act.loc[df_act["ID"].isna(), "llm"].unique())
            raise KeyError(f"Models missing from llm_info: {missing}")
        df_act = df_act.sort_values(by="ID", ascending=True)

        cols = ["ID"] + [col for col in df_act.columns if col != "ID"]
        df_act = df_act[cols]
        output = output_dir / f"{prefix}_{act}.csv"
        df_act.to_csv(output, index=False)
        print(f"Saved: {output}")


def main():
    output_dir = REPO_ROOT / "output"
    df = pd.read_csv(output_dir / "p_value_matrices.csv")
    split(df, "p_value", folder=output_dir)


if __name__ == "__main__":
    main()

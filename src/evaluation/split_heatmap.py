import pandas as pd
import json 
import os

import pandas as pd, re
from pathlib import Path

def _sanitize_predicate(pred: str) -> str:
    s = pred
    s = s.replace("?", "")
    s = s.replace("=", "eq").replace(">", "gt").replace("<", "lt").replace("+", "plus")
    s = s.replace("∅", "disj").replace("|", "or")
    return re.sub(r"[^\w]+", "_", s).strip("_")

def dataframe_to_heatmap_csvs(df: pd.DataFrame, outdir: Path,
                              action: str | None = None,
                              dataset: str | None = None,
                              index_order: list[str] | None = None) -> pd.DataFrame:
    """
    Convert a wide p-value dataframe with columns:
        ['ID','action','dataset','predicate','llm', <model1>, <model2>, ...]
    into one heatmap CSV per predicate with columns: i,j,val
    Also writes model_index_map.csv (idx, letter, llm).
    Returns the index map DataFrame.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    known = ["ID","action","dataset","predicate","llm"]
    model_cols = [c for c in df.columns if c not in known]

    if action is not None:
        df = df[df["action"] == action]
    if dataset is not None:
        df = df[df["dataset"] == dataset]

    if index_order is None:
        seen = []
        for name in df["llm"].tolist():
            if name not in seen:
                seen.append(name)
        index_order = seen

    if set(index_order) != set(model_cols):
        raise ValueError("Row/column model sets differ; supply a full index_order.")

    idx_map = {name: i+1 for i, name in enumerate(index_order)}
    letters = [chr(ord('A') + i) for i in range(len(index_order))]
    index_map_df = pd.DataFrame({
        "idx": [i+1 for i in range(len(index_order))],
        "letter": letters,
        "llm": index_order
    })
    index_map_df.to_csv(outdir / "model_index_map.csv", index=False)

    for pred in df["predicate"].unique():
        sub = df[df["predicate"] == pred]
        long = sub.melt(id_vars=["llm"], value_vars=model_cols,
                        var_name="col_model", value_name="val")
        long["i"] = long["llm"].map(idx_map)
        long["j"] = long["col_model"].map(idx_map)
        long["val"] = pd.to_numeric(long["val"], errors="coerce")
        out = long.dropna(subset=["i","j","val"])[["i","j","val"]].sort_values(["i","j"])
        out.to_csv(outdir / f"heatmap_{_sanitize_predicate(pred)}.csv", index=False)

    return index_map_df


def main():
    root_dir = os.path.dirname(os.path.abspath(__name__))
    folder = root_dir + "/output/"
    # file = "summary_2025-09-23_16-04.csv"
    file = "p_value_matrices_2025-09-24_08-48.csv"
    prefix = "p_value"
    df = pd.read_csv(folder + file)
    # split(df, prefix)
    dataframe_to_heatmap_csvs(df, folder + "/heatmaps", action="zero-shot", dataset="overall")


if __name__=="__main__":
    main()

# Now split_dfs['action_value'] gives you the DataFrame for that action

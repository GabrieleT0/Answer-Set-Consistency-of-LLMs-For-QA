"""
Summary with idk-as-worst-case imputation.

For every test instance where any relevant prompt received "idk" (or no
answer), the metric is set to the worst possible value before averaging:

  Consistency checks  (?A1=A2, ?A1>A3, …)      → 0   (not consistent)
  Contradiction-free (?SC(A1=A2), …)            → 0   (not contradiction-free)
  Jaccard equivalence (J(A1-A2), J(A1-A34), …)  → 0.0 (no overlap)
  Jaccard disjointness J(A3-A4)                  → 1.0 (fully overlapping)

Output: output/summary_idk_worst.csv
"""

from src.evaluation.paths import REPO_ROOT

import ast
import os

import numpy as np
import pandas as pd

from src.evaluation.statistics.eval_pvalue import compute_pvals
from src.evaluation.tables.split import split


# ── Worst-case values per metric ─────────────────────────────────────────────

WORST_CASE: dict[str, float] = {
    # Consistency (1 = correct, 0 = wrong)
    "?A1=A2":     0,
    "?A1=A3+A4":  0,
    "?A1>A3":     0,
    "?A1>A4":     0,
    "?A3∅A4":     0,
    "?A4=A1|3":   0,  # Joint partition consistency (legacy column name).
    "?A4=A1-A3":  0,  # Pure difference, diagnostic only.
    "?A3<=A1(same-context)": 0,
    "?A1=A1*":    0,
    "?A1=A1**":   0,
    "?A1*=A1**":  0,
    # Contradiction-free outcomes (legacy ?SC names; 1 = holds)
    "?SC(A1=A2)":   0,
    "?SC(A1>A3)":   0,
    "?SC(A1>A4)":   0,
    "?SC(A3∅A4)":   0,
    "?SC(A4=A1|3)": 0,
    # Jaccard — equivalence (higher is better → worst = 0)
    "J(A1-A2)":    0.0,
    "J(A1-A34)":   0.0,
    "J(A4-A1|3)":  0.0,
    "J(A1-A1*)":   0.0,
    "J(A1-A1**)":  0.0,
    "J(A1*-A1**)": 0.0,
    # Jaccard — disjointness (lower is better → worst = 1)
    "J(A3-A4)": 1.0,
}

# Which idk flags affect each metric
METRIC_IDK_MAP: dict[str, list[str]] = {
    "?A1=A2":       ["idk_A1", "idk_A2"],
    "J(A1-A2)":     ["idk_A1", "idk_A2"],
    "?SC(A1=A2)":   ["idk_A1", "idk_A2"],
    "?A1=A3+A4":    ["idk_A1", "idk_A3", "idk_A4"],
    "J(A1-A34)":    ["idk_A1", "idk_A3", "idk_A4"],
    "?A1>A3":       ["idk_A1", "idk_A3"],
    "?SC(A1>A3)":   ["idk_A1", "idk_A3"],
    "?A1>A4":       ["idk_A1", "idk_A4"],
    "?SC(A1>A4)":   ["idk_A1", "idk_A4"],
    "?A3∅A4":       ["idk_A3", "idk_A4"],
    "J(A3-A4)":     ["idk_A3", "idk_A4"],
    "?SC(A3∅A4)":   ["idk_A3", "idk_A4"],
    "J(A4-A1|3)":   ["idk_A4", "idk_A1", "idk_A3"],
    "?A4=A1|3":     ["idk_A4", "idk_A1", "idk_A3"],
    "?A4=A1-A3":    ["idk_A4", "idk_A1", "idk_A3"],
    "?A3<=A1(same-context)": ["idk_A1", "idk_A3"],
    "?SC(A4=A1|3)": ["idk_A4", "idk_A1", "idk_A3"],
    "?A1=A1*":      ["idk_A1", "idk_A1*"],
    "J(A1-A1*)":    ["idk_A1", "idk_A1*"],
    "?A1=A1**":     ["idk_A1", "idk_A1**"],
    "J(A1-A1**)":   ["idk_A1", "idk_A1**"],
    "?A1*=A1**":    ["idk_A1*", "idk_A1**"],
    "J(A1*-A1**)":  ["idk_A1*", "idk_A1**"],
}



def is_null_answer(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return False
    if isinstance(cell, str):
        text = cell.strip()
        if text.lower() in {"idk", "no answer"}:
            return True
        try:
            cell = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return False
    if isinstance(cell, (list, tuple, set)):
        return not cell or any(str(x).strip().lower() in {"idk", "no answer"} for x in cell)
    return False


def add_idk_flags(df):
    out = df.copy()
    for answer in ["A1", "A2", "A3", "A4", "A1*", "A1**", "A3_minus"]:
        flag = f"idk_{answer}"
        if answer in out:
            out[flag] = out[answer].map(is_null_answer).astype(int)
        elif flag not in out:
            out[flag] = 0
    return out


def metric_idk_mask(df, metric):
    columns = METRIC_IDK_MAP.get(metric, [])
    present = [c for c in columns if c in df]
    mask = df[present].fillna(0).astype(bool).any(axis=1) if present else pd.Series(False, index=df.index)
    if "action" not in df:
        return mask
    contextual = df["action"].isin(["classification", "fixing", "classification-no-idk", "fixing-no-idk"])
    contextual_columns = {
        "?A1>A3": ["idk_A1*", "idk_A3"],
        "?SC(A1>A3)": ["idk_A1*", "idk_A3"],
        "?A1>A4": ["idk_A1**", "idk_A4"],
        "?SC(A1>A4)": ["idk_A1**", "idk_A4"],
        "?A1=A3+A4": ["idk_A1**", "idk_A3_minus", "idk_A4"],
        "J(A1-A34)": ["idk_A1**", "idk_A3_minus", "idk_A4"],
        "?A3∅A4": ["idk_A3_minus", "idk_A4"],
        "?SC(A3∅A4)": ["idk_A3_minus", "idk_A4"],
        "J(A3-A4)": ["idk_A3_minus", "idk_A4"],
        "?A4=A1|3": ["idk_A1**", "idk_A3_minus", "idk_A4"],
        "?A4=A1-A3": ["idk_A1**", "idk_A3_minus", "idk_A4"],
        "?A3<=A1(same-context)": ["idk_A1**", "idk_A3_minus"],
        "?SC(A4=A1|3)": ["idk_A1**", "idk_A3_minus", "idk_A4"],
        "J(A4-A1|3)": ["idk_A1**", "idk_A3_minus", "idk_A4"],
    }.get(metric)
    # Legacy CSVs did not retain A3 from the minus task. Do not invent it.
    if contextual_columns and all(c in df for c in contextual_columns) and "A3_minus" in df:
        mask.loc[contextual] = df.loc[contextual, contextual_columns].fillna(0).astype(bool).any(axis=1)
    return mask


def impute_worst_case(df):
    out = add_idk_flags(df)
    for metric, worst in WORST_CASE.items():
        if metric in out:
            out.loc[out[metric].notna() & metric_idk_mask(out, metric), metric] = worst
    return out

# Backward-compatible helper names.
_is_idk = is_null_answer
_add_idk_flags = add_idk_flags

# ── Summary ───────────────────────────────────────────────────────────────────

def _group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [c for c in WORST_CASE if c in df.columns]
    idk_cols    = [c for c in df.columns if c.startswith("idk_")]
    agg_cols    = metric_cols + idk_cols

    out = (
        df.groupby(group_cols)[agg_cols]
        .mean()
        .reset_index()
        .round(4)
    )
    counts = df.groupby(group_cols).size().rename("n_instances")
    repeats = (df.assign(_repeat_available=df["action"].eq("zero-shot") & df["?A1=A1*"].notna()).groupby(group_cols)["_repeat_available"].sum().rename("n_repeat")) if "?A1=A1*" in df else None
    out = out.merge(counts, on=group_cols)
    if repeats is not None:
        out = out.merge(repeats, on=group_cols)
    return out


def summary_idk_worst(df_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(dataset, action, llm) and overall-(action, llm) averages
    after imputing worst-case values for idk answers.
    """
    df = impute_worst_case(df_analysis[df_analysis["action"] != "wikidata"].copy())

    group_cols = ["dataset", "action", "llm"]
    df_per = _group_summary(df, group_cols)

    # Overall (across all datasets)
    df_all = _group_summary(df, ["action", "llm"])
    df_all["dataset"] = "overall"

    df_summary = pd.concat([df_per, df_all], ignore_index=True)

    # Aggregate "cross-classification" averages (same as existing summaries)
    for col_group, new_col in [
        (["?A1=A1*", "?A1=A1**", "?A1*=A1**"],          "?A1=A1(ave)"),
        (["J(A1-A1*)", "J(A1-A1**)", "J(A1*-A1**)"],    "J_A1_ave"),
    ]:
        present = [c for c in col_group if c in df_summary.columns]
        if present:
            df_summary[new_col] = (
                df_summary[present]
                .apply(pd.to_numeric, errors="coerce")
                .mean(axis=1)
                .round(4)
            )

    # Average idk rate across all four prompts
    idk_base = [c for c in ["idk_A1", "idk_A2", "idk_A3", "idk_A4"]
                if c in df_summary.columns]
    if idk_base:
        df_summary["idk"] = df_summary[idk_base].mean(axis=1).round(4)

    return df_summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    root_dir      = str(REPO_ROOT)
    analysis_path = os.path.join(root_dir, "output", "analysis.csv")
    output_dir    = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {analysis_path} …")
    df_analysis = pd.read_csv(analysis_path)
    print(f"  {len(df_analysis):,} rows, "
          f"{df_analysis['llm'].nunique()} models, "
          f"{df_analysis['dataset'].nunique()} datasets")

    print("Imputing worst-case values for idk answers …")
    df_summary = summary_idk_worst(df_analysis)

    # Compute and merge p-values
    print("Computing p-values …")
    df_pval = compute_pvals(impute_worst_case(df_analysis))
    df_summary = df_summary.merge(df_pval, on=["dataset", "llm", "action"], how="left")

    out_path = os.path.join(output_dir, "summary_idk_worst.csv")
    df_summary.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

    # Split by action (generates summary_idk_worst_zero-shot.csv, etc.)
    split(df_summary, "summary_idk_worst", folder=output_dir)

    # Quick console preview: overall rows only
    print("\n=== Overall averages per model (dataset=overall) ===")
    cols_show = ["action", "llm",
                 "?A1=A2", "J(A1-A2)", "?A3∅A4", "J(A3-A4)",
                 "?A1>A3", "?A1>A4", "idk"]
    cols_show = [c for c in cols_show if c in df_summary.columns]
    overall = df_summary[df_summary["dataset"] == "overall"].sort_values(["action", "llm"])
    print(overall[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()

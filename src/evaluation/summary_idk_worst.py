"""
Summary with idk-as-worst-case imputation.

For every test instance where any relevant prompt received "idk" (or no
answer), the metric is set to the worst possible value before averaging:

  Consistency checks  (?A1=A2, ?A1>A3, …)      → 0   (not consistent)
  Self-contradiction  (?SC(A1=A2), …)            → 1   (contradiction exists)
  Jaccard equivalence (J(A1-A2), J(A1-A34), …)  → 0.0 (no overlap)
  Jaccard disjointness J(A3-A4)                  → 1.0 (fully overlapping)

Output: output/summary_idk_worst.csv
"""

import ast
import os

import numpy as np
import pandas as pd

from eval_pvalue import compute_pvals
from split import split



# ── Worst-case values per metric ─────────────────────────────────────────────

WORST_CASE: dict[str, float] = {
    # Consistency (1 = correct, 0 = wrong)
    "?A1=A2":     0,
    "?A1=A3+A4":  0,
    "?A1>A3":     0,
    "?A1>A4":     0,
    "?A3∅A4":     0,
    "?A4=A1|3":   0,
    "?A1=A1*":    0,
    "?A1=A1**":   0,
    "?A1*=A1**":  0,
    # Self-contradiction (1 = contradiction, 0 = no contradiction)
    "?SC(A1=A2)":   1,
    "?SC(A1>A3)":   1,
    "?SC(A1>A4)":   1,
    "?SC(A3∅A4)":   1,
    "?SC(A4=A1|3)": 1,
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
    "?SC(A4=A1|3)": ["idk_A4", "idk_A1", "idk_A3"],
    "?A1=A1*":      ["idk_A1", "idk_A1*"],
    "J(A1-A1*)":    ["idk_A1", "idk_A1*"],
    "?A1=A1**":     ["idk_A1", "idk_A1**"],
    "J(A1-A1**)":   ["idk_A1", "idk_A1**"],
    "?A1*=A1**":    ["idk_A1*", "idk_A1**"],
    "J(A1*-A1**)":  ["idk_A1*", "idk_A1**"],
}


# ── idk flag helpers ──────────────────────────────────────────────────────────

def _is_idk(cell) -> bool:
    """
    True if cell represents an answered-but-empty / idk answer set.
    NaN means the answer column was not populated for this row (e.g. A1* is
    never stored for zero-shot rows) — that is NOT the same as answering idk.
    """
    if cell is None:
        return False
    if isinstance(cell, float) and np.isnan(cell):
        return False   # NaN = not applicable / not stored, not an idk answer
    if isinstance(cell, str):
        if cell.strip() == "" or cell.strip().lower() == "idk":
            return True
        # try to parse stringified list
        try:
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, list):
                return len(parsed) == 0 or parsed == ["idk"]
        except Exception:
            pass
    if isinstance(cell, list):
        return len(cell) == 0 or cell == ["idk"]
    return False


def _add_idk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure idk_A1, idk_A2, idk_A3, idk_A4, idk_A1*, idk_A1** are present.
    Re-computes from the answer columns so values are trustworthy even if
    the CSV was produced by an older pipeline version.
    """
    df = df.copy()
    for col, flag in [("A1", "idk_A1"), ("A2", "idk_A2"),
                      ("A3", "idk_A3"), ("A4", "idk_A4"),
                      ("A1*", "idk_A1*"), ("A1**", "idk_A1**")]:
        if col in df.columns:
            df[flag] = df[col].apply(_is_idk).astype(int)
        else:
            df[flag] = 0   # column absent → treat as non-idk
    return df


# ── Imputation ────────────────────────────────────────────────────────────────

def impute_worst_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df where, for every metric column in WORST_CASE, rows
    with ANY relevant idk answer are replaced by the worst-case value.
    """
    df = _add_idk_flags(df)
    df = df.copy()

    for metric, worst in WORST_CASE.items():
        if metric not in df.columns:
            continue
        idk_cols = [c for c in METRIC_IDK_MAP.get(metric, []) if c in df.columns]
        if not idk_cols:
            continue
        applicable = df[metric].notna()          # metric was actually computed for this row
        idk_mask   = df[idk_cols].any(axis=1)   # ANY relevant answer is idk
        df.loc[applicable & idk_mask, metric] = worst

    return df


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

    # Blend ?A1=A1* and J(A1-A1*) from classification into zero-shot overall row
    # (mirrors the same logic in summary_xidk)
    blend_cols = ["?A1=A1*", "J(A1-A1*)"]
    mask_zs  = (df_summary["dataset"] == "overall") & (df_summary["action"] == "zero-shot")
    mask_clf = (df_summary["dataset"] == "overall") & (df_summary["action"] == "classification")
    a = df_summary.loc[mask_zs,  blend_cols].copy()
    b = df_summary.loc[mask_clf, blend_cols]
    if not b.empty:
        for col in blend_cols:
            a[col] = np.where(
                a[col].isna(),
                b[col].values,
                (a[col] + b[col].values) / 2,
            )
        df_summary.loc[mask_zs, blend_cols] = a

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
    root_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    analysis_path = os.path.join(root_dir, "output", "analysis_old.csv")
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
    df_pval = compute_pvals(df_analysis)
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

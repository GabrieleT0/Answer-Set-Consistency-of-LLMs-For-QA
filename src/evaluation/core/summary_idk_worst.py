"""
Summary with idk-as-worst-case imputation.

For every test instance where any relevant prompt received an unavailable
answer under the selected recognition rule, the metric is set to the worst
possible value before averaging. The paper-default rule is sole standardized
"idk" or an empty answer set; broader recognition is opt-in:

  Consistency checks  (?A1=A2, ?A1>A3, …)      → 0   (not consistent)
  Contradiction-free (?SC(A1=A2), …)            → 0   (not contradiction-free)
  Jaccard equivalence (J(A1-A2), J(A1-A34), …)  → 0.0 (no overlap)
  Jaccard disjointness J(A3-A4)                  → 1.0 (fully overlapping)

Output: output/summary_idk_worst.csv
"""

from src.evaluation.paths import REPO_ROOT

import ast
import os
import unicodedata
from functools import lru_cache

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



ABSTENTION_RECOGNITIONS = ("standard", "robust", "extended")
_NULL_PHRASES = {"", "idk", "no answer", "i don't know", "i do not know"}
_NULL_TRANSLATION = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "`": "'"})


def _null_phrase(cell):
    if not isinstance(cell, str):
        return None
    return _normalize_null_text(cell)


@lru_cache(maxsize=8192)
def _normalize_null_text(cell):
    text = unicodedata.normalize("NFKC", cell).translate(_NULL_TRANSLATION)
    text = " ".join(text.casefold().split())
    while True:
        previous = text
        text = text.rstrip(".!?").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1].strip()
        if text == previous:
            return text


def is_null_answer(cell, recognition="standard"):
    """Recognize observed abstentions without inventing missing responses.

    Robust requires the whole answer to be empty or contain only null phrases.
    Standard preserves the previous empty-set/sole-idk rule. Extended also
    flags explicit null phrases mixed with entities, for sensitivity analysis.
    None/NaN remain missing observations in every mode.
    """
    if recognition not in ABSTENTION_RECOGNITIONS:
        raise ValueError(f"Unknown abstention recognition: {recognition}")
    if cell is None or cell is pd.NA or (isinstance(cell, (float, np.floating)) and np.isnan(cell)):
        return False
    if isinstance(cell, str):
        text = cell.strip()
        if (text.lower() == "idk" if recognition == "standard" else _null_phrase(text) in _NULL_PHRASES):
            return True
        try:
            cell = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return False
    if isinstance(cell, (list, tuple, set)):
        if not cell:
            return True
        if recognition == "standard":
            return len(cell) == 1 and str(next(iter(cell))).strip().lower() == "idk"
        if recognition == "robust":
            return all(_null_phrase(value) in _NULL_PHRASES for value in cell)
        phrases = [_null_phrase(value) for value in cell]
        return all(phrase == "" for phrase in phrases) or any(
            phrase in _NULL_PHRASES - {""} for phrase in phrases)
    return False


def add_idk_flags(df, recognition="standard"):
    out = df.copy()
    for answer in ["A1", "A2", "A3", "A4", "A1*", "A1**", "A3_minus"]:
        flag = f"idk_{answer}"
        if answer in out:
            out[flag] = out[answer].map(lambda cell: is_null_answer(cell, recognition)).astype(int)
        elif flag not in out:
            out[flag] = 0
    return out


def metric_idk_mask(df, metric, recognition=None):
    if recognition is not None:
        df = add_idk_flags(df, recognition)
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


def impute_worst_case(df, recognition="standard"):
    out = add_idk_flags(df, recognition)
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


def apply_repeat_aggregation(summary, frame, summarize, aggregation="condition_mean",
                             recognition="standard", exclude_nulls=False):
    """Retain the paper's overall equal-condition mean, or paired diagnostics."""
    if aggregation not in {"condition_mean", "per_question"}:
        raise ValueError(f"Unknown repeat aggregation: {aggregation}")
    out = summary.copy()
    out["repeat_aggregation"] = "per_question"
    metrics = [c for c in ["?A1=A1*", "J(A1-A1*)"] if c in out]
    if aggregation == "per_question" or not metrics:
        return out
    isolated = frame.copy()
    missing = isolated.action.eq("zero-shot") & (
        isolated.repeat_source.ne("Base repeat") if "repeat_source" in isolated
        else isolated[metrics[0]].isna())
    isolated.loc[missing, metrics] = float("nan")
    if "A1*" in isolated:
        isolated.loc[missing, "A1*"] = None
    controls = summarize(isolated)
    controls = controls[controls.dataset.eq("overall")]
    flagged = add_idk_flags(isolated, recognition)
    for index, row in out[out.dataset.eq("overall") & out.action.eq("zero-shot")].iterrows():
        model = row.llm
        for metric in metrics:
            values = controls.loc[controls.llm.eq(model) & controls.action.isin(
                ["zero-shot", "classification"]), metric]
            out.at[index, metric] = values.mean()
        for action, column in [("zero-shot", "n_repeat"), ("classification", "n_repeat_cte")]:
            group = flagged[flagged.llm.eq(model) & flagged.action.eq(action)]
            available = group[metrics[0]].notna()
            if exclude_nulls:
                available &= ~metric_idk_mask(group, metrics[0])
            out.at[index, column] = int(available.sum())
        out.at[index, "repeat_aggregation"] = "condition_mean"
        for metric, aggregate in [("?A1=A1*", "?A1=A1(ave)"), ("J(A1-A1*)", "J_A1_ave")]:
            if aggregate in out and metric in out:
                out.at[index, aggregate] = out.at[index, metric]
        for metric in metrics:
            count = "n_scored_" + metric
            if count in out:
                out.at[index, count] = out.at[index, "n_repeat"]
    return out


def summary_idk_worst(df_analysis: pd.DataFrame, recognition="standard",
                      repeat_aggregation="condition_mean") -> pd.DataFrame:
    """
    Compute per-(dataset, action, llm) and overall-(action, llm) averages
    after imputing worst-case values for idk answers.
    """
    df = impute_worst_case(df_analysis[df_analysis["action"] != "wikidata"].copy(), recognition)

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

    return apply_repeat_aggregation(df_summary, df_analysis,
        lambda frame: summary_idk_worst(frame, recognition, "per_question"),
        repeat_aggregation, recognition)


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

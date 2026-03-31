"""
Computes how often the 'fixing' action corrects an answer that was inconsistent
under 'zero-shot', for each consistency metric.

For each (Q_ID, dataset, llm) pair that has both a zero-shot and a fixing row:
  - "wrong in zero-shot"  : consistency metric == 0 for zero-shot
  - "corrected by fixing" : same metric == 1 for fixing AND == 0 for zero-shot
"""

import os
import pandas as pd

CONSISTENCY_COLS = [
    "?A1=A2", "?A1=A3+A4",
    "?A1>A3", "?A1>A4",
    "?A3∅A4", "?A4=A1|3",
    "?A1=A1*", "?A1=A1**", "?A1*=A1**",
]

NON_STAR_COLS = [
    "?A1=A2",
    "?A1>A3", "?A1>A4",
    "?A3∅A4", "?A4=A1|3",
]

JOIN_KEYS = ["Q_ID", "dataset", "llm"]


def load_and_prepare(analysis_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(analysis_path)
    df = df[df["action"] != "wikidata"]

    zs = df[df["action"] == "zero-shot"][JOIN_KEYS + CONSISTENCY_COLS].copy()
    fx = df[df["action"] == "fixing"][JOIN_KEYS + CONSISTENCY_COLS].copy()

    # keep only metrics that have non-null values in both frames
    zs = zs.rename(columns={c: f"zs_{c}" for c in CONSISTENCY_COLS})
    fx = fx.rename(columns={c: f"fx_{c}" for c in CONSISTENCY_COLS})

    paired = zs.merge(fx, on=JOIN_KEYS, how="inner")
    return paired


def compute_correction_rates(paired: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    records = []
    for group_vals, grp in paired.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        row = dict(zip(group_cols, group_vals))
        row["n_pairs"] = len(grp)
        for metric in CONSISTENCY_COLS:
            zs_col = f"zs_{metric}"
            fx_col = f"fx_{metric}"
            # drop rows where either side is NaN for this metric
            valid = grp[[zs_col, fx_col]].dropna()
            wrong_zs = valid[valid[zs_col] == 0]
            corrected = wrong_zs[wrong_zs[fx_col] == 1]
            n_wrong = len(wrong_zs)
            n_corrected = len(corrected)
            n_total = len(valid)
            pct = round(n_corrected / n_total * 100, 2) if n_total > 0 else None
            row[f"wrong_zs_{metric}"] = n_wrong
            row[f"corrected_{metric}"] = n_corrected
            row[f"pct_corrected_{metric}"] = pct
        records.append(row)
    return pd.DataFrame(records)


def print_summary(df_detail: pd.DataFrame, df_overall: pd.DataFrame):
    print("\n=== Fixing correction rate (per dataset & LLM) ===")
    cols_to_show = ["dataset", "llm", "n_pairs"] + [f"pct_corrected_{m}" for m in CONSISTENCY_COLS]
    print(df_detail[cols_to_show].to_string(index=False))

    print("\n=== Overall (across all datasets) ===")
    cols_to_show_overall = ["llm", "n_pairs"] + [f"pct_corrected_{m}" for m in CONSISTENCY_COLS] + ["avg_pct_corrected"]
    print(df_overall[cols_to_show_overall].to_string(index=False))


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    analysis_path = os.path.join(root_dir, "output", "analysis_old.csv")
    output_folder = os.path.join(root_dir, "output")
    os.makedirs(output_folder, exist_ok=True)

    paired = load_and_prepare(analysis_path)
    print(f"Paired zero-shot / fixing rows: {len(paired)}")

    # Per dataset and LLM
    df_detail = compute_correction_rates(paired, group_cols=["dataset", "llm"])
    df_detail.to_csv(os.path.join(output_folder, "fixing_correction_rate.csv"), index=False)

    # Overall (collapse datasets, keep per LLM)
    df_overall = compute_correction_rates(paired, group_cols=["llm"])
    non_star_pct_cols = [f"pct_corrected_{m}" for m in NON_STAR_COLS]
    df_overall["avg_pct_corrected"] = df_overall[non_star_pct_cols].mean(axis=1).round(2)
    df_overall.to_csv(os.path.join(output_folder, "fixing_correction_rate_overall.csv"), index=False)

    # Grand overall (single row per metric)
    records = []
    row = {"n_pairs": len(paired)}
    for metric in CONSISTENCY_COLS:
        zs_col = f"zs_{metric}"
        fx_col = f"fx_{metric}"
        valid = paired[[zs_col, fx_col]].dropna()
        wrong_zs = valid[valid[zs_col] == 0]
        corrected = wrong_zs[wrong_zs[fx_col] == 1]
        n_wrong = len(wrong_zs)
        n_corrected = len(corrected)
        n_total = len(valid)
        pct = round(n_corrected / n_total * 100, 2) if n_total > 0 else None
        row[f"wrong_zs_{metric}"] = n_wrong
        row[f"corrected_{metric}"] = n_corrected
        row[f"pct_corrected_{metric}"] = pct
    records.append(row)
    df_grand = pd.DataFrame(records)
    df_grand.to_csv(os.path.join(output_folder, "fixing_correction_rate_grand.csv"), index=False)

    print_summary(df_detail, df_overall)


    print("\n=== Grand overall ===")
    for metric in CONSISTENCY_COLS:
        wrong = df_grand[f"wrong_zs_{metric}"].values[0]
        corrected = df_grand[f"corrected_{metric}"].values[0]
        pct = df_grand[f"pct_corrected_{metric}"].values[0]
        print(f"  {metric:20s}  wrong_zs={wrong:5d}  corrected={corrected:5d}  pct={pct}%")

    print(f"\nCSVs saved to {output_folder}/")


if __name__ == "__main__":
    main()

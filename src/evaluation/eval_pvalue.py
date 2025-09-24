import os
import pandas as pd
from scipy.stats import binomtest
import json

# Extended predicates and column names
PREDICATES = ['?A1=A2', '?A1=A3+A4', '?A1>A3', '?A1>A4', '?A3∅A4', '?A4=A1|3']
P_COLS     = ['p(A1=A2)', 'p(A1=A3+A4)', 'p(A1>A3)', 'p(A1>A4)', 'p(A3∅A4)', 'p(A4=A1|3)']

def one_sided_mcnemar(n10, n01):
    """
    One-sided McNemar exact test.
    H0: LLM_i is not better than LLM_j (p <= 0.5)
    H1: LLM_i is better than LLM_j (p > 0.5)
    
    Returns one-sided p-value.
    """
    discordant = n10 + n01
    if discordant == 0:
        return 1.0
    # Binomial test: probability that i wins at least n10 if p=0.5
    res = binomtest(n10, discordant, 0.5, alternative="greater")
    return res.pvalue

def _compare_group(g, dataset_label):
    """Run action-vs-zero-shot one-sided McNemar for all predicates within group g."""
    out_rows = []
    if 'zero-shot' not in set(g['action']):
        return pd.DataFrame(out_rows)

    base = g[g['action'] == 'zero-shot'][['Q_ID',"dataset"] + PREDICATES].copy()

    for action, g_act in g.groupby('action'):
        merged = base.merge(
            g_act[['Q_ID',"dataset"] + PREDICATES], 
            on=['Q_ID',"dataset"], 
            suffixes=('_zero', '_act')
        )
        if merged.empty:
            continue

        row = {
            'dataset': dataset_label,
            'llm': g['llm'].iloc[0],
            'action': action
        }

        for pred, pcol in zip(PREDICATES, P_COLS):
            a = merged[f'{pred}_zero']
            z = merged[f'{pred}_act']
            n10 = ((z == 1) & (a == 0)).sum()  # i correct, j wrong
            n01 = ((z == 0) & (a == 1)).sum()  # j correct, i wrong
            pval= one_sided_mcnemar(n10, n01)
            row[pcol] = round(pval,4)  # format to 4 decimals
          
        out_rows.append(row)
    return pd.DataFrame(out_rows)

def compute_pvals(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (dataset, llm, action) compute one-sided McNemar p(action > zero-shot).
    Includes zero-shot itself (p=1.0000).
    Also adds an 'overall' dataset (pooled across datasets per llm).
    Returns a wide DataFrame with formatted p-values.
    """
    frames = []

    # per-dataset
    for (dataset, llm), g in df.groupby(['dataset', 'llm']):
        frames.append(_compare_group(g, dataset))

    # overall (pool datasets) per llm
    for llm, g in df.groupby('llm'):
        frames.append(_compare_group(g, 'overall'))

    res = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['dataset','llm','action'] + P_COLS
    )

    return res.sort_values(['dataset', 'llm', 'action'], ignore_index=True)


def compute_pval_matrix(df: pd.DataFrame, llms, action_filter="zero-shot"):
    """
    For each dataset (and overall), compute one-sided p-value matrices between LLMs
    for the given action (default: zero-shot).
    
    Returns a dict of {(dataset, predicate): DataFrame(matrix)}.
    """
    results = {}
    datasets = list(df['dataset'].unique()) + ["overall"]

    for dataset in datasets:
        if dataset == "overall":
            g = df[df['action'] == action_filter]
        else:
            g = df[(df['dataset'] == dataset) & (df['action'] == action_filter)]
        if g.empty:
            continue

        # llms = sorted(g['llm'].unique())
        for pred in PREDICATES:
            mat = pd.DataFrame(1.0, index=llms, columns=llms, dtype=float)

            for i, llm_i in enumerate(llms):
                gi = g[g['llm'] == llm_i][['Q_ID', pred]]
                for j, llm_j in enumerate(llms):
                    if i == j:
                        continue
                    gj = g[g['llm'] == llm_j][['Q_ID', pred]]

                    merged = gi.merge(gj, on='Q_ID', suffixes=('_i', '_j'))
                    if merged.empty:
                        pval = 1.0
                    else:
                        z, a = merged[f"{pred}_i"], merged[f"{pred}_j"]
                        n10 = ((z == 1) & (a == 0)).sum()  # i correct, j wrong
                        n01 = ((z == 0) & (a == 1)).sum()  # j correct, i wrong
                        pval = one_sided_mcnemar(n10, n01)
                    mat.loc[llm_i, llm_j] = round(pval, 4)

            results[(dataset, pred)] = mat

    return results


def p_value_matrixs(df_analysis: pd.DataFrame, actions):
    """
    Wrapper to compute p-value matrices for all datasets and predicates.
    Saves results in wide format CSV.
    """
    rows_wide = []
    root_dir = os.path.dirname(os.path.abspath(__name__))
    llm_path = f"{root_dir}/data/llm_info.json"
    with open(llm_path, "r", encoding="utf-8") as f:
        llm_info = json.load(f)
    llms = list(llm_info.keys())
    # df = df.sort_values(by="llm_ID", ascending=True)
    # name_id_map = {key: llm_info[key]["ID"] for key in llm_info}
    for action in actions:
        pval_matrices = compute_pval_matrix(df_analysis, llms, action_filter=action)

        for (dataset, pred), mat in pval_matrices.items():
            mat  = mat.loc[llms, llms].astype(float).round(4)
        
            for row_llm in llms:
                vals = mat.loc[row_llm, llms].tolist()
                row = {"action":action,"dataset": dataset, "predicate": pred, "llm": row_llm}
                row.update({f"{llms[i]}": vals[i] for i in range(len(llms))})
                rows_wide.append(row)

    df_pvalue = pd.DataFrame(rows_wide).sort_values(["action","dataset", "predicate"], ignore_index=True)
    return df_pvalue

if __name__ == "__main__":
    # Example usage
    root_dir = os.path.dirname(os.path.abspath(__name__))
    folder = root_dir + "/output/"
    df_analysis = pd.read_csv(folder + "analysis.csv")
    actions = ["zero-shot", "fixing", "classification"]
    # df_pvalue = p_value_matrixs(df_analysis, actions)
    # df_pvalue.to_csv(os.path.join(folder, "p_value_matrices.csv"), index=False)
    df_pval = compute_pvals(df_analysis)
    print(df_pval.shape)
import os
import json
import pandas as pd
import numpy as np
# from statsmodels.stats.contingency_tables import mcnemar


def jaccard_similarity(list1, list2):
    """Calculate the Jaccard similarity between two sets."""
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

def get_answer_set(df, q_serie, task):
    match = df[(df["Q_serie"] == q_serie) & (df["task"] == task)]
    if not match.empty:
        return set(match["Answer"].values[0])
    return set()

def load_question(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
    # df["type"] = df["type"].apply(lambda x: str(x) if not pd.isna(x) else "0")
    return df

def load_all_questions(root_dir, datasets, languages):
    """
    Load and merge question files from multiple datasets and languages.

    Args:
        root_dir (str): Base directory containing the question files.
        datasets (list): List of dataset names.
        languages (list): List of language codes.
        load_questions_fn (Callable): Function to load a TSV file into a DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame with original index stored as 'q_index',
                      and columns 'dataset' and 'lang' added.
    """
    all_dfs = []

    for dataset in datasets:
        dataset_stem = os.path.splitext(dataset)[0]
        for lang in languages:
            candidates = [
                os.path.join(root_dir, "data", "ASCB", lang, f"{dataset_stem}.tsv"),
                os.path.join(root_dir, "data", "ASCB", lang, f"{dataset_stem.lower()}.tsv"),
                os.path.join(root_dir, "data", "ASCB", f"{dataset_stem}.tsv"),
                os.path.join(root_dir, "data", "ASCB", f"{dataset_stem.lower()}.tsv"),
                os.path.join(root_dir, "data", "Dataset", lang, f"{dataset_stem}.tsv"),
                os.path.join(root_dir, "data", "Dataset", f"{dataset_stem}.tsv"),
            ]
            candidates = list(dict.fromkeys(candidates))
            question_path = next((path for path in candidates if os.path.exists(path)), None)
            if question_path is None:
                print(f"File not found. Tried: {candidates}")
                continue

            df = load_question(question_path)
            df = df.copy()
            df["q_index"] = df["ID"].astype(int) if "ID" in df.columns else df.index
            df["dataset"] = dataset
            df["lang"] = lang

            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


########Answer Analysis ########
def load_answers(folder: str, datasets, llms, actions, tasks, languages, questions) -> pd.DataFrame:
    answer_frames = []

    json_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder)
        for file in files if file.endswith(".json")
    ]

    print(f"JSON files found: {len(json_files)}")

    for file in json_files:
        if not file.split("/")[-1].startswith("Q"):
            continue
        elements = file.replace("_", "/").replace(".json", "").split("/")
        elements_lower = [e.lower() for e in elements]
        question = next((q for q in questions if q in elements), None)
        action = _infer_action(elements_lower, actions)
        task = next((t for t in tasks if t in elements), None)
        dataset = next((d for d in datasets if d.lower() in elements_lower), None)
        lang = next((l for l in languages if l in elements), None)
        llm = next((l for l in llms if l in elements), None)

        if all([question, action, task, dataset, llm]):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame([{"Q_ID": key, "Answer": value} for key, value in data.items()])
            df["Q_serie"] = question
            df["action"] = action
            df["task"] = task
            df["dataset"] = dataset
            df["llm"] = llm
            df["lang"] = lang
            answer_frames.append(df)

    columns = ["Q_ID", "Answer", "Q_serie", "action", "task", "dataset", "llm", "lang"]
    if not answer_frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(answer_frames, ignore_index=True)


def _infer_action(elements_lower, actions):
    actions_lower = {a.lower(): a for a in actions}
    if "oracle" in elements_lower or "fixing" in elements_lower:
        return actions_lower.get("fixing", "fixing")
    if "cte" in elements_lower or "classandanswer" in elements_lower or "classification" in elements_lower:
        return actions_lower.get("classification", "classification")
    if "chain-of-thought" in elements_lower or "chain" in elements_lower:
        return actions_lower.get("chain-of-thought", "chain-of-thought")
    if "star" in elements_lower:
        return actions_lower.get("star", "star")
    if "wikidata" in elements_lower:
        return actions_lower.get("wikidata", actions_lower.get("zero-shot", "zero-shot"))
    return actions_lower.get("zero-shot", "zero-shot")

def enrich_answers(df_answers, df_questions):
    required_question_cols = {"q_index", "dataset", "Q1", "Q2", "Q3", "Q4"}
    missing_question_cols = required_question_cols - set(df_questions.columns)
    if missing_question_cols:
        raise ValueError(
            "Question files were not loaded correctly. "
            f"Missing columns: {sorted(missing_question_cols)}"
        )

    question_lookup = df_questions.melt(
        id_vars=["q_index", "dataset"],
        value_vars=["Q1", "Q2", "Q3", "Q4"],
        var_name="Q_serie",
        value_name="Question",
    )
    question_lookup["Q_ID"] = question_lookup["q_index"].astype(int)

    df_answers = df_answers.copy()
    df_answers["Q_ID"] = pd.to_numeric(df_answers["Q_ID"], errors="coerce").astype("Int64")
    df_answers = df_answers.merge(
        question_lookup[["Q_ID", "dataset", "Q_serie", "Question"]],
        on=["Q_ID", "dataset", "Q_serie"],
        how="left",
    )

    df_answers.drop_duplicates(
        subset=["Q_ID", "Q_serie", "action", "task", "dataset", "llm"],
        inplace=True
    )
    df_answers["Answer"] = df_answers["Answer"].apply(lambda x: x if isinstance(x, list) else [])
    df_answers.reset_index(drop=True, inplace=True)
    return df_answers


def analysis(df):
    rows = []
    group_keys = ["Q_ID", "action", "dataset", "llm"]
    grouped = df.groupby(group_keys)

    for keys, group in grouped: 
        if set(group["Q_serie"]) >= {"Q1", "Q2", "Q3", "Q4"}:
            action = group["action"].values[0]
            llm = group["llm"].values[0]
            dataset = group["dataset"].values[0]
            qid = group["Q_ID"].values[0]
            if action in ["zero-shot", "chain-of-thought"]:
                A1 = get_answer_set(group, "Q1", "equal")
                A2 = get_answer_set(group, "Q2", "equal")
                A3 = get_answer_set(group, "Q3", "sup-sub")
                A4 = get_answer_set(group, "Q4", "minus")
                A1_star = None
                if action == "zero-shot":
                    row = df[(df["action"]=="star") & (df["llm"]==llm)&(df["dataset"]==dataset)&(df["Q_ID"]==qid)]
                    if not row.empty:
                        A1_star = set(row["Answer"].values[0])
                A1_prime = None
                A1_double_prime = None

                similarities = {
                    "J(A1-A2)": round(jaccard_similarity(A1, A2), 4),
                    "J(A1-A34)": round(jaccard_similarity(A1, A3.union(A4)), 4),
                    "J(A3-A4)": round(jaccard_similarity(A3, A4), 4),
                    "J(A4-A1|3)":round(jaccard_similarity(A4, A1 - A3),4),
                    "J(A1-A1*)": round(jaccard_similarity(A1, A1_star), 4) if A1_star is not None else None,
                    "J(A1-A1**)": None,
                    "J(A1*-A1**)": None
                    }
                consistency = {
                    "?A1=A2": int(A1 == A2),
                    "?A1=A3+A4": int(A1 == A3.union(A4)),
                    "?A1>A3": int(A3.issubset(A1)),
                    "?A1>A4": int(A4.issubset(A1)),
                    "?A3∅A4": int(A3.isdisjoint(A4)),
                    "?A4=A1|3": int(A1 == A3.union(A4) and A3.isdisjoint(A4)),
                    "?A1=A1*": int(A1 == A1_star) if A1_star is not None else None,
                    "?A1=A1**": None,
                    "?A1*=A1**": None
                    }
            elif action in ['classification','fixing']:
                # Usage
                A1_equal = get_answer_set(group, "Q1", "equal")
                A1_contain = get_answer_set(group, "Q1", "sup-sub")
                A1_minus = get_answer_set(group, "Q1", "minus")
                A2_equal = get_answer_set(group, "Q2", "equal")
                A3_contain = get_answer_set(group, "Q3", "sup-sub")
                A3_minus = get_answer_set(group, "Q3", "minus")
                if not A3_contain:
                    A3_contain = A3_minus
                A4_minus = get_answer_set(group, "Q4", "minus")
                similarities = {
                    "J(A1-A2)": round(jaccard_similarity(A1_equal, A2_equal), 4),
                    "J(A1-A34)": round(jaccard_similarity(A1_minus, A3_minus.union(A4_minus)), 4),
                    "J(A3-A4)": round(jaccard_similarity(A3_minus, A4_minus), 4),
                    "J(A4-A1|3)":round(jaccard_similarity(A4_minus, A1_minus - A3_minus),4),
                    "J(A1-A1*)": round(jaccard_similarity(A1_equal, A1_contain), 4),
                    "J(A1-A1**)": round(jaccard_similarity(A1_equal, A1_minus), 4),
                    "J(A1*-A1**)": round(jaccard_similarity(A1_contain, A1_minus), 4)
                    }
                consistency = {
                    "?A1=A2": int(A1_equal == A2_equal),
                    "?A1=A3+A4": int(A1_minus == A3_minus.union(A4_minus)),
                    "?A1>A3": int(A3_contain.issubset(A1_contain)),
                    "?A1>A4": int(A4_minus.issubset(A1_minus)),
                    "?A3∅A4": int(A3_minus.isdisjoint(A4_minus)),
                    "?A4=A1|3": int(A1_minus == A3_minus.union(A4_minus) and A3_minus.isdisjoint(A4_minus)),
                    "?A1=A1*": int(A1_equal == A1_contain),
                    "?A1=A1**": int(A1_equal == A1_minus),
                    "?A1*=A1**": int(A1_contain == A1_minus)
                    }

                A1 = A1_equal
                A2 = A2_equal
                A3 = A3_contain
                A4 = A4_minus
                A1_prime = list(A1_contain)
                A1_double_prime = list(A1_minus)
                
            q_map = {
                row["Q_serie"]: row["Question"]
                for _, row in group.iterrows()
                if row["Q_serie"] in {"Q1", "Q2", "Q3", "Q4"}
            }

            row = {
                "Q_ID": keys[0], "action": keys[1], "dataset": keys[2], "llm": keys[3],
                **consistency, **similarities,
                "Q1": q_map.get("Q1", ""), "Q2": q_map.get("Q2", ""),
                "Q3": q_map.get("Q3", ""), "Q4": q_map.get("Q4", ""),
                "A1": list(A1), "A2": list(A2), "A3": list(A3), "A4": list(A4),
                "A1*": A1_prime, "A1**": A1_double_prime,
                "idk_A1": 1 if len(A1) == 0 or ("idk" in A1) else 0,
                "idk_A2": 1 if len(A2) == 0 or ("idk" in A2) else 0,
                "idk_A3": 1 if len(A3) == 0 or ("idk" in A3) else 0,
                "idk_A4": 1 if len(A4) == 0 or ("idk" in A4) else 0
            }
            rows.append(row)
    df_analysis = pd.DataFrame(rows)
    return df_analysis



def summary(df_analysis):
    group_cols = ["dataset", "action", "llm"]
    consistency_cols = ["?A1=A2", "?A1=A3+A4", "?A1>A3", "?A1>A4", "?A3∅A4", "?A4=A1|3", "?A1=A1*", "?A1=A1**","?A1*=A1**"]
    jaccard_cols = ["J(A1-A2)", "J(A1-A34)", "J(A3-A4)","J(A4-A1|3)","J(A1-A1*)", "J(A1-A1**)","J(A1*-A1**)"]
    self_contradition_cols = ["?SC(A1=A2)","?SC(A1>A3)","?SC(A1>A4)","?SC(A3∅A4)","?SC(A4=A1|3)"]
    pval_cols = [col for col in df_analysis.columns if col.startswith("p_value_")]
    metric_cols = consistency_cols + jaccard_cols + pval_cols + self_contradition_cols

    for a in ["A1", "A2", "A3", "A4"]:
        df_analysis[f"idk_{a}"] = df_analysis[a].apply(lambda x: int(
        (isinstance(x, list) and len(x) == 0)       # []
        or (x == "idk")                             # "idk"
        or (isinstance(x, list) and x == ["idk"])   # ["idk"]
    ))

    empty_cols = [f"idk_{a}" for a in ["A1", "A2", "A3", "A4"]]


    df_summary = (
        df_analysis
        .groupby(group_cols)[metric_cols + empty_cols]
        .mean()
        .reset_index()
        .round(4)
    )
    group_cols_overall = ["action", "llm"]
    df_summary_extend = (
        df_analysis
        .groupby(group_cols_overall)[metric_cols + empty_cols]
        .mean()
        .reset_index()
        .round(4)
    )
    df_summary_extend["dataset"] = "overall"
    
    df_summary = pd.concat([df_summary, df_summary_extend], ignore_index=True)
    df_summary["?A1=A1(ave)"] = df_summary[["?A1=A1*", "?A1=A1**","?A1*=A1**"]].mean(axis=1).round(4)
    df_summary["J_A1_ave"] = df_summary[["J(A1-A1*)", "J(A1-A1**)", "J(A1*-A1**)"]].mean(axis=1).round(4)
    
    col = ["?A1=A1*","J(A1-A1*)"]
    # source values indexed by (llm, dataset) from classification rows
    # src = df_summary.query('action == "classification"').set_index(['llm', 'dataset'])[col]

    # # assign to matching zero-shot rows
    # mask = df_summary['action'].eq('zero-shot')
    # zero_idx = pd.MultiIndex.from_frame(df_summary.loc[mask, ['llm', 'dataset']])
    # df_summary.loc[mask, col] = src.reindex(zero_idx).to_numpy()
    mask1 = (df_summary["dataset"] == "overall") & (df_summary["action"] == "zero-shot")
    mask2 = (df_summary["dataset"] == "overall") & (df_summary["action"] == "classification")
    a = df_summary.loc[mask1, col].copy()
    b = df_summary.loc[mask2, col]

    # Vectorized conditional assignment
    for column in col:
        # Where a[column] is NaN, use b[column], otherwise use (a[column] + b[column]) / 2
        a[column] = np.where(a[column].isna(), 
                            b[column].values, 
                            (a[column] + b[column].values) / 2)

    # Assign back to original dataframe
    df_summary.loc[mask1, col] = a

    idk_col = ["idk_A1","idk_A2","idk_A3","idk_A4"]
    df_summary["idk"] = df_summary[idk_col].mean(axis=1)
    return df_summary

def summary_xidk(df_analysis):
    group_cols = ["dataset", "action", "llm"]
    consistency_cols = ["?A1=A2", "?A1=A3+A4", "?A1>A3", "?A1>A4", "?A3∅A4", "?A4=A1|3", "?A1=A1*", "?A1=A1**","?A1*=A1**"]
    jaccard_cols = ["J(A1-A2)", "J(A1-A34)", "J(A3-A4)","J(A4-A1|3)","J(A1-A1*)", "J(A1-A1**)" ,"J(A1*-A1**)"]
    self_contradition_cols = ["?SC(A1=A2)","?SC(A1>A3)","?SC(A1>A4)","?SC(A3∅A4)","?SC(A4=A1|3)"]
    pval_cols = [col for col in df_analysis.columns if col.startswith("p_value_")]
    metric_cols = consistency_cols + jaccard_cols + pval_cols + self_contradition_cols

    # for a in ["A1", "A2", "A3", "A4"]:
    #     df_analysis[f"idk_{a}"] = df_analysis[a].apply(lambda x: int(
    #     (isinstance(x, list) and len(x) == 0)       # []
    #     or (x == "idk")                             # "idk"
    #     or (isinstance(x, list) and x == ["idk"])   # ["idk"]
    # ))

    empty_cols = [f"idk_{a}" for a in ["A1", "A2", "A3", "A4"]]

    # Define which idk columns to use for each metric
    metric_idk_map = {
        "?A1=A2": ["idk_A1", "idk_A2"],
        "J(A1-A2)": ["idk_A1", "idk_A2"],
        "?A1=A3+A4": ["idk_A1", "idk_A3", "idk_A4"],
        "J(A1-A34)": ["idk_A1", "idk_A3", "idk_A4"],
        "?A1>A3": ["idk_A1", "idk_A3"],
        "?A1>A4": ["idk_A1", "idk_A4"],
        "?A3∅A4": ["idk_A3", "idk_A4"],
        "J(A3-A4)": ["idk_A3", "idk_A4"],
        "J(A4-A1|3)": ["idk_A4", "idk_A1", "idk_A3"],
        "?A4=A1|3": ["idk_A4", "idk_A1", "idk_A3"],
        "?A1=A1*": ["idk_A1", "idk_A1*"],
        "J(A1-A1*)": ["idk_A1", "idk_A1*"],
        "?A1=A1**": ["idk_A1", "idk_A1**"],
        "J(A1-A1**)": ["idk_A1", "idk_A1**"],
        "?A1*=A1**": ["idk_A1*", "idk_A1**"],
        "J(A1*-A1**)": ["idk_A1*", "idk_A1**"],
    }

    # Compute summary per metric, filtering rows where all relevant idk columns are 1
    summary_dict = {col: [] for col in metric_cols + empty_cols}
    grouped = df_analysis.groupby(group_cols)
    for name, group in grouped:
        for col in metric_cols:
            idk_cols = metric_idk_map.get(col, [])
            # Only use idk columns that exist in the group
            idk_cols_existing = [c for c in idk_cols if c in group.columns]
            if idk_cols_existing:
                mask = ~(group[idk_cols_existing].all(axis=1))
                # if len(mask[mask==False]) > 0:
                #     print(f"Computing {col} for group {name} with {len(mask[mask==False])} idk rows filtered out.")
                filtered = group.loc[mask, col]
            else:
                filtered = group[col]
            summary_dict[col].append(filtered.mean())
        for col in empty_cols:
            summary_dict[col].append(group[col].mean())
    df_summary = pd.DataFrame({"dataset": [x[0] for x in grouped.groups.keys()],
                              "action": [x[1] for x in grouped.groups.keys()],
                              "llm": [x[2] for x in grouped.groups.keys()]})
    for col in metric_cols + empty_cols:
        df_summary[col] = summary_dict[col]
    df_summary = df_summary.round(4)

    # Overall summary
    group_cols_overall = ["action", "llm"]
    summary_dict_overall = {col: [] for col in metric_cols + empty_cols}
    grouped_overall = df_analysis.groupby(group_cols_overall)
    for name, group in grouped_overall:
        for col in metric_cols:
            idk_cols = metric_idk_map.get(col, [])
            idk_cols_existing = [c for c in idk_cols if c in group.columns]
            if idk_cols_existing:
                mask = ~(group[idk_cols_existing].all(axis=1))
                filtered = group.loc[mask, col]
            else:
                filtered = group[col]
            summary_dict_overall[col].append(filtered.mean())
        for col in empty_cols:
            summary_dict_overall[col].append(group[col].mean())
    df_summary_extend = pd.DataFrame({"action": [x[0] for x in grouped_overall.groups.keys()],
                                     "llm": [x[1] for x in grouped_overall.groups.keys()],
                                     "dataset": "overall"})
    for col in metric_cols + empty_cols:
        df_summary_extend[col] = summary_dict_overall[col]
    df_summary_extend = df_summary_extend.round(4)

    df_summary = pd.concat([df_summary, df_summary_extend], ignore_index=True)
    # Ensure columns are numeric before mean/round to avoid TypeError
    for col_group, new_col in [
        (["?A1=A1*", "?A1=A1**","?A1*=A1**"], "?A1=A1(ave)"),
        (["J(A1-A1*)", "J(A1-A1**)", "J(A1*-A1**)"], "J_A1_ave")
    ]:
        numeric_cols = df_summary[col_group].apply(pd.to_numeric, errors='coerce')
        df_summary[new_col] = numeric_cols.mean(axis=1).round(4)

    col = ["?A1=A1*","J(A1-A1*)"]
    mask1 = (df_summary["dataset"] == "overall") & (df_summary["action"] == "zero-shot")
    mask2 = (df_summary["dataset"] == "overall") & (df_summary["action"] == "classification")
    a = df_summary.loc[mask1, col].copy()
    b = df_summary.loc[mask2, col]

    for column in col:
        a[column] = np.where(a[column].isna(), 
                            b[column].values, 
                            (a[column] + b[column].values) / 2)

    df_summary.loc[mask1, col] = a

    idk_col = ["idk_A1","idk_A2","idk_A3","idk_A4"]
    df_summary["idk"] = df_summary[idk_col].mean(axis=1)
    return df_summary

if __name__ == "__main__":
    # load dfanalysis 
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    analysis_path = os.path.join(root_dir, "output", "analysis.csv")
    df_analysis = pd.read_csv(analysis_path)
    # get summary_xidk
    df_summary_xidk = summary_xidk(df_analysis)
    # save
    output_folder = os.path.join(root_dir, "output")
    os.makedirs(output_folder, exist_ok=True)
    df_summary_xidk.to_csv(os.path.join(output_folder, "summary_xidk.csv"), index=False)

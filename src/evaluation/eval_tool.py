import os
import json
import pandas as pd
import datetime


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
        for lang in languages:
            question_path = os.path.join(root_dir, "data", "Dataset", lang, f"{dataset}.tsv")
            if not os.path.exists(question_path):
                print(f"File not found: {question_path}")
                continue

            df = load_question(question_path)
            df = df.copy()
            df["q_index"] = df.index
            df["dataset"] = dataset
            df["lang"] = lang

            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


#####Relation Classification #####
def load_relations(root_dir, datasets, llms):
    """
        DataFrame with columns: ["Q_ID", "dataset", "llm", "R(1-2)", "R(1-3)", "R(1-4)", "R(3-4)", "R(1-34)"]
    """
    # find JSON files
    json_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(root_dir)
        for file in files
        if file.startswith("Relation") and file.endswith(".json")
    ]
    print(f"JSON files found: {len(json_files)}")

    # initialize dataframe
    df_relation = pd.DataFrame(
        columns=["Q_ID", "dataset", "llm", "R(1-2)", "R(1-3)", "R(1-4)", "R(3-4)", "R(1-34)"]
    )

    for file in json_files:
        elements = file.replace("_", "/").replace(".json", "").split("/")
        dataset = next((d for d in datasets if d in elements), None)
        llm = next((l for l in llms if l in elements), None)

        if all([dataset, llm]):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # transform dict into rows
            rows = [
                {
                    "dataset": dataset,
                    "llm": llm,
                    "Q_ID": key,
                    "R(1-2)": value[0],
                    "R(1-3)": value[1],
                    "R(1-4)": value[2],
                    "R(3-4)": value[3],
                    "R(1-34)": value[4],
                }
                for key, value in data.items()
            ]
            df_relation = pd.concat([df_relation, pd.DataFrame(rows)], ignore_index=True)

    return df_relation


CANONICAL_LABELS = [
    "Equivalence", "Contains", "ContainedBy", "Overlap", "Disjoint", "Unknown", "Else"
]

GT = {
    "R(1-2)":  "Equivalence",
    "R(1-3)":  "Contains",
    "R(1-4)":  "Contains",
    "R(3-4)":  "Disjoint",
    "R(1-34)": "Equivalence",
}


def _normalize_pred(x: object) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    if s in CANONICAL_LABELS:
        return s
    return "Else"

def per_model_confusions(
    df_relation: pd.DataFrame,
    relation_cols=None,
    include_overall: bool = True,
    round_digits: int = 4,
):
    """
    Build complete confusion matrices per (llm, dataset) and per relation column.
    Adds an 'overall' dataset per llm if include_overall=True.
    
    Returns
    -------
    cms_counts : pd.DataFrame
        MultiIndex rows: (llm, dataset, relation, True)
        Columns: Equivalence, Contains, ContainedBy, Overlap, Disjoint, Unknown, Else (counts)
    cms_ratio : pd.DataFrame
        Same shape, row-normalized ratios.
    """
    if relation_cols is None:
        relation_cols = list(GT.keys())

    needed = {"dataset", "llm", *relation_cols}
    missing = needed - set(df_relation.columns)
    if missing:
        raise ValueError(f"df_relation missing columns: {missing}")

    rows_counts, rows_ratio, idx = [], [], []

    # 1) Per (llm, dataset)
    for (llm, dataset), group in df_relation.groupby(["llm", "dataset"], dropna=False):
        n_group = len(group)
        for rel in relation_cols:
            truth = GT[rel]
            y_pred = group[rel].map(_normalize_pred)
            counts = y_pred.value_counts()
            row_counts = [int(counts.get(lbl, 0)) for lbl in CANONICAL_LABELS]
            row_ratio = [(c / n_group) if n_group > 0 else 0.0 for c in row_counts]
            rows_counts.append(row_counts)
            rows_ratio.append(row_ratio)
            idx.append((llm, dataset, rel, truth))

    # 2) Per llm (overall across datasets)
    if include_overall:
        for llm, group in df_relation.groupby("llm", dropna=False):
            n_group = len(group)
            for rel in relation_cols:
                truth = GT[rel]
                y_pred = group[rel].map(_normalize_pred)
                counts = y_pred.value_counts()
                row_counts = [int(counts.get(lbl, 0)) for lbl in CANONICAL_LABELS]
                row_ratio = [(c / n_group) if n_group > 0 else 0.0 for c in row_counts]
                rows_counts.append(row_counts)
                rows_ratio.append(row_ratio)
                idx.append((llm, "overall", rel, truth))

    index = pd.MultiIndex.from_tuples(idx, names=["llm", "dataset", "relation", "True"])
    cms_counts = pd.DataFrame(rows_counts, index=index, columns=CANONICAL_LABELS)
    cms_ratio  = pd.DataFrame(rows_ratio,  index=index, columns=CANONICAL_LABELS)
    if round_digits is not None:
        cms_ratio = cms_ratio.round(round_digits)

    return cms_counts, cms_ratio


def build_confusion_table(cms_counts: pd.DataFrame,
                          cms_ratio: pd.DataFrame,
                          round_digits: int = 4) -> pd.DataFrame:
    """
    Create one tidy table:
      llm | dataset | relation | True | Accuracy | Size | Equivalence | Contains | ... | Else
    where each label column is 'ratio(count)' and Accuracy is for the True label as 'ratio(count)'.
    """
    records = []
    for idx in cms_counts.index:
        llm, dataset, relation, true_label = idx
        counts_row = cms_counts.loc[idx]
        ratio_row  = cms_ratio.loc[idx]

        N = int(counts_row.sum())
        acc_ratio = float(ratio_row.get(true_label, 0.0))
        acc_count = int(counts_row.get(true_label, 0))

        row = {
            "llm": llm,
            "dataset": dataset,
            "relation": relation,
            "True": true_label,
            "Accuracy": f"{acc_ratio:.{round_digits}f}({acc_count})",
            "Size": N,
        }

        # Add each predicted label as ratio(count)
        for lbl in CANONICAL_LABELS:
            r = float(ratio_row.get(lbl, 0.0))
            c = int(counts_row.get(lbl, 0))
            row[lbl] = f"{r:.{round_digits}f}({c})"

        records.append(row)

    out = pd.DataFrame.from_records(records)
    # nice ordering
    cols = ["llm", "dataset", "relation", "True", "Accuracy", "Size"] + CANONICAL_LABELS
    return out[cols]


def relation_summary(df_relation: pd.DataFrame,
                            relation_cols=None,
                            include_overall: bool = True,
                            round_digits: int = 4) -> pd.DataFrame:
    """
    Convenience wrapper: calls your per_model_confusions(...) then builds the table.
    """
    # uses the per_model_confusions you already have
    cms_counts, cms_ratio = per_model_confusions(
        df_relation,
        relation_cols=relation_cols,
        include_overall=include_overall,
        round_digits=round_digits,
    )
    return build_confusion_table(cms_counts, cms_ratio, round_digits)

########Answer Analysis ########
def load_answers(folder: str, datasets, llms, actions, tasks, languages, questions) -> pd.DataFrame:
    df_answers = pd.DataFrame(columns=["Q_ID", "Q_serie", "action", "task", "dataset", "lang","llm"])

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
        question = next((q for q in questions if q in elements), None)
        action = next((a for a in actions if a in elements), "zero-shot")
        task = next((t for t in tasks if t in elements), None)
        dataset = next((d for d in datasets if d in elements), None)
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
            df_answers = pd.concat([df_answers, df], ignore_index=True)

    return df_answers

def enrich_answers(df_answers, df_questions):
    df_answers["Question"] = df_answers.apply(
        lambda x: df_questions.loc[
            (df_questions["q_index"] == int(x["Q_ID"])) &
            (df_questions["dataset"] == x["dataset"])
        ][x["Q_serie"]].values[0]
        if not df_questions.loc[
            (df_questions["q_index"] == int(x["Q_ID"])) &
            (df_questions["dataset"] == x["dataset"]) 
        ].empty else None,
        axis=1
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
            if action in ["zero-shot", "wikidata"]:
                A1 = get_answer_set(group, "Q1", "equal")
                A2 = get_answer_set(group, "Q2", "equal")
                A3 = get_answer_set(group, "Q3", "sup-sub")
                A4 = get_answer_set(group, "Q4", "minus")

                A1_prime = None
                A1_double_prime = None

                similarities = {
                    "J(A1-A2)": round(jaccard_similarity(A1, A2), 4),
                    "J(A1-A34)": round(jaccard_similarity(A1, A3.union(A4)), 4),
                    "J(A1-A1*)": None,
                    "J(A1-A1**)": None,
                    "J(A1*-A1**)": None
                    }
                consistency = {
                    "?A1=A2": int(A1 == A2),
                    "?A1=A3+A4": int(A1 == A3.union(A4)),
                    "?A1>A3": int(A3.issubset(A1)),
                    "?A1>A4": int(A4.issubset(A1)),
                    "?A3∅A4": int(A3.isdisjoint(A4))
                    }
            elif action in ['classification','fixing']:
                # Usage
                A1_equal = get_answer_set(group, "Q1", "equal")
                A1_contain = get_answer_set(group, "Q1", "sup-sub")
                A1_minus = get_answer_set(group, "Q1", "minus")
                A2_equal = get_answer_set(group, "Q2", "equal")
                A3_contain = get_answer_set(group, "Q3", "sup-sub")
                A3_minus = get_answer_set(group, "Q3", "minus")
                A4_minus = get_answer_set(group, "Q4", "minus")
                similarities = {
                    "J(A1-A2)": round(jaccard_similarity(A1_equal, A2_equal), 4),
                    "J(A1-A34)": round(jaccard_similarity(A1_minus, A3_minus.union(A4_minus)), 4),
                    "J(A1-A1*)": round(jaccard_similarity(A1_equal, A1_contain), 4),
                    "J(A1-A1**)": round(jaccard_similarity(A1_equal, A1_minus), 4),
                    "J(A1*-A1**)": round(jaccard_similarity(A1_contain, A1_minus), 4)
                    }
                consistency = {
                    "?A1=A2": int(A1_equal == A2_equal),
                    "?A1=A3+A4": int(A1_minus == A3_minus.union(A4_minus)),
                    "?A1>A3": int(A3_contain.issubset(A1_contain)),
                    "?A1>A4": int(A4_minus.issubset(A1_minus)),
                    "?A3∅A4": int(A3_minus.isdisjoint(A4_minus))
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
                "A1*": A1_prime, "A1**": A1_double_prime
            }
            rows.append(row)

    return pd.DataFrame(rows)

def summary(df_analysis):
    group_cols = ["dataset", "action", "llm"]
    consistency_cols = ["?A1=A2", "?A1=A3+A4", "?A1>A3", "?A1>A4", "?A3∅A4"]
    jaccard_cols = ["J(A1-A2)", "J(A1-A34)", "J(A1-A1*)", "J(A1-A1**)","J(A1*-A1**)"]
    pval_cols = [col for col in df_analysis.columns if col.startswith("p_value_")]
    metric_cols = consistency_cols + jaccard_cols + pval_cols

    for a in ["A1", "A2", "A3", "A4"]:
        df_analysis[f"{a}_empty_ratio"] = df_analysis[a].apply(lambda x: int(
        (isinstance(x, list) and len(x) == 0)       # []
        or (x == "idk")                             # "idk"
        or (isinstance(x, list) and x == ["idk"])   # ["idk"]
    ))

    empty_cols = [f"{a}_empty_ratio" for a in ["A1", "A2", "A3", "A4"]]

    df_valid = df_analysis.copy()
    for col in jaccard_cols:
        df_valid = df_valid[df_valid[col] != -1]

    df_summary = (
        df_valid
        .groupby(group_cols)[metric_cols + empty_cols]
        .mean()
        .reset_index()
        .round(4)
    )
    group_cols_overall = ["action", "llm"]
    df_summary_extend = (
        df_valid
        .groupby(group_cols_overall)[metric_cols + empty_cols]
        .mean()
        .reset_index()
        .round(4)
    )
    df_summary_extend["dataset"] = "overall"

    return pd.concat([df_summary, df_summary_extend], ignore_index=True)


# if __name__ == "__main__":
#     root_dir = os.path.dirname(os.path.abspath(__name__))
#     datasets=["spinach", "qawiki",'synthetic']
#     llms = ['gpt-4.1-2025-04-14', 'gpt-4.1-mini-2025-04-14', 'gpt-4.1-nano-2025-04-14', 
#             'gpt-4o','o3','gpt-5-nano',"gpt-5-mini","gpt-5",
#             "gemini-2.0-flash","gemini-2.5-flash","gemini-2.5-pro",
#             "grok-3-mini","deepseek-chat","deepseek-reasoner","llama3.1:8b","llama3.3:70b"]
#     actions = ["fixing", "classification", "wikidata"]
#     tasks = ['equal', 'sup-sub', "minus"]
#     languages = ['en']

#     df_questions = load_all_questions(root_dir, datasets, languages)
#     df_answers = load_answers(
#         folder=root_dir + "/data/answers/",
#         datasets = datasets,
#         llms=llms,
#         actions=actions,
#         tasks=tasks,
#         languages=languages,
#         questions=["Q1", "Q2", "Q3", "Q4"]
#     )

#     df_answers = enrich_answers(df_answers, df_questions)
#     df_analysis = analysis(df_answers)
#     df_summary = summary(df_analysis)

#     # Define output folder path
#     output_folder = os.path.join(root_dir, "data", "Analysis")
#     os.makedirs(output_folder, exist_ok=True)

#     # Save results
#     analysis_file_format = datetime.datetime.now().strftime("analysis_%Y-%m-%d_%H-%M.csv")
#     summary_file_format = datetime.datetime.now().strftime("summary_%Y-%m-%d_%H-%M.csv")
#     df_analysis.to_csv(os.path.join(output_folder, analysis_file_format), index=False)
#     df_summary.to_csv(os.path.join(output_folder, summary_file_format), index=False)

#     print("Analysis and summary saved to:", output_folder)


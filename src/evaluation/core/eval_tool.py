
from src.evaluation.paths import REPO_ROOT
import ast
import os
import json
import pandas as pd
import numpy as np
from src.evaluation.core.summary_idk_worst import add_idk_flags, is_null_answer, metric_idk_mask
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
        for root, _, files in sorted(os.walk(folder))
        for file in sorted(files) if file.endswith(".json")
    ]

    # Older CtE generators saved the answer to dataset Q3 as Q2 in the
    # containment task. Prefer explicitly named Q3 files when both exist.
    json_files.sort(key=lambda path: (os.path.basename(path).startswith("Q2_sup-sub"), path))
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

        if task == "sup-sub" and question == "Q2" and action in ["classification", "classification-no-idk"]:
            question = "Q3"
        if action in actions and all([question, action, task, dataset, llm]):
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
    # Check ablation directories before filename tokens such as "fixing" or
    # "classAndAnswer" can map them back to the original action.
    for ablation_action in [
        "zero-shot-no-idk",
        "fixing-no-idk",
        "classification-no-idk",
    ]:
        if ablation_action in elements_lower:
            return actions_lower.get(ablation_action, ablation_action)
    if "oracle" in elements_lower or "fixing" in elements_lower:
        return actions_lower.get("fixing", "fixing")
    if "cte" in elements_lower or "classandanswer" in elements_lower or "classification" in elements_lower:
        return actions_lower.get("classification", "classification")
    if "chain-of-thought" in elements_lower or "chain" in elements_lower:
        return actions_lower.get("chain-of-thought", "chain-of-thought")
    if "star" in elements_lower:
        return actions_lower.get("star", "star")
    if "wikidata" in elements_lower:
        return actions_lower.get("wikidata", "wikidata")
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


def partition_metrics(a1, a3, a4):
    """Joint partition score plus its two same-answer diagnostic components."""
    difference = int(a4 == a1 - a3)
    containment = int(a3.issubset(a1))
    joint = int(a1 == a3.union(a4) and a3.isdisjoint(a4))
    if joint != (difference & containment):
        raise AssertionError("Partition identity failed")
    return {"?A4=A1|3": joint, "?A4=A1-A3": difference,
            "?A3<=A1(same-context)": containment}


def analysis(df):
    rows = []
    group_keys = ["Q_ID", "action", "dataset", "llm"]
    grouped = df.groupby(group_keys)
    repeat_answers = {
        (r.Q_ID, r.dataset, r.llm): set(r.Answer)
        for r in df[df["action"].eq("star")].itertuples()
    }
    # Explicit fallback for the two models without isolated Q1 repeats.
    # Use the CtE equality-task Q1, never an aggregate CtE consistency score.
    fallback_models = {"llama3.1:70b", "grok-3-mini"}
    cte_q1_answers = {
        (r.Q_ID, r.dataset, r.llm): set(r.Answer)
        for r in df.loc[df["action"].eq("classification")
                        & df["Q_serie"].eq("Q1") & df["task"].eq("equal")
                        & df["llm"].isin(fallback_models)].itertuples()
    }

    for keys, group in grouped: 
        if set(group["Q_serie"]) >= {"Q1", "Q2", "Q3", "Q4"}:
            action = group["action"].values[0]
            llm = group["llm"].values[0]
            dataset = group["dataset"].values[0]
            qid = group["Q_ID"].values[0]
            repeat_source = None
            if action in ["zero-shot", "zero-shot-no-idk", "chain-of-thought"]:
                A1 = get_answer_set(group, "Q1", "equal")
                A2 = get_answer_set(group, "Q2", "equal")
                A3 = get_answer_set(group, "Q3", "sup-sub")
                A4 = get_answer_set(group, "Q4", "minus")
                A1_star = repeat_answers.get((qid, dataset, llm)) if action == "zero-shot" else None
                if A1_star is not None:
                    repeat_source = "Base repeat"
                elif action == "zero-shot":
                    A1_star = cte_q1_answers.get((qid, dataset, llm))
                    if A1_star is not None:
                        repeat_source = "CtE Q1 (equal) fallback"
                A1_prime = list(A1_star) if A1_star is not None else None
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
                    **partition_metrics(A1, A3, A4),
                    "?A1=A1*": int(A1 == A1_star) if A1_star is not None else None,
                    "?A1=A1**": None,
                    "?A1*=A1**": None
                    }
            elif action in [
                "classification",
                "fixing",
                "classification-no-idk",
                "fixing-no-idk",
            ]:
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
                    **partition_metrics(A1_minus, A3_minus, A4_minus),
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
                
            else:
                continue

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
                "repeat_source": repeat_source,
                "A3_minus": list(A3_minus) if action in ["classification", "fixing", "classification-no-idk", "fixing-no-idk"] else None,
                "idk_A1": 1 if len(A1) == 0 or ("idk" in A1) else 0,
                "idk_A2": 1 if len(A2) == 0 or ("idk" in A2) else 0,
                "idk_A3": 1 if len(A3) == 0 or ("idk" in A3) else 0,
                "idk_A4": 1 if len(A4) == 0 or ("idk" in A4) else 0
            }
            rows.append(row)
    df_analysis = pd.DataFrame(rows)
    return add_idk_flags(df_analysis)



def summary(df_analysis):
    group_cols = ["dataset", "action", "llm"]
    consistency_cols = ["?A1=A2", "?A1=A3+A4", "?A1>A3", "?A1>A4", "?A3∅A4", "?A4=A1|3", "?A1=A1*", "?A1=A1**","?A1*=A1**"]
    consistency_cols += [c for c in ["?A4=A1-A3", "?A3<=A1(same-context)"] if c in df_analysis]
    jaccard_cols = ["J(A1-A2)", "J(A1-A34)", "J(A3-A4)","J(A4-A1|3)","J(A1-A1*)", "J(A1-A1**)","J(A1*-A1**)"]
    self_contradition_cols = ["?SC(A1=A2)","?SC(A1>A3)","?SC(A1>A4)","?SC(A3∅A4)","?SC(A4=A1|3)"]
    pval_cols = [col for col in df_analysis.columns if col.startswith("p_value_")]
    metric_cols = consistency_cols + jaccard_cols + pval_cols + self_contradition_cols

    df_analysis = add_idk_flags(df_analysis)

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
    
    idk_col = ["idk_A1","idk_A2","idk_A3","idk_A4"]
    df_summary["idk"] = df_summary[idk_col].mean(axis=1)
    return df_summary

def summary_empty_answers(df_analysis):
    """Table 4: replace each observed null answer with an empty set, then score.

    Retain all observed instances, including null answers. Missing observations
    remain unavailable; this does not invent 600 records or missing repeats.
    Use each task's own answer sets and the existing J(empty, empty) = 0 rule.
    Original abstention flags are retained for reporting the IDK rate.
    """
    def answer(cell):
        if is_null_answer(cell):
            return set()
        if isinstance(cell, str):
            if not cell.strip():
                return None
            cell = ast.literal_eval(cell)
        if isinstance(cell, (list, tuple, set)):
            return set(cell)
        if cell is None or pd.isna(cell):
            return None
        raise ValueError(f'Unsupported answer-set cell type: {type(cell).__name__}')

    frame = add_idk_flags(df_analysis[df_analysis.action.ne('wikidata')].copy())
    scores = []
    contextual_actions = {'classification', 'fixing', 'classification-no-idk', 'fixing-no-idk'}
    for row in frame.to_dict('records'):
        a1, a2, a3, a4, repeat, minus_a1, minus_a3 = [
            answer(row.get(name)) for name in ['A1', 'A2', 'A3', 'A4', 'A1*', 'A1**', 'A3_minus']]
        contextual = row['action'] in contextual_actions
        contain_a1 = repeat if contextual else a1
        diff_a1 = minus_a1 if contextual else a1
        diff_a3 = minus_a3 if contextual else a3
        values = {}
        pairs = [
            ('?A1=A2', 'J(A1-A2)', a1, a2, lambda left, right: left == right),
            ('?A1>A3', None, contain_a1, a3, lambda left, right: right.issubset(left)),
            ('?A1>A4', None, diff_a1, a4, lambda left, right: right.issubset(left)),
            ('?A3∅A4', 'J(A3-A4)', diff_a3, a4, lambda left, right: left.isdisjoint(right)),
            ('?A1=A1*', 'J(A1-A1*)', a1, repeat, lambda left, right: left == right),
        ]
        for binary, similarity, left, right, predicate in pairs:
            available = left is not None and right is not None
            values[binary] = int(predicate(left, right)) if available else float('nan')
            if similarity:
                values[similarity] = round(jaccard_similarity(left, right), 4) if available else float('nan')
        if all(a is not None for a in [diff_a1, diff_a3, a4]):
            values.update(partition_metrics(diff_a1, diff_a3, a4))
            values['J(A4-A1|3)'] = round(jaccard_similarity(a4, diff_a1 - diff_a3), 4)
        else:
            for metric in ['?A4=A1|3', '?A4=A1-A3', '?A3<=A1(same-context)', 'J(A4-A1|3)']:
                values[metric] = float('nan')
        scores.append(values)
    if frame.empty:
        raise ValueError('No observed instances available for Table 4')
    scored = pd.DataFrame(scores, index=frame.index)
    frame[scored.columns] = scored
    metrics = list(scored.columns)
    flags = ['idk_A1', 'idk_A2', 'idk_A3', 'idk_A4']

    def aggregate(keys):
        grouped = frame.groupby(keys)
        result = grouped[metrics + flags].mean().round(4)
        result['n_instances'] = grouped.size()
        counts = grouped[metrics].count().rename(columns=lambda name: 'n_scored_' + name)
        result = result.join(counts)
        result['n_repeat'] = grouped['?A1=A1*'].count()
        result = result.reset_index()
        result.loc[result.action.ne('zero-shot'), 'n_repeat'] = 0
        result['idk'] = result[flags].mean(axis=1)
        result['abstention_policy'] = 'individual_empty_set'
        return result

    per_dataset = aggregate(['dataset', 'action', 'llm'])
    overall = aggregate(['action', 'llm'])
    overall['dataset'] = 'overall'
    return pd.concat([per_dataset, overall], ignore_index=True)


def summary_xidk(df_analysis):
    group_cols = ["dataset", "action", "llm"]
    consistency_cols = ["?A1=A2", "?A1=A3+A4", "?A1>A3", "?A1>A4", "?A3∅A4", "?A4=A1|3", "?A1=A1*", "?A1=A1**","?A1*=A1**"]
    consistency_cols += [c for c in ["?A4=A1-A3", "?A3<=A1(same-context)"] if c in df_analysis]
    jaccard_cols = ["J(A1-A2)", "J(A1-A34)", "J(A3-A4)","J(A4-A1|3)","J(A1-A1*)", "J(A1-A1**)" ,"J(A1*-A1**)"]
    self_contradition_cols = ["?SC(A1=A2)","?SC(A1>A3)","?SC(A1>A4)","?SC(A3∅A4)","?SC(A4=A1|3)"]
    pval_cols = [col for col in df_analysis.columns if col.startswith("p_value_")]
    metric_cols = consistency_cols + jaccard_cols + pval_cols + self_contradition_cols

    df_analysis = add_idk_flags(df_analysis)

    empty_cols = [f"idk_{a}" for a in ["A1", "A2", "A3", "A4"]]

    # Compute each metric only on rows where every required answer is present.
    # If any answer used by that metric is IDK/empty, skip the row.
    summary_dict = {col: [] for col in metric_cols + empty_cols}
    grouped = df_analysis.groupby(group_cols)
    for name, group in grouped:
        for col in metric_cols:
            filtered = group.loc[~metric_idk_mask(group, col), col]
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
            filtered = group.loc[~metric_idk_mask(group, col), col]
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

    idk_col = ["idk_A1","idk_A2","idk_A3","idk_A4"]
    df_summary["idk"] = df_summary[idk_col].mean(axis=1)
    return df_summary

if __name__ == "__main__":
    # load dfanalysis 
    root_dir = str(REPO_ROOT)
    analysis_path = os.path.join(root_dir, "output", "analysis.csv")
    df_analysis = pd.read_csv(analysis_path)
    # get summary_xidk
    df_summary_xidk = summary_xidk(df_analysis)
    # save
    output_folder = os.path.join(root_dir, "output")
    os.makedirs(output_folder, exist_ok=True)
    df_summary_xidk.to_csv(os.path.join(output_folder, "summary_xidk.csv"), index=False)

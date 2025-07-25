import os
import json
import pandas as pd
from utils import jaccard_similarity


def load_questions(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
    df["type"] = df["type"].apply(lambda x: str(x) if not pd.isna(x) else "0")
    return df


def load_answers(folder: str, datasets, llms, actions, tasks, questions) -> pd.DataFrame:
    df_answers = pd.DataFrame(columns=["Q_ID", "Q_serie", "action", "task", "dataset", "llm"])

    json_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder)
        for file in files if file.endswith(".json")
    ]

    print(f"JSON files found: {len(json_files)}")

    for file in json_files:
        elements = file.replace("_", "/").replace(".json", "").split("/")
        question = next((q for q in questions if q in elements), None)
        action = next((a for a in actions if a in elements), "zero-shot")
        task = next((t for t in tasks if t in elements), None)
        dataset = next((d for d in datasets if d in elements), None)
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
            df_answers = pd.concat([df_answers, df], ignore_index=True)

    return df_answers


def enrich_answers(df_answers, df_questions):
    df_answers["Question"] = df_answers.apply(
        lambda x: df_questions.at[int(x["Q_ID"]), x["Q_serie"]] if int(x["Q_ID"]) in df_questions.index else None,
        axis=1
    )

    group_keys = ["Q_ID", "Q_serie", "action", "dataset", "llm"]
    df_answers["Answer_serie"] = (
        df_answers.groupby(group_keys).cumcount().apply(lambda x: x + 1)
    )

    df_answers.drop_duplicates(
        subset=["Q_ID", "Q_serie", "action", "task", "dataset", "llm"],
        inplace=True
    )

    df_answers.reset_index(drop=True, inplace=True)
    return df_answers


def analysis(df):
    summaries = []
    group_keys = ["Q_ID", "action", "dataset", "llm"]
    grouped = df[df["Answer_serie"] == 1].groupby(group_keys)

    for keys, group in grouped:
        if set(group["Q_serie"]) >= {"Q1", "Q2", "Q3", "Q4"}:
            A1 = set(group[group["Q_serie"] == "Q1"]["Answer"].values[0])
            A2 = set(group[group["Q_serie"] == "Q2"]["Answer"].values[0])
            A3 = set(group[group["Q_serie"] == "Q3"]["Answer"].values[0])
            A4 = set(group[group["Q_serie"] == "Q4"]["Answer"].values[0])

            q_map = {
                row["Q_serie"]: row["Question"]
                for _, row in group.iterrows()
                if row["Q_serie"] in {"Q1", "Q2", "Q3", "Q4"}
            }

            df_serie2 = df[(df["Answer_serie"] == 2) & (df["Q_serie"] == "Q1")]
            df_serie3 = df[(df["Answer_serie"] == 3) & (df["Q_serie"] == "Q1")]
            for col, val in zip(group_keys, keys):
                df_serie2 = df_serie2[df_serie2[col] == val]
                df_serie3 = df_serie3[df_serie3[col] == val]

            A1_prime = set(df_serie2["Answer"].values[0]) if not df_serie2.empty else set()
            A1_double_prime = set(df_serie3["Answer"].values[0]) if not df_serie3.empty else set()

            consistency = {
                "?A1=A2": int(A1 == A2),
                "?A1=A3+A4": int(A1 == A3.union(A4)),
                "?A1>A3": int(A3.issubset(A1)),
                "?A1>A4": int(A4.issubset(A1)),
                "?A3∅A4": int(A3.isdisjoint(A4)),
            }

            similarities = {
                "J(A1-A2)": round(jaccard_similarity(A1, A2), 4),
                "J(A1-A34)": round(jaccard_similarity(A1, A3.union(A4)), 4),
                "J(A1-A1*)": round(jaccard_similarity(A1, A1_prime), 4),
                "J(A1-A1**)": round(jaccard_similarity(A1, A1_double_prime), 4)
            }

            summary_row = {
                "Q_ID": keys[0], "action": keys[1], "dataset": keys[2], "llm": keys[3],
                **consistency, **similarities,
                "Q1": q_map.get("Q1", ""), "Q2": q_map.get("Q2", ""),
                "Q3": q_map.get("Q3", ""), "Q4": q_map.get("Q4", ""),
                "A1": list(A1), "A2": list(A2), "A3": list(A3), "A4": list(A4),
                "A1*": list(A1_prime), "A1**": list(A1_double_prime)
            }
            summaries.append(summary_row)

    return pd.DataFrame(summaries)


def summary(df_analysis):
    group_cols = ["dataset", "action", "llm"]
    consistency_cols = ["?A1=A2", "?A1=A3+A4", "?A1>A3", "?A1>A4", "?A3∅A4"]
    jaccard_cols = ["J(A1-A2)", "J(A1-A34)", "J(A1-A1*)", "J(A1-A1**)"]
    pval_cols = [col for col in df_analysis.columns if col.startswith("p_value_")]
    metric_cols = consistency_cols + jaccard_cols + pval_cols

    for a in ["A1", "A2", "A3", "A4"]:
        df_analysis[f"{a}_empty_ratio"] = df_analysis[a].apply(lambda x: int(isinstance(x, list) and len(x) == 0))

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

    return df_summary


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__name__))
    question_path = root_dir + "/data/Dataset/en/spinach.tsv"
    df_questions = load_questions(question_path)

    df_answers = load_answers(
        folder=root_dir + "/data/answers/",
        datasets=["spinach"],
        llms=['gpt-4.1-2025-04-14', 'gpt-4.1-mini-2025-04-14', 'gpt-4.1-nano-2025-04-14', 'gpt-4o', "o3"],
        actions=["fixing", "classification", "wikidata"],
        tasks=['equal', 'sup-sub', "minus"],
        questions=["Q1", "Q2", "Q3", "Q4"]
    )

    df_answers = enrich_answers(df_answers, df_questions)
    df_analysis = analysis(df_answers)
    df_summary = summary(df_analysis)

    # Define output folder path
    output_folder = os.path.join(root_dir, "data", "Analysis")
    os.makedirs(output_folder, exist_ok=True)

    # Save results
    df_analysis.to_csv(os.path.join(output_folder, "analysis.csv"), index=False)
    df_summary.to_csv(os.path.join(output_folder, "summary.csv"), index=False)

    print("✅ Analysis and summary saved to:", output_folder)

    # Optional: Save as Parquet (if needed)
    try:
        df_analysis.to_parquet(os.path.join(output_folder, "analysis.parquet"), index=False)
        df_summary.to_parquet(os.path.join(output_folder, "summary.parquet"), index=False)
    except ImportError:
        print("⚠️ Skipped Parquet export — install `pyarrow` or `fastparquet` to enable it.")

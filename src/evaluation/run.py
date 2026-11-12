"""Aggregate evaluation entry point (saved responses only)."""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.core.eval_tool import load_all_questions, load_answers, enrich_answers, analysis, summary, summary_xidk
from src.evaluation.core.eval_relation import load_relations, load_relation_clf, relation_summary, merge_relations_by_action, update_summary_by_relations
from src.evaluation.statistics.eval_pvalue import compute_pvals, p_value_matrixs
from src.evaluation.tables.split_heatmap import dataframe_to_heatmap_csvs
from src.evaluation.tables.split import split
from src.evaluation.paths import REPO_ROOT
from src.evaluation.core.summary_idk_worst import summary_idk_worst, impute_worst_case

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    # This dependency was already absent before the directory reorganization.
    # Check before evaluating or writing outputs, rather than silently skipping it.
    try:
        from src.evaluation.reviewer_abstention_audit import regenerate as regenerate_paper_tables
    except ModuleNotFoundError as exc:
        if exc.name != "src.evaluation.reviewer_abstention_audit":
            raise
        raise RuntimeError(
            "Aggregate evaluation requires the missing reviewer_abstention_audit module. "
            "Restore it before running this pipeline; table exports remain available separately."
        ) from exc
    root_dir = str(REPO_ROOT)
    print("Root directory:", root_dir)
    datasets=["spinach", "qawiki",'synthetic','LC-QuAD']

    llm_path = f"{root_dir}/data/llm_info.json"
    with open(llm_path, "r", encoding="utf-8") as f:
        llms = json.load(f).keys()
    actions = [
        "fixing",
        "classification",
        "star",
        "zero-shot",
        "chain-of-thought",
        "zero-shot-no-idk",
        "fixing-no-idk",
        "classification-no-idk",
    ]
    tasks = ['equal', 'sup-sub', "minus"]
    languages = ['en']


    # Define output folder path
    output_folder = os.path.join(root_dir, "output")
    os.makedirs(output_folder, exist_ok=True)

    # Relation Classification
    df_relation = load_relations(root_dir + "/data/answers/zero-shot/", datasets, llms)
    df_relation_clf = load_relation_clf(root_dir + "/data/answers/CtE/", datasets, llms, tasks)
    # From your df_relation:
    df_relation_summery = relation_summary(df_relation, include_overall=True, round_digits=4)

    # Save if you want:
    time = datetime.datetime.now()
    relation_file_format = time.strftime("relations_%Y-%m-%d_%H-%M.csv")
    summary_file_format = time.strftime("relation_summary_%Y-%m-%d_%H-%M.csv")
    summary_file_format_excel = time.strftime("relation_summary_%Y-%m-%d_%H-%M.xlsx")

    # df_relation.to_csv(os.path.join(output_folder, relation_file_format), index=False)
    df_relation_summery.to_csv(os.path.join(output_folder, "relation_summery.csv"), index=False)
    # df_relation_summery.to_excel(os.path.join(output_folder, summary_file_format_excel), index=False)


    df_questions = load_all_questions(root_dir, datasets, languages)
    df_answers = load_answers(
        folder=root_dir + "/data/answers/",
        datasets = datasets,
        llms=llms,
        actions=actions,
        tasks=tasks,
        languages=languages,
        questions=["Q1", "Q2", "Q3", "Q4"]
    )

    df_answers = enrich_answers(df_answers, df_questions)
    df_analysis = analysis(df_answers)
    df_analysis = merge_relations_by_action(df_analysis, df_relation, df_relation_clf)
    
    
    # Save results
    # analysis_file_format = time.strftime("analysis_%Y-%m-%d_%H-%M.csv")
    df_analysis.to_csv(os.path.join(output_folder, "analysis.csv"), index=False)

    regenerate_paper_tables(
        Path(root_dir) / "data" / "answers",
        Path(llm_path),
        Path(output_folder) / "reviewer_abstention",
        questions_root=Path(root_dir) / "data" / "ASCB" / "en",
    )

    # p-values
    df_scored = impute_worst_case(df_analysis)
    df_pval = compute_pvals(df_scored)

    df_summary = summary(df_analysis)
    
    df_summary = update_summary_by_relations(df_analysis, df_summary, task="zero-shot")
    df_summary = update_summary_by_relations(df_analysis, df_summary, task="classification")
    df_summary = df_summary.merge(compute_pvals(df_analysis), on=["dataset","llm","action"], how="left")
    df_summary.to_csv(os.path.join(output_folder, "summary.csv"), index=False)

    # summary_file_format = time.strftime("summary_%Y-%m-%d_%H-%M.csv")
    # summary_file_format_excel = time.strftime("summary_%Y-%m-%d_%H-%M.xlsx")


    df_summary_xidk = summary_xidk(df_analysis)
    df_summary_xidk = update_summary_by_relations(df_analysis, df_summary_xidk, task="zero-shot")
    df_summary_xidk = update_summary_by_relations(df_analysis, df_summary_xidk, task="classification")
    # This legacy diagnostic excludes metric instances and is not paper
    # Table 4. Paper Table 4 is regenerated above from individual answers.

    
    df_summary_xidk.to_csv(os.path.join(output_folder, "summary_xidk.csv"), index=False)
    # df_summary.to_excel(os.path.join(output_folder, summary_file_format_excel), index=False)

    summary_actions = [action for action in actions if action != "star"]
    split(df_summary, "summary", actions=summary_actions)
    split(df_summary_xidk, "summary_xidk", actions=summary_actions)
    df_worst = summary_idk_worst(df_analysis).merge(df_pval, on=["dataset", "llm", "action"], how="left")
    df_worst.to_csv(os.path.join(output_folder, "summary_idk_worst.csv"), index=False)
    split(df_worst, "summary_idk_worst", actions=summary_actions)
    df_pvalue = p_value_matrixs(df_scored, summary_actions)
    p_value_matrixs_file_format = time.strftime("p_value_matrices_%Y-%m-%d_%H-%M.csv")
    df_pvalue.to_csv(os.path.join(output_folder, "p_value_matrices.csv"), index=False)
    dataframe_to_heatmap_csvs(df_pvalue, output_folder)
    from src.evaluation.tables.export_supplementary_tables import build_tables, write_tables
    write_tables(build_tables(df_analysis, llms), output_folder + "/tables_9_11", source="raw")
    print("Analysis and summary saved to:", output_folder)


if __name__ == "__main__":
    main()

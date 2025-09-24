from eval_tool import load_all_questions, load_answers, enrich_answers, analysis, summary
from eval_relation import load_relations, load_relation_clf, relation_summary, merge_relations_by_action, update_summary_by_relations
from eval_pvalue import compute_pvals, p_value_matrixs
from split_heatmap import dataframe_to_heatmap_csvs
from split import split
import os
import json
import datetime
import pandas as pd

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__name__))
    print("Root directory:", root_dir)
    datasets=["spinach", "qawiki",'synthetic','lc-quad']

    llm_path = f"{root_dir}/data/llm_info.json"
    with open(llm_path, "r", encoding="utf-8") as f:
        llms = json.load(f).keys()
    actions = ["wikidata", "fixing", "classification","star","zero-shot"]
    tasks = ['equal', 'sup-sub', "minus"]
    languages = ['en']


    # Define output folder path
    output_folder = os.path.join(root_dir, "output")
    os.makedirs(output_folder, exist_ok=True)

    df_analysis = pd.read_csv(root_dir + "/output/analysis.csv")

    # df_analysis = merge_relations_by_action(df_analysis, df_relation, df_relation_clf)
    # Save results
    # analysis_file_format = time.strftime("analysis_%Y-%m-%d_%H-%M.csv")
    df_analysis.to_csv(os.path.join(output_folder, "analysis.csv"), index=False)

    # p-values
    df_pval = compute_pvals(df_analysis)

    df_summary = summary(df_analysis)

    

    df_summary = update_summary_by_relations(df_analysis, df_summary, task="zero-shot")
    df_summary = update_summary_by_relations(df_analysis, df_summary, task="classification")
    df_summary = df_summary.merge(df_pval, on=["dataset","llm","action"], how="left")

    # summary_file_format = time.strftime("summary_%Y-%m-%d_%H-%M.csv")
    # summary_file_format_excel = time.strftime("summary_%Y-%m-%d_%H-%M.xlsx")


    df_summary.to_csv(os.path.join(output_folder, "summary.csv"), index=False)
    # df_summary.to_excel(os.path.join(output_folder, summary_file_format_excel), index=False)

    split(df_summary, "summary")
    
    df_pvalue = p_value_matrixs(df_analysis, actions)
    # p_value_matrixs_file_format = time.strftime("p_value_matrices_%Y-%m-%d_%H-%M.csv")
    df_pvalue.to_csv(os.path.join(output_folder, "p_value_matrices.csv"), index=False)
    dataframe_to_heatmap_csvs(df_pvalue, output_folder)
    print("Analysis and summary saved to:", output_folder)
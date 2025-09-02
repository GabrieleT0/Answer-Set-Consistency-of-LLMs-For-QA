from eval_tool import load_all_questions, load_answers, enrich_answers, analysis, summary, load_relations, relation_summary
import os
import datetime

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__name__))
    print("Root directory:", root_dir)
    datasets=["spinach", "qawiki",'synthetic']
    llms = ['gpt-4.1-2025-04-14', 'gpt-4.1-mini-2025-04-14', 'gpt-4.1-nano-2025-04-14', 
            'gpt-4o','o3','gpt-5-nano',"gpt-5-mini","gpt-5",
            "gemini-2.0-flash","gemini-2.5-flash","gemini-2.5-pro",
            "grok-3-mini","deepseek-chat","deepseek-reasoner","llama3.1:8b","llama3.3:70b"]
    actions = ["fixing", "classification", "wikidata"]
    tasks = ['equal', 'sup-sub', "minus"]
    languages = ['en']


    # Define output folder path
    output_folder = os.path.join(root_dir, "output")
    os.makedirs(output_folder, exist_ok=True)

    # Relation Classification
    df_relation = load_relations(root_dir + "/data/answers/zero-shot/", datasets, llms)
    # df_relation_summery = relation_summary(df_relation)
    # From your df_relation:
    df_relation_summery = relation_summary(df_relation, include_overall=True, round_digits=4)

    # Save if you want:
    relation_file_format = datetime.datetime.now().strftime("relations_%Y-%m-%d_%H-%M.csv")
    summary_file_format = datetime.datetime.now().strftime("relation_summary_%Y-%m-%d_%H-%M.csv")
    summary_file_format_excel = datetime.datetime.now().strftime("relation_summary_%Y-%m-%d_%H-%M.xlsx")

    df_relation.to_csv(os.path.join(output_folder, relation_file_format), index=False)
    df_relation_summery.to_csv(os.path.join(output_folder, summary_file_format), index=False)
    df_relation_summery.to_excel(os.path.join(output_folder, summary_file_format_excel), index=False)


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
    df_summary = summary(df_analysis)

    # Save results
    analysis_file_format = datetime.datetime.now().strftime("analysis_%Y-%m-%d_%H-%M.csv")
    summary_file_format = datetime.datetime.now().strftime("summary_%Y-%m-%d_%H-%M.csv")
    summary_file_format_excel = datetime.datetime.now().strftime("summary_%Y-%m-%d_%H-%M.xlsx")
    df_analysis.to_csv(os.path.join(output_folder, analysis_file_format), index=False)
    df_summary.to_csv(os.path.join(output_folder, summary_file_format), index=False)
    df_summary.to_excel(os.path.join(output_folder, summary_file_format_excel), index=False)

    print("Analysis and summary saved to:", output_folder)



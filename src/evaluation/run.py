"""Aggregate evaluation entry point (saved responses only)."""
import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.core.eval_tool import load_all_questions, load_answers, enrich_answers, analysis, summary, summary_empty_answers
from src.evaluation.core.eval_relation import load_relations, load_relation_clf, relation_summary, merge_relations_by_action, update_summary_by_relations
from src.evaluation.statistics.eval_pvalue import compute_pvals, p_value_matrixs
from src.evaluation.paths import (
    REPO_ROOT,
    create_run_directory,
    display_path,
    finish_run_manifest,
    start_run_manifest,
)
from src.evaluation.core.summary_idk_worst import summary_idk_worst, impute_worst_case, ABSTENTION_RECOGNITIONS

def _run(args, output_folder):
    recognition = args.recognition
    root_dir = str(REPO_ROOT)
    print("Repository root: .")
    datasets=["spinach", "qawiki",'synthetic','LC-QuAD']

    llm_path = f"{root_dir}/data/llm_info.json"
    with open(llm_path, "r", encoding="utf-8") as f:
        llm_info = json.load(f)
    llms = llm_info.keys()
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


    os.makedirs(output_folder, exist_ok=True)

    # Relation Classification
    df_relation = load_relations(root_dir + "/data/answers/zero-shot/", datasets, llms)
    df_relation_clf = load_relation_clf(root_dir + "/data/answers/CtE/", datasets, llms, tasks)
    # From your df_relation:
    df_relation_summery = relation_summary(df_relation, include_overall=True, round_digits=4)

    df_relation_summery.to_csv(os.path.join(output_folder, "relation_summery.csv"), index=False)


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
    df_analysis = analysis(df_answers, recognition)
    df_analysis = merge_relations_by_action(df_analysis, df_relation, df_relation_clf, recognition)
    
    
    df_analysis.to_csv(os.path.join(output_folder, "analysis.csv"), index=False)

    # Table 4 retains instances and replaces individual null answers by empty
    # sets. Keep the old exclusion-based diagnostic as summary_xidk below.
    df_summary_empty = summary_empty_answers(df_analysis, recognition)
    df_summary_empty.to_csv(os.path.join(output_folder, "summary_empty_answers.csv"), index=False)

    # p-values
    df_scored = impute_worst_case(df_analysis, recognition)
    df_pval = compute_pvals(df_scored)

    df_summary = summary(df_analysis, recognition)
    
    df_summary = update_summary_by_relations(df_analysis, df_summary, task="zero-shot")
    df_summary = update_summary_by_relations(df_analysis, df_summary, task="classification")
    df_summary = df_summary.merge(df_pval, on=["dataset","llm","action"], how="left")
    df_summary.to_csv(os.path.join(output_folder, "summary.csv"), index=False)

    summary_actions = [action for action in actions if action != "star"]
    df_worst = summary_idk_worst(df_analysis, recognition).merge(df_pval, on=["dataset", "llm", "action"], how="left")
    df_worst.to_csv(os.path.join(output_folder, "summary_idk_worst.csv"), index=False)
    df_pvalue = p_value_matrixs(df_scored, summary_actions)
    df_pvalue.to_csv(os.path.join(output_folder, "p_value_matrices.csv"), index=False)

    # Export all manuscript tables from this exact in-memory analysis snapshot.
    # Keeping them in one subdirectory avoids duplicate standalone runs and
    # makes the data/configuration version unambiguous.
    from src.evaluation.tables.export_paper_results import TABLES, export
    export(
        REPO_ROOT,
        Path(output_folder) / "paper_results",
        list(TABLES),
        recognition,
        frame=df_analysis,
        relations=df_relation,
        info=llm_info,
    )
    print("Analysis and summary saved to:", display_path(output_folder))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recognition', choices=ABSTENTION_RECOGNITIONS, default='robust')
    parser.add_argument(
        '--output-dir', type=Path,
        help='Exact version directory. The default is timestamped under output/evaluation_runs/versions',
    )
    args = parser.parse_args()
    if args.output_dir is None:
        output_dir, run_id = create_run_directory(
            REPO_ROOT / 'output' / 'evaluation_runs' / 'versions',
            'evaluation-version', args.recognition,
        )
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = output_dir.name
    input_files = [path for path in (REPO_ROOT / 'data').rglob('*')
                   if path.is_file() and path.suffix in {'.json', '.csv'}]
    manifest = start_run_manifest(
        output_dir, run_id=run_id, run_type='evaluation-version',
        config={
            'recognition': args.recognition,
            'datasets': ['spinach', 'qawiki', 'synthetic', 'LC-QuAD'],
            'actions': [
                'fixing', 'classification', 'star', 'zero-shot', 'chain-of-thought',
                'zero-shot-no-idk', 'fixing-no-idk', 'classification-no-idk',
            ],
            'tasks': ['equal', 'sup-sub', 'minus'],
            'languages': ['en'],
            'paper_results': [
                'main', 'filtered', 'classification', 'significance',
                'correlation', 'cfr', 'correct', 'incorrect',
            ],
            'paper_results_directory': 'paper_results',
        }, recognition=args.recognition,
        input_files=input_files, input_root=REPO_ROOT,
        entity_normalization='none',
    )
    try:
        _run(args, str(output_dir))
    except Exception as error:
        finish_run_manifest(output_dir, manifest, status='failed', error=repr(error))
        raise
    finish_run_manifest(output_dir, manifest)
    print('Run manifest:', display_path(output_dir / 'manifest.json'))


if __name__ == "__main__":
    main()

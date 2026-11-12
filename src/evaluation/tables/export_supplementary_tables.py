"""Rebuild Tables 9–11 from saved model outputs, or export the archived numbers.

No model/API calls. See --help for the two explicit data sources.
"""

from src.evaluation.paths import REPO_ROOT
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.evaluation.core.eval_tool import load_all_questions, load_answers, enrich_answers, analysis
from src.evaluation.core.eval_relation import load_relations, load_relation_clf, merge_relations_by_action, GT
from src.evaluation.core.summary_idk_worst import impute_worst_case

ROOT = REPO_ROOT
# relation label column, binary score, similarity score, contradiction-free score
SPECS = [
    ("R(1-2)", "?A1=A2", "J(A1-A2)", "?SC(A1=A2)"),
    ("R(1-3)", "?A1>A3", None, "?SC(A1>A3)"),
    ("R(1-4)", "?A1>A4", None, "?SC(A1>A4)"),
    ("R(3-4)", "?A3∅A4", "J(A3-A4)", "?SC(A3∅A4)"),
    ("R(1-34)", "?A4=A1|3", "J(A4-A1|3)", "?SC(A4=A1|3)"),
]
BINARY = [s[1] for s in SPECS]
SIMILARITIES = ["J(A1-A2)", "J(A3-A4)", "J(A4-A1|3)"]
LABELS = {9: "tab:cRi_by_id_act", 10: "tab:cRi+", 11: "tab:cRi-"}
ACTION_NAMES = {"zero-shot": "Base", "classification": "CtE"}
DISPLAY = {"deepseek-chat": "DeepSeek-V3.1", "deepseek-reasoner": "DeepSeek-reasoner",
           "o3": "GPT-o3", "gpt-oss:20b": "GPT-oss-20b", "llama3.1:8b": "Llama-3.1-8b",
           "llama3.1:70b": "Llama-3.1-70b", "mistral-small:24b": "Mistral-small:24b"}


def rebuild_analysis(root, models):
    for directory in [root / "data/answers/zero-shot", root / "data/answers/CtE"]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing answer/classification directory: {directory}")
    datasets = ["LC-QuAD", "qawiki", "spinach", "synthetic"]
    tasks = ["equal", "sup-sub", "minus"]
    questions = load_all_questions(str(root), datasets, ["en"])
    answers = load_answers(str(root / "data/answers"), datasets, models,
                           list(ACTION_NAMES), tasks, ["en"], ["Q1", "Q2", "Q3", "Q4"])
    frame = analysis(enrich_answers(answers, questions))
    base_relations = load_relations(str(root / "data/answers/zero-shot"), datasets, models)
    cte_relations = load_relation_clf(str(root / "data/answers/CtE"), datasets, models, tasks)
    return merge_relations_by_action(frame, base_relations, cte_relations)


def build_tables(frame, models):
    """Long-form percent scores, with denominators for every table cell."""
    frame = impute_worst_case(frame)
    rows = []
    for model in models:
        for action in ACTION_NAMES:
            group = frame[frame.llm.eq(model) & frame.action.eq(action)]
            for relation, binary, similarity, cfr in SPECS:
                observed = group[relation].notna()
                for table in [9, 10, 11]:
                    mask = observed if table == 9 else observed & (group[relation].eq(GT[relation]) if table == 10 else group[relation].ne(GT[relation]))
                    selected = group.loc[mask]
                    metrics = [cfr] if table == 9 else [binary] + ([similarity] if similarity else [])
                    for metric in metrics:
                        values = pd.to_numeric(selected[metric], errors="coerce")
                        rows.append(dict(table=table, llm=model, action=action, metric=metric,
                                         score_percent=100*values.mean(), n_total=len(group),
                                         n_classified=int(observed.sum()), n_selected=len(selected),
                                         n_scored=int(values.notna().sum()), source_column=metric))
    return pd.DataFrame(rows)


def archived_tables(summary, models):
    """Reproduce published numbers, exposing their original union-column mapping."""
    summary = summary[summary.dataset.eq("overall")].set_index(["llm", "action"], verify_integrity=True)
    rows = []
    for model in models:
        for action in ACTION_NAMES:
            row = summary.loc[(model, action)]
            for table in [9, 10, 11]:
                metrics = [s[3] for s in SPECS] if table == 9 else BINARY + SIMILARITIES
                for metric in metrics:
                    column = metric
                    if table != 9:
                        # These are the columns the original notebook actually used.
                        column = {"?A4=A1|3": "?A1=A3+A4", "J(A1-A2)": "J(1-2)",
                                  "J(A3-A4)": "J(3-4)", "J(A4-A1|3)": "J(1-34)"}.get(metric, metric)
                        sign = "+" if table == 10 else "-"
                        column += sign if column.startswith("J(") else f"({sign})"
                    rows.append(dict(table=table, llm=model, action=action, metric=metric,
                                     score_percent=100*row[column], n_total=None, n_classified=None,
                                     n_selected=None, n_scored=None, source_column=column))
    return pd.DataFrame(rows)


def render_table(frame, number, models, source):
    metrics = [s[3] for s in SPECS] if number == 9 else BINARY + SIMILARITIES
    headers = [r"\fcfr{"+r+"}" for r in [r"\rone",r"\rtwo",r"\rthree",r"\rfour",r"\rfive"]] if number == 9 else [r"\fconr{"+r+"}" for r in [r"\rone",r"\rtwo",r"\rthree",r"\rfour",r"\rfive"]]+[r"\fjacs{"+r+"}" for r in [r"\rone",r"\rfour",r"\rfive"]]
    caption = ("Contradiction-free rates for the relations predicted by each model." if number == 9 else
               "Consistency and similarity when the predicted relation is "+("correct." if number == 10 else "incorrect."))
    caption += " Values use a 0--100 scale; -- denotes no observations."
    if source == "archived":
        caption += " Historical export: Table 9 retains the archived gold-predicate agreement scores; the final columns in Tables 10--11 use union equality and union similarity."
    else:
        caption += (" Explicit classification abstentions and null answers score zero." if number == 9 else
                    " Missing classifications are excluded; explicit classification abstentions belong to the incorrect subset. Null answers receive the main-table worst score. The final binary score is joint partition consistency; its similarity diagnoses the difference component.")
    lines = [r"\begin{table}[!t]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\caption{"+caption+"}",
             r"\begin{tabular}{ll"+"r"*len(metrics)+"}", r"\toprule",
             r"\textbf{Model} & \textbf{Str} & "+" & ".join("$"+h+"$" for h in headers)+r" \\", r"\midrule"]
    lookup = frame[frame.table.eq(number)].set_index(["llm", "action", "metric"])
    for model in models:
        name = DISPLAY.get(model, re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model))
        for prefix, brand in [("gpt-", "GPT-"), ("gemini-", "Gemini-"), ("grok-", "Grok-")]:
            if name.startswith(prefix):
                name = brand + name[len(prefix):]
                break
        for action in ACTION_NAMES:
            values = [lookup.loc[(model, action, metric), "score_percent"] for metric in metrics]
            lines.append(name+" & "+ACTION_NAMES[action]+" & "+" & ".join("--" if pd.isna(v) else f"{v:.2f}" for v in values)+r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\label{"+LABELS[number]+"}", r"\end{table}"]
    return "\n".join(lines)


def write_tables(frame, output, source="raw", models=None, manuscript=None):
    if manuscript and source != "raw":
        raise ValueError("Only corrected raw-data tables may replace the manuscript tables")
    models = list(models) if models is not None else list(frame.llm.unique())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    frame["source"] = source
    frame.to_csv(output.with_suffix(".csv"), index=False)
    blocks = {number: render_table(frame, number, models, source) for number in LABELS}
    output.with_suffix(".tex").write_text("\n\n".join(blocks.values())+"\n")
    if manuscript:
        if source != "raw":
            raise ValueError("Only corrected raw-data tables may replace the manuscript tables")
        path = Path(manuscript)
        text = path.read_text()
        # Resolve and validate all ranges before writing the manuscript once.
        replacements = []
        for number, label in LABELS.items():
            marker = r"\label{"+label+"}"
            if text.count(marker) != 1:
                raise ValueError(f"Expected one {marker}")
            position = text.index(marker)
            start = text.rfind(r"\begin{table}", 0, position)
            end = text.index(r"\end{table}", position)+len(r"\end{table}")
            if start < 0:
                raise ValueError(f"Missing table environment for {label}")
            replacements.append((start,end,blocks[number]))
        for start,end,block in sorted(replacements, reverse=True):
            text = text[:start]+block+text[end:]
        path.write_text(text)
    print(f"Wrote {output.with_suffix('.csv')} and {output.with_suffix('.tex')} ({source})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["raw", "archived"], default="raw")
    parser.add_argument("--output", type=Path, help="Output stem; archived exports use a separate default")
    parser.add_argument("--update-manuscript", type=Path, help="Replace only Tables 9–11 in this appendix source")
    args = parser.parse_args()
    if args.source == "archived" and args.update_manuscript:
        parser.error("Archived exports cannot update the manuscript")
    info = json.loads((ROOT/"data/llm_info.json").read_text())
    models = sorted(info, key=lambda model: info[model]["ID"])
    if args.source == "raw":
        tables = build_tables(rebuild_analysis(ROOT, models), models)
    else:
        tables = archived_tables(pd.read_csv(ROOT/"data/evaluation_results/summary_filtered_idk_filtered.csv"), models)
    output = args.output or ROOT / "output" / ("tables_9_11_archived" if args.source == "archived" else "tables_9_11")
    write_tables(tables, output, args.source, models, args.update_manuscript)


if __name__ == "__main__":
    main()

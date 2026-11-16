"""Score paired relation-aware ablation experiments.

    venv/bin/python -m src.evaluation.analyses.eval_relation_aware_ablation

Writes per_item.csv, summary.csv, and manifest.json to one timestamped run
directory by default. A relevant idk/empty answer is scored as inconsistent
under the selected recognition rule. The primary partition score jointly
requires the difference identity and same-context containment; both components
are also reported as diagnostics. The correction comparison is reported both
on initially inconsistent dialogues and on all dialogues (unchanged initial
answers for the latter's initially consistent cases).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from src.evaluation.core.summary_idk_worst import ABSTENTION_RECOGNITIONS, is_null_answer
from src.evaluation.paths import (
    create_run_directory,
    display_path,
    finish_run_manifest,
    start_run_manifest,
)
from src.get_answers.relation_aware_ablation import ROOT, TASKS


PRIMARY_METRICS = {
    "equal": ("E1,2",),
    "sup-sub": ("N3,1",),
    "minus": ("N4,1", "D3,4", "E4,1\\3"),
}
DIAGNOSTIC_METRICS = {
    "equal": (),
    "sup-sub": (),
    "minus": ("E4_difference_only", "N3,1_same_context"),
}
METRICS = {
    task: PRIMARY_METRICS[task] + DIAGNOSTIC_METRICS[task]
    for task in PRIMARY_METRICS
}


def scores(task: str, answers: dict[str, list[str]], recognition: str = "standard") -> dict[str, int]:
    a = {q: set(answers[q]) for q in TASKS[task]}

    def valid(*questions: str) -> bool:
        return not any(is_null_answer(answers[q], recognition) for q in questions)

    if task == "equal":
        return {"E1,2": int(valid("Q1", "Q2") and a["Q1"] == a["Q2"])}
    if task == "sup-sub":
        return {"N3,1": int(valid("Q1", "Q3") and a["Q3"].issubset(a["Q1"]))}
    available = valid("Q1", "Q3", "Q4")
    difference = available and a["Q4"] == a["Q1"] - a["Q3"]
    containment = available and a["Q3"].issubset(a["Q1"])
    return {
        "N4,1": int(valid("Q1", "Q4") and a["Q4"].issubset(a["Q1"])),
        "D3,4": int(valid("Q3", "Q4") and a["Q3"].isdisjoint(a["Q4"])),
        # The paper's primary partition score is joint: the difference
        # identity and same-context containment must both hold.
        "E4,1\\3": int(difference and containment),
        "E4_difference_only": int(difference),
        "N3,1_same_context": int(containment),
    }


def updated_answers(task: str, initial: dict, branch: dict | None) -> dict[str, list[str]]:
    answers = {q: list(values) for q, values in initial["answers"].items()}
    if branch is not None:
        answers[TASKS[task][-1]] = branch["answer"]
    return answers


def metric_abstention(task: str, metric: str, answers: dict[str, list[str]],
                      recognition: str = "standard") -> int:
    relevant = {
        "E1,2": ("Q1", "Q2"),
        "N3,1": ("Q1", "Q3"),
        "N4,1": ("Q1", "Q4"),
        "D3,4": ("Q3", "Q4"),
        "E4,1\\3": ("Q1", "Q3", "Q4"),
        "E4_difference_only": ("Q1", "Q3", "Q4"),
        "N3,1_same_context": ("Q1", "Q3"),
    }[metric]
    return int(any(is_null_answer(answers[q], recognition) for q in relevant))


def pair_rows(comparison: str, dataset: str, qid: str, task: str,
              left: dict[str, list[str]], right: dict[str, list[str]],
              initial_inconsistent: bool, recognition: str = "standard") -> list[dict]:
    left_scores = scores(task, left, recognition)
    right_scores = scores(task, right, recognition)
    return [
        {
            "comparison": comparison,
            "dataset": dataset,
            "question_id": qid,
            "task": task,
            "metric": metric,
            "initial_inconsistent": int(initial_inconsistent),
            "left": left_scores[metric],
            "right": right_scores[metric],
            "left_abstention": metric_abstention(task, metric, left, recognition),
            "right_abstention": metric_abstention(task, metric, right, recognition),
        }
        for metric in METRICS[task]
    ]


def read_rows(input_dir: Path, model: str | None = None,
              recognition: str = "standard") -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped = 0
    paths = sorted(input_dir.glob("**/*.json"))
    if not paths:
        raise ValueError(f"No checkpoint JSON files found under {input_dir}")
    models = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        if "records" not in checkpoint or "task" not in checkpoint:
            continue
        if model is not None and checkpoint["model"] != model:
            continue
        models.add(checkpoint["model"])
        dataset, task = checkpoint["dataset"], checkpoint["task"]
        for record in checkpoint["records"].values():
            initial = record.get("initial")
            qid = str(record["question_id"])
            if "cte" in record and "sequential" in record:
                rows.extend(pair_rows(
                    "sequential_vs_cte", dataset, qid, task,
                    record["sequential"]["answers"], record["cte"]["answers"],
                    bool(initial and initial["eligible_for_revision"]),
                    recognition,
                ))
            else:
                skipped += 1
            if not initial:
                skipped += 1
                continue
            eligible = initial["eligible_for_revision"]
            if eligible and ("recheck" not in record or "relation" not in record):
                skipped += 1
                continue
            recheck = updated_answers(task, initial, record.get("recheck"))
            relation = updated_answers(task, initial, record.get("relation"))
            rows.extend(pair_rows(
                "recheck_vs_relation_all", dataset, qid, task,
                recheck, relation, eligible,
                recognition,
            ))
            if eligible:
                rows.extend(pair_rows(
                    "recheck_vs_relation_failed", dataset, qid, task,
                    recheck, relation, True,
                    recognition,
                ))
    if not rows:
        raise ValueError("No complete paired records found")
    if model is None and len(models) > 1:
        raise ValueError("Multiple models found; pass --model or use a model-specific input directory")
    return rows, skipped


def mcnemar_exact(left: list[int], right: list[int]) -> tuple[int, int, float]:
    worsened = sum(a == 1 and b == 0 for a, b in zip(left, right))
    improved = sum(a == 0 and b == 1 for a, b in zip(left, right))
    n = worsened + improved
    if not n:
        return worsened, improved, 1.0
    tail = sum(math.comb(n, k) for k in range(min(worsened, improved) + 1))
    return worsened, improved, min(1.0, 2 * tail / (1 << n))


def paired_ci(deltas: list[int], *, samples: int = 2000, seed: int = 2027) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    boot = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    )
    return boot[int(0.025 * samples)], boot[int(0.975 * samples) - 1]


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["comparison"], row["dataset"], row["metric"])].append(row)
        grouped[(row["comparison"], "overall", row["metric"])].append(row)
    summaries = []
    for (comparison, dataset, metric), group in sorted(grouped.items()):
        left = [int(r["left"]) for r in group]
        right = [int(r["right"]) for r in group]
        deltas = [b - a for a, b in zip(left, right)]
        worsened, improved, p = mcnemar_exact(left, right)
        lo, hi = paired_ci(deltas)
        summaries.append({
            "comparison": comparison,
            "dataset": dataset,
            "metric": metric,
            "n_pairs": len(group),
            "left_rate": sum(left) / len(group),
            "right_rate": sum(right) / len(group),
            "delta_pp": 100 * sum(deltas) / len(group),
            "ci_low_pp": 100 * lo,
            "ci_high_pp": 100 * hi,
            "worsened": worsened,
            "improved": improved,
            "mcnemar_p_two_sided": p,
            "left_abstention_rate": sum(int(r["left_abstention"]) for r in group) / len(group),
            "right_abstention_rate": sum(int(r["right_abstention"]) for r in group) / len(group),
        })
    by_setting: dict[tuple[str, str], list[dict]] = defaultdict(list)
    primary = {metric for metrics in PRIMARY_METRICS.values() for metric in metrics}
    for entry in summaries:
        if entry["metric"] in primary:
            by_setting[(entry["comparison"], entry["dataset"])].append(entry)
    for (comparison, dataset), entries in sorted(by_setting.items()):
        if {entry["metric"] for entry in entries} != primary:
            continue
        summaries.append({
            "comparison": comparison,
            "dataset": dataset,
            "metric": "CON_mean_5",
            "n_pairs": "",
            "left_rate": sum(entry["left_rate"] for entry in entries) / 5,
            "right_rate": sum(entry["right_rate"] for entry in entries) / 5,
            "delta_pp": sum(entry["delta_pp"] for entry in entries) / 5,
            "ci_low_pp": "",
            "ci_high_pp": "",
            "worsened": "",
            "improved": "",
            "mcnemar_p_two_sided": "",
            "left_abstention_rate": sum(entry["left_abstention_rate"] for entry in entries) / 5,
            "right_abstention_rate": sum(entry["right_abstention_rate"] for entry in entries) / 5,
        })
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "output" / "relation_aware_ablation")
    parser.add_argument("--model", help="Required if input-dir contains multiple models")
    parser.add_argument("--recognition", choices=ABSTENTION_RECOGNITIONS, default="robust")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Exact run directory. By default a timestamped directory is created under output/evaluation_runs/relation_aware_ablation",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_paths = sorted(args.input_dir.glob("**/*.json"))
    if args.output_dir is None:
        label = args.model or args.input_dir.name
        output_dir, run_id = create_run_directory(
            ROOT / "output" / "evaluation_runs" / "relation_aware_ablation",
            "relation-aware-ablation", args.recognition, label,
        )
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = output_dir.name
    config = {
        "input_dir": args.input_dir,
        "model": args.model,
        "recognition": args.recognition,
        "primary_partition_metric": "A4 = A1 - A3 AND A3 subseteq A1",
        "diagnostic_partition_metrics": ["A4 = A1 - A3", "A3 subseteq A1"],
    }
    manifest = start_run_manifest(
        output_dir, run_id=run_id, run_type="relation-aware-ablation",
        config=config, recognition=args.recognition,
        input_files=input_paths, input_root=args.input_dir,
        entity_normalization="none",
        evaluation_policies={
            "repeat_aggregation": "macro mean of five primary relation rates",
            "pairing_policy": "complete left-right pairs only",
            "correction_selection": "historical generation trigger used difference-only failure",
            "table4_abstention_scoring": "not applicable",
        },
    )
    try:
        rows, skipped = read_rows(args.input_dir, args.model, args.recognition)
        write_csv(output_dir / "per_item.csv", rows)
        write_csv(output_dir / "summary.csv", summarize(rows))
        manifest["result_counts"] = {
            "paired_metric_rows": len(rows),
            "skipped_incomplete_task_records": skipped,
        }
    except Exception as error:
        finish_run_manifest(output_dir, manifest, status="failed", error=repr(error))
        raise
    finish_run_manifest(output_dir, manifest)
    print(f"Scored {len(rows)} paired metric rows; skipped {skipped} incomplete task records")
    print(f"Run manifest: {display_path(output_dir / 'manifest.json')}")
    print(f"Results: {display_path(output_dir / 'summary.csv')}")


if __name__ == "__main__":
    main()

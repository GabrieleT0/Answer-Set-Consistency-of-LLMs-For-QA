"""Score paired relation-aware ablation experiments.

    venv/bin/python -m src.evaluation.eval_relation_aware_ablation

Writes per_item.csv and summary.csv beside the generation checkpoints. A
relevant idk/empty answer is scored as inconsistent, as in the paper's
worst-case summary. The correction comparison is reported both on initially
inconsistent dialogues and on all dialogues (unchanged initial answers for
the latter's initially consistent cases).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from src.get_answers.relation_aware_ablation import ROOT, TASKS, is_abstention


METRICS = {
    "equal": ("E1,2",),
    "sup-sub": ("N3,1",),
    "minus": ("N4,1", "D3,4", "E4,1\\3"),
}


def scores(task: str, answers: dict[str, list[str]]) -> dict[str, int]:
    a = {q: set(answers[q]) for q in TASKS[task]}

    def valid(*questions: str) -> bool:
        return not any(is_abstention(answers[q]) for q in questions)

    if task == "equal":
        return {"E1,2": int(valid("Q1", "Q2") and a["Q1"] == a["Q2"])}
    if task == "sup-sub":
        return {"N3,1": int(valid("Q1", "Q3") and a["Q3"].issubset(a["Q1"]))}
    return {
        "N4,1": int(valid("Q1", "Q4") and a["Q4"].issubset(a["Q1"])),
        "D3,4": int(valid("Q3", "Q4") and a["Q3"].isdisjoint(a["Q4"])),
        "E4,1\\3": int(valid("Q1", "Q3", "Q4") and a["Q4"] == a["Q1"] - a["Q3"]),
    }


def updated_answers(task: str, initial: dict, branch: dict | None) -> dict[str, list[str]]:
    answers = {q: list(values) for q, values in initial["answers"].items()}
    if branch is not None:
        answers[TASKS[task][-1]] = branch["answer"]
    return answers


def metric_abstention(task: str, metric: str, answers: dict[str, list[str]]) -> int:
    relevant = {
        "E1,2": ("Q1", "Q2"),
        "N3,1": ("Q1", "Q3"),
        "N4,1": ("Q1", "Q4"),
        "D3,4": ("Q3", "Q4"),
        "E4,1\\3": ("Q1", "Q3", "Q4"),
    }[metric]
    return int(any(is_abstention(answers[q]) for q in relevant))


def pair_rows(comparison: str, dataset: str, qid: str, task: str,
              left: dict[str, list[str]], right: dict[str, list[str]],
              initial_inconsistent: bool) -> list[dict]:
    left_scores, right_scores = scores(task, left), scores(task, right)
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
            "left_abstention": metric_abstention(task, metric, left),
            "right_abstention": metric_abstention(task, metric, right),
        }
        for metric in METRICS[task]
    ]


def read_rows(input_dir: Path, model: str | None = None) -> tuple[list[dict], int]:
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
            ))
            if eligible:
                rows.extend(pair_rows(
                    "recheck_vs_relation_failed", dataset, qid, task,
                    recheck, relation, True,
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
    for entry in summaries:
        by_setting[(entry["comparison"], entry["dataset"])].append(entry)
    for (comparison, dataset), entries in sorted(by_setting.items()):
        if {entry["metric"] for entry in entries} != {
            metric for metrics in METRICS.values() for metric in metrics
        }:
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
    parser.add_argument("--output-dir", type=Path, help="Defaults to input-dir")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows, skipped = read_rows(args.input_dir, args.model)
    output_dir = args.output_dir or args.input_dir
    write_csv(output_dir / "per_item.csv", rows)
    write_csv(output_dir / "summary.csv", summarize(rows))
    print(f"Scored {len(rows)} paired metric rows; skipped {skipped} incomplete task records")
    print(f"Results: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired Base vs entity-list injection on 92 ASCB groups.

Create lists: venv/bin/python -m src.get_answers.gold_fact_injection_experiment entities
Preview: venv/bin/python -m src.get_answers.gold_fact_injection_experiment run
Execute: venv/bin/python -m src.get_answers.gold_fact_injection_experiment run --execute
Analyze: venv/bin/python -m src.get_answers.gold_fact_injection_experiment analyze

Each question is asked in a fresh context. The injected file contains only
entity names, with no question text, answer labels, or subset flags. The
candidate-answer TSV supplies names for these files and is used for scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output" / "entity_context"
DEFAULT_CANDIDATES = ROOT / "data/ASCB/en/unified_dataset_w_candidate_answer.tsv"
DEFAULT_QUESTIONS = ROOT / "data/ASCB/en/unified-dataset.tsv"
QUESTIONS = ("Q1", "Q2", "Q3", "Q4")
ARMS = ("base", "entities")


def read_plan(candidates: Path, questions: Path) -> dict:
    sys.path.insert(0, str(ROOT / "src/evaluation"))
    from gold_answer_reviewer_audit import gold_groups

    groups, coverage = gold_groups(candidates, questions)
    with questions.open(newline="", encoding="utf-8") as handle:
        question_rows = {
            (r["source"].casefold(), int(r["source_id"])): r
            for r in csv.DictReader(handle, delimiter="\t")
        }
    planned = []
    for (dataset, source_id), gold in sorted(groups.items()):
        row = question_rows[dataset.casefold(), source_id]
        planned.append({
            "dataset": dataset, "source_id": source_id,
            "questions": {q: row[q] for q in QUESTIONS},
            "candidate": {q: sorted(gold[q]) for q in QUESTIONS},
        })
    return {
        "schema_version": 1,
        "reference_status": "provisional_candidate_answers_not_independently_adjudicated",
        "n_included": len(planned),
        "n_excluded": int((~coverage.included).sum()),
        "question_groups": planned,
    }


def entity_path(directory: Path, group: dict) -> Path:
    name = group["dataset"].replace("/", "-").replace(" ", "-")
    return directory / f"{name}_{group['source_id']}.txt"


def make_entity_files(args: argparse.Namespace) -> None:
    plan = read_plan(args.candidates, args.questions)
    directory = args.entity_dir
    directory.mkdir(parents=True, exist_ok=True)
    existing = [entity_path(directory, group) for group in plan["question_groups"]
                if entity_path(directory, group).exists()]
    if existing:
        raise FileExistsError(f"{existing[0]} exists; refusing to overwrite entity files")
    for group in plan["question_groups"]:
        path = entity_path(directory, group)
        path.write_text("\n".join(group["candidate"]["Q1"]) + "\n",
                        encoding="utf-8")
    print(f"Created {len(plan['question_groups'])} entity-only text files in {directory}")


def load_entity_files(directory: Path, groups: list[dict]) -> dict[tuple[str, int], str]:
    values = {}
    for group in groups:
        path = entity_path(directory, group)
        if not path.exists():
            raise FileNotFoundError(f"missing entity list: {path}; run the entities command")
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or any(not line.strip() or line != line.strip() for line in lines):
            raise ValueError(f"{path} must have one nonempty entity per line")
        if len(lines) != len(set(lines)):
            raise ValueError(f"{path} contains duplicate entity names")
        values[group["dataset"], group["source_id"]] = "\n".join(lines)
    return values


def entity_digest(directory: Path, groups: list[dict]) -> str:
    digest = hashlib.sha256()
    for group in groups:
        path = entity_path(directory, group)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def injected_question(group: dict, question_id: str, entity_text: str) -> str:
    return (
        "The following text file contains candidate entities:\n"
        + entity_text
        + "\n\nChoose from these entities those that answer the question. "
          "The file may contain entities that do not answer this particular "
          "question. Do not add entities absent from the file. "
          "If you cannot determine which candidates answer it, return 'idk'.\n\n"
        + group["questions"][question_id]
    )


def call_key(row: dict) -> tuple:
    return (row["model"], row["dataset"], int(row["source_id"]),
            row["question_id"], row["arm"], int(row["repeat"]))


def load_journal(path: Path) -> dict[tuple, dict]:
    latest = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[call_key(row)] = row
    return latest


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_run_config(existing: dict | None, requested: dict) -> dict:
    """Keep the shared protocol fixed while adding newly requested models."""
    if existing is None:
        return requested
    if {k: v for k, v in existing.items() if k != "models"} != {
            k: v for k, v in requested.items() if k != "models"}:
        raise ValueError(
            "run protocol changed (plan, entity files, prompt, repeats, "
            "temperature, or seed); use another output directory")
    return {**existing, "models": list(dict.fromkeys(
        [*existing["models"], *requested["models"]]))}


def bootstrap_mean_ci(values: list[float], *, seed: int = 2027,
                      draws: int = 2000) -> tuple[float, float, float]:
    """Resample question groups, never individual responses or relations."""
    if not values:
        raise ValueError("no groups to bootstrap")
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    sampled = sorted(statistics.mean(rng.choices(values, k=len(values)))
                     for _ in range(draws))
    return mean, sampled[int(0.025 * draws)], sampled[int(0.975 * draws) - 1]


def run(args: argparse.Namespace) -> None:
    output = args.output_dir
    plan_path = output / "plan.json"
    if not plan_path.exists():
        if (output / "responses.jsonl").exists():
            raise FileNotFoundError("responses exist but plan.json is missing")
        write_json(plan_path, read_plan(args.candidates, args.questions))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entity_texts = load_entity_files(args.entity_dir, plan["question_groups"])
    config_path = output / "run_config.json"
    requested = {
        "schema_version": 1, "models": args.models, "repeats": args.repeats,
        "temperature": args.temperature, "seed": args.seed,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "entity_files_sha256": entity_digest(args.entity_dir, plan["question_groups"]),
        "prompt_sha256": hashlib.sha256(
            (ROOT / "src/get_answers/prompts.yaml").read_bytes()).hexdigest(),
    }
    if not config_path.exists() and (output / "responses.jsonl").exists():
        raise FileNotFoundError(
            f"{config_path} is missing, but responses.jsonl exists; "
            "restore the original run configuration or use another output directory")
    config = merge_run_config(
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists() else None, requested)
    tasks = [
        {"model": model, "dataset": g["dataset"], "source_id": g["source_id"],
         "question_id": q, "arm": arm, "repeat": rep,
         "question": (g["questions"][q] if arm == "base" else injected_question(
             g, q, entity_texts[g["dataset"], g["source_id"]]))}
        for model in args.models for g in plan["question_groups"]
        for q in QUESTIONS for arm in ARMS for rep in range(1, args.repeats + 1)
    ]
    random.Random(args.seed).shuffle(tasks)
    done = load_journal(output / "responses.jsonl")
    pending = [t for t in tasks if call_key(t) not in done or
               (args.retry_errors and done[call_key(t)]["status"] == "error")]
    print(f"{len(plan['question_groups'])} groups; {len(tasks)} total calls; {len(pending)} pending")
    if not args.execute:
        print("Preview only; add --execute to call providers.")
        return
    write_json(config_path, config)
    from dotenv import load_dotenv
    load_dotenv(ROOT / "src/.env")
    from .llms import return_chat_model
    from .single_question_benchmark import get_prompt
    from .utils import convert_response_to_set

    prompt = get_prompt("standard", "en")
    chains = {}
    attempted = 0
    errors = 0
    with (output / "responses.jsonl").open("a", encoding="utf-8") as handle:
        for task in pending:
            if args.max_calls is not None and attempted >= args.max_calls:
                break
            model = task["model"]
            row = {k: task[k] for k in
                   ("model", "dataset", "source_id", "question_id", "arm", "repeat")}
            row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                if model not in chains:
                    chains[model] = prompt | return_chat_model(model, temperature=args.temperature)
                answer = chains[model].invoke({"question": task["question"]})
                raw = answer.content if hasattr(answer, "content") else str(answer)
                if not isinstance(raw, str):
                    raise TypeError("non-text model response")
                row.update(status="ok", raw_answer=raw,
                           parsed_answer=convert_response_to_set(raw))
                errors = 0
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__,
                           error_message=str(exc)[:300])
                errors += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            attempted += 1
            print(f"{attempted}: {model} {task['dataset']}:{task['source_id']} "
                  f"{task['question_id']} {task['arm']} -> {row['status']}", flush=True)
            if errors >= args.max_consecutive_errors:
                print("Stopping after consecutive provider errors; inspect the journal.")
                break


def analyze(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(ROOT / "src/evaluation"))
    from gold_answer_reviewer_audit import answer_set
    from reviewer_abstention_audit import _score

    output = args.output_dir
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    groups = {(g["dataset"], g["source_id"]): g for g in plan["question_groups"]}
    records = load_journal(output / "responses.jsonl")
    answers = {}
    quality = []
    for key, row in sorted(records.items()):
        model, dataset, source_id, q, arm, rep = key
        gold = set(groups[dataset, source_id]["candidate"][q])
        value = answer_set(row.get("parsed_answer")) if row["status"] == "ok" else None
        answers[key] = value
        quality.append({
            "model": model, "dataset": dataset, "source_id": source_id,
            "question_id": q, "arm": arm, "repeat": rep,
            "available": int(value is not None),
            "exact": int(value == gold) if value is not None else 0,
            "recall": len(value & gold) / len(gold) if value is not None and gold else "",
            "omitted": len(gold - value) if value is not None else "",
            "extra": len(value - gold) if value is not None else "",
        })
    write_csv(output / "answer_quality.csv", quality)
    pair_rows = []
    for model in config["models"]:
        for (dataset, source_id), group in groups.items():
            for arm in ARMS:
                for rep in range(1, config["repeats"] + 1):
                    values = {q: answers.get((model, dataset, source_id, q, arm, rep))
                              for q in QUESTIONS}
                    gold = {q: set(group["candidate"][q]) for q in QUESTIONS}
                    for narrow, metric in (("Q3", "N_CON_3_1"), ("Q4", "N_CON_4_1")):
                        required = [(model, dataset, source_id, q, arm, rep)
                                    for q in ("Q1", narrow)]
                        if not all(key in answers for key in required):
                            continue
                        broad_answer, narrow_answer = values["Q1"], values[narrow]
                        available = broad_answer is not None and narrow_answer is not None
                        conflict = narrow_answer - broad_answer if available else set()
                        pair_rows.append({
                            "model": model, "dataset": dataset, "source_id": source_id,
                            "arm": arm, "repeat": rep, "relation": metric,
                            "available": int(available),
                            "consistent": int(bool(available and _score(metric, [broad_answer, narrow_answer]))),
                            "verified_broad_omissions": len(conflict & gold[narrow]),
                            "unsupported_narrow_entities": len(conflict - gold[narrow]),
                            "both_exact": int(available and broad_answer == gold["Q1"]
                                              and narrow_answer == gold[narrow]),
                        })
    write_csv(output / "containment_pairs.csv", pair_rows)
    summary = []
    for model in config["models"]:
        for arm in ARMS:
            subset = [r for r in pair_rows if r["model"] == model and r["arm"] == arm]
            qs = [r for r in quality if r["model"] == model and r["arm"] == arm]
            recalls = [float(r["recall"]) for r in qs if r["recall"] != ""]
            if not subset or not qs:
                continue
            summary.append({
                "model": model, "arm": arm, "n_pairs": len(subset),
                "containment_pass_rate": statistics.mean(r["consistent"] for r in subset),
                "verified_omission_violation_rate": statistics.mean(
                    r["verified_broad_omissions"] > 0 for r in subset),
                "unsupported_entity_violation_rate": statistics.mean(
                    r["unsupported_narrow_entities"] > 0 for r in subset),
                "n_questions": len(qs),
                "answer_exact_rate": statistics.mean(r["exact"] for r in qs),
                "answer_available_rate": statistics.mean(r["available"] for r in qs),
                "answer_recall": statistics.mean(recalls) if recalls else "",
            })
    write_csv(output / "summary.csv", summary)
    by_pair = {(r["model"], r["dataset"], r["source_id"], r["repeat"],
                r["relation"], r["arm"]): r for r in pair_rows}
    paired = []
    for key, base in by_pair.items():
        if key[-1] != "base":
            continue
        fact = by_pair.get((*key[:-1], "entities"))
        if fact is None:
            continue
        paired.append({
            "model": key[0], "dataset": key[1], "source_id": key[2],
            "repeat": key[3], "relation": key[4],
            "base_consistent": base["consistent"],
            "entities_consistent": fact["consistent"],
            "consistency_gain": fact["consistent"] - base["consistent"],
            "base_verified_broad_omission": int(base["verified_broad_omissions"] > 0),
            "entities_verified_broad_omission": int(fact["verified_broad_omissions"] > 0),
        })
    write_csv(output / "paired_containment.csv", paired)
    paired_summary = []
    for model in config["models"]:
        grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for row in paired:
            if row["model"] == model:
                grouped[row["dataset"], row["source_id"]].append(row)
        if not grouped:
            continue
        gains = [statistics.mean(r["consistency_gain"] for r in rows)
                 for rows in grouped.values()]
        mean, low, high = bootstrap_mean_ci(gains)
        paired_summary.append({
            "model": model, "n_paired_groups": len(grouped),
            "entities_minus_base_containment_pass": mean,
            "ci_low": low, "ci_high": high,
        })
    write_csv(output / "paired_summary.csv", paired_summary)
    if summary:
        print(f"Wrote {output / 'summary.csv'}")
    else:
        print("No complete containment pairs to summarize yet.")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("entities", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--entity-dir", type=Path)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--models", nargs="+", default=["gpt-4.1-mini-2025-04-14"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    args = parser.parse_args()
    if args.entity_dir is None:
        args.entity_dir = args.output_dir / "entities"
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    {"entities": make_entity_files, "run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()

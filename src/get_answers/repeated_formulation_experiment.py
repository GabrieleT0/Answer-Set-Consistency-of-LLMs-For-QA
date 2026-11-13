#!/usr/bin/env python3
"""Repeat every ASCB formulation on the usable candidate-annotated groups.

Commands:
  plan     Select all usable groups by default (no provider calls).
  run      Preview calls; add --execute to call models and append results.
  analyze  Compare within-formulation variability, wording effects, and relations.

Run from the repository root with venv/bin/python. Only Base/standard prompting
is used, matching the existing single-question benchmark. Each invocation is
stateless. The script never writes into data/answers/.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CANDIDATES = REPO_ROOT / "data/ASCB/en/unified_dataset_w_candidate_answer.tsv"
DEFAULT_QUESTIONS = REPO_ROOT / "data/ASCB/en/unified-dataset.tsv"
sys.path.insert(0, str(REPO_ROOT / "src/evaluation"))
from gold_answer_reviewer_audit import answer_set, gold_groups, normalize  # noqa: E402
from reviewer_abstention_audit import PRIMARY, _score  # noqa: E402

QUESTIONS = ("Q1", "Q2", "Q3", "Q4")
RELATION_QUESTIONS = {
    "E_CON_1_2": ("Q1", "Q2"),
    "N_CON_3_1": ("Q1", "Q3"),
    "N_CON_4_1": ("Q1", "Q4"),
    "D_CON_3_4": ("Q3", "Q4"),
    "E_CON_4_1_minus_3": ("Q1", "Q3", "Q4"),
}
assert tuple(RELATION_QUESTIONS) == PRIMARY


def size_bin(n: int) -> str:
    return "1-2" if n <= 2 else "3-5" if n <= 5 else "6+"


def dataset_quotas(counts: dict[str, int], n_groups: int,
                   min_per_dataset: int) -> dict[str, int]:
    """Proportional allocation with a small coverage floor per dataset."""
    if n_groups < 1 or n_groups > sum(counts.values()):
        raise ValueError("--groups must be between 1 and the number of valid groups")
    if min_per_dataset < 0 or n_groups < len(counts) * min_per_dataset:
        raise ValueError("--min-per-dataset is incompatible with --groups")
    if any(n < min_per_dataset for n in counts.values()):
        raise ValueError("A dataset has fewer groups than --min-per-dataset")
    ideal = {dataset: n_groups * n / sum(counts.values()) for dataset, n in counts.items()}
    quota = {dataset: max(min_per_dataset, int(ideal[dataset])) for dataset in counts}
    while sum(quota.values()) < n_groups:
        choice = max((d for d in counts if quota[d] < counts[d]),
                     key=lambda d: (ideal[d] - quota[d], d))
        quota[choice] += 1
    while sum(quota.values()) > n_groups:
        choice = max((d for d in counts if quota[d] > min_per_dataset),
                     key=lambda d: (quota[d] - ideal[d], d))
        quota[choice] -= 1
    return quota


def select_groups(groups: dict[tuple[str, int], dict[str, set[str]]],
                  n_groups: int, min_per_dataset: int,
                  seed: int) -> list[dict[str, Any]]:
    """Sample datasets and Q1 answer-size strata approximately proportionally."""
    rng = random.Random(seed)
    strata: dict[str, dict[str, list[tuple[str, int]]]] = defaultdict(
        lambda: {"1-2": [], "3-5": [], "6+": []})
    for key, gold in sorted(groups.items()):
        strata[key[0]][size_bin(len(gold["Q1"]))].append(key)
    quotas = dataset_quotas(
        {dataset: sum(map(len, bins.values())) for dataset, bins in strata.items()},
        n_groups, min_per_dataset)
    chosen = []
    for dataset in sorted(strata):
        bins = strata[dataset]
        nonempty = {label: len(pool) for label, pool in bins.items() if pool}
        bin_quotas = dataset_quotas(
            nonempty, quotas[dataset],
            min_per_dataset=1 if quotas[dataset] >= len(nonempty) else 0)
        for label, n_chosen in bin_quotas.items():
            rng.shuffle(bins[label])
            chosen.extend(bins[label][:n_chosen])
    return [
        {"dataset": dataset, "source_id": source_id,
         "size_bin": size_bin(len(groups[dataset, source_id]["Q1"])),
         "candidate": {q: sorted(groups[dataset, source_id][q]) for q in QUESTIONS}}
        for dataset, source_id in sorted(chosen)
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_plan(args: argparse.Namespace) -> dict[str, Any]:
    groups, coverage = gold_groups(args.candidates, args.questions)
    n_groups = len(groups) if args.groups is None else args.groups
    selected = select_groups(groups, n_groups, args.min_per_dataset, args.seed)
    questions = pd.read_csv(args.questions, sep="\t", dtype=str, keep_default_na=False)
    lookup = {(row.source.casefold(), int(row.source_id)): row
              for row in questions.itertuples(index=False)}
    for group in selected:
        row = lookup[group["dataset"].casefold(), group["source_id"]]
        group["questions"] = {q: getattr(row, q) for q in QUESTIONS}
    plan = {
        "schema_version": 1,
        "selection_seed": args.seed,
        "n_groups_requested": n_groups,
        "min_per_dataset": args.min_per_dataset,
        "candidate_groups_available": len(groups),
        "candidate_groups_excluded": int((~coverage.included).sum()),
        "dataset_population": {name: int(count) for name, count in
                               pd.Series([key[0] for key in groups]).value_counts().items()},
        "question_groups": selected,
    }
    path = args.output_dir / "plan.json"
    if path.exists() and not args.force:
        raise FileExistsError(f"{path} exists; use --force to replace the selection")
    _write_json(path, plan)
    pd.DataFrame([
        {"dataset": g["dataset"], "source_id": g["source_id"],
         "size_bin": g["size_bin"],
         **{f"n_{q}": len(g["candidate"][q]) for q in QUESTIONS},
         **{q: g["questions"][q] for q in QUESTIONS}}
        for g in selected
    ]).to_csv(args.output_dir / "selected_groups.csv", index=False)
    print(f"Selected {len(selected)} of {len(groups)} valid annotated groups -> {path}")
    print(pd.Series([g["dataset"] for g in selected]).value_counts().sort_index().to_string())
    return plan


def call_key(row: dict[str, Any]) -> tuple[str, str, int, str, int]:
    return (row["model"], row["dataset"], int(row["source_id"]),
            row["question_id"], int(row["replicate"]))


def load_journal(path: Path) -> dict[tuple[str, str, int, str, int], dict[str, Any]]:
    latest = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid journal line {line_no} in {path}") from error
            latest[call_key(row)] = row
    return latest


def planned_calls(plan: dict[str, Any], models: list[str], repeats: int,
                  seed: int) -> list[dict[str, Any]]:
    if repeats < 2:
        raise ValueError("At least two repeats are needed to measure variation")
    tasks = [
        {"model": model, "dataset": group["dataset"],
         "source_id": group["source_id"], "question_id": q,
         "replicate": replicate, "question": group["questions"][q]}
        for model in models for group in plan["question_groups"]
        for q in QUESTIONS for replicate in range(1, repeats + 1)
    ]
    random.Random(seed).shuffle(tasks)
    return tasks


def merge_run_config(existing: dict[str, Any] | None,
                     requested: dict[str, Any]) -> dict[str, Any]:
    """Add models to an existing run while preserving its shared protocol."""
    if existing is None:
        return requested
    protocol = lambda config: {k: v for k, v in config.items() if k != "models"}
    if protocol(existing) != protocol(requested):
        raise ValueError(
            "The plan, prompt, repeats, temperature, or call-order seed differs "
            "from run_config.json; use a new --output-dir for a different protocol.")
    return {**existing, "models": list(dict.fromkeys(
        [*existing["models"], *requested["models"]]))}


def run_experiment(args: argparse.Namespace) -> None:
    plan_path = args.output_dir / "plan.json"
    if not plan_path.exists():
        if (args.output_dir / "run_config.json").exists() or \
                (args.output_dir / "responses.jsonl").exists():
            raise FileNotFoundError(
                f"{plan_path} is missing, but this output directory contains run data. "
                "Restore its original plan or use a new --output-dir.")
        make_plan(argparse.Namespace(
            candidates=DEFAULT_CANDIDATES, questions=DEFAULT_QUESTIONS,
            groups=None, min_per_dataset=3, seed=2027,
            output_dir=args.output_dir, force=False))
    plan = _read_json(plan_path)
    models = list(dict.fromkeys(args.models))
    tasks = planned_calls(plan, models, args.repeats, args.seed)
    prompt_path = REPO_ROOT / "src/get_answers/prompts.yaml"
    requested_config = {
        "schema_version": 1, "models": models, "repeats": args.repeats,
        "temperature_requested": args.temperature, "call_order_seed": args.seed,
        "prompt_type": "standard", "language": "en", "strategy": "Base",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "prompts_yaml_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
    }
    config_path = args.output_dir / "run_config.json"
    if not config_path.exists() and (args.output_dir / "responses.jsonl").exists():
        raise FileNotFoundError(
            f"{config_path} is missing, but responses.jsonl exists. "
            "Restore the original run configuration before continuing.")
    config = merge_run_config(
        _read_json(config_path) if config_path.exists() else None,
        requested_config)
    latest = load_journal(args.output_dir / "responses.jsonl")
    pending = [task for task in tasks if call_key(task) not in latest
               or (args.retry_errors and latest[call_key(task)]["status"] != "ok")]
    print(f"Plan: {len(plan['question_groups'])} groups, {len(models)} models, "
          f"{args.repeats} repeats, {len(tasks)} total calls; {len(pending)} pending")
    if not args.execute:
        print("Preview only. Add --execute to call model providers.")
        return
    _write_json(config_path, config)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from get_answers.llms import return_chat_model  # noqa: E402
    from get_answers.single_question_benchmark import get_prompt  # noqa: E402
    from get_answers.utils import convert_response_to_set  # noqa: E402

    prompt = get_prompt("standard", "en")
    chains = {}
    journal = args.output_dir / "responses.jsonl"
    attempted = 0
    consecutive_errors = 0
    with journal.open("a", encoding="utf-8") as handle:
        for task in pending:
            if args.max_calls is not None and attempted >= args.max_calls:
                break
            model = task["model"]
            row = {k: task[k] for k in ("model", "dataset", "source_id", "question_id", "replicate")}
            row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                if model not in chains:
                    chains[model] = prompt | return_chat_model(model, temperature=args.temperature)
                response = chains[model].invoke({"question": task["question"]})
                raw = response.content if hasattr(response, "content") else str(response)
                if not isinstance(raw, str):
                    raw = str(raw)
                row.update(status="ok", raw_answer=raw,
                           parsed_answer=convert_response_to_set(raw))
            except Exception as error:
                row.update(status="error", error_type=type(error).__name__,
                           error_message=str(error)[:500])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            attempted += 1
            consecutive_errors = consecutive_errors + 1 if row["status"] == "error" else 0
            print(f"{attempted}: {model} {task['dataset']}:{task['source_id']} "
                  f"{task['question_id']} repeat {task['replicate']} -> {row['status']}")
            if args.max_consecutive_errors and consecutive_errors >= args.max_consecutive_errors:
                print("Stopped after consecutive provider failures; inspect the journal "
                      "and use --retry-errors after correcting the issue.")
                break
    print(f"Recorded {attempted} new calls in {journal}")


def _jaccard(left: set[str] | None, right: set[str] | None) -> float:
    if left is None or right is None:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _candidate_disagreement(left: set[str] | None, right: set[str] | None,
                            candidate: list[str]) -> float | None:
    """Fraction of expected entities included in exactly one answer."""
    if not candidate:
        return None
    return sum((entity in left if left is not None else False) !=
               (entity in right if right is not None else False)
               for entity in candidate) / len(candidate)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _difference(left: object, right: object) -> float | None:
    return None if pd.isna(left) or pd.isna(right) else float(left) - float(right)


def bootstrap_mean_ci(values: list[float], seed: int = 2027,
                      draws: int = 2000) -> tuple[float, float, float]:
    """Bootstrap question groups, never dependent answer combinations."""
    if not values:
        raise ValueError("Cannot summarize an empty comparison")
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    estimates = sorted(statistics.mean(rng.choices(values, k=len(values)))
                       for _ in range(draws))
    return mean, estimates[int(0.025 * draws)], estimates[int(0.975 * draws) - 1]


def analyze_experiment(args: argparse.Namespace) -> None:
    plan = _read_json(args.output_dir / "plan.json")
    config = _read_json(args.output_dir / "run_config.json")
    latest = load_journal(args.output_dir / "responses.jsonl")
    groups = {(g["dataset"], g["source_id"]): g for g in plan["question_groups"]}
    answers: dict[tuple[str, str, int, str], list[set[str] | None]] = defaultdict(list)
    response_rows = []
    for key, row in sorted(latest.items()):
        model, dataset, source_id, q, replicate = key
        if model not in config["models"] or (dataset, source_id) not in groups:
            continue
        target = set(groups[dataset, source_id]["candidate"][q])
        provider_ok = row["status"] == "ok"
        value = answer_set(row.get("parsed_answer")) if provider_ok else None
        if provider_ok:
            answers[model, dataset, source_id, q].append(value)
        omitted = len(target - value) if value is not None else len(target)
        extra = len(value - target) if value is not None else 0
        response_rows.append({
            "model": model, "dataset": dataset, "source_id": source_id,
            "question_id": q, "replicate": replicate, "status": row["status"],
            "available": provider_ok and value is not None, "n_gold": len(target),
            "n_omitted": omitted if provider_ok else None,
            "n_extra": extra if provider_ok else None,
            "omission_fraction": omitted / len(target) if provider_ok and target else None,
            "candidate_jaccard": _jaccard(value, target) if provider_ok else None,
            "candidate_exact": (int(value == target) if value is not None else 0)
            if provider_ok else None,
        })
    response_frame = pd.DataFrame(response_rows)
    response_frame.to_csv(args.output_dir / "response_quality.csv", index=False)
    if not answers:
        print("No successful provider calls to analyze; see response_quality.csv")
        return
    within_rows = []
    for (model, dataset, source_id, q), values in sorted(answers.items()):
        pairs = list(itertools.combinations(values, 2))
        candidate_entities = groups[dataset, source_id]["candidate"][q]
        flip_rates = []
        for entity in candidate_entities:
            inclusion = sum(entity in value for value in values if value is not None) / len(values)
            flip_rates.append(2 * inclusion * (1 - inclusion))
        all_records = response_frame.loc[(response_frame.model == model)
                                         & (response_frame.dataset == dataset)
                                         & (response_frame.source_id == source_id)
                                         & (response_frame.question_id == q)]
        subset = all_records.loc[all_records.status == "ok"]
        within_rows.append({
            "model": model, "dataset": dataset, "source_id": source_id,
            "question_id": q, "n_recorded": len(values),
            "n_expected": config["repeats"],
            "n_provider_errors": len(all_records) - len(subset),
            "n_gold": int(subset.n_gold.iloc[0]),
            "within_exact": _mean([float(a == b) if a is not None and b is not None else 0.0
                                   for a, b in pairs]),
            "within_jaccard": _mean([_jaccard(a, b) for a, b in pairs]),
            "gold_entity_flip_rate": _mean(flip_rates),
            "candidate_jaccard": subset.candidate_jaccard.mean(),
            "candidate_exact": subset.candidate_exact.mean(),
            "omission_fraction": subset.omission_fraction.mean(),
            "mean_extra_count": subset.n_extra.mean(),
            "unavailable_rate": 1 - subset.available.mean(),
        })
    within = pd.DataFrame(within_rows)
    within.to_csv(args.output_dir / "within_formulation.csv", index=False)
    contrast_rows = []
    for (model, dataset, source_id), frame in within.groupby(
            ["model", "dataset", "source_id"], sort=True):
        by_q = frame.set_index("question_id")
        if "Q1" not in by_q.index:
            continue
        base = by_q.loc["Q1"]
        for q in ("Q2", "Q3", "Q4"):
            if q not in by_q.index:
                continue
            other = by_q.loc[q]
            contrast_rows.append({
                "model": model, "dataset": dataset, "source_id": source_id,
                "compare_q": q, "q1_n_gold": base.n_gold, "other_n_gold": other.n_gold,
                "within_exact_difference": _difference(other.within_exact, base.within_exact),
                "within_jaccard_difference": _difference(other.within_jaccard, base.within_jaccard),
                "gold_entity_flip_difference": _difference(
                    other.gold_entity_flip_rate, base.gold_entity_flip_rate),
                "omission_fraction_difference": _difference(other.omission_fraction,
                                                            base.omission_fraction),
            })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(args.output_dir / "formulation_contrasts.csv", index=False)
    contrast_summary = []
    if not contrasts.empty:
        for (model, q), frame in contrasts.groupby(["model", "compare_q"], sort=True):
            row = {"model": model, "compare_q": q, "n_groups": len(frame)}
            for metric in ("within_exact_difference", "within_jaccard_difference",
                           "gold_entity_flip_difference", "omission_fraction_difference"):
                values = frame[metric].dropna().tolist()
                if values:
                    mean, low, high = bootstrap_mean_ci(values)
                    row.update({metric: mean, metric + "_ci_low": low,
                                metric + "_ci_high": high})
            contrast_summary.append(row)
    pd.DataFrame(contrast_summary).to_csv(
        args.output_dir / "formulation_contrast_summary.csv", index=False)
    equivalence_rows = []
    relation_rows = []
    relation_error_rows = []
    for model in config["models"]:
        for (dataset, source_id), group in groups.items():
            key = lambda q: (model, dataset, source_id, q)
            if all(key(q) in answers and len(answers[key(q)]) >= 2 for q in ("Q1", "Q2")):
                q1, q2 = answers[key("Q1")], answers[key("Q2")]
                candidate_entities = group["candidate"]["Q1"]
                inclusion_shift = _mean([
                    abs(sum(a is not None and entity in a for a in q1) / len(q1)
                        - sum(a is not None and entity in a for a in q2) / len(q2))
                    for entity in candidate_entities
                ])
                cross = list(itertools.product(q1, q2))
                w1 = within.loc[(within.model == model) & (within.dataset == dataset)
                                & (within.source_id == source_id) & (within.question_id == "Q1")].iloc[0]
                w2 = within.loc[(within.model == model) & (within.dataset == dataset)
                                & (within.source_id == source_id) & (within.question_id == "Q2")].iloc[0]
                within_exact = (w1.within_exact + w2.within_exact) / 2
                within_jaccard = (w1.within_jaccard + w2.within_jaccard) / 2
                within_candidate_disagreement = _mean([
                    score for values in (q1, q2)
                    for a, b in itertools.combinations(values, 2)
                    if (score := _candidate_disagreement(
                        a, b, candidate_entities)) is not None
                ])
                between_candidate_disagreement = _mean([
                    score for a, b in cross
                    if (score := _candidate_disagreement(
                        a, b, candidate_entities)) is not None
                ])
                between_exact = _mean([float(a == b) if a is not None and b is not None else 0.0
                                       for a, b in cross])
                between_jaccard = _mean([_jaccard(a, b) for a, b in cross])
                equivalence_rows.append({
                    "model": model, "dataset": dataset, "source_id": source_id,
                    "n_q1": len(q1), "n_q2": len(q2),
                    "within_exact": within_exact, "between_exact": between_exact,
                    "wording_gap_exact": within_exact - between_exact,
                    "within_jaccard": within_jaccard, "between_jaccard": between_jaccard,
                    "wording_gap_jaccard": within_jaccard - between_jaccard,
                    "mean_abs_gold_inclusion_shift": inclusion_shift,
                    "within_candidate_disagreement": within_candidate_disagreement,
                    "between_candidate_disagreement": between_candidate_disagreement,
                    "wording_gap_candidate_disagreement": _difference(
                        between_candidate_disagreement, within_candidate_disagreement),
                })
            for metric, required in RELATION_QUESTIONS.items():
                if not all(key(q) in answers and len(answers[key(q)]) >= 2 for q in required):
                    continue
                combinations = itertools.product(*(answers[key(q)] for q in required))
                scores = []
                strata: dict[tuple[str, str], list[float]] = defaultdict(list)
                for combo in combinations:
                    if any(a is None for a in combo):
                        score = 0.0
                        stratum = ("unavailable", "unavailable")
                    else:
                        score = float(_score(metric, list(combo)))
                        omissions = sum(len(set(group["candidate"][q]) - a)
                                        for q, a in zip(required, combo))
                        extras = sum(len(a - set(group["candidate"][q]))
                                     for q, a in zip(required, combo))
                        stratum = ("0" if omissions == 0 else "1" if omissions == 1 else "2+",
                                   "0" if extras == 0 else "1+")
                    scores.append(score)
                    strata[stratum].append(score)
                relation_rows.append({
                    "model": model, "dataset": dataset, "source_id": source_id,
                    "relation_metric": metric, "n_combinations": len(scores),
                    "pass_probability": _mean(scores),
                    "score_varies": int(0 < sum(scores) < len(scores)),
                })
                for (omission_bin, extra_bin), values in strata.items():
                    relation_error_rows.append({
                        "model": model, "dataset": dataset, "source_id": source_id,
                        "relation_metric": metric, "omission_bin": omission_bin,
                        "extra_bin": extra_bin, "n_combinations": len(values),
                        "pass_probability": _mean(values),
                    })
    equivalent = pd.DataFrame(equivalence_rows)
    equivalent.to_csv(args.output_dir / "equivalence_wording_effect.csv", index=False)
    wording_summary = []
    if not equivalent.empty:
        for model, frame in equivalent.groupby("model", sort=True):
            row = {"model": model, "n_groups": len(frame)}
            for metric in ("wording_gap_exact", "wording_gap_jaccard",
                           "mean_abs_gold_inclusion_shift",
                           "wording_gap_candidate_disagreement"):
                values = frame[metric].dropna().tolist()
                if not values:
                    continue
                mean, low, high = bootstrap_mean_ci(values)
                row.update({metric: mean, metric + "_ci_low": low,
                            metric + "_ci_high": high})
            wording_summary.append(row)
    pd.DataFrame(wording_summary).to_csv(args.output_dir / "wording_effect_summary.csv", index=False)
    relations = pd.DataFrame(relation_rows)
    relations.to_csv(args.output_dir / "relation_probabilities.csv", index=False)
    pd.DataFrame(relation_error_rows).to_csv(
        args.output_dir / "relation_error_strata.csv", index=False)
    if not within.empty:
        within.groupby(["model", "question_id"], sort=True).agg(
            n_groups=("source_id", "size"),
            mean_within_exact=("within_exact", "mean"),
            mean_within_jaccard=("within_jaccard", "mean"),
            mean_gold_entity_flip_rate=("gold_entity_flip_rate", "mean"),
            mean_candidate_jaccard=("candidate_jaccard", "mean"),
            mean_n_gold=("n_gold", "mean"),
            mean_omission_fraction=("omission_fraction", "mean"),
            mean_unavailable_rate=("unavailable_rate", "mean"),
        ).reset_index().to_csv(args.output_dir / "variability_by_formulation.csv", index=False)
    if not relations.empty:
        relations.groupby(["model", "relation_metric"], sort=True).agg(
            n_groups=("source_id", "size"),
            mean_pass_probability=("pass_probability", "mean"),
            fraction_groups_with_variable_score=("score_varies", "mean"),
        ).reset_index().to_csv(args.output_dir / "relation_summary.csv", index=False)
    print(f"Analyzed {len(response_frame)} recorded calls, {len(within)} formulation groups, "
          f"{len(relations)} model-group relations -> {args.output_dir}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "output/repeated_formulations")
    plan = sub.add_parser("plan", parents=[common], help="Select all usable groups by default")
    plan.add_argument("--candidates", type=Path,
                      default=DEFAULT_CANDIDATES)
    plan.add_argument("--questions", type=Path,
                      default=DEFAULT_QUESTIONS)
    plan.add_argument("--groups", type=int, default=None,
                      help="Optional smaller stratified subset; default: all usable groups")
    plan.add_argument("--min-per-dataset", type=int, default=3)
    plan.add_argument("--seed", type=int, default=2027)
    plan.add_argument("--force", action="store_true")
    run = sub.add_parser("run", parents=[common], help="Preview or execute model calls")
    run.add_argument("--models", nargs="+", required=True)
    run.add_argument("--repeats", type=int, default=10)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=2027)
    run.add_argument("--max-calls", type=int)
    run.add_argument("--max-consecutive-errors", type=int, default=5)
    run.add_argument("--retry-errors", action="store_true")
    run.add_argument("--execute", action="store_true")
    sub.add_parser("analyze", parents=[common], help="Analyze recorded responses")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "plan":
        make_plan(args)
    elif args.command == "run":
        run_experiment(args)
    else:
        analyze_experiment(args)


if __name__ == "__main__":
    main()

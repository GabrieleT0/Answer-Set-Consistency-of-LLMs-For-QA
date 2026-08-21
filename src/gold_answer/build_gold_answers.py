#!/usr/bin/env python3
"""Build ASCB gold answers by voting over relation-valid answer quadruples."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTION_DIR = REPO_ROOT / "data" / "ASCB" / "en"
DEFAULT_ANSWER_DIR = REPO_ROOT / "data" / "answers"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "gold_answer"

MODEL_PRIORITY = (
    ("GPT-5", "gpt-5"),
    ("GPT-o3", "o3"),
    ("GPT-5-mini", "gpt-5-mini"),
    ("Gemini-2.5-pro", "gemini-2.5-pro"),
    ("DeepSeek-R", "deepseek-reasoner"),
)
METHODS = (
    ("zero-shot", "zero-shot", "{question}_{relation}_answers_{model}.json"),
    ("CtE", "CtE", "{question}_{relation}_answers_classAndAnswer_{model}.json"),
    ("Oracle", "Oracle", "{question}_{relation}_answers_fixing_{model}.json"),
)
DATASETS = (
    ("LC-QuAD", "LC-QuAD.tsv"),
    ("qawiki", "qawiki.tsv"),
    ("spinach", "spinach.tsv"),
    ("synthetic", "synthetic.tsv"),
)
QUESTION_RELATIONS = {
    "Q1": "equal",
    "Q2": "equal",
    "Q3": "sup-sub",
    "Q4": "minus",
}
METHOD_DATASET_DIR = {
    "zero-shot": {"LC-QuAD": "LC-QuAD"},
    "CtE": {"LC-QuAD": "lc-quad"},
    "Oracle": {"LC-QuAD": "lc-quad"},
}


def normalize_entity(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"\s+", " ", text).casefold()


def clean_answer(value: Any) -> list[str] | None:
    """Normalize and deduplicate one answer; None means missing/IDK."""
    if not isinstance(value, list):
        return None
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        display = unicodedata.normalize("NFKC", str(item)).strip()
        key = normalize_entity(display)
        if key == "idk":
            return None
        if display and key not in seen:
            seen.add(key)
            result.append(display)
    return result


def answer_key(answer: list[str]) -> tuple[str, ...]:
    return tuple(sorted(normalize_entity(item) for item in answer))


def quadruple_key(answers: dict[str, list[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(answer_key(answers[q]) for q in QUESTION_RELATIONS)


def validate_relations(answers: dict[str, list[str]]) -> dict[str, bool]:
    sets = {name: set(answer_key(value)) for name, value in answers.items()}
    return {
        "A1_equals_A2": sets["Q1"] == sets["Q2"],
        "A3_subset_A1": sets["Q3"].issubset(sets["Q1"]),
        "A4_subset_A1": sets["Q4"].issubset(sets["Q1"]),
        "A3_disjoint_A4": sets["Q3"].isdisjoint(sets["Q4"]),
        "A1_equals_A3_union_A4": sets["Q1"] == sets["Q3"] | sets["Q4"],
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_outputs(answer_dir: Path, dataset: str) -> tuple[dict, list[str]]:
    outputs: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    missing: list[str] = []
    for method, method_dir, pattern in METHODS:
        dataset_dir = METHOD_DATASET_DIR.get(method, {}).get(dataset, dataset)
        for model_name, file_model in MODEL_PRIORITY:
            source = (model_name, method)
            outputs[source] = {}
            for question, relation in QUESTION_RELATIONS.items():
                filename = pattern.format(question=question, relation=relation, model=file_model)
                path = answer_dir / method_dir / dataset_dir / relation / filename
                # Match the existing evaluator: absent CtE Q3/sup-sub files use
                # the corresponding Q3/minus output.
                if not path.exists() and method == "CtE" and question == "Q3":
                    relation = "minus"
                    filename = pattern.format(question=question, relation=relation, model=file_model)
                    fallback = answer_dir / method_dir / dataset_dir / relation / filename
                    if fallback.exists():
                        path = fallback
                if path.exists():
                    outputs[source][question] = load_json(path)
                else:
                    outputs[source][question] = {}
                    missing.append(str(path.relative_to(REPO_ROOT)))
    return outputs, missing


def read_questions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def source_priority() -> list[tuple[str, str]]:
    """Model priority first, method priority second."""
    return [
        (model, method)
        for model, _ in MODEL_PRIORITY
        for method, _, _ in METHODS
    ]


def select_valid_quadruple(candidates: list[dict]) -> tuple[dict | None, int]:
    """Filter relation-valid candidates, then select their exact-set mode."""
    return select_quadruple(
        [candidate for candidate in candidates if candidate["relation_valid"]]
    )


def select_quadruple(candidates: list[dict]) -> tuple[dict | None, int]:
    """Select the exact-set mode, using source priority for ties."""
    if not candidates:
        return None, 0
    counts = Counter(quadruple_key(candidate["answers"]) for candidate in candidates)
    highest = max(counts.values())
    winning_keys = {key for key, count in counts.items() if count == highest}
    priority = source_priority()
    selected = next(
        candidate
        for source in priority
        for candidate in candidates
        if candidate["source"] == source
        and quadruple_key(candidate["answers"]) in winning_keys
    )
    return selected, highest


def build_candidates(outputs: dict, source_id: str) -> list[dict]:
    candidates = []
    for source in source_priority():
        answers: dict[str, list[str]] = {}
        for question in QUESTION_RELATIONS:
            raw = outputs[source][question].get(source_id)
            cleaned = clean_answer(raw)
            if cleaned is not None:
                answers[question] = cleaned
        complete = len(answers) == 4
        checks = validate_relations(answers) if complete else None
        candidates.append({
            "source": source,
            "selected_from": f"{source[1]}+{source[0]}",
            "answers": answers,
            "complete": complete,
            "relation_checks": checks,
            "relation_valid": bool(complete and all(checks.values())),
        })
    return candidates


def build(question_dir: Path, answer_dir: Path) -> tuple[list[dict], list[dict], dict]:
    records: list[dict] = []
    rejected: list[dict] = []
    missing_files: list[str] = []
    for dataset, filename in DATASETS:
        outputs, missing = load_outputs(answer_dir, dataset)
        missing_files.extend(missing)
        for row in read_questions(question_dir / filename):
            source_id = str(row["ID"])
            candidates = build_candidates(outputs, source_id)
            selected, confidence = select_valid_quadruple(candidates)
            is_gold = selected is not None
            if selected is None:
                selected, confidence = select_quadruple(
                    [candidate for candidate in candidates if candidate["complete"]]
                )
            record = {
                "dataset": dataset,
                "source_id": source_id,
                "status": "gold" if is_gold else "needs_review",
                "selected_from": selected["selected_from"] if selected else None,
                "confidence": confidence if selected else None,
                "relation-pass": "yes" if is_gold else "no",
                "questions": {q: row[q] for q in QUESTION_RELATIONS},
                "answers": {
                    q: {
                        "candidate_answer": selected["answers"][q] if selected else None,
                    }
                    for q in QUESTION_RELATIONS
                },
                "relation_checks": selected["relation_checks"] if selected else None,
            }
            if not is_gold:
                record["candidate_relation_audit"] = [
                    {
                        "selected_from": candidate["selected_from"],
                        "answers": {
                            q: candidate["answers"].get(q) for q in QUESTION_RELATIONS
                        },
                        "complete": candidate["complete"],
                        "relation_checks": candidate["relation_checks"],
                    }
                    for candidate in candidates
                ]
                rejected.append(record)
            records.append(record)

    accepted = sum(record["status"] == "gold" for record in records)
    scores = Counter(str(record["confidence"]) for record in records if record["confidence"])
    summary = {
        "selection_unit": "complete A1-A4 quadruple",
        "selection_rule": "stage 1: mode among relation-valid quadruples; stage 2 fallback: mode among all complete quadruples",
        "models_in_priority_order": [name for name, _ in MODEL_PRIORITY],
        "methods_in_priority_order": [name for name, _, _ in METHODS],
        "confidence_unit": "number of identical candidates within the active selection stage",
        "total_quadruples": len(records),
        "accepted_quadruples": accepted,
        "rejected_quadruples": len(rejected),
        "total_question_slots": len(records) * 4,
        "gold_questions": accepted * 4,
        "review_question_slots": len(rejected) * 4,
        "confidence_distribution": dict(sorted(scores.items(), key=lambda item: int(item[0]))),
        "missing_input_files": sorted(set(missing_files)),
    }
    return records, rejected, summary


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_question_tsv(path: Path, records: list[dict]) -> None:
    fields = [
        "dataset", "source_id", "question_id", "question", "status",
        "confidence", "selected_from", "relation-pass", "candidate_answer",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for record in records:
            for question in QUESTION_RELATIONS:
                answer = record["answers"][question]
                writer.writerow({
                    "dataset": record["dataset"],
                    "source_id": record["source_id"],
                    "question_id": question,
                    "question": record["questions"][question],
                    "status": record["status"],
                    "confidence": record["confidence"] or "",
                    "selected_from": record["selected_from"] or "",
                    "relation-pass": record["relation-pass"],
                    "candidate_answer": json.dumps(answer["candidate_answer"], ensure_ascii=False),
                })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-dir", type=Path, default=DEFAULT_QUESTION_DIR)
    parser.add_argument("--answer-dir", type=Path, default=DEFAULT_ANSWER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    records, rejected, summary = build(args.question_dir, args.answer_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "gold_answers.json", records)
    write_question_tsv(args.output_dir / "gold_answers.tsv", records)
    write_json(args.output_dir / "rejected_quadruples.json", rejected)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

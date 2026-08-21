#!/usr/bin/env python3
"""Build ASCB gold answers by majority agreement among selected LLMs.

For each question, an exact (after conservative text normalization) answer-set
majority is selected. Ties are resolved by MODEL_PRIORITY. A quadruple is only
emitted as gold when its four selected sets satisfy the ASCB relations.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTION_DIR = REPO_ROOT / "data" / "ASCB" / "en"
DEFAULT_ANSWER_DIR = REPO_ROOT / "data" / "answers" / "zero-shot"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "gold_answer"

# Priority is also the deterministic tie-break order requested for the project.
MODEL_PRIORITY = (
    ("GPT-5", "gpt-5"),
    ("GPT-o3", "o3"),
    ("GPT-5-mini", "gpt-5-mini"),
    ("Gemini-2.5-pro", "gemini-2.5-pro"),
    ("DeepSeek-R", "deepseek-reasoner"),
)

DATASETS = (
    ("LC-QuAD", "LC-QuAD.tsv"),
    ("qawiki", "qawiki.tsv"),
    ("spinach", "spinach.tsv"),
    ("synthetic", "synthetic.tsv"),
)

QUESTION_FILES = {
    "Q1": ("equal", "Q1_equal_answers_{model}.json"),
    "Q2": ("equal", "Q2_equal_answers_{model}.json"),
    "Q3": ("sup-sub", "Q3_sup-sub_answers_{model}.json"),
    "Q4": ("minus", "Q4_minus_answers_{model}.json"),
}


def normalize_entity(value: Any) -> str:
    """Conservatively normalize an entity for set comparison."""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def clean_answer(value: Any) -> list[str] | None:
    """Return a deduplicated answer list, or None for an IDK abstention."""
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        display = unicodedata.normalize("NFKC", str(item)).strip()
        if normalize_entity(display) == "idk":
            return None
        key = normalize_entity(display)
        if display and key not in seen:
            seen.add(key)
            cleaned.append(display)
    return cleaned


def answer_key(answer: list[str]) -> tuple[str, ...]:
    """Canonical, order-insensitive key for an answer set."""
    return tuple(sorted(normalize_entity(item) for item in answer))


@dataclass
class Consensus:
    answer: list[str]
    confidence: int
    agreeing_models: list[str]
    available_models: int
    selected_from: str


def choose_consensus(model_answers: dict[str, list[str] | None]) -> Consensus | None:
    """Choose the modal answer set, breaking ties by MODEL_PRIORITY."""
    valid = {
        model: answer
        for model, answer in model_answers.items()
        if answer is not None
    }
    if not valid:
        return None
    keys = {model: answer_key(answer) for model, answer in valid.items()}
    counts = Counter(keys.values())
    highest = max(counts.values())
    winning_keys = {key for key, count in counts.items() if count == highest}
    selected_model = next(
        name for name, _ in MODEL_PRIORITY
        if name in keys and keys[name] in winning_keys
    )
    selected_key = keys[selected_model]
    agreeing = [
        name for name, _ in MODEL_PRIORITY
        if keys.get(name) == selected_key
    ]
    return Consensus(
        answer=valid[selected_model] or [],
        confidence=highest,
        agreeing_models=agreeing,
        available_models=len(valid),
        selected_from=selected_model,
    )


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


def load_model_outputs(answer_dir: Path, dataset: str) -> tuple[dict, list[str]]:
    outputs: dict[str, dict[str, dict[str, Any]]] = {}
    missing: list[str] = []
    for model_name, file_model in MODEL_PRIORITY:
        outputs[model_name] = {}
        for question, (relation_dir, pattern) in QUESTION_FILES.items():
            path = answer_dir / dataset / relation_dir / pattern.format(model=file_model)
            if path.exists():
                outputs[model_name][question] = load_json(path)
            else:
                outputs[model_name][question] = {}
                missing.append(str(path.relative_to(REPO_ROOT)))
    return outputs, missing


def read_questions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build(question_dir: Path, answer_dir: Path) -> tuple[list[dict], list[dict], dict]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    missing_files: list[str] = []

    for dataset, filename in DATASETS:
        rows = read_questions(question_dir / filename)
        outputs, missing = load_model_outputs(answer_dir, dataset)
        missing_files.extend(missing)
        for row in rows:
            source_id = str(row["ID"])
            consensus: dict[str, Consensus] = {}
            unavailable: dict[str, list[str]] = {}
            raw_by_question: dict[str, dict[str, list[str] | None]] = {}
            for question in QUESTION_FILES:
                model_answers: dict[str, list[str] | None] = {}
                unavailable[question] = []
                for model_name, _ in MODEL_PRIORITY:
                    raw = outputs[model_name][question].get(source_id, "__missing__")
                    if raw == "__missing__":
                        unavailable[question].append(model_name)
                        model_answers[model_name] = None
                    else:
                        model_answers[model_name] = clean_answer(raw)
                        if model_answers[model_name] is None:
                            unavailable[question].append(model_name)
                raw_by_question[question] = model_answers
                chosen = choose_consensus(model_answers)
                if chosen is not None:
                    consensus[question] = chosen

            base = {
                "dataset": dataset,
                "source_id": source_id,
                "questions": {q: row[q] for q in QUESTION_FILES},
            }
            if len(consensus) != 4:
                rejected.append({
                    **base,
                    "reason": "no_valid_candidate",
                    "missing_questions": [q for q in QUESTION_FILES if q not in consensus],
                    "unavailable_models": unavailable,
                })
                continue

            answers = {q: consensus[q].answer for q in QUESTION_FILES}
            checks = validate_relations(answers)
            record = {
                **base,
                "answers": {
                    q: asdict(consensus[q]) for q in QUESTION_FILES
                },
                "relation_checks": checks,
            }
            if all(checks.values()):
                accepted.append(record)
            else:
                rejected.append({
                    **record,
                    "reason": "set_relation_violation",
                    "failed_relations": [name for name, ok in checks.items() if not ok],
                })

    confidence_counts = Counter(
        str(item["answers"][q]["confidence"])
        for item in accepted for q in QUESTION_FILES
    )
    summary = {
        "models_in_priority_order": [name for name, _ in MODEL_PRIORITY],
        "normalization": "Unicode NFKC, trim/collapse whitespace, case-insensitive exact set equality",
        "total_quadruples": len(accepted) + len(rejected),
        "accepted_quadruples": len(accepted),
        "rejected_quadruples": len(rejected),
        "gold_questions": len(accepted) * 4,
        "confidence_distribution": dict(sorted(confidence_counts.items())),
        "missing_input_files": sorted(set(missing_files)),
    }
    return accepted, rejected, summary


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_question_tsv(path: Path, accepted: list[dict]) -> None:
    fields = [
        "dataset", "source_id", "question_id", "question", "gold_answer",
        "confidence", "agreeing_models", "available_models", "selected_from",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for item in accepted:
            for question in QUESTION_FILES:
                answer = item["answers"][question]
                writer.writerow({
                    "dataset": item["dataset"],
                    "source_id": item["source_id"],
                    "question_id": question,
                    "question": item["questions"][question],
                    "gold_answer": json.dumps(answer["answer"], ensure_ascii=False),
                    "confidence": answer["confidence"],
                    "agreeing_models": ",".join(answer["agreeing_models"]),
                    "available_models": answer["available_models"],
                    "selected_from": answer["selected_from"],
                })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-dir", type=Path, default=DEFAULT_QUESTION_DIR)
    parser.add_argument("--answer-dir", type=Path, default=DEFAULT_ANSWER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted, rejected, summary = build(args.question_dir, args.answer_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "gold_answers.json", accepted)
    write_question_tsv(args.output_dir / "gold_answers.tsv", accepted)
    write_json(args.output_dir / "rejected_quadruples.json", rejected)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


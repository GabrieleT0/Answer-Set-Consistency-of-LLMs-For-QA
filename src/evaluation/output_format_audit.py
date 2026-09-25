#!/usr/bin/env python3
"""Audit saved answer sets for detectable output-format violations.

The answer-generation pipeline splits raw model responses on ``|`` before
writing JSON. Consequently, the saved files cannot reveal every violation of
the requested raw format (for example, whether a clean-looking answer used the
wrong delimiter). This audit therefore reports a conservative lower bound:
only artifacts that remain detectable after parsing are counted.

By default, the script audits the answer files consumed by the main
consistency analysis:

* Base: Q1/equal, Q2/equal, Q3/sup-sub, and Q4/minus.
* CtE and Oracle: Q1/equal, Q1/sup-sub, Q1/minus, Q2/equal,
  Q3/sup-sub, Q3/minus, and Q4/minus.

The unused legacy CtE Q2/sup-sub files, Wikidata variants, no-IDK ablations,
and repeated-Q1 files are excluded. Repeated-Q1 files can be included with
``--include-repeat``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

ACTION_DIRS = {
    "Base": Path("data/answers/zero-shot"),
    "CtE": Path("data/answers/CtE"),
    "Oracle": Path("data/answers/Oracle"),
}

SCORED_SLOTS = {
    "Base": {
        ("Q1", "equal"),
        ("Q2", "equal"),
        ("Q3", "sup-sub"),
        ("Q4", "minus"),
    },
    "CtE": {
        ("Q1", "equal"),
        ("Q1", "sup-sub"),
        ("Q1", "minus"),
        ("Q2", "equal"),
        ("Q3", "sup-sub"),
        ("Q3", "minus"),
        ("Q4", "minus"),
    },
    "Oracle": {
        ("Q1", "equal"),
        ("Q1", "sup-sub"),
        ("Q1", "minus"),
        ("Q2", "equal"),
        ("Q3", "sup-sub"),
        ("Q3", "minus"),
        ("Q4", "minus"),
    },
}

ANSWER_FILE_RE = re.compile(
    r"^(Q[1-4])_(equal|sup-sub|minus)_answers_(.+)\.json$"
)

# These patterns are deliberately high precision. A clean entity can be a long
# phrase, so length or sentence punctuation alone is not treated as an error.
FORMAT_PATTERNS = {
    "reasoning_or_markup_tag": re.compile(
        r"</?(?:think|analysis|final|answer)>|```", re.IGNORECASE
    ),
    "role_or_prompt_echo": re.compile(
        r"(?:^|\n)\s*(?:human|assistant|ai|user)\s*:"
        r"|(?:the\s+)?AI\s+(?:responds?|should\s+respond)"
        r"|return\s+(?:me\s+)?only"
        r"|separat(?:e|ed)\s+(?:each|the|all)\s+answer",
        re.IGNORECASE,
    ),
    "markdown_or_numbered_list": re.compile(
        # At the start of an element, strings such as "1. FC Köln" are valid
        # entity names. A marker following a newline is unambiguously an
        # embedded list because the requested format is pipe-delimited.
        r"\n\s*(?:[-*•]\s+|\d+[.)]\s+)"
    ),
    "question_or_answer_label": re.compile(
        r"(?:^|\n)\s*(?:Q[1-4]|answer)\s*:", re.IGNORECASE
    ),
    "explanatory_preamble": re.compile(
        r"^\s*(?:"
        r"here(?:'s|\s+is)\s+(?:a|an|the|my|your|our|one|what)\b"
        r"|here\s+are\s+(?:a|some|the|my|your|our)\b"
        r"|according\s+to"
        r"|based\s+on"
        r"|after\s+(?:researching|checking|reviewing)"
        r"|please\s+note"
        r"|note\s*:"
        r"|i\s+apologize"
        r"|i\s+understand\s+(?:your|the|that|why)"
        r"|i\s+(?:cannot|can't)\s+"
        r"(?:answer|provide|find|determine|identify|access|verify|confirm|list|help)"
        r"|i\s+can\s+provide"
        r"|i\s+(?:do\s+not|don't)\s+have\s+"
        r"(?:(?:enough|sufficient)\s+)?(?:information|access|data)"
        r"|i(?:'m| am)\s+(?:unable|not\s+able)"
        r"|the\s+(?:answer\s+(?:is|to)|following\s+(?:is|are))"
        r"|i\s+appreciate\s+you"
        r"|you\s+are\s+(?:correct|right)"
        r"|to\s+(?:answer|provide)"
        r"|sure\s*[,!:]"
        r"|certainly\s*[,!:]"
        r")",
        re.IGNORECASE,
    ),
}

VALID_NULL_RESPONSES = {"idk", "no answer"}


@dataclass(frozen=True)
class AnswerFile:
    action: str
    dataset: str
    question: str
    relation: str
    model: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--answers-root",
        type=Path,
        default=REPO_ROOT / "data" / "answers",
        help="Directory containing zero-shot, CtE, and Oracle answer folders.",
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=REPO_ROOT / "data" / "llm_info.json",
        help="JSON object whose keys are the paper's model names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "format_audit",
        help="Directory in which audit CSVs are written.",
    )
    parser.add_argument(
        "--include-repeat",
        action="store_true",
        help="Also audit the repeated Q1 (Q1*) Base responses.",
    )
    return parser.parse_args()


def load_models(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by model.")
    return list(data)


def infer_model(filename: str, models: Iterable[str]) -> str | None:
    # Longest first prevents, e.g., gpt-5 from matching gpt-5-mini.
    aliases = {"deeepseek-reasoner": "deepseek-reasoner"}
    for alias, canonical in aliases.items():
        if alias in filename and canonical in models:
            return canonical
    return next(
        (model for model in sorted(models, key=len, reverse=True) if model in filename),
        None,
    )


def infer_dataset(path: Path) -> str:
    aliases = {
        "lc-quad": "LC-QuAD",
        "qawiki": "QAWiki",
        "spinach": "SPINACH",
        "synthetic": "Synthetic",
    }
    for part in path.parts:
        if part.lower() in aliases:
            return aliases[part.lower()]
    return "unknown"


def discover_answer_files(
    answers_root: Path, models: list[str], include_repeat: bool
) -> list[AnswerFile]:
    discovered: list[AnswerFile] = []
    relative_dirs = {
        action: answers_root / relative.name
        for action, relative in ACTION_DIRS.items()
    }

    for action, directory in relative_dirs.items():
        for path in sorted(directory.rglob("Q*.json")):
            name_lower = path.name.lower()
            if "wikidata" in name_lower:
                continue

            is_repeat = "_answers_star_" in name_lower
            if is_repeat:
                if action != "Base" or not include_repeat:
                    continue
                question, relation, output_action = "Q1", "equal", "Base-repeat"
            else:
                match = ANSWER_FILE_RE.match(path.name)
                if not match:
                    continue
                question, relation = match.group(1), match.group(2)
                if (question, relation) not in SCORED_SLOTS[action]:
                    continue
                output_action = action

            model = infer_model(path.name, models)
            if model is None:
                continue
            discovered.append(
                AnswerFile(
                    action=output_action,
                    dataset=infer_dataset(path),
                    question=question,
                    relation=relation,
                    model=model,
                    path=path,
                )
            )
    return discovered


def normalized_null(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().strip("'\"")
    return normalized if normalized in VALID_NULL_RESPONSES else None


def inspect_text(text: str) -> set[str]:
    stripped = text.strip()
    if normalized_null(stripped):
        return set()

    reasons: set[str] = set()
    if re.fullmatch(r"i\s+(?:do\s+not|don't)\s+know[.!]?", stripped, re.IGNORECASE):
        reasons.add("noncanonical_abstention")
    if "\n" in text or "\r" in text:
        reasons.add("line_break")
    for reason, pattern in FORMAT_PATTERNS.items():
        if pattern.search(text):
            reasons.add(reason)

    lowered = stripped.lower()
    if re.search(r"\bidk\b", lowered):
        reasons.add("embedded_idk")
    if "no answer" in lowered:
        reasons.add("embedded_no_answer")
    return reasons


def inspect_response(value: Any) -> tuple[set[str], set[str]]:
    """Return (format errors, storage-schema anomalies) for one response."""
    format_reasons: set[str] = set()
    schema_reasons: set[str] = set()

    if isinstance(value, list):
        null_items = [
            item for item in value if normalized_null(item) in VALID_NULL_RESPONSES
        ]
        if null_items and len(value) > 1:
            format_reasons.add("mixed_null_and_answers")
        for item in value:
            if isinstance(item, str):
                format_reasons.update(inspect_text(item))
            else:
                # Numeric entities can be legitimate answers. They violate the
                # expected saved-list schema, but are not attributed to the LLM
                # as a format error without the original raw response.
                schema_reasons.add("non_string_list_item")
        return format_reasons, schema_reasons

    if isinstance(value, str):
        schema_reasons.add("top_level_string")
        # A bare idk/no-answer is valid model output despite inconsistent
        # storage. Other strings can still contain detectable response errors.
        format_reasons.update(inspect_text(value))
        return format_reasons, schema_reasons

    schema_reasons.add(f"top_level_{type(value).__name__}")
    return format_reasons, schema_reasons


def load_records(
    files: list[AnswerFile], repo_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    file_anomalies: list[dict[str, Any]] = []

    for answer_file in files:
        try:
            with answer_file.path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            file_anomalies.append(
                {
                    "model": answer_file.model,
                    "action": answer_file.action,
                    "dataset": answer_file.dataset,
                    "file": str(answer_file.path.relative_to(repo_root)),
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
            )
            continue

        if not isinstance(data, dict):
            file_anomalies.append(
                {
                    "model": answer_file.model,
                    "action": answer_file.action,
                    "dataset": answer_file.dataset,
                    "file": str(answer_file.path.relative_to(repo_root)),
                    "reason": "top_level_json_not_object",
                    "detail": type(data).__name__,
                }
            )
            continue

        for question_id, answer in data.items():
            format_reasons, schema_reasons = inspect_response(answer)
            records.append(
                {
                    "model": answer_file.model,
                    "action": answer_file.action,
                    "dataset": answer_file.dataset,
                    "question": answer_file.question,
                    "relation": answer_file.relation,
                    "question_id": question_id,
                    "detectable_format_error": int(bool(format_reasons)),
                    "format_reasons": ";".join(sorted(format_reasons)),
                    "storage_schema_anomaly": int(bool(schema_reasons)),
                    "schema_reasons": ";".join(sorted(schema_reasons)),
                    "file": str(answer_file.path.relative_to(repo_root)),
                    "parsed_answer_json": json.dumps(answer, ensure_ascii=False),
                }
            )
    return records, file_anomalies


def summarize(
    records: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(str(record[field]) for field in group_fields)].append(record)

    result = []
    for key, group in groups.items():
        checked = len(group)
        errors = sum(int(row["detectable_format_error"]) for row in group)
        anomalies = sum(int(row["storage_schema_anomaly"]) for row in group)
        reason_counts = Counter(
            reason
            for row in group
            for reason in str(row["format_reasons"]).split(";")
            if reason
        )
        summary_row: dict[str, Any] = dict(zip(group_fields, key))
        summary_row.update(
            {
                "responses_checked": checked,
                "detectable_format_errors": errors,
                "detectable_format_error_rate": round(errors / checked, 6)
                if checked
                else 0.0,
                "storage_schema_anomalies": anomalies,
                "format_reason_counts": ";".join(
                    f"{reason}={count}" for reason, count in sorted(reason_counts.items())
                ),
            }
        )
        result.append(summary_row)
    return sorted(result, key=lambda row: tuple(row[field] for field in group_fields))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    answers_root = args.answers_root.resolve()
    repo_root = REPO_ROOT.resolve()
    models = load_models(args.models_file)
    files = discover_answer_files(answers_root, models, args.include_repeat)
    records, file_anomalies = load_records(files, repo_root)

    if not records:
        raise SystemExit("No answer records matched the requested audit scope.")

    errors = [row for row in records if row["detectable_format_error"]]
    schema_anomalies = [row for row in records if row["storage_schema_anomaly"]]
    by_model = summarize(records, ("model",))
    by_model_action = summarize(records, ("model", "action"))

    detail_fields = [
        "model",
        "action",
        "dataset",
        "question",
        "relation",
        "question_id",
        "detectable_format_error",
        "format_reasons",
        "storage_schema_anomaly",
        "schema_reasons",
        "file",
        "parsed_answer_json",
    ]
    summary_fields = [
        "model",
        "responses_checked",
        "detectable_format_errors",
        "detectable_format_error_rate",
        "storage_schema_anomalies",
        "format_reason_counts",
    ]
    action_summary_fields = [
        "model",
        "action",
        "responses_checked",
        "detectable_format_errors",
        "detectable_format_error_rate",
        "storage_schema_anomalies",
        "format_reason_counts",
    ]

    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "output_format_errors.csv", errors, detail_fields)
    write_csv(
        output_dir / "output_format_errors_by_model.csv", by_model, summary_fields
    )
    write_csv(
        output_dir / "output_format_errors_by_model_action.csv",
        by_model_action,
        action_summary_fields,
    )
    write_csv(
        output_dir / "storage_schema_anomalies.csv",
        schema_anomalies,
        detail_fields,
    )
    write_csv(
        output_dir / "unreadable_or_invalid_files.csv",
        file_anomalies,
        ["model", "action", "dataset", "file", "reason", "detail"],
    )

    total = len(records)
    error_count = len(errors)
    print(
        f"Audited {total:,} saved responses from {len(files):,} files; "
        f"found {error_count:,} detectable format errors "
        f"({error_count / total:.2%})."
    )
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()

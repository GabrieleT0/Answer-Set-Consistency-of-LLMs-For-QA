"""Generate the two paired controls.

Run from the repository root, for example::

    venv/bin/python -m src.get_answers.relation_aware_ablation \
        --model gpt-4.1-mini-2025-04-14 --max-items 10

Each task has an independent conversation, matching the original CtE/Oracle
scripts. Checkpoints retain the *raw* initial dialogue so the two correction
arms can be resumed and replayed from precisely the same context.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from . import llms, utils


ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = Path(__file__).with_name("prompts.yaml")
DATASETS = ("LC-QuAD.tsv", "qawiki.tsv", "spinach.tsv", "synthetic.tsv")
TASKS = {
    "equal": ("Q1", "Q2"),
    "sup-sub": ("Q1", "Q3"),
    "minus": ("Q1", "Q3", "Q4"),
}
RELATION_FEEDBACK = {
    "equal": "The first and second questions have equivalent answer sets.",
    "sup-sub": "The answer set of the second question must be a subset of that of the first question.",
    "minus": "The answer set of the third question must equal the first answer set minus the second answer set.",
}
RECHECK_PROMPT = "Please check and answer again."
SCHEMA_VERSION = 2


class UnusableModelResponse(RuntimeError):
    """A response could not be used after retrying the same dialogue turn."""


def _empty_provider_response(error: Exception) -> bool:
    """Recognize providers that fail while converting null content to text."""
    get_errors = getattr(error, "errors", None)
    if not callable(get_errors):
        return False
    try:
        details = get_errors()
    except Exception:
        return False
    return any(
        tuple(detail.get("loc", ())[:1]) == ("content",)
        and detail.get("input") is None
        for detail in details
    )


def answer_list(raw: str) -> list[str]:
    if raw.strip().lower() == "idk":
        return ["idk"]
    if raw.strip().lower() == "no answer":
        return []
    return utils.convert_response_to_set(raw)


def is_abstention(answer: list[str]) -> bool:
    return not answer or any(value.strip().lower() == "idk" for value in answer)


def is_consistent(task: str, answers: dict[str, list[str]]) -> bool:
    if any(is_abstention(answers[q]) for q in TASKS[task]):
        return False
    sets = {q: set(answers[q]) for q in TASKS[task]}
    if task == "equal":
        return sets["Q1"] == sets["Q2"]
    if task == "sup-sub":
        return sets["Q3"].issubset(sets["Q1"])
    return sets["Q4"] == sets["Q1"] - sets["Q3"]


def new_conversation(chat, turns: list[dict[str, str]] | None = None):
    """Recreate ConversationChain history, including raw assistant replies."""
    memory = ConversationBufferMemory()
    for turn in turns or []:
        memory.chat_memory.add_user_message(turn["user"])
        memory.chat_memory.add_ai_message(turn["assistant"])
    return ConversationChain(llm=chat, memory=memory)


def ask(chat, turns: list[dict[str, str]], prompt: str, *,
        attempts: int = 3, retry_delay: float = 1.0) -> str:
    """Retry an unusable completion from the last *valid* transcript.

    ConversationChain records blank replies in its memory. Rebuilding the
    chain on each attempt prevents a failed completion from entering the next
    request's context, while keeping every successful turn unchanged.
    """
    for attempt in range(attempts):
        conversation = new_conversation(chat, turns)
        try:
            raw = conversation.predict(input=prompt)
        except Exception as error:
            if not _empty_provider_response(error):
                raise
            raw = None
        if isinstance(raw, str) and raw.strip():
            turns.append({"user": prompt, "assistant": raw})
            return raw
        if attempt + 1 < attempts:
            time.sleep(retry_delay * (attempt + 1))
    raise UnusableModelResponse(f"Empty or non-text model response after {attempts} attempts")


def classification_prompt(task: str, questions: dict[str, str], prompts: dict) -> str:
    if task == "minus":
        return prompts["relation_classification_minus"]["en"]["template_classification"].format(
            q1=questions["Q1"], q2=questions["Q3"], q3=questions["Q4"]
        )
    return prompts["relation_classification"]["en"]["template_classification"].format(
        q1=questions["Q1"], q2=questions[TASKS[task][1]]
    )


def run_enumeration(chat, task: str, questions: dict[str, str], prompts: dict, *,
                    classify: bool, attempts: int = 3, retry_delay: float = 1.0) -> dict:
    turns: list[dict[str, str]] = []
    classification = None
    if classify:
        classification = ask(chat, turns, classification_prompt(task, questions, prompts),
                             attempts=attempts, retry_delay=retry_delay)
    suffix = prompts["relation_classification"]["en"]["template"]
    raw_answers = {
        q: ask(chat, turns, questions[q] + suffix, attempts=attempts, retry_delay=retry_delay)
        for q in TASKS[task]
    }
    return {
        "turns": turns,
        "classification": classification,
        "raw_answers": raw_answers,
        "answers": {q: answer_list(raw) for q, raw in raw_answers.items()},
    }


def run_initial(chat, task: str, questions: dict[str, str], prompts: dict, *,
                attempts: int = 3, retry_delay: float = 1.0) -> dict:
    turns: list[dict[str, str]] = []
    suffix = prompts["fix"]["en"]["template"]
    raw_answers = {
        q: ask(chat, turns, questions[q] + "\n" + suffix,
               attempts=attempts, retry_delay=retry_delay)
        for q in TASKS[task]
    }
    answers = {q: answer_list(raw) for q, raw in raw_answers.items()}
    return {
        "turns": turns,
        "raw_answers": raw_answers,
        "answers": answers,
        "consistent": is_consistent(task, answers),
        "eligible_for_revision": (
            all(not is_abstention(answers[q]) for q in TASKS[task])
            and not is_consistent(task, answers)
        ),
    }


def run_branch(chat, initial: dict, prompt: str, *,
               attempts: int = 3, retry_delay: float = 1.0) -> dict:
    # This function never mutates initial, so both arms receive identical history.
    turns = [dict(turn) for turn in initial["turns"]]
    raw = ask(chat, turns, prompt, attempts=attempts, retry_delay=retry_delay)
    return {"prompt": prompt, "raw_answer": raw, "answer": answer_list(raw)}


def relation_prompt(task: str) -> str:
    return RELATION_FEEDBACK[task] + " " + RECHECK_PROMPT


def load_checkpoint(path: Path, metadata: dict) -> dict:
    if not path.exists():
        return {**metadata, "records": {}}
    with path.open(encoding="utf-8") as handle:
        saved = json.load(handle)
    for key, expected in metadata.items():
        if saved.get(key) != expected:
            raise ValueError(f"Checkpoint {path} has a different {key}; use a new output directory")
    return saved


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run(args, *, chat=None) -> None:
    load_dotenv(ROOT / "src" / ".env")
    prompts_bytes = PROMPTS_PATH.read_bytes()
    prompts = yaml.safe_load(prompts_bytes)
    chat = chat or llms.return_chat_model(args.model, temperature=0)
    attempts = getattr(args, "response_attempts", 3)
    retry_delay = getattr(args, "retry_delay", 1.0)
    output_dir = Path(args.output_dir)
    for dataset in args.datasets:
        path = ROOT / "data" / "ASCB" / "en" / dataset
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if args.max_items is not None:
            rows = rows[args.start : args.start + args.max_items]
        else:
            rows = rows[args.start :]
        for task in args.tasks:
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "model": args.model,
                "dataset": dataset,
                "task": task,
                "prompts_sha256": hashlib.sha256(prompts_bytes).hexdigest(),
                "feedback_sha256": hashlib.sha256(
                    (RECHECK_PROMPT + relation_prompt(task)).encode("utf-8")
                ).hexdigest(),
            }
            checkpoint_path = output_dir / utils.output_model_name(args.model) / dataset.replace(".tsv", "") / f"{task}.json"
            checkpoint = load_checkpoint(checkpoint_path, metadata)
            for offset, row in enumerate(rows, start=args.start):
                key = str(offset)
                questions = {q: row[q] for q in TASKS[task]}
                record = checkpoint["records"].setdefault(key, {
                    "row_index": offset,
                    "question_id": row.get("ID", key),
                    "questions": questions,
                })
                if record["questions"] != questions:
                    raise ValueError(f"Questions changed for {dataset} row {key}")

                def complete(field: str, value: dict) -> None:
                    record[field] = value
                    record.get("unusable_responses", {}).pop(field, None)
                    save_checkpoint(checkpoint_path, checkpoint)

                def try_complete(field: str, generate) -> bool:
                    if field in record:
                        return True
                    try:
                        value = generate()
                    except UnusableModelResponse as error:
                        record.setdefault("unusable_responses", {})[field] = str(error)
                        save_checkpoint(checkpoint_path, checkpoint)
                        print(f"{dataset} {task} {key} {field}: {error}; continuing", flush=True)
                        return False
                    complete(field, value)
                    return True

                try_complete("sequential", lambda: run_enumeration(
                    chat, task, questions, prompts, classify=False,
                    attempts=attempts, retry_delay=retry_delay,
                ))
                try_complete("cte", lambda: run_enumeration(
                    chat, task, questions, prompts, classify=True,
                    attempts=attempts, retry_delay=retry_delay,
                ))
                try_complete("initial", lambda: run_initial(
                    chat, task, questions, prompts,
                    attempts=attempts, retry_delay=retry_delay,
                ))
                if "initial" in record and record["initial"]["eligible_for_revision"]:
                    if "branch_order" not in record:
                        record["branch_order"] = (
                            ["recheck", "relation"] if offset % 2 == 0
                            else ["relation", "recheck"]
                        )
                        save_checkpoint(checkpoint_path, checkpoint)
                    for branch in record["branch_order"]:
                        prompt = RECHECK_PROMPT if branch == "recheck" else relation_prompt(task)
                        try_complete(branch, lambda prompt=prompt: run_branch(
                            chat, record["initial"], prompt,
                            attempts=attempts, retry_delay=retry_delay,
                        ))
                required = {"sequential", "cte", "initial"}
                if record.get("initial", {}).get("eligible_for_revision"):
                    required.update(("recheck", "relation"))
                missing = sorted(required - record.keys())
                status = "complete" if not missing else f"incomplete ({', '.join(missing)})"
                print(f"{dataset} {task} {key}: {status}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None, help="Rows per dataset; omit for all 600 groups")
    parser.add_argument("--response-attempts", type=int, default=3,
                        help="Attempts for an empty/null completion before recording a failure")
    parser.add_argument("--retry-delay", type=float, default=1.0,
                        help="Seconds between retries of an unusable completion")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "relation_aware_ablation"))
    args = parser.parse_args(argv)
    if args.start < 0 or (args.max_items is not None and args.max_items < 1):
        parser.error("--start must be nonnegative and --max-items must be positive")
    if args.response_attempts < 1 or args.retry_delay < 0:
        parser.error("--response-attempts must be positive and --retry-delay nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())

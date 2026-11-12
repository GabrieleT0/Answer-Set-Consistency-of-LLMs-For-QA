"""Generate the two paired controls.

Run from the repository root, for example::

    .venv/bin/python -m src.get_answers.relation_aware_ablation \
        --model gpt-4.1-mini-2025-04-14 --max-items 10 --workers 8

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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock, local

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


class ContentFilteredResponse(RuntimeError):
    """The provider blocked a request; no answer is available for this stage."""


def _content_filtered(error: Exception) -> bool:
    # Azure's LangChain adapter raises this ValueError for filtered completions.
    if isinstance(error, ValueError) and (
        "Azure has not provided the response due to a content filter being triggered"
        in str(error)
    ):
        return True
    # Prompt filtering can instead arrive as a structured provider API error.
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict):
            codes = [details.get("code")]
            inner = details.get("innererror")
            if isinstance(inner, dict):
                codes.append(inner.get("code"))
            return any(str(code).lower() in {"content_filter", "responsibleaipolicyviolation"}
                       for code in codes)
    return False


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
            if _content_filtered(error):
                raise ContentFilteredResponse("Provider content filter blocked this request") from error
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


def _process_record(job, *, dataset, task, prompts, get_chat, publish,
                    attempts, retry_delay, stop):
    offset, record = job
    key = str(offset)
    questions = record["questions"]

    def complete(field: str, value: dict) -> None:
        record[field] = value
        record.get("unusable_responses", {}).pop(field, None)
        record.get("blocked_responses", {}).pop(field, None)
        publish(record)

    def try_complete(field: str, generate) -> bool:
        if stop.is_set():
            return False
        if field in record:
            return True
        try:
            value = generate()
        except ContentFilteredResponse as error:
            record.setdefault("blocked_responses", {})[field] = str(error)
            record.get("unusable_responses", {}).pop(field, None)
            publish(record)
            print(f"{dataset} {task} {key} {field}: content-filtered; continuing", flush=True)
            return False
        except UnusableModelResponse as error:
            record.setdefault("unusable_responses", {})[field] = str(error)
            record.get("blocked_responses", {}).pop(field, None)
            publish(record)
            print(f"{dataset} {task} {key} {field}: {error}; continuing", flush=True)
            return False
        complete(field, value)
        return True

    try_complete("sequential", lambda: run_enumeration(
        get_chat(), task, questions, prompts, classify=False,
        attempts=attempts, retry_delay=retry_delay,
    ))
    try_complete("cte", lambda: run_enumeration(
        get_chat(), task, questions, prompts, classify=True,
        attempts=attempts, retry_delay=retry_delay,
    ))
    try_complete("initial", lambda: run_initial(
        get_chat(), task, questions, prompts,
        attempts=attempts, retry_delay=retry_delay,
    ))
    if "initial" in record and record["initial"]["eligible_for_revision"]:
        if "branch_order" not in record:
            record["branch_order"] = (
                ["recheck", "relation"] if offset % 2 == 0
                else ["relation", "recheck"]
            )
            publish(record)
        for branch in record["branch_order"]:
            prompt = RECHECK_PROMPT if branch == "recheck" else relation_prompt(task)
            try_complete(branch, lambda prompt=prompt: run_branch(
                get_chat(), record["initial"], prompt,
                attempts=attempts, retry_delay=retry_delay,
            ))
    required = {"sequential", "cte", "initial"}
    if record.get("initial", {}).get("eligible_for_revision"):
        required.update(("recheck", "relation"))
    missing = sorted(required - record.keys())
    status = "complete" if not missing else f"incomplete ({', '.join(missing)})"
    blocked = sorted(set(missing) & record.get("blocked_responses", {}).keys())
    if blocked:
        status += f"; content-filtered: {', '.join(blocked)}"
    print(f"{dataset} {task} {key}: {status}", flush=True)


def _run_jobs(jobs, process, workers, stop):
    """Keep at most workers records in flight, including after an interruption."""
    if workers == 1:
        for job in jobs:
            process(job)
        return
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qa")
    pending = set()
    remaining = iter(jobs)
    try:
        for _ in range(workers):
            job = next(remaining, None)
            if job is None:
                break
            pending.add(executor.submit(process, job))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            for _ in done:
                job = next(remaining, None)
                if job is not None:
                    pending.add(executor.submit(process, job))
    except BaseException:
        stop.set()
        for future in pending:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def run(args, *, chat=None, chat_factory=None) -> None:
    workers = getattr(args, "workers", 8)
    if workers < 1:
        raise ValueError("workers must be positive")
    if chat is not None and workers != 1:
        raise ValueError("For multiple workers use chat_factory, not a shared chat instance")
    load_dotenv(ROOT / "src" / ".env")
    prompts_bytes = PROMPTS_PATH.read_bytes()
    prompts = yaml.safe_load(prompts_bytes)
    attempts = getattr(args, "response_attempts", 3)
    retry_delay = getattr(args, "retry_delay", 1.0)
    clients = local()

    def get_chat():
        if not hasattr(clients, "chat"):
            if chat is not None:
                clients.chat = chat
            elif chat_factory is not None:
                clients.chat = chat_factory()
            else:
                clients.chat = llms.return_chat_model(args.model, temperature=0)
        return clients.chat

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
            jobs = []
            for offset, row in enumerate(rows, start=args.start):
                key = str(offset)
                questions = {q: row[q] for q in TASKS[task]}
                record = deepcopy(checkpoint["records"].get(key, {
                    "row_index": offset,
                    "question_id": row.get("ID", key),
                    "questions": questions,
                }))
                if record["questions"] != questions:
                    raise ValueError(f"Questions changed for {dataset} row {key}")
                jobs.append((offset, record))

            checkpoint_lock = Lock()
            stop = Event()

            def publish(record):
                # Workers own their local records. Only this locked section may
                # update the shared checkpoint or replace its temporary file.
                with checkpoint_lock:
                    checkpoint["records"][str(record["row_index"])] = deepcopy(record)
                    save_checkpoint(checkpoint_path, checkpoint)

            def process(job):
                _process_record(job, dataset=dataset, task=task, prompts=prompts,
                                get_chat=get_chat, publish=publish, attempts=attempts,
                                retry_delay=retry_delay, stop=stop)

            _run_jobs(jobs, process, workers, stop)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--workers", type=int, default=8,
                        help="Independent question groups processed concurrently (default: 8)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None, help="Rows per dataset; omit for all 600 groups")
    parser.add_argument("--response-attempts", type=int, default=3,
                        help="Attempts for an empty/null completion before recording a failure")
    parser.add_argument("--retry-delay", type=float, default=1.0,
                        help="Seconds between retries of an unusable completion")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "relation_aware_ablation"))
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.start < 0 or (args.max_items is not None and args.max_items < 1):
        parser.error("--start must be nonnegative and --max-items must be positive")
    if args.response_attempts < 1 or args.retry_delay < 0:
        parser.error("--response-attempts must be positive and --retry-delay nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())

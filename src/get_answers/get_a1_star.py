#!/usr/bin/env python3
"""Complete isolated Q1* Base answers, including partially finished models.

Preview: venv/bin/python -m src.get_answers.get_a1_star
Execute: venv/bin/python -m src.get_answers.get_a1_star --execute --workers 4

Runs one model at a time, with bounded concurrent questions per model. Each
question uses a fresh stateless prompt. Do not run two instances against the
same output directory simultaneously.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from threading import local

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("LC-QuAD", "qawiki", "spinach", "synthetic")
DEFAULT_PROVIDER_MODELS = {
    "gpt-4.1-nano-2025-04-14": "openai/gpt-4.1-nano",
    "gemini-2.5-flash": "google/gemini-2.5-flash:nitro",
    "llama3.1:8b": "meta-llama/llama-3.1-8b-instruct:nitro",
    "gpt-oss:20b": "openai/gpt-oss-20b:nitro",
}
PROVIDER_OUTPUT_ALIASES = {
    "meta-llama/llama-3.1-70b-instruct": "llama3.1:70b",
}


def load_questions(dataset: str) -> list[str]:
    path = ROOT / "data/ASCB/en" / f"{dataset}.tsv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or any(not row.get("Q1", "").strip() for row in rows):
        raise ValueError(f"Missing Q1 question in {path}")
    return [row["Q1"] for row in rows]


def answer_path(output_dir: Path, dataset: str, model: str) -> Path:
    if "/" in model or "\\" in model:
        raise ValueError(
            "Output model names cannot contain path separators; use the saved "
            "model name with --models and the provider ID with --provider-model")
    return (output_dir / dataset.lower() / "equal" /
            f"Q1_equal_answers_star_{model}.json")


def load_answers(path: Path, n_questions: int) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    answers = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(answers, dict):
        raise ValueError(f"Expected an object in {path}")
    for key, value in answers.items():
        if not key.isdigit() or not 0 <= int(key) < n_questions or not isinstance(value, list):
            raise ValueError(f"Unexpected answer entry {key!r} in {path}")
    return answers


def save_answers(path: Path, answers: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(answers, ensure_ascii=False, indent=4) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pending_indices(answers: dict[str, list[str]], count: int,
                    retry_empty: bool = False) -> list[int]:
    return [i for i in range(count)
            if str(i) not in answers or (retry_empty and not answers[str(i)])]


def collect_answers(questions: list[str], indices: list[int], answers: dict[str, list[str]],
                    path: Path, request, workers: int,
                    max_consecutive_errors: int, *,
                    provenance_path: Path | None = None,
                    output_model: str | None = None,
                    provider_model: str | None = None,
                    dataset: str | None = None) -> tuple[int, int, bool]:
    """Bound requests and checkpoint each success from the main thread."""
    completed = errors = consecutive_errors = 0
    remaining = iter(indices)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}

        def submit_next() -> None:
            try:
                index = next(remaining)
            except StopIteration:
                return
            active[pool.submit(request, questions[index])] = index

        for _ in range(min(workers, len(indices))):
            submit_next()
        stopping = False
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                index = active.pop(future)
                try:
                    value = future.result()
                    if not isinstance(value, list):
                        raise TypeError("Parsed answer is not a list")
                except Exception as exc:
                    errors += 1
                    consecutive_errors += 1
                    print(f"{path.name} Q1[{index}]: {type(exc).__name__}: {exc}", flush=True)
                else:
                    answers[str(index)] = value
                    save_answers(path, answers)
                    if provenance_path is not None:
                        record = {
                            "output_model": output_model,
                            "provider_model": provider_model,
                            "dataset": dataset,
                            "question_id": "Q1",
                            "index": index,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        with provenance_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed += 1
                    consecutive_errors = 0
                    print(f"{path.name} Q1[{index}]: saved", flush=True)
                if consecutive_errors >= max_consecutive_errors:
                    stopping = True
                if not stopping:
                    submit_next()
            if stopping:
                # Running calls still finish and get saved; no new calls are sent.
                print(f"Stopping new calls for {path.name} after repeated errors.", flush=True)
    return completed, errors, stopping


def make_request(model: str):
    """Give each worker thread its own model client and stateless prompt chain."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / "src/.env")
    if __package__:
        from .llms import return_chat_model
        from .single_question_benchmark import get_prompt
        from .utils import convert_response_to_set
    else:
        from llms import return_chat_model
        from single_question_benchmark import get_prompt
        from utils import convert_response_to_set

    prompt = get_prompt("standard", "en")
    state = local()

    def request(question: str) -> list[str]:
        if not hasattr(state, "chain"):
            state.chain = prompt | return_chat_model(model, temperature=0)
        response = state.chain.invoke({"question": question})
        raw = response.content if hasattr(response, "content") else str(response)
        if not isinstance(raw, str):
            raise TypeError("Model returned non-text content")
        return convert_response_to_set(raw)

    return request


def provider_model_id(model: str) -> str:
    """Accept a raw OpenRouter model ID as well as the explicit prefix."""
    return model if "/" not in model or model.startswith("openrouter:") else (
        "openrouter:" + model)


def recover_misrouted_answers(
        output_dir: Path, dataset: str, output_model: str, provider_model: str,
        answers: dict[str, list[str]], n_questions: int, *,
        save: bool) -> tuple[dict[str, list[str]], int, int]:
    """Copy completed provider-ID answers without replacing established answers.

    Earlier versions interpreted a slash in --models as a path separator. The
    misplaced JSON is read as a snapshot, so this also works while that older
    process continues to checkpoint it. Calling again imports newly saved keys.
    """
    raw_provider = provider_model.removeprefix("openrouter:")
    if "/" not in raw_provider:
        return answers, 0, 0
    legacy_path = (output_dir / dataset.lower() / "equal" /
                   f"Q1_equal_answers_star_{raw_provider}.json")
    if not legacy_path.exists():
        return answers, 0, 0
    legacy = load_answers(legacy_path, n_questions)
    added = {key: value for key, value in legacy.items() if key not in answers}
    preserved = sum(key in answers for key in legacy)
    merged = {**answers, **added}
    if save and added:
        save_answers(answer_path(output_dir, dataset, output_model), merged)
        provenance = output_dir / "Q1_star_provenance.jsonl"
        with provenance.open("a", encoding="utf-8") as handle:
            for key in sorted(added, key=int):
                handle.write(json.dumps({
                    "output_model": output_model,
                    "provider_model": provider_model,
                    "dataset": dataset,
                    "question_id": "Q1",
                    "index": int(key),
                    "source": "recovered_misrouted_file",
                    "source_path": str(legacy_path),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
    return merged, len(added), preserved


def output_model_id(selected: str) -> str:
    """Map known provider IDs to the stable output names used by evaluation."""
    if "/" not in selected and not selected.startswith("openrouter:"):
        return selected
    provider = provider_model_id(selected)
    if output_model := PROVIDER_OUTPUT_ALIASES.get(provider.removeprefix("openrouter:")):
        return output_model
    for output_model, configured_provider in DEFAULT_PROVIDER_MODELS.items():
        if provider_model_id(configured_provider) == provider:
            return output_model
    raise ValueError(
        f"No saved output model is mapped to {selected!r}; pass its saved name "
        "with --models and use --provider-model for the provider ID")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        help="Default: all models in data/llm_info.json")
    parser.add_argument("--provider-model",
                        help="Provider model for a single --models output name; "
                             "raw OpenRouter IDs are accepted")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/answers/zero-shot")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent Q1 calls on the same model (default: 1)")
    parser.add_argument("--max-calls", type=int, help="Limit new calls per model")
    parser.add_argument("--max-consecutive-errors", type=int, default=5)
    parser.add_argument("--retry-empty", action="store_true",
                        help="Re-query saved empty lists")
    parser.add_argument("--recover-misrouted", action="store_true",
                        help="Copy saved answers from older provider-ID folders "
                             "into the established model files without API calls")
    parser.add_argument("--execute", action="store_true",
                        help="Call providers; otherwise only preview")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_consecutive_errors < 1:
        parser.error("--workers and --max-consecutive-errors must be positive")
    if args.max_calls is not None and args.max_calls < 1:
        parser.error("--max-calls must be positive")
    selected = args.models or json.loads(
        (ROOT / "data/llm_info.json").read_text(encoding="utf-8")).keys()
    try:
        models = {}
        for selected_model in selected:
            output_model = output_model_id(selected_model)
            if output_model not in models:
                provider = (selected_model if "/" in selected_model or
                            selected_model.startswith("openrouter:") else
                            DEFAULT_PROVIDER_MODELS.get(output_model, output_model))
                models[output_model] = provider_model_id(provider)
    except ValueError as exc:
        parser.error(str(exc))
    if args.provider_model and len(models) != 1:
        parser.error("--provider-model requires exactly one --models name")
    total_pending = 0
    for model, default_provider in models.items():
        provider = provider_model_id(args.provider_model or default_provider)
        if provider != model:
            print(f"Saving as {model}; calling provider model {provider}", flush=True)
        model_calls = 0
        request = None
        for dataset in args.datasets:
            questions = load_questions(dataset)
            path = answer_path(args.output_dir, dataset, model)
            answers = load_answers(path, len(questions))
            answers, recovered, preserved = recover_misrouted_answers(
                args.output_dir, dataset, model, provider, answers, len(questions),
                save=args.execute or args.recover_misrouted)
            if recovered or preserved:
                verb = "recovered" if args.execute or args.recover_misrouted else "recoverable"
                print(f"{model} {dataset}: {recovered} {verb} from provider-ID folder; "
                      f"{preserved} existing answers retained", flush=True)
            pending = pending_indices(answers, len(questions), args.retry_empty)
            if args.max_calls is not None:
                pending = pending[:max(0, args.max_calls - model_calls)]
            total_pending += len(pending)
            print(f"{model} {dataset}: {len(answers)}/{len(questions)} saved, "
                  f"{len(pending)} calls selected", flush=True)
            if args.execute and pending:
                if request is None:
                    request = make_request(provider)
                completed, errors, stopped = collect_answers(
                    questions, pending, answers, path, request, args.workers,
                    args.max_consecutive_errors,
                    provenance_path=(args.output_dir / "Q1_star_provenance.jsonl"
                                     if provider != model else None),
                    output_model=model, provider_model=provider, dataset=dataset)
                print(f"{model} {dataset}: {completed} saved, {errors} errors", flush=True)
                if stopped:
                    print(f"Skipping remaining datasets for {model}; rerun after "
                          "checking provider access.", flush=True)
                    break
            model_calls += len(pending)
    print(f"Selected {total_pending} missing Q1* calls across {len(models)} models."
          + ("" if args.execute else " Add --execute to run them."))


if __name__ == "__main__":
    main()

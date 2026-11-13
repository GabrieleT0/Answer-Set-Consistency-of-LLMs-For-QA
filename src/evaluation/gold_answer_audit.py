#!/usr/bin/env python3
"""Gold-answer audit for the stochasticity and relation-scoring.

Run from the repository root:
    python src/evaluation/gold_answer_audit.py
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from output_format_audit import REPO_ROOT, discover_answer_files, load_models
from reviewer_abstention_audit import unavailable


SLOTS = {"Q1": "Q1/equal", "Q2": "Q2/equal", "Q3": "Q3/sup-sub", "Q4": "Q4/minus"}
DATASETS = {"lc-quad": "LC-QuAD", "qawiki": "QAWiki", "spinach": "SPINACH", "synthetic": "Synthetic"}


def normalize(value: str) -> str:
    """Normalize typography only; do not silently equate different entities."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def parse_set(value: str) -> set[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError("Candidate answer must be a JSON list of strings")
    result = {normalize(x) for x in parsed}
    if "" in result:
        raise ValueError("Candidate answer has an empty entity")
    return result


def answer_set(value: object) -> set[str] | None:
    if unavailable(value):
        return None
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return None
    result = {normalize(x) for x in value}
    return result if "" not in result else None


def gold_groups(path: Path, questions_path: Path) -> tuple[dict[tuple[str, int], dict[str, set[str]]], pd.DataFrame]:
    candidates = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    questions = pd.read_csv(questions_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"dataset", "source_id", "task_id", "question_id", "question", "candidate_answer"}
    if not required <= set(candidates):
        raise ValueError(f"Candidate TSV lacks columns: {sorted(required - set(candidates))}")
    if candidates.duplicated(["dataset", "source_id", "question_id"]).any():
        raise ValueError("Duplicate candidate question rows")
    lookup = {(r.source.casefold(), r.source_id): r for r in questions.itertuples(index=False)}
    valid = {}
    audit = []
    for (dataset, source_id), frame in candidates.groupby(["dataset", "source_id"], sort=True):
        reason = []
        source = lookup.get((dataset.casefold(), source_id))
        if source is None:
            reason.append("missing_benchmark_group")
        if set(frame.question_id) != set(SLOTS):
            reason.append("incomplete_candidate_group")
        gold = {}
        for row in frame.itertuples(index=False):
            if row.question_id not in SLOTS:
                reason.append("unknown_question_id")
                continue
            if source is not None and normalize(row.question) != normalize(getattr(source, row.question_id)):
                reason.append("question_text_mismatch")
            try:
                gold[row.question_id] = parse_set(row.candidate_answer)
            except (ValueError, json.JSONDecodeError):
                reason.append("invalid_candidate_answer")
        if set(gold) == set(SLOTS):
            if gold["Q1"] != gold["Q2"]:
                reason.append("gold_equivalence_fails")
            if gold["Q1"] != gold["Q3"] | gold["Q4"]:
                reason.append("gold_partition_fails")
            if not gold["Q3"].isdisjoint(gold["Q4"]):
                reason.append("gold_disjointness_fails")
        key = (DATASETS[dataset.casefold()], int(source_id))
        audit.append({"dataset": key[0], "source_id": key[1], "task_id": frame.task_id.iloc[0],
                      "included": not reason, "reason": ";".join(sorted(set(reason)))})
        if not reason:
            valid[key] = gold
    return valid, pd.DataFrame(audit)


def load_base(root: Path, models_file: Path) -> dict[tuple[str, str, int], dict[str, object]]:
    records = defaultdict(dict)
    files = discover_answer_files(root, load_models(models_file), include_repeat=True)
    for file in files:
        if file.action not in {"Base", "Base-repeat"}:
            continue
        slot = "Q1/repeat" if file.action == "Base-repeat" else f"{file.question}/{file.relation}"
        with file.path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        for qid, response in saved.items():
            key = (file.dataset, file.model, int(qid))
            if slot in records[key]:
                raise ValueError(f"Duplicate saved response for {key}, {slot}")
            records[key][slot] = response
    return records


def response_metrics(answer: set[str] | None, gold: set[str]) -> dict[str, object]:
    if answer is None:
        return {"available": False, "n_gold": len(gold), "n_answer": None,
                "n_correct": None, "n_omitted": None, "n_extra": None,
                "precision": None, "recall": None, "f1": None,
                "gold_set": json.dumps(sorted(gold)), "answer_set": None,
                "omitted_set": None, "extra_set": None}
    correct = len(answer & gold)
    precision = correct / len(answer) if answer else 0.0
    recall = correct / len(gold) if gold else None
    return {"available": True, "n_gold": len(gold), "n_answer": len(answer),
            "n_correct": correct, "n_omitted": len(gold - answer), "n_extra": len(answer - gold),
            "precision": precision, "recall": recall,
            "f1": (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            if recall is not None else None,
            "gold_set": json.dumps(sorted(gold)), "answer_set": json.dumps(sorted(answer)),
            "omitted_set": json.dumps(sorted(gold - answer)),
            "extra_set": json.dumps(sorted(answer - gold))}


def pair_metrics(left: set[str] | None, right: set[str] | None,
                 gold_left: set[str], gold_right: set[str], kind: str) -> dict[str, object]:
    result = {"available": left is not None and right is not None}
    if not result["available"]:
        return result
    assert left is not None and right is not None
    omissions_left, omissions_right = gold_left - left, gold_right - right
    relation = {"equivalence": left == right,
                "containment": right <= left,
                "disjointness": left.isdisjoint(right)}[kind]
    true_positive_relation = {"equivalence": (left & gold_left) == (right & gold_right),
                              "containment": (right & gold_right) <= (left & gold_left),
                              "disjointness": (left & gold_left).isdisjoint(right & gold_right)}[kind]
    result.update({
        "relation_holds": relation,
        "any_omission": bool(omissions_left or omissions_right),
        "both_exact_gold": left == gold_left and right == gold_right,
        "both_no_extras": left <= gold_left and right <= gold_right,
        "different_gold_facts_listed": bool((left & gold_left) ^ (right & gold_right)) if kind == "equivalence" else None,
        "different_gold_omissions": omissions_left != omissions_right if kind == "equivalence" else None,
        "n_gold_facts_differ": len((left & gold_left) ^ (right & gold_right)) if kind == "equivalence" else None,
        "n_omitted_total": len(omissions_left) + len(omissions_right),
        "n_extra_total": len(left - gold_left) + len(right - gold_right),
        "relation_on_true_positives": true_positive_relation,
        "n_cross_answer_overlap": len(left & right),
        "mean_recall": (len(left & gold_left) / len(gold_left) + len(right & gold_right) / len(gold_right)) / 2
        if gold_left and gold_right else None,
    })
    return result


def summarize(pairs: pd.DataFrame, responses: pd.DataFrame, quality_path: Path) -> pd.DataFrame:
    rows = []
    for (model, relation), frame in pairs.groupby(["model", "relation"], sort=True):
        valid = frame.loc[frame.available]
        d = {"model": model, "relation": relation, "n_gold_groups": len(frame), "n_available_pairs": len(valid)}
        if len(valid):
            holds = valid.relation_holds.astype(bool)
            omission = valid.any_omission.astype(bool)
            no_extras = valid.both_no_extras.astype(bool)
            different_facts = valid.different_gold_facts_listed.eq(True)
            for col in ["relation_holds", "any_omission", "both_exact_gold", "both_no_extras"]:
                d[col + "_rate"] = valid[col].astype(bool).mean()
            d["n_consistent_with_omission"] = int((holds & omission).sum())
            d["n_inconsistent_with_different_gold_facts"] = int((~holds & different_facts).sum())
            if relation == "disjoint_Q3_Q4":
                d["n_disjoint_with_omission_no_extras"] = int((holds & omission & no_extras).sum())
        rows.append(d)
    summary = pd.DataFrame(rows)
    # Per-question quality gives a relation-independent companion to relation predicates.
    quality = responses.loc[responses.available].groupby(["model", "question_id"])[["precision", "recall", "f1"]].mean()
    quality.to_csv(quality_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=REPO_ROOT / "data/ASCB/en/unified_dataset_w_candidate_answer.tsv")
    parser.add_argument("--questions", type=Path, default=REPO_ROOT / "data/ASCB/en/unified-dataset.tsv")
    parser.add_argument("--answers-root", type=Path, default=REPO_ROOT / "data/answers")
    parser.add_argument("--models-file", type=Path, default=REPO_ROOT / "data/llm_info.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output/gold_answer")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gold, coverage = gold_groups(args.candidates, args.questions)
    coverage.to_csv(args.output_dir / "gold_coverage.csv", index=False)
    candidates = pd.read_csv(args.candidates, sep="\t", dtype=str, keep_default_na=False)
    question_text = {(DATASETS[row.dataset.casefold()], int(row.source_id), row.question_id): row.question
                     for row in candidates.itertuples(index=False)}
    saved = load_base(args.answers_root, args.models_file)
    response_rows, pair_rows = [], []
    for (dataset, model, qid), slots in sorted(saved.items()):
        if (dataset, qid) not in gold:
            continue
        target = gold[dataset, qid]
        answers = {q: answer_set(slots.get(slot)) for q, slot in SLOTS.items()}
        repeat = answer_set(slots.get("Q1/repeat"))
        for q, answer in answers.items():
            response_rows.append({"dataset": dataset, "model": model, "source_id": qid,
                                  "question_id": q, "question": question_text[dataset, qid, q],
                                  **response_metrics(answer, target[q])})
        if "Q1/repeat" in slots:
            response_rows.append({"dataset": dataset, "model": model, "source_id": qid,
                                  "question_id": "Q1_repeat", "question": question_text[dataset, qid, "Q1"],
                                  **response_metrics(repeat, target["Q1"])})
        for label, left, right, gl, gr, kind in [
            ("paraphrase_Q1_Q2", answers["Q1"], answers["Q2"], target["Q1"], target["Q2"], "equivalence"),
            ("containment_Q3_in_Q1", answers["Q1"], answers["Q3"], target["Q1"], target["Q3"], "containment"),
            ("containment_Q4_in_Q1", answers["Q1"], answers["Q4"], target["Q1"], target["Q4"], "containment"),
            ("disjoint_Q3_Q4", answers["Q3"], answers["Q4"], target["Q3"], target["Q4"], "disjointness"),
        ] + ([("repeat_Q1_Q1star", answers["Q1"], repeat, target["Q1"], target["Q1"], "equivalence")]
             if "Q1/repeat" in slots else []):
            pair_rows.append({"dataset": dataset, "model": model, "source_id": qid, "relation": label,
                              **pair_metrics(left, right, gl, gr, kind)})
    responses = pd.DataFrame(response_rows)
    pairs = pd.DataFrame(pair_rows)
    responses.to_csv(args.output_dir / "per_response.csv", index=False)
    pairs.to_csv(args.output_dir / "per_pair.csv", index=False)
    # A reviewable list of the exact model–question cases behind the omission
    # count, with a separate unique-question summary.
    disjoint_omissions = pairs.loc[
        pairs.relation.eq("disjoint_Q3_Q4") & pairs.available &
        pairs.relation_holds.eq(True) & pairs.any_omission.eq(True),
        ["dataset", "source_id", "model", "both_no_extras"],
    ]
    for q in ("Q3", "Q4"):
        detail = responses.loc[responses.question_id.eq(q),
                               ["dataset", "source_id", "model", "question", "gold_set",
                                "answer_set", "omitted_set", "extra_set"]]
        detail = detail.rename(columns={name: f"{q.lower()}_{name}" for name in
                                        ["question", "gold_set", "answer_set", "omitted_set", "extra_set"]})
        disjoint_omissions = disjoint_omissions.merge(
            detail, on=["dataset", "source_id", "model"], how="left", validate="one_to_one")
    disjoint_omissions.to_csv(args.output_dir / "disjoint_pass_with_omissions.csv", index=False)
    question_summary = disjoint_omissions.groupby(["dataset", "source_id", "q3_question", "q4_question"],
                                                   as_index=False).agg(
        n_models=("model", "size"),
        models=("model", lambda x: "; ".join(sorted(x))),
        n_without_extra_entities=("both_no_extras", "sum"),
    )
    question_summary.to_csv(args.output_dir / "disjoint_omission_question_summary.csv", index=False)
    summary = summarize(pairs, responses, args.output_dir / "per_model_question_quality.csv")
    summary.to_csv(args.output_dir / "per_model_relation.csv", index=False)
    pooled_rows = []
    for relation, group in pairs.groupby("relation", sort=True):
        valid = group.loc[group.available]
        holds = valid.relation_holds.astype(bool)
        omission = valid.any_omission.astype(bool)
        no_extras = valid.both_no_extras.astype(bool)
        different_facts = valid.different_gold_facts_listed.eq(True)
        pooled_rows.append({
            "relation": relation, "n_pairs": len(group), "n_available": len(valid),
            "n_holds": int(holds.sum()),
            "n_holds_with_omission": int((holds & omission).sum()),
            "n_holds_with_omission_no_extras": int((holds & omission & no_extras).sum()),
            "n_fails_due_to_different_gold_facts": int((~holds & different_facts).sum()),
        })
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(args.output_dir / "pooled_relation.csv", index=False)
    # Match the two relations on exactly the same item and model, avoiding a
    # sample-composition explanation for their raw score difference.
    comparable = pairs.loc[pairs.relation.isin(["paraphrase_Q1_Q2", "disjoint_Q3_Q4"]) & pairs.available]
    common = comparable.pivot(index=["dataset", "model", "source_id"], columns="relation",
                              values=["relation_holds", "both_exact_gold", "mean_recall"]).dropna(
                                  subset=[("relation_holds", "paraphrase_Q1_Q2"),
                                          ("relation_holds", "disjoint_Q3_Q4")])
    if len(common):
        pd.DataFrame([{
            "n_matched": len(common),
            "equivalence_consistency": common[("relation_holds", "paraphrase_Q1_Q2")].astype(bool).mean(),
            "disjointness_consistency": common[("relation_holds", "disjoint_Q3_Q4")].astype(bool).mean(),
            "equivalence_both_exact_gold": common[("both_exact_gold", "paraphrase_Q1_Q2")].astype(bool).mean(),
            "disjointness_both_exact_gold": common[("both_exact_gold", "disjoint_Q3_Q4")].astype(bool).mean(),
            "equivalence_mean_recall": common[("mean_recall", "paraphrase_Q1_Q2")].mean(),
            "equivalence_recall_n": int(common[("mean_recall", "paraphrase_Q1_Q2")].notna().sum()),
            "disjointness_mean_recall": common[("mean_recall", "disjoint_Q3_Q4")].mean(),
            "disjointness_recall_n": int(common[("mean_recall", "disjoint_Q3_Q4")].notna().sum()),
        }]).to_csv(args.output_dir / "matched_relation_comparison.csv", index=False)
    # Compare paraphrase and repeat only on matched model/group pairs with both observed.
    matched = pairs.loc[pairs.relation.isin(["paraphrase_Q1_Q2", "repeat_Q1_Q1star"]) & pairs.available]
    wide = matched.pivot(index=["dataset", "model", "source_id"], columns="relation", values="relation_holds").dropna()
    if len(wide):
        p = wide["paraphrase_Q1_Q2"].astype(bool)
        r = wide["repeat_Q1_Q1star"].astype(bool)
        b = int((~p & r).sum())
        c = int((p & ~r).sum())
        pd.DataFrame([{"n_matched": len(wide), "paraphrase_consistency": p.mean(),
                       "repeat_consistency": r.mean(), "paraphrase_only_failure": b,
                       "repeat_only_failure": c}]).to_csv(
            args.output_dir / "matched_repeat_vs_paraphrase.csv", index=False)
    print(f"Valid gold groups: {len(gold)}/{len(coverage)}; pair rows: {len(pairs)}")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()

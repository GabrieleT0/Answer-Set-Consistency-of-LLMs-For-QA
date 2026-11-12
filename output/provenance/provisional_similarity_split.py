"""Provisional editorial split of source-derived ASCB questions.

This is an operational similarity/answer-condition review, not a record of
author intent or a proof of answer-set equivalence. It labels all 1,800 slots
so that the counts can be inspected and revised row by row.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent

# First-pass manual decisions on Q1 after viewing nearest raw candidates.
# Only clearly answer-preserving changes are placed here; every other changed
# Q1 is treated as substantially revised/no close source counterpart.
LC_Q1_PARAPHRASE = {
    3, 5, 10, 13, 15, 17, 23, 26, 31, 32, 35, 44, 61, 62, 69, 92, 95,
    97, 101, 114, 116, 142,
}
QAWIKI_Q1_PARAPHRASE = {19, 50}

# Resolve the 13 uncertain direct-anchor SPINACH Q1 pairs under the same
# conservative, answer-condition-aware rule. Reviewers may revise these.
SPINACH_Q1_UNCERTAIN_PARAPHRASE = {36, 61, 62, 83, 119}

# Nonverbatim Q3/Q4 that are close, answer-preserving rewrites of a distinct
# raw QAWiki question. Most Q3/Q4 questions change the Q1 answer condition.
QAWIKI_Q34_PARAPHRASE = {
    (7, "Q3"), (9, "Q3"), (26, "Q3"), (26, "Q4"),
    (34, "Q3"), (53, "Q4"), (113, "Q4"), (134, "Q3"),
}


def main():
    with (HERE / "semantic_review_worklist.tsv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["source"], int(row["source_id"]))][row["slot"]] = row

    q1_labels = {}
    for (source, group_id), slots in groups.items():
        q1 = slots["Q1"]
        if q1["text_match_status"] in ("literal", "normalized"):
            label = "copy_or_formatting"
        elif source == "LC-QuAD":
            label = "paraphrase" if group_id in LC_Q1_PARAPHRASE else "substantially_modified"
        elif source == "qawiki":
            label = "paraphrase" if group_id in QAWIKI_Q1_PARAPHRASE else "substantially_modified"
        else:
            prior = q1["provisional_semantic_label"]
            if prior == "paraphrase":
                label = "paraphrase"
            elif prior == "adapted_answer_condition":
                label = "substantially_modified"
            elif prior == "uncertain":
                label = ("paraphrase" if group_id in SPINACH_Q1_UNCERTAIN_PARAPHRASE
                         else "substantially_modified")
            else:
                raise ValueError((source, group_id, prior))
        q1_labels[(source, group_id)] = label

    output = []
    for row in rows:
        source, group_id, slot = row["source"], int(row["source_id"]), row["slot"]
        status = row["text_match_status"]
        if status == "literal":
            label, basis = "literal_copy", "exact text in original raw source"
        elif status == "normalized":
            label, basis = "formatting_only", "case/whitespace normalization"
        elif slot == "Q1":
            label = q1_labels[(source, group_id)]
            basis = "manual Q1 source-candidate review; provisional"
        elif slot == "Q2":
            label = ("paraphrase" if q1_labels[(source, group_id)] in
                     ("copy_or_formatting", "paraphrase") else "substantially_modified")
            basis = "Q2 intended equivalent to Q1; inherits Q1 source-condition judgment"
        elif source == "qawiki" and (group_id, slot) in QAWIKI_Q34_PARAPHRASE:
            label, basis = "paraphrase", "manual review of close raw QAWiki question"
        else:
            label, basis = "substantially_modified", (
                "Q3/Q4 adds a subset or complement condition to Q1; no literal raw match"
            )
        output.append({**row, "editorial_label": label, "label_basis": basis})

    with (HERE / "provisional_similarity_split.tsv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(output)
    counts = defaultdict(Counter)
    for row in output:
        counts[row["source"]][row["editorial_label"]] += 1
        counts["total"][row["editorial_label"]] += 1
    for source in ("LC-QuAD", "qawiki", "spinach", "total"):
        print(source, dict(counts[source]), "total", sum(counts[source].values()))


if __name__ == "__main__":
    main()

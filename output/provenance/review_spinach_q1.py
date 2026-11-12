"""Record a provisional manual semantic review of source-linked Q1 edits.

Labels apply only to the 150 ASCB rows in the `spinach` subset. They are a
single review pass and should be adjudicated by the paper's authors.
"""

import csv
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent

# The reviewer judged these changed-text pairs to preserve the answer set.
PARAPHRASE = {
    2, 6, 8, 10, 12, 14, 15, 16, 17, 31, 32, 35, 45, 47, 48, 54, 57,
    59, 60, 63, 69, 73, 78, 80, 81, 82, 86, 87, 88, 94, 103, 115, 117,
    120, 131, 139, 140, 142, 145,
}

# These changed-text pairs clearly change entities, time, scope, or filters.
ADAPTED = {
    3, 7, 13, 18, 25, 30, 33, 34, 40, 50, 53, 58, 64, 65, 68, 70,
    72, 74, 76, 85, 92, 93, 95, 100, 101, 104, 105, 112, 114, 124,
    127, 130,
}


def main():
    with (HERE / "spinach_source_anchors.tsv").open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != 150:
        raise ValueError("Expected 150 source-linked Q1 rows")
    for row in rows:
        index = int(row["source_id"])
        if row["Q1_status"] in ("literal", "normalized"):
            label = "copy_or_formatting"
        elif index in PARAPHRASE:
            label = "paraphrase"
        elif index in ADAPTED:
            label = "adapted_answer_condition"
        else:
            label = "uncertain"
        row["provisional_semantic_label"] = label
    with (HERE / "spinach_q1_semantic_review.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(Counter(row["provisional_semantic_label"] for row in rows))


if __name__ == "__main__":
    main()

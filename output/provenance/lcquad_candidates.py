"""Suggest LC-QuAD 2.0 source rows for manual Q1 provenance review.

Lexical retrieval only: candidates and scores do not establish semantic derivation.
"""

import csv
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("ASCB_RAW_DIR", "/tmp/ascb-raw"))
STOP = set("a an and are as at be by did do does for from have how in is it its list of on or the to was were what which who whom whose with".split())


def words(text):
    ascii_text = unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()
    return {x for x in re.findall(r"[a-z0-9]+", ascii_text) if x not in STOP}


def main():
    originals = []
    for split in ("train", "test"):
        for row in json.loads((RAW / f"lcquad2_{split}.json").read_text()):
            for field in ("question", "paraphrased_question"):
                if row.get(field):
                    originals.append((split, str(row["uid"]), field, row[field]))
    token_sets = [words(row[3]) for row in originals]
    postings = defaultdict(list)
    for i, token_set in enumerate(token_sets):
        for token in token_set:
            postings[token].append(i)
    n = len(originals)
    idf = {token: math.log((n + 1) / (len(indices) + 1)) + 1
           for token, indices in postings.items()}
    lengths = [math.sqrt(sum(idf[token] ** 2 for token in token_set))
               for token_set in token_sets]
    with (ROOT / "data/ASCB/en/unified-dataset.tsv").open(newline="") as f:
        questions = [r for r in csv.DictReader(f, delimiter="\t") if r["source"] == "LC-QuAD"]
    rows = []
    for question in questions:
        tokens = words(question["Q1"])
        q_length = math.sqrt(sum(idf.get(token, 0) ** 2 for token in tokens))
        dot = Counter()
        for token in tokens:
            weight = idf.get(token, 0) ** 2
            for index in postings.get(token, ()):
                dot[index] += weight
        scores = [(value / (q_length * lengths[index]), index)
                  for index, value in dot.items() if q_length and lengths[index]]
        for rank, (score, index) in enumerate(sorted(scores, reverse=True)[:5], start=1):
            split, uid, field, original = originals[index]
            rows.append({"ascb_source_id": question["source_id"],
                         "ascb_Q1": question["Q1"], "rank": rank,
                         "source_split": split, "source_uid": uid,
                         "source_field": field, "source_question": original,
                         "token_cosine": f"{score:.4f}"})
    with (Path(__file__).parent / "lcquad_q1_candidates.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", len(rows), "candidate rows for", len(questions), "ASCB Q1 questions")


if __name__ == "__main__":
    main()

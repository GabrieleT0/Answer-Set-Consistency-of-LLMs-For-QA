"""Join exact matches, direct anchors, and retrieval candidates for review.

The output deliberately leaves semantic labels blank where they have not been
reviewed. Nearest-neighbor candidates do not establish provenance.
"""

import csv
from pathlib import Path

from question_provenance_audit import RAW


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def main():
    exact = read_tsv(HERE / "exact_matches.tsv")
    candidates = {
        (row["index"], row["slot"]): row
        for row in read_tsv(HERE / "semantic_candidates.tsv")
        if row["rank"] == "1"
    }
    spinach = {
        row["source_id"]: row
        for row in read_tsv(HERE / "spinach_q1_semantic_review.tsv")
    }
    qawiki = {
        row["ID"]: row
        for row in read_tsv(ROOT / "data/ASCB/en/qawiki.tsv")
    }
    raw_qawiki = {
        row["questionId"]: row
        for row in read_tsv(RAW / "qawiki-v1-simple-2025-09-09.tsv")
    }

    output = []
    for row in exact:
        index, slot, source = row["index"], row["slot"], row["source"]
        if source == "synthetic":
            continue
        source_id = str(int(index) % 150)
        anchor_question = anchor_record = ""
        review_label = "copy" if row["match_status"] in ("literal", "normalized") else ""
        if source == "spinach":
            anchor = spinach[source_id]
            anchor_question = anchor["original_question"]
            anchor_record = f'{anchor["origin"]}/{anchor["source_file"]}:{anchor["source_pool_index"]}'
            if slot == "Q1":
                review_label = anchor["provisional_semantic_label"]
        elif source == "qawiki":
            source_row = qawiki[source_id]
            url = source_row["q2"] if slot == "Q2" and source_row["q2"] else source_row["q1"]
            if url in raw_qawiki:
                anchor_record = url
                english = [v.rsplit("@", 1)[0] for v in raw_qawiki[url]["questions"].split("|")
                           if v.endswith("@en") or v.endswith("@en-GB")]
                anchor_question = english[0] if english else ""
        nearest = candidates[(index, slot)]
        output.append({
            "index": index, "source": source, "source_id": source_id, "slot": slot,
            "ascb_question": row["question"], "text_match_status": row["match_status"],
            "direct_anchor_record": anchor_record, "direct_anchor_question": anchor_question,
            "nearest_raw_record": nearest["source_record"],
            "nearest_raw_question": nearest["source_question"],
            "nearest_cosine": nearest["cosine"],
            "provisional_semantic_label": review_label,
        })
    if len(output) != 1800:
        raise ValueError(f"Expected 1,800 source-derived slots, got {len(output)}")
    with (HERE / "semantic_review_worklist.tsv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()

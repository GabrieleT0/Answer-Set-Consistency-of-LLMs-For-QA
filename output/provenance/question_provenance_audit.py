"""Audit ASCB question text against the available source question text.

This counts verbatim matches, not semantic derivation or author effort.
"""

import csv
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("ASCB_RAW_DIR", "/tmp/ascb-raw"))
OUT = Path(__file__).resolve().parent


def norm(text):
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"\s+", " ", text)


def source_texts():
    records = defaultdict(list)
    for split in ("train", "test"):
        for row in json.loads((RAW / f"lcquad2_{split}.json").read_text()):
            for field in ("question", "paraphrased_question"):
                if row.get(field):
                    records["LC-QuAD"].append((row[field], f"{split}:{row['uid']}:{field}"))
    with (RAW / "qawiki-v1-simple-2025-09-09.tsv").open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            for field in ("questions", "paraphrases"):
                for variant in row[field].split("|"):
                    if variant.endswith("@en") or variant.endswith("@en-GB"):
                        records["qawiki"].append((variant.rsplit("@", 1)[0], row["questionId"]))
    origins = {
        "train.json": "QALD-9-plus",
        "test (3).json": "QALD-9-plus",
        "test (1).json": "QALD-10",
        "test (2).json": "QALD-7",
        "dev.json": "WikiWebQuestions",
        "test (4).json": "WikiWebQuestions",
    }
    for name, origin in origins.items():
        for row in json.loads((ROOT / "data" / "Spinach" / name).read_text()):
            if row.get("question"):
                records["spinach"].append((row["question"], f"{origin}/{name}:{row.get('id', '')}"))
    for split in ("dev", "test"):
        for row in json.loads((RAW / f"spinach_{split}.json").read_text()):
            if row.get("question"):
                records["spinach"].append((row["question"], f"SPINACH/{split}:{row.get('id', '')}"))
    return records


def write_spinach_anchors():
    """Reconstruct the saved source pool and follow its numeric `usparql` IDs."""
    ordered_files = (
        ("spinach_output_test (1).json", "QALD-7"),
        ("spinach_output_test.json", "QALD-10"),
        ("dev.json", "WikiWebQuestions"),
        ("spinach_output_test (2).json", "QALD-9-plus"),
        ("test (4).json", "WikiWebQuestions"),
        ("test (2).json", "QALD-7"),
        ("train.json", "QALD-9-plus"),
        ("spinach_output_dev.json", "WikiWebQuestions"),
        ("test (1).json", "QALD-10"),
        ("test (3).json", "QALD-9-plus"),
    )
    pool = []
    for filename, origin in ordered_files:
        for item in json.loads((ROOT / "data/Spinach" / filename).read_text()):
            if isinstance(item, dict) and "question" in item:
                pool.append((item["question"], filename, origin))
    with (ROOT / "data/Spinach/spinach_output.tsv").open(newline="") as f:
        saved = list(csv.DictReader(f, delimiter="\t"))
    if len(pool) != len(saved) or any(p[0] != s["question"] for p, s in zip(pool, saved)):
        raise ValueError("Saved SPINACH construction pool does not match the raw JSON sequence")

    with (ROOT / "data/ASCB/en/spinach.tsv").open(newline="") as f:
        benchmark = list(csv.DictReader(f, delimiter="\t"))
    anchors = []
    for row in benchmark:
        source_index = int(row["usparql"].removeprefix("no_usparql_"))
        source_question, filename, origin = pool[source_index]
        status = ("literal" if row["Q1"] == source_question else
                  "normalized" if norm(row["Q1"]) == norm(source_question) else "changed")
        anchors.append({
            "source_id": row["ID"], "source_pool_index": source_index,
            "origin": origin, "source_file": filename,
            "original_question": source_question, "ascb_Q1": row["Q1"],
            "Q1_status": status,
        })
    with (OUT / "spinach_source_anchors.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=anchors[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(anchors)
    print("spinach source origins", dict(Counter(x["origin"] for x in anchors)))
    print("spinach anchored Q1 status", dict(Counter(x["Q1_status"] for x in anchors)))


def main():
    write_spinach_anchors()
    records = source_texts()
    literal = {s: defaultdict(list) for s in records}
    normalized = {s: defaultdict(list) for s in records}
    for source, texts in records.items():
        for text, identifier in texts:
            literal[source][text].append(identifier)
            normalized[source][norm(text)].append(identifier)

    with (ROOT / "data/ASCB/en/unified-dataset.tsv").open(newline="") as f:
        benchmark = list(csv.DictReader(f, delimiter="\t"))
    counts = defaultdict(Counter)
    outrows = []
    for row in benchmark:
        source = row["source"]
        for slot in ("Q1", "Q2", "Q3", "Q4"):
            text = row[slot]
            ids = literal.get(source, {}).get(text, [])
            norm_ids = normalized.get(source, {}).get(norm(text), [])
            status = "literal" if ids else "normalized" if norm_ids else "no_text_match"
            counts[source][status] += 1
            counts[source][slot + ":" + status] += 1
            outrows.append({
                "index": row["index"], "source": source, "slot": slot,
                "question": text, "match_status": status,
                "source_records": "|".join(dict.fromkeys(ids or norm_ids)),
            })

    with (OUT / "exact_matches.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=outrows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(outrows)
    for source, tally in counts.items():
        print(source, "source texts", len(records.get(source, [])),
              "literal", tally["literal"], "normalized-only", tally["normalized"],
              "no text match", tally["no_text_match"])
        for slot in ("Q1", "Q2", "Q3", "Q4"):
            print(" ", slot, *(f"{status}={tally[slot + ':' + status]}"
                                  for status in ("literal", "normalized", "no_text_match")))


if __name__ == "__main__":
    main()

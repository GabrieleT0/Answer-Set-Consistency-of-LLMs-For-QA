"""Retrieve semantically similar raw questions for manual provenance review.

Embedding similarity retrieves candidates; it does not establish that two
questions have the same answer set or that one was derived from the other.
Requires fastembed and numpy installed separately from the project venv.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from question_provenance_audit import ROOT, source_texts


OUT = Path(__file__).resolve().parent
MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5",
                      cache_dir="/tmp/ascb-models", threads=4)


def vectors(texts, cache_name):
    cache = Path("/tmp/ascb-raw") / f"{cache_name}_bge.npy"
    if cache.exists():
        array = np.load(cache)
        if array.shape == (len(texts), 384):
            return array
    print("Embedding", cache_name, len(texts), "questions", flush=True)
    array = np.stack(list(MODEL.embed(texts, batch_size=128))).astype("float32")
    array /= np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)
    np.save(cache, array)
    return array


def main():
    sources = source_texts()
    with (ROOT / "data/ASCB/en/unified-dataset.tsv").open(newline="") as f:
        benchmark = list(csv.DictReader(f, delimiter="\t"))
    output = []
    for source in ("LC-QuAD", "qawiki", "spinach"):
        unique = defaultdict(list)
        for question, identifier in sources[source]:
            unique[question].append(identifier)
        source_questions = list(unique)
        source_vectors = vectors(source_questions, source.replace("-", "_"))
        query_rows = [(row, slot) for row in benchmark if row["source"] == source
                      for slot in ("Q1", "Q2", "Q3", "Q4")]
        query_questions = [row[slot] for row, slot in query_rows]
        query_vectors = vectors(query_questions, "query_" + source.replace("-", "_"))
        print("Retrieving", source, len(query_rows), "against", len(source_questions), flush=True)
        for start in range(0, len(query_rows), 32):
            scores = query_vectors[start:start + 32] @ source_vectors.T
            for i, score_row in enumerate(scores):
                top = np.argpartition(score_row, -5)[-5:]
                top = sorted(top, key=lambda j: float(score_row[j]), reverse=True)
                row, slot = query_rows[start + i]
                for rank, j in enumerate(top, 1):
                    source_question = source_questions[j]
                    output.append({
                        "index": row["index"], "source": source,
                        "source_id": row["source_id"], "slot": slot,
                        "ascb_question": row[slot], "rank": rank,
                        "source_question": source_question,
                        "source_record": "|".join(dict.fromkeys(unique[source_question])),
                        "cosine": f"{float(score_row[j]):.4f}",
                    })
    with (OUT / "semantic_candidates.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(output)
    print("Wrote", len(output), "candidate pairs", flush=True)


if __name__ == "__main__":
    main()

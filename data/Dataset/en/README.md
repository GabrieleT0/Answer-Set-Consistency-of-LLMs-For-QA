# English Benchmark Collection

This directory contains the English benchmark collection used to evaluate answer-set consistency in question answering. Each item is organized as four related questions:

- `Q1`: primary question
- `Q2`: intended paraphrase or equivalent reformulation of `Q1`
- `Q3`: intended constrained subset of `Q1`
- `Q4`: intended complement or paired remainder relative to `Q3`

The collection is designed for evaluation, not for general-purpose training.

## Files

- `LC-QuAD.tsv`: 150 data rows derived from LC-QuAD-style source materials
- `qawiki.tsv`: 150 data rows derived from QAWiki-style entity and relation materials
- `spinach.tsv`: 150 data rows derived from SPARQL-seeded question materials
- `synthetic.tsv`: 150 data rows created synthetically for controlled consistency tests

Total size: 596 benchmark items.

## Intended evaluative use

This collection is intended to support:

1. Consistency evaluation across semantically related question variants.
2. Analysis of answer-set equivalence, subset, and complement behavior.
3. Benchmarking prompting, retrieval, or post-processing methods that aim to improve consistency.

It is not intended to support claims about broad factual QA quality, demographic fairness, or deployment readiness on its own.

## Construction overview

The collection combines four sources of question materials. Source items were normalized into a shared `Q1`-`Q4` format so that each row encodes an intended answer-set relation pattern. The synthetic split was created to provide controlled test cases with known relation structure.

## Known limitations and biases

- Topic coverage is influenced by knowledge-base-style QA sources and may over-represent well-documented entities and relations.
- The collection is English-only.
- Some intended semantic relations remain approximate because natural-language interpretation can be ambiguous.
- The synthetic subset may be cleaner and more regular than naturally occurring user queries.

## Sensitive data and social impact

The released files contain benchmark questions only and are not intended to include personal or sensitive user data. The main positive use is better evaluation of QA reliability under semantically related prompts. A misuse risk is over-claiming that good performance here implies general truthfulness, safety, or fairness.

## License

This collection is released under the MIT license. See the repository license file for the governing license text.

## NeurIPS 2026 ED submission notes

For the NeurIPS 2026 Evaluations and Datasets track, treat this directory as a dataset collection page. Submit the per-dataset Croissant files from the `temp/` directory rather than only the collection-level Croissant file.
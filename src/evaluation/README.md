# Evaluation Code

The evaluation package is organized by responsibility. Run commands from the repository root using `python -m ...` (or `.venv/bin/python -m ...` with the local environment).

```text
src/evaluation/
├── run.py                         # Versioned analysis plus complete paper-results export
├── export_paper_results.py         # Compatibility CLI for table exports
├── export_supplementary_tables.py  # Optional historical supplementary export
├── paths.py                       # Shared repository path
├── core/                          # Answer loading, set scoring, relation scoring, summaries
├── statistics/                    # Exact McNemar tests and model comparison matrices
├── tables/                        # Table exports, revision marking, CSV splitting
├── analyses/                      # Ablations and targeted audits
├── legacy/                        # Historical/manual scripts, outside the main pipeline
└── notebooks/                     # Historical exploratory notebook
```

Automated regression tests are stored outside the runtime source tree, in
`tests/evaluation/`.

## Where to make changes

- `core/eval_tool.py`: load answers, align question groups, compute set predicates and Jaccard scores.
- `core/eval_relation.py`: normalize and evaluate relation predictions.
- `core/summary_idk_worst.py`: shared abstention policy and pessimistic summaries.
- `core/eval_tool.summary_empty_answers`: Table 4 sensitivity analysis, replacing individual observed null answers with empty sets while retaining instances. `summary_xidk` remains the separate historical exclusion diagnostic. The default export writes only the current `filtered` table; `--include-table4-comparison` adds the exclusion reference and paired comparison.
- `statistics/eval_pvalue.py`: paired significance tests and model comparison matrices.
- `tables/export_paper_results.py`: export Tables 3 and 4 by default; other tables are explicit.
- `tables/export_supplementary_tables.py`: optional raw or archived historical supplementary exports.
- `tables/table_revisions.py`: freeze the original-paper baseline and generate clean/blue versions.
- `tables/split.py`: aggregate per-strategy CSV exports.
- `analyses/`: no-IDK and relation-aware ablations, correction rates, lexical negation, output format, partial matches, and entity normalization.

The move does not change scoring definitions or regenerate experimental outputs. Repository paths are resolved through `paths.py` rather than inferred from each file's directory depth.

## Common commands

Recompute core Tables 3 and 4 from saved answers:

```bash
python -m src.evaluation.tables.export_paper_results
```

Generate clean/blue versions from a directory containing the two manuscript-formatted current tables:

```bash
python -m src.evaluation.tables.export_paper_results --source reviewed --current-dir path/to/current_tables
```

Replace `path/to/current_tables` with a directory containing `main.tex` and `filtered.tex`. The frozen original-paper baseline must already exist in `output/paper_tables/base/` (or be supplied with `--base-dir`). This command exports `clean/` and `blue/` variants without recalculating scores. See the table-generation instructions for initializing a baseline if those generated files have been removed.

Optionally export historical supplementary tables 9–11:

```bash
python -m src.evaluation.tables.export_supplementary_tables --source raw
```

The previously documented `python -m src.evaluation.export_paper_results` and `python src/evaluation/export_supplementary_tables.py` entry points remain available as thin CLI wrappers. Other moved modules use the new package paths; old flat Python import paths are not retained.

Run targeted analyses:

```bash
python -m src.evaluation.analyses.eval_no_idk_ablation --help
python -m src.evaluation.analyses.eval_relation_aware_ablation --help
python -m src.evaluation.analyses.lexical_negation_audit --help
python -m src.evaluation.analyses.output_format_audit --help
python -m src.evaluation.analyses.entity_normalization_sensitivity --help
```

The main evaluation entry points create UTC-timestamped directories under
`output/evaluation_runs/` unless an exact `--output-dir` is supplied. Each run
contains `manifest.json` with schema `ascb-evaluation-run/v1`, the effective
CLI/configuration, input fingerprints, the selected `standard`, `robust`, or
`extended` IDK-recognition mode, entity-normalization mode, scoring and
aggregation policies, completion status, and output hashes. Main runs default
to `recognition=robust` and `entity_normalization=none`; cached Wikidata-alias normalization is emitted as a
separate sensitivity run.
Git metadata is intentionally omitted so anonymous run artifacts cannot be
linked to a development repository.
Manifest paths are repository-relative; paths outside the repository are
recorded only with anonymous `<external>/...` labels. Evaluation artifacts do
not store usernames or absolute home-directory paths.

`python -m src.evaluation.run` creates a self-contained directory under
`output/evaluation_runs/versions/`. It writes the shared analysis and summaries
at the version root, then derives all manuscript tables from that in-memory
analysis under `paper_results/`. The table exporter does not reload the answer
files for this nested export. The standalone paper command remains intentionally
limited to Tables 3 and 4 for IDK-policy comparisons.

Run all regression tests:

```bash
python -m unittest discover -s tests/evaluation -t .
```

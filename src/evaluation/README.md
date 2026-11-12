# Evaluation Code

The evaluation package is organized by responsibility. Run commands from the repository root using `python -m ...` (or `.venv/bin/python -m ...` with the local environment).

```text
src/evaluation/
├── run.py                         # Aggregate evaluation entry point
├── export_paper_results.py         # Compatibility CLI for table exports
├── export_supplementary_tables.py  # Compatibility CLI for supplementary tables
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
- `statistics/eval_pvalue.py`: paired significance tests and model comparison matrices.
- `tables/export_paper_results.py`: export all eight result tables.
- `tables/export_supplementary_tables.py`: raw or archived supplementary table exports.
- `tables/table_revisions.py`: freeze the original-paper baseline and generate clean/blue versions.
- `tables/split.py` and `tables/split_heatmap.py`: aggregate CSV exports.
- `analyses/`: no-IDK and relation-aware ablations, correction rates, lexical negation, output format, partial matches, and entity normalization.

The move does not change scoring definitions or regenerate experimental outputs. Repository paths are resolved through `paths.py` rather than inferred from each file's directory depth.

## Common commands

Recompute all eight tables from saved answers:

```bash
python -m src.evaluation.tables.export_paper_results
```

Generate clean/blue versions from a directory containing the eight manuscript-formatted current tables:

```bash
python -m src.evaluation.tables.export_paper_results --source reviewed --current-dir path/to/current_tables
```

Replace `path/to/current_tables` with a directory containing all eight current `.tex` table files. The frozen original-paper baseline must already exist in `output/paper_tables/base/` (or be supplied with `--base-dir`). This command exports `clean/` and `blue/` variants without recalculating scores. See the table-generation instructions for initializing a baseline if those generated files have been removed.

Export supplementary tables:

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

Run all regression tests:

```bash
python -m unittest discover -s tests/evaluation -t .
```

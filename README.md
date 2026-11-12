# Answer Set Consistency of LLMs for Question Answering

This repository contains the code, ASCB benchmark data, model outputs, evaluation scripts, aggregate results, and figures supporting a NeurIPS submission on answer-set consistency in large language models.

The project evaluates whether an LLM gives mutually consistent answer sets for groups of related natural-language questions. Each ASCB item contains four questions:

- `Q1`: the base question, with answer set `A1`.
- `Q2`: a paraphrase of `Q1`, expected to have the same answer set, `A2 = A1`.
- `Q3`: a restricted version of `Q1`, expected to satisfy `A3 ⊆ A1`.
- `Q4`: the complementary restriction, expected to satisfy `A4 ⊆ A1`, `A3 ∩ A4 = ∅`, and `A1 = A3 ∪ A4`.

The evaluation reports exact set-consistency predicates, Jaccard similarities, IDK/empty-answer rates, relation-classification summaries, pairwise p-values, and the figures used in the paper.

## Repository Layout

```text
.
├── data/
│   ├── ASCB/
│   │   └── en/                         # ASCB benchmark TSV files and Croissant metadata
│   ├── Spinach/                        # Intermediate SPINACH-derived generation files
│   ├── answers/                        # Raw model answer JSON files
│   ├── evaluation_results/             # Committed paper-facing aggregate CSV results
│   ├── llm_info.json                   # Model metadata used by evaluation and plots
│   └── wikidata_cache.json             # Cached Wikidata lookups
├── src/
│   ├── question_pipeline/              # Scripts for deriving related questions
│   ├── get_answers/                    # LLM prompting and answer-generation scripts
│   ├── evaluation/                     # Metrics, summaries, p-values, and heatmap CSVs
│   └── visualization/                  # Figure-generation scripts
├── charts/                             # Generated figures
├── docs/                               # Static GitHub Pages table of ASCB questions
├── output/                             # Regenerated local evaluation outputs, if produced
├── requirements.txt
├── LICENSE
└── README.md
```

## ASCB Data

The ASCB question files are under `data/ASCB/en/`.

| File | Description |
| --- | --- |
| `LC-QuAD.tsv` | ASCB question groups derived from LC-QuAD-style knowledge-base questions. |
| `qawiki.tsv` | ASCB question groups derived from QA-Wiki items. |
| `spinach.tsv` | ASCB question groups derived from SPINACH items. |
| `synthetic.tsv` | Synthetic ASCB question groups created for controlled logical relations. |
| `unified-dataset.tsv` | Unified ASCB table combining the benchmark sources. |
| `croissant_unified_ASCB_hf.json` | Croissant metadata for the unified ASCB dataset. |

Each TSV row contains a related question group with at least `ID`, `Q1`, `Q2`, `Q3`, and `Q4`. Some files also include provenance or type columns.

The static website table in `docs/index.html` displays ASCB as grouped rows: each row contains the related `Q1`, `Q2`, `Q3`, and `Q4`. The table data is stored in `docs/questions.json` and can be regenerated with:


## Model Answers and Results

Raw model outputs are committed under `data/answers/`.

```text
data/answers/
├── zero-shot/      # Direct answer generation and relation-classification outputs
├── CtE/            # Classification-then-enumerate outputs
└── Oracle/         # Relation-aware fixing/oracle outputs
```

The committed aggregate results used for analysis are under `data/evaluation_results/`.

Important files include:

- `analysis.csv`: per-question, per-model, per-action answer-set metrics.
- `analysis_idk_filtered.csv`: per-question metrics after filtering IDK/empty answers.
- `summary_filtered_idk.csv`: summary metrics with IDK/empty answers filtered from consistency scores.
- `summary_idk_worst.csv`: summary metrics where IDK/empty answers are treated pessimistically.
- `summary_*_{zero-shot,classification,fixing}.csv`: per-action summary splits.
- `p_value_matrices.csv`: pairwise model-comparison p-values.
- `heatmaps/heatmap_*.csv`: heatmap-ready CSV files derived from `p_value_matrices.csv`.
- `tradeoff/*.csv`: IDK/consistency trade-off summaries.

Local reproduction scripts write fresh outputs to `output/`. These files may differ from the committed `data/evaluation_results/` if the raw answers, loaders, model list, or filtering policy have changed.

## Setup

Python 3.10 or newer is required. The repository has been smoke-tested with the local virtual environment in this workspace.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run commands from the repository.

Live model generation requires provider credentials. The answer-generation code reads credentials from environment variables and/or `.env` files used by `src/get_answers/llms.py`.

Common variables include:

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export DEEPSEEK_API_KEY=...
export XAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=...
export AZURE_API_VERSION=...
```

See `src/env.example` for the minimal example used by this repository.

## Reproducing the Experiment

The complete experiment has three stages: answer generation, evaluation, and visualization. If you only need the paper aggregate results, start from the committed CSV files in `data/evaluation_results/`. If you want to regenerate results from raw model JSON files, run the evaluation stage.

### 1. Generate or Update Model Answers

The answer-generation scripts are in `src/get_answers/`. The main configuration file is:

```text
src/get_answers/config.json
```

It specifies the language, datasets, model list, prompt types, and relation labels. The current dataset entries correspond to the ASCB TSV files in `data/ASCB/en/`.

The top-level runner is:

```bash
python src/get_answers/run.py
```

The individual generation entry points are:

```bash
python src/get_answers/single_question_benchmark.py
python src/get_answers/get_a1_star.py
python src/get_answers/relation_classification.py
python src/get_answers/relation_classification_and_questions.py
python src/get_answers/try_fix_llm_response.py
```

These scripts call external LLM APIs and require valid credentials. They write JSON outputs under `data/answers/` using the folder and filename conventions expected by the evaluation loaders.

#### Run the paired relation-aware controls with multiple workers

The `relation_aware_ablation` generator supports parallel question groups:

```bash
.venv/bin/python -m src.get_answers.relation_aware_ablation --model gpt-5-mini --workers 8
```

With the virtual environment activated, `python` can replace `.venv/bin/python`.
`--workers` defaults to 8, processing up to eight independent question groups
concurrently. Use `--workers 1` for serial execution. Turns within each dialogue remain sequential,
and the two correction arms replay the same saved initial dialogue. Each worker
has its own model client and conversation history. Dataset/task batches retain
their existing order; this option applies to this generator, not all generation
scripts.

Results are saved under `output/relation_aware_ablation/`. Checkpoint writes are
serialized and atomic. Rerunning with the same model, output directory, and
compatible prompts skips saved fields and retries missing fields; the worker
count can change between runs. If an earlier serial process is still using that
directory, stop it before restarting with multiple workers. On interruption or
an unhandled request error, pending jobs are cancelled and in-flight work is
allowed to finish its current stage and save progress.

Provider content-filter blocks are recorded separately under `blocked_responses`.
The affected stage remains incomplete, and the other stages and question groups
continue. Filtered responses are not converted into answers or IDK observations.
Rerunning retries missing stages using the original prompts and preserves saved
complete stages. LangChain deprecation warnings are distinct from provider errors.

Parallelism can reduce time spent waiting for API responses; it does not guarantee
an eightfold speedup. Reduce `--workers` if the provider's rate limits become the
bottleneck. No live API benchmark is included in the offline regression tests.

Run the generation and evaluation regression tests without model calls:

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

### 2. Regenerate Evaluation CSV Files

Evaluation code is grouped into `core/`, `statistics/`, `tables/`, `analyses/`,
`legacy/`, and `notebooks/`. Regression tests are outside `src/`, in
`tests/evaluation/`. See the [evaluation package guide](src/evaluation/README.md)
for the directory layout, current module paths, and regression-test command.
The common table-export CLI entry points remain available.

To recompute answer-set consistency metrics from `data/answers/` and `data/ASCB/en/`, run:

```bash
python src/evaluation/run.py
```

This command:

- loads ASCB questions from `data/ASCB/en/`;
- loads raw model answers from `data/answers/`;
- computes per-question consistency metrics and Jaccard similarities;
- merges relation-classification information when available;
- computes pairwise p-values;
- writes regenerated CSV files to `output/`.

Main regenerated files:

```text
output/analysis.csv
output/summary.csv
output/summary_xidk.csv
output/summary_zero-shot.csv
output/summary_classification.csv
output/summary_fixing.csv
output/p_value_matrices.csv
output/heatmap_*.csv
```

To recompute only the p-value matrix from an existing `analysis.csv`, run:

```bash
python -m src.evaluation.statistics.eval_pvalue
```

The script uses `output/analysis.csv` when it exists; otherwise it falls back to `data/evaluation_results/analysis.csv`.

### Reproduce Supplementary Tables 9–11

Rebuild these tables directly from saved Base/CtE answers and classification
outputs, without model calls or notebook setup:

```bash
python src/evaluation/export_supplementary_tables.py
```

This writes `output/tables_9_11.csv` (percent scores, source columns, and
per-cell denominators) and `output/tables_9_11.tex` (the three complete tables).
The regular evaluation runner also exports these tables. To replace only
these tables in the local manuscript, use:

```bash
python src/evaluation/export_supplementary_tables.py --update-manuscript TEMP/appendix.tex
```

The corrected export reads CtE labels from `data/answers/CtE/`, uses joint
partition consistency and difference-component Jaccard in Tables 10–11, and
checks the actual predicted relation for Table 9. Null answers and explicit
classification abstentions receive zero contradiction-free credit. Missing
classification observations are excluded, not classified as incorrect; their
coverage is visible in the CSV. In the current raw files, the two DeepSeek
models have CtE classifications for only 150 of their 600 answer groups;
missing labels are not filled from Base or inferred from aggregate scores.
For Tables 10–11, explicit classification
abstentions belong to the incorrect subset, and answer scores use the same
worst-case null-answer policy as the main table.

To recover the originally published numbers from the committed aggregate
snapshot instead of recomputing them, explicitly select the archived mode:

```bash
python src/evaluation/export_supplementary_tables.py --source archived --output /tmp/ascb_archived_tables_9_11
```

Archived exports retain the historical Table 9 agreement scores and disclose
the original union-column mapping for the final columns of Tables 10–11 and cannot overwrite the manuscript. They are for
historical comparison, not corrected evaluation. The old `src/evaluation/notebooks/onboard.ipynb` is
not required for either path.

### 3. Regenerate Figures

To regenerate the charts from the available aggregate CSV files, run:

```bash
python src/visualization/run.py
```

The visualization runner uses:

- `output/` if regenerated summaries are present;
- otherwise `data/evaluation_results/`.

It writes figures under `charts/`, including line charts, p-value heatmaps, positive/negative split plots, and bubble scatter plots.

For the older standalone chart set, including `charts/action_pvalue_heatmaps/` and
`charts/llms_pvalue_heatmaps/`, run:

```bash
python src/visualization/generate_charts.py
```

This script reads `output/summary_idk_worst.csv` and
`output/p_value_matrices.csv` when present. If those regenerated files do not
exist, it falls back to the committed results in `data/evaluation_results/`. It
writes bar charts and p-value heatmaps under `charts/`.

## Generate the updated paper tables

Run these commands from the repository root using the existing `.venv` environment
(or replace `.venv/bin/python` with the Python executable for your environment).
They use saved results and do not call model APIs. Golden Answers and other
separately managed new experiments are outside this workflow.

### Recompute all eight result tables from raw answers

```bash
.venv/bin/python -m src.evaluation.export_paper_results
```

Outputs go to `output/reproduced_tables/`, including CSVs, plain LaTeX tables,
coverage counts, response provenance, and input SHA-256 hashes. This command
recomputes results; it does not update the manuscript-formatted tables in
`output/paper_tables/clean/`.

### Generate manuscript-formatted clean and blue versions

```bash
.venv/bin/python -m src.evaluation.export_paper_results --source reviewed --current-dir path/to/current_tables
```

Replace `path/to/current_tables` with the directory containing the eight current
`.tex` table inputs. This compares them against the frozen original-paper baseline in
`output/paper_tables/base/` (which must already exist). It writes `clean/` and
`blue/` tables, plus a
`changes.json` report. Existing blue marks are removed before comparison; the
baseline is never updated by a normal export. This step does not automatically
insert the newly recomputed raw-mode results: update and check the current input
tables before generating the two versions.

### Regenerate charts from aggregate results

```bash
.venv/bin/python src/visualization/run.py
```

For the older standalone bar charts and heatmaps, also run:

```bash
.venv/bin/python src/visualization/generate_charts.py
```

Charts are written under `charts/`. These commands read aggregate evaluation CSVs,
not `output/reproduced_tables/` or the manuscript table fragments. They may fall
back to archived results in `data/evaluation_results/` when fresh aggregate files
are absent; running them alone does not guarantee figures based on the latest
raw answers.

### Current full-refresh limitation

There is currently no single verified command that refreshes raw evaluation,
manuscript-formatted tables, both revision variants, and all figures together.
The aggregate evaluation entry point is `python src/evaluation/run.py`, but it
currently imports the missing `reviewer_abstention_audit` module. That dependency
must be restored or the entry point repaired before using it for a full refresh.
Simply chaining the commands above does not close the gaps between their inputs.

### Can `output/reproduced_tables/` be deleted?

Yes. It contains generated outputs, not required source inputs. Deleting it does
not remove raw answers, the frozen baseline, current manuscript table inputs, or
the clean/blue versions. The raw-mode command recreates it on the next run.
Several existing outputs are tracked in Git, so deleting them appears as file
deletions in `git status`. Keep or archive their provenance/hash files if you need
to identify that exact past data snapshot; regenerating after inputs change may
produce different values and hashes.

For individual table commands, label mappings, and baseline initialization, see
[the table generation instructions](output/table_generation_instructions.md).

## Metrics

For each dataset, model, and action, the evaluation reports:

- binary set predicates such as `?A1=A2`, `?A1=A3+A4`, `?A1>A3`, `?A1>A4`, `?A3∅A4`, and `?A4=A1|3`;
- Jaccard similarities such as `J(A1-A2)`, `J(A1-A34)`, `J(A3-A4)`, and `J(A4-A1|3)`;
- IDK and empty-answer indicators for `A1`, `A2`, `A3`, and `A4`;
- relation-classification summaries for predicted logical relations;
- pairwise model p-values for each predicate/action/dataset combination.

Answers are parsed as sets from the committed JSON outputs. Empty outputs and `idk` answers are tracked explicitly so that filtered and pessimistic summaries can both be reported.

## Notes on Reproducibility

- Run scripts from the repository root unless a script explicitly says otherwise.
- The evaluation expects dataset names matching the ASCB sources: `LC-QuAD`, `qawiki`, `spinach`, and `synthetic`.
- Re-running evaluation or visualization may overwrite files in `output/` and `charts/`.

## License

The code in this repository is released under the MIT License. See `LICENSE`.

The ASCB dataset is released under the GNU General Public License v3.0 (GPL-3.0).

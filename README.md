# Answer Set Consistency of LLMs for Question Answering

This repository contains the code, ASCB benchmark data, model outputs, evaluation scripts, aggregate results, and figures supporting a ICLR submission on answer-set consistency in large language models.

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
│   ├── evaluation/                     # Metrics, summaries, p-values, and paper tables
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

Score one model directory at a time with
`python -m src.evaluation.analyses.eval_relation_aware_ablation --input-dir output/relation_aware_ablation/<model>`.
By default, each evaluation is written to its own UTC-timestamped directory under
`output/evaluation_runs/relation_aware_ablation/`. Pass `--output-dir` only when
an exact destination is required. Every run includes `manifest.json` with the
manifest schema version, command and configuration, IDK-recognition rule,
entity-normalization mode, evaluation policies, input hashes, completion
status, and output hashes. Git metadata is deliberately excluded to avoid
linking anonymous artifacts to a development repository.
Paths stored in manifests and printed by evaluation entry points are
repository-relative. External locations use anonymous `<external>/...` labels,
so run artifacts do not disclose usernames or home directories.
The primary set-partition score jointly requires `A4 = A1 - A3` and `A3 <= A1`;
the two components are also exported as diagnostics. The root-level
`per_item.csv` and `summary.csv` are the initial GPT-4.1-mini snapshot that used
the difference-only ablation score and are retained only for run provenance.

### Standardized evaluation factors

The default evaluation configuration uses `recognition=robust` and
`entity_normalization=none`. Use `--recognition standard` to reproduce the
original narrower IDK rule. Cached Wikidata alias normalization is a separate
sensitivity run:

```bash
.venv/bin/python -m src.evaluation.analyses.entity_normalization_sensitivity
```

Its default output is a timestamped directory under
`output/evaluation_runs/entity-normalization/`. In addition to IDK recognition
and entity normalization, every run manifest records the answer representation,
joint partition criterion, repeated-control aggregation, main and Table 4
abstention policies, missing-record policy, denominator policy, pairing policy
where applicable, and all input hashes. Compare these fields before treating
numbers from different runs as directly comparable.

The repeated question result is an auxiliary stability control only. It asks
the same `Q1` again to separate variability under identical wording from the
effect of the equivalent rephrasing (Q_2). It is excluded from the five primary
relations and from `CON`. The default manifest records the upstream publication
rule. The control is the model-level equal-weight mean of the isolated
Base-repeat condition and the CtE `Q1` control condition. Missing Base
repetitions are not filled from CtE on a per-question basis, and the control is
not published unless all 600 isolated Base repetitions are available.

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
- writes a self-contained version under `output/evaluation_runs/versions/`.

Main regenerated files:

```text
<version>/analysis.csv
<version>/summary.csv
<version>/summary_empty_answers.csv
<version>/summary_idk_worst.csv
<version>/p_value_matrices.csv
<version>/paper_results/*.csv
<version>/paper_results/*.tex
```

To recompute only the p-value matrix from an existing `analysis.csv`, run:

```bash
python -m src.evaluation.statistics.eval_pvalue
```

The script uses `output/analysis.csv` when it exists; otherwise it falls back to `data/evaluation_results/analysis.csv`.

### Optional historical Supplementary Tables 9–11

Rebuild these tables directly from saved Base/CtE answers and classification
outputs, without model calls or notebook setup:

```bash
python src/evaluation/export_supplementary_tables.py
```

This writes `output/tables_9_11.csv` (percent scores, source columns, and
per-cell denominators) and `output/tables_9_11.tex` (the three complete tables).
These tables are not generated by the regular evaluation runner or the default
paper-table export. To rebuild them explicitly for historical comparison, use
the standalone command above. To replace only these tables in a local
manuscript, use:

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

For a single entry point, use:

```bash
./run_evaluation.sh
```

The default `core` scope creates one timestamped evaluation version with the
shared analysis and all manuscript result tables under its `paper_results/`
subdirectory. Use `--scope aggregate` for the same version build,
`--scope paper` for a standalone Tables 3–4 export,
`--scope entity` for entity normalization, or `--scope ablation --model gpt-5`
for one saved ablation model. Run `./run_evaluation.sh --help` for all options.

Recommended workflow: create the canonical complete version once with
`./run_evaluation.sh --scope core` (robust by default). For IDK sensitivity,
use `--scope paper --recognition standard` or `extended`; those standalone runs
generate only Tables 3 and 4 and therefore do not duplicate the IDK-independent
paper tables.

### Standalone Tables 3 and 4 export

```bash
.venv/bin/python -m src.evaluation.export_paper_results
```

Outputs go to a new UTC-timestamped directory under
`output/evaluation_runs/paper-results/`, including CSVs, plain LaTeX tables,
and `manifest.json`. Add `--include-audit-details` only when separate coverage,
repeat provenance, and source-fingerprint files are needed.
For a complete self-contained evaluation version, use `./run_evaluation.sh`.
That command writes the shared `analysis.csv` and summaries once, then derives
all paper tables into the sibling `paper_results/` folder without rebuilding
the answer analysis.
This command
recomputes results; it does not update the manuscript-formatted tables in
`output/paper_tables/clean/`.

Paper-table exports use the `robust` recognition rule by default: empty strings,
blank-only lists, `no answer`, and `I don't know` / `I do not know` are treated
as unavailable after normalization. Use `--recognition standard` to preserve
the original empty-set/sole-`idk` rule. The model-level equal-weight Base/CtE
aggregation is retained for both repeated-control columns, but the columns are
published only for models with all 600 isolated Base repetitions.
Verified implementation and input-selection corrections remain applied.

The robust default can be selected explicitly as follows:

```bash
.venv/bin/python -m src.evaluation.run --recognition robust
```

`robust` also recognizes empty strings, blank-only lists, `no answer`, and
`I don't know` / `I do not know`, normalizing surrounding quotes and terminal
punctuation. Valid-entity lists are retained; `None` and `NaN` remain missing.
`extended` additionally flags explicit null phrases mixed with valid entities.
The paper-table exporter also accepts `--recognition robust` or
`--recognition extended`. Each invocation produces only Tables 3 and 4 for the
selected rule in a distinct timestamped run directory. This avoids duplicating
unrelated tables across IDK modes. To explicitly compare all three modes in one
diagnostic file, add `--include-idk-sensitivity`.
Keeping the original protocol does not retain incorrect source mixing or imply
that corrected results must exactly equal the submitted numbers.

To regenerate both Table 4 policies together, run:

```bash
.venv/bin/python -m src.evaluation.export_paper_results --table filtered \
  --include-table4-comparison --output-dir output/table4_versions
```

Despite the historical key `filtered`, Table 4 now replaces each observed
abstention or empty answer individually with the empty set and retains the
instance. It preserves companion answers, uses each task's own context, and
retains original abstention flags. This is a sensitivity analysis: empty sets
can create trivial consistency. Missing observations are not filled, and
Jaccard similarity of two empty sets remains zero. Per-metric denominators are
included in the CSV; 600 is valid only when all 600 observations are available.
Table 3 continues to use worst-case scores for abstentions. The old exclusion
policy remains available through `core.eval_tool.summary_xidk` for comparison.
With `--include-table4-comparison`, both policies are exported on the same data
snapshot, with independent files:

- `filtered.csv` / `filtered.tex`: individual empty-set replacement (current Table 4).
- `filtered_exclusion.csv` / `filtered_exclusion.tex`: the previous metric-instance exclusion method.
- `policy_comparison.csv`: paired score changes and denominators for both methods.

Each CSV includes an explicit `abstention_policy`. The exclusion reference uses
a distinct LaTeX label so both tables can be included in the same document.
With `--include-audit-details`, `source_manifest.json` identifies the input snapshot.
The default core export does not generate the legacy Table 4 exclusion reference.

### Generate manuscript-formatted clean and blue versions

```bash
.venv/bin/python -m src.evaluation.export_paper_results --source reviewed --current-dir path/to/current_tables
```

Replace `path/to/current_tables` with the directory containing the two current
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
not timestamped paper-result runs or the manuscript table fragments. They may fall
back to archived results in `data/evaluation_results/` when fresh aggregate files
are absent; running them alone does not guarantee figures based on the latest
raw answers.

### Current full-refresh limitation

The aggregate evaluation entry point `python src/evaluation/run.py` now refreshes
the shared analysis and all numerical paper tables in one version directory.
It does not update manuscript source files, clean/blue revision variants, or
figures automatically; those remain deliberate follow-up steps.

### Can generated evaluation-run directories be deleted?

Yes. They contain generated outputs, not required source inputs. Deleting them does
not remove raw answers, the frozen baseline, current manuscript table inputs, or
the clean/blue versions. The raw-mode command creates a new timestamped directory
on the next run. Preserve `manifest.json` with any result snapshot that must remain
auditable; regenerating after inputs or code change may produce different values.

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

# Answer Set Consistency of LLMs for Question Answering

Code, datasets, model outputs, evaluation files, and figures supporting a submission on answer-set consistency in large language models.

The repository studies whether LLM answers respect simple logical relations between related natural-language questions. Each benchmark item contains four questions:

- `Q1`: the original question, with answer set `A1`.
- `Q2`: a paraphrase of `Q1`, expected to have the same answer set `A2 = A1`.
- `Q3`: a more restrictive version of `Q1`, expected to satisfy `A3 ⊆ A1`.
- `Q4`: the complementary restriction, expected to satisfy `A4 ⊆ A1`, `A3 ∩ A4 = ∅`, and `A1 = A3 ∪ A4`.

The evaluation computes exact set-consistency indicators, Jaccard similarities, relation-classification summaries, IDK/empty-answer rates, pairwise p-values, and the visualizations used in the paper.

## Repository Layout

```text
.
├── data/
│   ├── ASCB/                     # Benchmark question sets, one TSV per source dataset
│   ├── Spinach/                  # Intermediate/generated SPINACH-derived question files
│   ├── answers/                  # Raw model answer JSON files
│   ├── evaluation_results/       # Aggregated CSV outputs used for analysis
│   ├── question_analysis/        # Golden-answer metadata and question-level figures
│   ├── llm_info.json             # Model metadata and plotting order
│   └── wikidata_cache.json       # Cached Wikidata lookups
├── src/
│   ├── question_pipeline/        # Pipeline for generating Q2/Q3/Q4 from seed questions
│   ├── get_answers/              # LLM prompting, answer extraction, fixing, and classification
│   ├── evaluation/               # Consistency metrics, relation analysis, p-values
│   └── visualization/            # Plot generation scripts
├── charts/                       # Generated charts used by the paper
├── docs/                         # Static ASCB question-set table for GitHub Pages
├── requirements.txt              # Python dependencies
└── LICENSE
```

## Data

The benchmark question files are in `data/ASCB/`.

| File | Examples | Description |
| --- | ---: | --- |
| `LC-QuAD.tsv` | 150 | Questions derived from LC-QuAD-style knowledge-base queries. |
| `qawiki.tsv` | 150 | Questions derived from QA-Wiki items. |
| `spinach.tsv` | 150 | Questions derived from SPINACH items. |
| `synthetic.tsv` | 150 | Synthetic question sets created for controlled logical relations. |

Each row contains at least `ID`, `Q1`, `Q2`, `Q3`, and `Q4`. Some source files include additional provenance or type columns.

The committed model-answer files are under `data/answers/`:

- `zero-shot/`: direct answering prompts, and `A1*` variants where present (same question but asked in a different context window).
- `CtE/`: classification-then-enumerate outputs, named with `classAndAnswer` in the JSON filenames.
- `Oracle/`: In the prompt, we show the actual relationship linking the questions, labeled fixing in the JSON filenames.

The main aggregate analysis files are in `data/evaluation_results/`, including:

- `analysis.csv`: per-question, per-model, per-action consistency and answer-set details.
- `summary_filtered_idk.csv`: summary metrics with IDK/empty-answer filtered out in the evaluation metrics.
- `summary_idk_worst*.csv`: summaries where IDK/empty answers are treated as completly 
- `p_value_matrices.csv`: Pairwise model-comparison p-values. This is used to verify H2 of the paper: whether an LLM with generally better performance performs significantly better than an LLM with generally worse performance.
- `heatmaps/`: Heatmap CSV files created from p_value_matrices.csv, used to generate the heatmaps under charts/p_value_heatmap*.*


## Answer Set Consistency Benchmark Website Table

The repository includes a static website table in `docs/index.html`. It displays ASCB as grouped question sets: each table row contains the related `Q1`, `Q2`, `Q3`, and `Q4` questions.
You can see the table at the following link: [https://anonymous.4open.science/w/Answer-Set-Consistency-of-LLMs-For-QA-5FE6/](https://anonymous.4open.science/w/Answer-Set-Consistency-of-LLMs-For-QA-5FE6/)

## Setup

Python 3.10 or newer is required; Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Live model generation requires the relevant provider credentials. The code reads keys from environment variables in `src/get_answers/llms.py`, including:

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=...
export AZURE_API_VERSION=...
```

Self-hosted models are configured in `src/get_answers/llms.py` through `SelfHostedAPIWrapper`.

## Running the Questions Pipelines

### 1. Generate LLM Answers

The answer-generation code is in `src/get_answers/`.

The main configuration file is `src/get_answers/config.json`; prompts are in `src/get_answers/prompts.yaml`. The individual entry points are:

```bash
python src/get_answers/single_question_benchmark.py
python src/get_answers/get_a1_star.py
python src/get_answers/relation_classification.py
python src/get_answers/relation_classification_and_questions.py
python src/get_answers/try_fix_llm_response.py
```

### 2. Evaluate Consistency

```bash
python src/evaluation/run.py
```

This script loads questions and model answers, computes consistency metrics, merges relation-classification results, computes p-values, and writes CSV outputs to `output/`.

The committed, paper-facing aggregate results are already available in `data/evaluation_results/`. If regenerating them from raw JSON files, keep the expected action tokens in the filenames or adapt the loaders in `src/evaluation/eval_tool.py` and `src/evaluation/eval_relation.py`.

The core evaluated predicates are:

- `?A1=A2`
- `?A1=A3+A4`
- `?A1>A3`
- `?A1>A4`
- `?A3∅A4`
- `?A4=A1|3`
- Classification/fixing-specific variants comparing `A1`, `A1*`, and `A1**`

### 4. Generate Figures

```bash
python src/visualization/run.py
```

The visualization scripts generate line charts, p-value heatmaps, positive/negative split plots, and bubble scatter plots. The generated paper figures are already committed in `charts/`.

Several plotting scripts expect timestamped filenames such as `summary_<time>.csv` and `p_value_matrices_<time>.csv`. Update the `time`, `folder`, and `out_dir` values in `src/visualization/run.py` or call the individual plotting modules with an explicit config.

## Metrics

For each model/action/dataset, the evaluation reports:

- Binary consistency predicates over answer sets.
- Jaccard similarities, for example `J(A1-A2)` and `J(A1-A34)`.
- IDK/empty-answer indicators for `A1`, `A2`, `A3`, and `A4`.
- Self-contradiction indicators derived from predicted question relations.
- Pairwise model p-values for each predicate/action/dataset combination.

Answers are parsed as sets from pipe-separated model outputs. Empty outputs and `idk` are tracked explicitly and summarized separately.

## License

The code in this repository is released under the MIT License. See `LICENSE`.

The ASCB dataset files under `data/ASCB/` are released under the GNU General Public License v3.0 (GPL-3.0).

### Initialize a baseline once on a fresh checkout


Initialization refuses to overwrite an existing baseline. `base/manifest.json` records original source and baseline-table SHA-256 hashes; subsequent exports verify that the frozen baseline has not been edited. To create a deliberately different baseline, use a new `--base-dir` rather than changing the existing files.

The generated files are manuscript fragments, not standalone documents. They use the manuscript's existing packages and metric macros. For example, when compiling from the repository root, use `\input{output/paper_tables/blue/main.tex}` for the marked version or `\input{output/paper_tables/clean/main.tex}` for the clean version. Adjust paths when compiling from another directory.

The commands below use the default `--source raw` mode to recompute results. Its simple LaTeX output remains separate from the reviewed baseline-comparison workflow.

## Environment and Main Entry Point

Use the project's existing Python environment. Required dependencies are pandas, numpy, scipy, and statsmodels, as listed in `requirements.txt`. If `.venv` is available, run:

```bash
.venv/bin/python -m src.evaluation.export_paper_results
```

This generates all eight tables in `output/reproduced_tables/`, with one CSV and one LaTeX file per table. To specify a different directory:

```bash
.venv/bin/python -m src.evaluation.export_paper_results --output-dir /tmp/paper-tables
```

**Raw-mode `.tex` files reproduce the numerical results using simple formatting, without blue revision marks.** Use the reviewed comparison workflow above for manuscript-formatted clean/blue versions. CSV files also retain sample counts or full p-values for verification.

Single-table commands also rebuild the shared analysis data, so their runtime is similar to generating all tables together. Running the main entry point is usually preferable.

## 1. Main Results Table (Table 3)

Label: `tab:consistency_per_action_by_id`

```bash
.venv/bin/python -m src.evaluation.export_paper_results --table main
```

- Inputs: Base, CtE, Oracle (action `fixing` in the code), and `star` responses under `data/answers/`; the four question datasets under `data/`; and `data/llm_info.json`.
- Calculation: `core.eval_tool.analysis` → `core.summary_idk_worst.summary_idk_worst`, selecting `dataset=overall`. Displayed scores are multiplied by 100. CON is the arithmetic mean of the five binary relation-consistency scores and excludes the repeated-question columns.
- The main score retains the joint partition definition: `A1 = A3 ∪ A4` and `A3 ∩ A4 = ∅`. The final Jaccard metric compares `A4` with `A1 − A3`.
- Outputs: `main.csv` and `main.tex`. Repeated-question results are displayed only in Base rows.

Repeated-question pairing prioritizes genuine `star` responses matched by model, dataset, and question ID. Under the currently specified rule, missing repeats for Llama-3.1-70B and Grok-3-mini use the CtE equality-task Q1 response as a fallback; this rule does not apply to other models. It does not use the consistency score between two CtE responses or average that score with the Base control score. Response provenance is saved in `repeat_provenance.csv` and is not added to the manuscript tables.

## 2. Results Excluding Abstentions (Table 4)

Label: `tab:consistency_per_action_filtering`

```bash
.venv/bin/python -m src.evaluation.export_paper_results --table filtered
```

- Inputs: the same as the main table.
- Calculation: `core.eval_tool.summary_xidk`, selecting `dataset=overall`. Each metric excludes instances with an abstention in any response required by that metric. Denominators can therefore differ across metrics; filtering does not remove an entire model or apply a single question-level exclusion to all metrics. CON remains the mean of the five relation scores.
- Outputs: `filtered.csv` and `filtered.tex`.

## 3. Relation Classification Accuracy

Label: `tab:classification`

```bash
.venv/bin/python -m src.evaluation.export_paper_results --table classification
```

- Inputs: `Relation*.json` files under `data/answers/zero-shot/`, with the model list from `data/llm_info.json`.
- Calculation: `core.eval_relation.load_relations` normalizes relation labels. Predictions for the five relations are compared with the expected labels `Equivalence / Contains / Contains / Disjoint / Equivalence`, respectively. Average is the mean of the five accuracies.
- Outputs: `classification.csv` and `classification.tex`. The CSV includes `n_items` and per-relation `_n` columns for checking coverage. Scores use a 0–100 scale.

## 4. Significance of CtE / Oracle Improvements over Base

Label: `tab:p-value_prompting`

```bash
.venv/bin/python -m src.evaluation.export_paper_results --table significance
```

- Inputs: per-question binary scores for Base, CtE, and Oracle in the shared analysis data.
- Calculation: apply `impute_worst_case` to align abstention handling with the main table, then call `statistics.eval_pvalue.compute_pvals`. Pair observations by model, dataset, and question ID. Test CtE and Oracle improvements over Base separately using a one-sided exact McNemar test.
- Outputs: `significance.csv` retains full p-values. `significance.tex` displays `* p<0.05`, `** p<0.01`, and `*** p<0.001`; other entries are `--`.
- Tests use neither rounded table percentages nor the denominators from the table excluding abstentions.

## 5. Pearson Correlations

Label: `tab:Pearson`

```bash
.venv/bin/python -m src.evaluation.export_paper_results --table correlation
```

- Inputs: recomputed Base main-table scores and the `score` values in `data/llm_info.json`.
- Calculation: include only models with a non-null `score`. Call `scipy.stats.pearsonr` separately for the five binary relation metrics, CON, and three Jaccard metrics. Use the main-table aggregation precision rather than reconstructing values from the manuscript's two-decimal displays.
- Outputs: `correlation.csv` and `correlation.tex`, including `r`, the two-sided Pearson test `p`, and `n_models` (currently 14).
- This entry point does not depend on the working-directory-relative paths in the older `pearson_test.py` script.

## 6. Contradiction-Free Rates (Historical Supplementary Table 9)

Label: `tab:cRi_by_id_act`

```bash
.venv/bin/python -m src.evaluation.export_paper_results --table cfr
```

- Inputs: Base / CtE relation predictions and their corresponding generated answers.
- Calculation: `core.eval_relation.add_contradiction_free_scores` checks whether the model's predicted relation holds between its answer sets. `tables.export_supplementary_tables.build_tables` then aggregates these scores. A score of 1 means contradiction-free; explicit abstentions receive 0; missing classification records are excluded from the denominator.
- Outputs: `cfr.csv` and `cfr.tex`. The CSV includes `n_total`, `n_classified`, `n_selected`, and `n_scored`.

## 7. Consistency and Similarity Given Correct Classification (Historical Supplementary Table 10)

Label: `tab:cRi+`

```bash
.venv/bin/python -m src.evaluation.export_paper_results --table correct
```

- Inputs: the same as the contradiction-free rate table.
- Calculation: for each relation, select instances where the predicted relation matches the expected label, then calculate the corresponding consistency and Jaccard scores. The final binary column uses joint partition consistency; the final Jaccard column evaluates the difference component. Apply the main-table worst-case abstention policy.
- Outputs: `correct.csv` and `correct.tex`. Entries with no eligible observations are displayed as `--`, not zero.

## 8. Consistency and Similarity Given Incorrect Classification (Historical Supplementary Table 11)

Label: `tab:cRi-`

```bash
.venv/bin/python -m src.evaluation.export_paper_results --table incorrect
```

- Inputs and score definitions: the same as the preceding table.
- Calculation: for each relation, select observed classifications that differ from the expected label. Explicit classification abstentions belong to the incorrect subset; missing classification records do not.
- Outputs: `incorrect.csv` and `incorrect.tex`, retaining the effective sample count for each score.

## Additional Verification Files and Tests

Each run also produces `coverage.csv` (sample counts by model and strategy), `repeat_provenance.csv` (per-question repeated-response sources), and `source_manifest.json` (SHA-256 hashes of input files). If inputs change during generation, the script raises an error; rerun after data updates finish.

**Data-version differences:** In the snapshot used for the previous copyable tables, Base Mistral/o3 had 590 groups each, and GPT-oss-20b Oracle had 300 groups. During the verification run on November 12, 2026, the current raw files yielded 600, 600, and 417 groups, respectively. Consequently, some results recomputed from the current files differ from the previous copyable tables. Other tables were not replaced in bulk based on these updates. Use each run's `coverage.csv` and input hashes to identify its data version.

Run the regression tests with:

```bash
.venv/bin/python -m unittest tests.evaluation.test_scoring_regressions
```

To generate only the three historical supplementary tables independently, the existing entry point is also available:

```bash
.venv/bin/python -m src.evaluation.export_supplementary_tables --source raw --output output/tables_9_11
```

`--source archived` exports historical summary values rather than recomputing the corrected metrics from raw responses. Outputs from these two sources represent different evaluation versions and should be distinguished.

## Evaluation package organization

Implementation modules are grouped into `core/`, `statistics/`, `tables/`, and `analyses/`; regression tests live outside `src/`, in `tests/evaluation/`. The common export commands above remain available as compatibility CLI entry points. See [the evaluation package guide](../src/evaluation/README.md) for the complete layout and canonical module commands.

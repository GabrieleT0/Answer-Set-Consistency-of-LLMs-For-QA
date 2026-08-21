# ASCB gold-answer construction

Run from the repository root:

```bash
python src/gold_answer/build_gold_answers.py
```

The builder reads the direct (`zero-shot`) answers of the following models in
priority order: GPT-5, GPT-o3, GPT-5-mini, Gemini-2.5-pro, and DeepSeek-R
(`deepseek-reasoner` in filenames).

For each question, it chooses the answer set supported by the most models. An
answer-set comparison is order-insensitive and applies Unicode NFKC,
whitespace, and case normalization. The selected output retains the spelling
and ordering from the highest-priority agreeing model. Equal-size ties are
resolved by the same model priority. `idk` is treated as abstention; an empty
list remains a valid empty answer set.

The confidence score is the number of the five models supporting the selected
set (1--5). A quadruple is included only if all of these hold:

- `A1 = A2`
- `A3 subseteq A1`
- `A4 subseteq A1`
- `A3 intersection A4 = empty set`
- `A1 = A3 union A4`

Outputs under `data/gold_answer/`:

- `gold_answers.json`: accepted quadruples with provenance and relation checks
- `gold_answers.tsv`: one row per question, suitable as the question/gold table
- `rejected_quadruples.json`: audit records and failed constraints
- `summary.json`: counts, confidence distribution, and missing input files


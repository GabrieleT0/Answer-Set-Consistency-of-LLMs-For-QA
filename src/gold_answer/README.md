# ASCB gold-answer construction

Run `python src/gold_answer/build_gold_answers.py` from the repository root.

Each of the 3 methods (`zero-shot`, `CtE`, `Oracle`) and 5 models contributes
one complete `(A1,A2,A3,A4)` candidate. Stage 1 groups relation-valid candidates
and selects the largest identical group. If none pass, Stage 2 selects the
largest identical group among all complete candidates for manual review.
Confidence is the selected group's size (1--15). Ties use model priority
followed by method priority.

`selected_from` combines method and model, for example `CtE+GPT-5`. Every one
of the 2400 question slots remains in the JSON and TSV. A group with no valid
candidate is marked `needs_review`. Both accepted and review records use the
single `candidate_answer` field; `relation-pass` is simply `yes` or `no`.

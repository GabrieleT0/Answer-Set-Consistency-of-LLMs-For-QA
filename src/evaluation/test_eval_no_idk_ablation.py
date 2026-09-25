import unittest

import pandas as pd

from src.evaluation.eval_no_idk_ablation import (
    apply_table3_idk_policy,
    build_cross_model_delta_table,
    build_no_idk_strategy_table,
)


class Table3IdkPolicyTest(unittest.TestCase):
    def test_trivial_idk_equivalence_is_scored_as_inconsistent(self):
        frame = pd.DataFrame(
            {
                "?A1=A2": [1],
                "J(A1-A2)": [1.0],
                "idk_A1": [1],
                "idk_A2": [1],
                "idk_A3": [0],
                "idk_A4": [0],
            }
        )

        scored = apply_table3_idk_policy(frame)

        self.assertEqual(scored.loc[0, "?A1=A2"], 0)
        self.assertEqual(scored.loc[0, "J(A1-A2)"], 0.0)
        # Preserve the flags so the ablation report can compare abstention.
        self.assertEqual(scored.loc[0, "idk_A1"], 1)
        self.assertEqual(scored.loc[0, "idk_A2"], 1)

    def test_only_metrics_using_the_idk_answer_are_changed(self):
        frame = pd.DataFrame(
            {
                "?A1=A2": [1],
                "?A1=A3+A4": [1],
                "?A1>A3": [1],
                "?A1>A4": [1],
                "?A3∅A4": [1],
                "?A4=A1|3": [1],
                "J(A1-A2)": [0.7],
                "J(A1-A34)": [0.8],
                "J(A3-A4)": [0.0],
                "J(A4-A1|3)": [0.9],
                "idk_A1": [0],
                "idk_A2": [0],
                "idk_A3": [1],
                "idk_A4": [0],
            }
        )

        scored = apply_table3_idk_policy(frame)

        # A1/A2 metrics do not use A3 and remain unchanged.
        self.assertEqual(scored.loc[0, "?A1=A2"], 1)
        self.assertEqual(scored.loc[0, "J(A1-A2)"], 0.7)
        # Every metric involving A3 receives its Table 3 worst case.
        self.assertEqual(scored.loc[0, "?A1=A3+A4"], 0)
        self.assertEqual(scored.loc[0, "?A1>A3"], 0)
        self.assertEqual(scored.loc[0, "?A3∅A4"], 0)
        self.assertEqual(scored.loc[0, "?A4=A1|3"], 0)
        self.assertEqual(scored.loc[0, "J(A1-A34)"], 0.0)
        self.assertEqual(scored.loc[0, "J(A3-A4)"], 1.0)
        self.assertEqual(scored.loc[0, "J(A4-A1|3)"], 0.0)
        # A1/A4 containment does not use A3.
        self.assertEqual(scored.loc[0, "?A1>A4"], 1)

    def test_literal_no_answer_is_null_for_nonempty_ascb_questions(self):
        frame = pd.DataFrame(
            {
                "A1": ["['no answer']"],
                "A2": ["['no answer']"],
                "?A1=A2": [1],
                "idk_A1": [0],
                "idk_A2": [0],
                "idk_A3": [0],
                "idk_A4": [0],
            }
        )

        scored = apply_table3_idk_policy(frame)

        self.assertEqual(scored.loc[0, "?A1=A2"], 0)
        self.assertEqual(scored.loc[0, "idk_A1"], 1)
        self.assertEqual(scored.loc[0, "idk_A2"], 1)


class NoIdkStrategyTableTest(unittest.TestCase):
    def test_strategies_use_the_same_common_question_groups(self):
        rows = []
        action_values = {
            "zero-shot-no-idk": {
                1: [0, 0, 0, 1, 0],
                2: [1, 1, 1, 1, 1],
            },
            "classification-no-idk": {
                1: [1, 1, 1, 1, 1],
                2: [1, 1, 1, 1, 1],
            },
            # Oracle has only Q_ID 1, so all strategies must use that group.
            "fixing-no-idk": {
                1: [1, 0, 1, 1, 1],
            },
        }
        metrics = ["?A1=A2", "?A1>A3", "?A1>A4", "?A3∅A4", "?A4=A1|3"]
        for action, question_values in action_values.items():
            for q_id, values in question_values.items():
                row = {
                    "Q_ID": q_id,
                    "dataset": "example",
                    "llm": "model-a",
                    "action": action,
                    "A1": "['answer']",
                    "A2": "['answer']",
                    "A3": "['answer']",
                    "A4": "['answer']",
                    "idk_A1": 0,
                    "idk_A2": 0,
                    "idk_A3": 0,
                    "idk_A4": 0,
                }
                row.update(dict(zip(metrics, values)))
                rows.append(row)

        table = build_no_idk_strategy_table(pd.DataFrame(rows))

        self.assertEqual(table["strategy"].tolist(), ["Base", "CtE", "Oracle"])
        self.assertTrue((table["n_common_question_groups"] == 1).all())
        base = table.loc[table["strategy"].eq("Base")].iloc[0]
        cte = table.loc[table["strategy"].eq("CtE")].iloc[0]
        self.assertEqual(base["E_CON_1,2"], 0.0)
        self.assertEqual(cte["E_CON_1,2"], 100.0)


class CrossModelDeltaTableTest(unittest.TestCase):
    def test_excludes_models_without_all_strategy_metric_cells(self):
        metrics = ["?A1=A2", "?A1>A3", "?A1>A4", "?A3∅A4", "?A4=A1|3"]
        rows = []
        for llm, actions in {
            "complete": ["zero-shot", "classification", "fixing"],
            "partial": ["zero-shot", "fixing"],
        }.items():
            for action in actions:
                for metric in metrics:
                    rows.append(
                        {
                            "dataset": "overall",
                            "llm": llm,
                            "original_action": action,
                            "metric": metric,
                            "delta_no_idk_minus_original": 0.1,
                            "p_value_two_sided": 0.01,
                        }
                    )

        table = build_cross_model_delta_table(pd.DataFrame(rows))

        self.assertTrue((table["n_models"] == 1).all())
        self.assertTrue((table["models"] == "complete").all())
        macro = table.loc[
            (table["strategy"].eq("Base"))
            & (table["paper_metric"].eq("MACRO_AVERAGE"))
        ].iloc[0]
        self.assertEqual(macro["mean_delta_pp"], 10.0)


if __name__ == "__main__":
    unittest.main()

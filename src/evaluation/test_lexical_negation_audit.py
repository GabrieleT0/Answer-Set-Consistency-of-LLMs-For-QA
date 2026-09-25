import unittest

import numpy as np
import pandas as pd

from src.evaluation import lexical_negation_audit as audit


class LexicalNegationAuditTest(unittest.TestCase):
    def test_markers_cover_variants_without_substring_false_positives(self):
        questions = pd.DataFrame(
            {
                "Q1": [
                    "Which artists were not painters?",
                    "Who never attended?",
                    "Known people",
                ],
                "Q2": [
                    "Which artists weren't painters?",
                    "Who attended at no time?",
                    "Notable people",
                ],
                "Q3": [
                    "Which non-European cities?",
                    "Who didn't attend?",
                    "Known people",
                ],
                "Q4": [
                    "Cities outside Europe",
                    "Who is no longer active?",
                    "Notable people",
                ],
            }
        )
        annotated = audit.annotate_questions(questions)

        self.assertEqual(
            annotated["Q1_explicit_marker"].tolist(),
            [True, True, False],
        )
        self.assertEqual(
            annotated["Q2_explicit_marker"].tolist(),
            [True, False, False],
        )
        self.assertEqual(
            annotated["Q3_explicit_marker"].tolist(),
            [True, True, False],
        )
        self.assertEqual(
            annotated["Q4_explicit_marker"].tolist(),
            [True, True, False],
        )
        self.assertEqual(annotated.loc[0, "Q3_matched_markers"], "non-*")
        self.assertEqual(annotated.loc[1, "Q3_matched_markers"], "n't")
        self.assertEqual(annotated.loc[2, "Q4_matched_markers"], "")

    def test_idk_is_zero_only_when_metric_is_applicable(self):
        frame = pd.DataFrame(
            {
                "?A1>A4": [1.0, 1.0, np.nan],
                "idk_A1": [0, 1, 1],
                "idk_A4": [0, 0, 0],
            }
        )
        values = audit.apply_idk_as_wrong(frame, audit.METRICS[1])

        self.assertEqual(values.iloc[0], 1.0)
        self.assertEqual(values.iloc[1], 0.0)
        self.assertTrue(np.isnan(values.iloc[2]))

    def test_equivalence_uses_markers_from_either_q1_or_q2(self):
        base = pd.DataFrame(
            {
                "dataset": ["test"] * 4,
                "Q_ID": [1, 1, 2, 2],
                "action": ["zero-shot"] * 4,
                "?A1=A2": [1, 0, 1, 1],
                "idk_A1": [0, 0, 0, 0],
                "idk_A2": [0, 0, 0, 0],
            }
        )
        annotations = pd.DataFrame(
            {
                "dataset": ["test", "test"],
                "Q_ID": [1, 2],
                "Q1_explicit_marker": [False, False],
                "Q1_matched_markers": ["", ""],
                "Q2_explicit_marker": [True, False],
                "Q2_matched_markers": ["not", ""],
            }
        )

        summary, group_rates = audit.analyze_metric(
            base,
            annotations,
            audit.METRICS[0],
        )

        self.assertEqual(summary["predicate"], "?A1=A2")
        self.assertEqual(summary["marker_questions"], "Q1|Q2")
        self.assertEqual(summary["marked_question_groups"], 1)
        self.assertEqual(summary["unmarked_question_groups"], 1)
        self.assertEqual(
            group_rates["explicit_marker"].tolist(),
            [True, False],
        )
        self.assertEqual(
            group_rates["Q2_explicit_marker"].tolist(),
            [True, False],
        )


if __name__ == "__main__":
    unittest.main()

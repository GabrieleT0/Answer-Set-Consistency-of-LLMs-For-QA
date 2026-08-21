import unittest

from src.gold_answer.build_gold_answers import select_valid_quadruple, validate_relations


def candidate(model, method, answers):
    checks = validate_relations(answers)
    return {"source": (model, method), "answers": answers, "relation_checks": checks, "relation_valid": all(checks.values())}


class GoldAnswerTests(unittest.TestCase):
    def setUp(self):
        self.valid = {"Q1": ["a", "b"], "Q2": ["B", "A"], "Q3": ["a"], "Q4": ["b"]}

    def test_relations(self):
        self.assertTrue(all(validate_relations(self.valid).values()))

    def test_invalid_candidates_are_filtered_before_voting(self):
        invalid = {"Q1": ["x"], "Q2": ["y"], "Q3": [], "Q4": []}
        candidates = [candidate("GPT-5", "zero-shot", invalid) for _ in range(10)]
        candidates.append(candidate("GPT-o3", "Oracle", self.valid))
        selected, score = select_valid_quadruple(candidates)
        self.assertEqual(selected["source"], ("GPT-o3", "Oracle"))
        self.assertEqual(score, 1)

    def test_score_counts_identical_valid_quadruples(self):
        candidates = [
            candidate("GPT-5", "zero-shot", self.valid),
            candidate("GPT-o3", "CtE", {"Q1": ["B", "A"], "Q2": ["a", "b"], "Q3": ["A"], "Q4": ["B"]}),
            candidate("GPT-5-mini", "Oracle", {"Q1": ["x"], "Q2": ["x"], "Q3": ["x"], "Q4": []}),
        ]
        selected, score = select_valid_quadruple(candidates)
        self.assertEqual(selected["source"], ("GPT-5", "zero-shot"))
        self.assertEqual(score, 2)

    def test_tie_uses_model_then_method_priority(self):
        other = {"Q1": ["x"], "Q2": ["x"], "Q3": ["x"], "Q4": []}
        candidates = [candidate("GPT-o3", "zero-shot", self.valid), candidate("GPT-5", "Oracle", other), candidate("GPT-5", "CtE", self.valid)]
        selected, _ = select_valid_quadruple(candidates)
        self.assertEqual(selected["source"], ("GPT-5", "CtE"))


if __name__ == "__main__":
    unittest.main()

"""Regression cases for the three paper/code discrepancies and paired IDs."""
import unittest
import tempfile
import json
from pathlib import Path
import pandas as pd
from src.evaluation.eval_tool import analysis, _infer_action, load_answers, partition_metrics
from src.evaluation.eval_pvalue import compute_pvals, compute_pval_matrix
from src.evaluation.summary_idk_worst import summary_idk_worst
from src.evaluation.summary_idk_worst import impute_worst_case


def answers(action, values):
    return pd.DataFrame([
        dict(Q_ID=0, action=action, dataset="test", llm="m", Q_serie=q,
             task=task, Answer=value, Question=q)
        for (q, task), value in values.items()
    ])


class ScoringRegressionTests(unittest.TestCase):
    def test_wikidata_variants_are_not_base_answers(self):
        self.assertEqual(_infer_action(["zero-shot", "wikidata"], ["zero-shot"]), "wikidata")


    def test_loader_keeps_standard_and_wikidata_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "zero-shot" / "test" / "equal"
            folder.mkdir(parents=True)
            (folder / "Q1_equal_answers_m.json").write_text(json.dumps({"0": ["entity"]}))
            (folder / "Q1_equal_answers_wikidata_m.json").write_text(json.dumps({"0": ["idk"]}))
            loaded = load_answers(directory, ["test"], ["m"], ["zero-shot"], ["equal"], ["en"], ["Q1"])
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.iloc[0].Answer, ["entity"])

    def test_legacy_containment_filename_is_q3_and_canonical_takes_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "CtE" / "test" / "sup-sub"
            folder.mkdir(parents=True)
            legacy = folder / "Q2_sup-sub_answers_classAndAnswer_m.json"
            legacy.write_text(json.dumps({"0": ["legacy"]}))
            loaded = load_answers(directory, ["test"], ["m"], ["classification"], ["sup-sub"], ["en"], ["Q2", "Q3"])
            self.assertEqual(loaded.iloc[0].Q_serie, "Q3")
            (folder / "Q3_sup-sub_answers_classAndAnswer_m.json").write_text(json.dumps({"0": ["canonical"]}))
            loaded = load_answers(directory, ["test"], ["m"], ["classification"], ["sup-sub"], ["en"], ["Q2", "Q3"])
            self.assertEqual(loaded.iloc[0].Answer, ["canonical"])

    def test_difference_does_not_require_containment(self):
        df = answers("zero-shot", {("Q1", "equal"): ["a", "b"],
            ("Q2", "equal"): ["a", "b"], ("Q3", "sup-sub"): ["a", "extra"],
            ("Q4", "minus"): ["b"]})
        row = analysis(df).iloc[0]
        self.assertEqual(row["?A4=A1|3"], 0)
        self.assertEqual(row["?A4=A1-A3"], 1)
        self.assertEqual(row["?A3<=A1(same-context)"], 0)
        self.assertEqual(row["?A1>A3"], 0)

    def test_contextual_difference_and_abstention_use_actual_task(self):
        df = answers("classification", {("Q1", "equal"): ["idk"],
            ("Q2", "equal"): ["idk"], ("Q1", "sup-sub"): ["a"],
            ("Q3", "sup-sub"): ["a"], ("Q1", "minus"): ["a", "b"],
            ("Q3", "minus"): ["a", "extra"], ("Q4", "minus"): ["b"]})
        row = impute_worst_case(analysis(df)).iloc[0]
        self.assertEqual(row["?A1=A2"], 0)
        self.assertEqual(row["?A1>A3"], 1)
        self.assertEqual(row["?A4=A1|3"], 0)
        self.assertEqual(row["?A4=A1-A3"], 1)
        self.assertEqual(row["?A3<=A1(same-context)"], 0)
        df.loc[(df.Q_serie == "Q3") & (df.task == "minus"), "Answer"] = pd.Series([["idk"]], index=df.index[(df.Q_serie == "Q3") & (df.task == "minus")])
        row = impute_worst_case(analysis(df)).iloc[0]
        self.assertEqual(row["?A1>A3"], 1)
        self.assertEqual(row["?A4=A1|3"], 0)
        self.assertEqual(row["J(A3-A4)"], 1)

    def test_partition_identity_exhaustively(self):
        subsets = [{i for i in range(3) if mask & (1 << i)} for mask in range(8)]
        for a1 in subsets:
            for a3 in subsets:
                for a4 in subsets:
                    result = partition_metrics(a1, a3, a4)
                    self.assertEqual(result["?A4=A1|3"], result["?A4=A1-A3"] & result["?A3<=A1(same-context)"])

    def test_partition_identity_after_abstention_scoring(self):
        rows = []
        for action in ["zero-shot", "classification"]:
            for a1 in [[], ["idk"], ["a", "b"]]:
                for a3 in [[], ["a"], ["a", "extra"]]:
                    for a4 in [[], ["idk"], ["b"]]:
                        row = dict(action=action, A1=a1, A3=a3, A4=a4,
                                   **partition_metrics(set(a1), set(a3), set(a4)))
                        if action == "classification":
                            row.update({"A1**": a1, "A3_minus": a3})
                        rows.append(row)
        scored = impute_worst_case(pd.DataFrame(rows))
        self.assertTrue((scored["?A4=A1|3"] == (scored["?A4=A1-A3"] * scored["?A3<=A1(same-context)"])).all())

    def test_repeat_answer_is_retained_and_missing_is_not_filled(self):
        df = answers("zero-shot", {("Q1", "equal"): ["a"], ("Q2", "equal"): ["a"],
            ("Q3", "sup-sub"): ["a"], ("Q4", "minus"): ["b"]})
        missing = analysis(df)
        clf = missing.copy(); clf["action"] = "classification"; clf["?A1=A1*"] = 1
        summary = summary_idk_worst(pd.concat([missing, clf], ignore_index=True))
        zs = summary[(summary.action == "zero-shot") & (summary.dataset == "overall")]
        self.assertTrue(zs["?A1=A1*"].isna().all())
        repeat = answers("star", {("Q1", "equal"): ["idk"]})
        result = analysis(pd.concat([df, repeat], ignore_index=True))
        self.assertEqual(result.iloc[0]["A1*"], ["idk"])
        self.assertEqual(result.iloc[0]["idk_A1*"], 1)
        self.assertEqual(impute_worst_case(result).iloc[0]["J(A1-A1*)"], 0)

    def test_no_answer_and_mixed_idk_are_failures(self):
        df = pd.DataFrame({"A1": [["no answer"], ["a", "IDK"]], "A2": [["no answer"], ["a", "IDK"]], "?A1=A2": [1, 1]})
        self.assertEqual(impute_worst_case(df)["?A1=A2"].tolist(), [0, 0])

    def test_overall_pairing_includes_dataset(self):
        rows = []
        for dataset in ["x", "y"]:
            for qid in range(6):
                for model in ["i", "j"]:
                    from src.evaluation.eval_pvalue import PREDICATES
                    rows.append(dict(dataset=dataset, Q_ID=qid, action="zero-shot", llm=model,
                        **{m: int((dataset == "x") == (model == "i")) for m in PREDICATES}))
        matrix = compute_pval_matrix(pd.DataFrame(rows), ["i", "j"])[("overall", "?A1=A2")]
        self.assertAlmostEqual(matrix.loc["i", "j"], 0.61279296875)

    def test_p_values_keep_threshold_precision(self):
        from src.evaluation.eval_pvalue import PREDICATES
        rows = [dict(dataset="x", Q_ID=i, llm="m", action=action,
                     **{m: int(action == "classification") for m in PREDICATES})
                for i in range(20) for action in ["zero-shot", "classification"]]
        result = compute_pvals(pd.DataFrame(rows))
        p = result.loc[(result.dataset == "overall") & (result.action == "classification"), "p(A1=A2)"].iloc[0]
        self.assertGreater(p, 0)
        self.assertAlmostEqual(p, 2**-20)


class SupplementaryReproductionTests(unittest.TestCase):
    def test_cte_loader_handles_dataset_case_and_model_filename_alias(self):
        from src.evaluation.eval_relation import load_relation_clf
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "lc-quad" / "sup-sub"
            folder.mkdir(parents=True)
            (folder / "Containment_sup-sub_relation_llama3.1-8b.json").write_text(json.dumps({"0": ["contains"]}))
            frame = load_relation_clf(directory, ["LC-QuAD"], ["llama3.1:8b"], ["sup-sub"])
            self.assertEqual(frame.iloc[0]["dataset"], "LC-QuAD")
            self.assertEqual(frame.iloc[0]["R(1-3)"], "Contains")

    def test_contradiction_free_checks_predicted_relation_not_gold_agreement(self):
        from src.evaluation.eval_relation import add_contradiction_free_scores
        frame = pd.DataFrame([dict(action="zero-shot", A1=["a", "b"], A2=["b", "c"],
                                   **{"R(1-2)": "Disjoint"})])
        result = add_contradiction_free_scores(frame)
        self.assertEqual(result.iloc[0]["?SC(A1=A2)"], 0)
        frame["R(1-2)"] = "Overlap"
        self.assertEqual(add_contradiction_free_scores(frame).iloc[0]["?SC(A1=A2)"], 1)

    def test_contradiction_free_uses_correct_dialogue_and_abstention_policy(self):
        from src.evaluation.eval_relation import add_contradiction_free_scores
        frame = pd.DataFrame([dict(action="classification", A1=["idk"], A2=["idk"], A3=["a"],
                                   **{"A1*": ["a", "b"], "R(1-3)": "Contains", "R(1-2)": "Unknown"})])
        result = add_contradiction_free_scores(frame)
        self.assertEqual(result.iloc[0]["?SC(A1>A3)"], 1)
        self.assertEqual(result.iloc[0]["?SC(A1=A2)"], 0)
        self.assertTrue(pd.isna(result.iloc[0]["?SC(A1>A4)"]))
        result["A1*"] = pd.Series([["idk"]])
        self.assertEqual(impute_worst_case(result).iloc[0]["?SC(A1>A3)"], 0)

    def test_table_groups_use_joint_metric_and_exclude_missing_predictions(self):
        from src.evaluation.export_supplementary_tables import build_tables
        frame = analysis(answers("zero-shot", {("Q1", "equal"): ["a", "b"],
            ("Q2", "equal"): ["a", "b"], ("Q3", "sup-sub"): ["a"], ("Q4", "minus"): ["a", "b"]}))
        from src.evaluation.eval_relation import GT, add_contradiction_free_scores
        for relation, truth in GT.items():
            frame[relation] = truth
        frame["R(1-3)"] = None
        frame = add_contradiction_free_scores(frame)
        tables = build_tables(frame, ["m"])
        joint = tables[(tables.table == 10) & (tables.action == "zero-shot") & (tables.metric == "?A4=A1|3")].iloc[0]
        self.assertEqual(joint.score_percent, 0)
        self.assertEqual(joint.n_selected, 1)
        from src.evaluation.eval_relation import update_summary_by_relations
        summary = pd.DataFrame([dict(dataset="overall", action="zero-shot", llm="m")])
        summary = update_summary_by_relations(frame, summary)
        self.assertEqual(summary.iloc[0]["?A4=A1|3(+)"], 0)
        self.assertEqual(summary.iloc[0]["?A1=A3+A4(+)"], 1)
        self.assertIn("J(4-1|3)+", summary.columns)
        missing = tables[(tables.table == 11) & (tables.action == "zero-shot") & (tables.metric == "?A1>A3")].iloc[0]
        self.assertEqual(missing.n_selected, 0)
        self.assertTrue(pd.isna(missing.score_percent))

    def test_manuscript_update_replaces_only_the_three_table_blocks(self):
        from src.evaluation.export_supplementary_tables import archived_tables, write_tables, LABELS
        root = Path(__file__).resolve().parents[2]
        tables = archived_tables(pd.read_csv(root / "data/evaluation_results/summary_filtered_idk_filtered.csv"), ["llama3.1:8b"])
        with tempfile.TemporaryDirectory() as directory:
            appendix = Path(directory) / "appendix.tex"
            original = "KEEP BEFORE\n" + "\n".join("\\begin{table}OLD\\label{"+label+"}\\end{table}" for label in LABELS.values()) + "\nKEEP AFTER"
            appendix.write_text(original)
            write_tables(tables, Path(directory) / "corrected", source="raw", manuscript=appendix)
            updated = appendix.read_text()
            self.assertTrue(updated.startswith("KEEP BEFORE"))
            self.assertTrue(updated.endswith("KEEP AFTER"))
            self.assertNotIn("OLD", updated)
            for label in LABELS.values():
                self.assertEqual(updated.count("\\label{"+label+"}"), 1)
            with self.assertRaises(ValueError):
                write_tables(tables, Path(directory) / "archived", source="archived", manuscript=appendix)
            self.assertEqual(appendix.read_text(), updated)

    def test_frozen_export_has_explicit_legacy_union_mapping(self):
        from src.evaluation.export_supplementary_tables import archived_tables, write_tables
        root = Path(__file__).resolve().parents[2]
        archived = pd.read_csv(root / "data/evaluation_results/summary_filtered_idk_filtered.csv")
        tables = archived_tables(archived, ["llama3.1:8b"])
        row = tables[(tables.table == 9) & (tables.action == "zero-shot") & (tables.metric == "?SC(A1=A2)")].iloc[0]
        self.assertAlmostEqual(row.score_percent, 93.83)
        row = tables[(tables.table == 10) & (tables.action == "zero-shot") & (tables.metric == "?A4=A1|3")].iloc[0]
        self.assertEqual(row.source_column, "?A1=A3+A4(+)")
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "tables"
            write_tables(tables, prefix, source="archived")
            self.assertTrue(prefix.with_suffix(".csv").exists())
            text = prefix.with_suffix(".tex").read_text()
            self.assertIn("Historical export", text)
            self.assertEqual(text.count("\\begin{table}"), 3)

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Audit explicit negation/exclusion wording and Base consistency.

The inferential analysis uses the question group as the statistical unit:

1. Mark Q1, Q2, Q3, and Q4 using a fixed, inspectable lexical rule.
2. Treat a relevant IDK response as inconsistent, matching Table 3.
3. Average the binary consistency outcome over models within each question
   group.
4. Apply the arcsine-square-root variance-stabilizing transformation to each
   question-group proportion.
5. Compare marked and unmarked groups with a two-sided Welch t-test.

Example
-------
python -m src.evaluation.analyses.lexical_negation_audit

By default, artifacts are written to ``output/lexical_negation_audit``.
"""

from __future__ import annotations

from src.evaluation.paths import REPO_ROOT

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


DEFAULT_QUESTIONS = REPO_ROOT / "data" / "ASCB" / "en" / "unified-dataset.tsv"
REGENERATED_ANALYSIS = REPO_ROOT / "output" / "analysis.csv"
COMMITTED_ANALYSIS = REPO_ROOT / "data" / "evaluation_results" / "analysis.csv"
DEFAULT_ANALYSIS = (
    REGENERATED_ANALYSIS if REGENERATED_ANALYSIS.exists() else COMMITTED_ANALYSIS
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "lexical_negation_audit"

# Keep the rules separate so every match can be traced to its marker.
# ``n't`` and ``no longer`` are included as ordinary English realizations of
# explicit negation.  They are needed to avoid restricting the audit to the
# uncontracted token "not".
MARKER_PATTERNS: dict[str, str] = {
    "not": r"\bnot\b",
    "never": r"\bnever\b",
    "excluding": r"\bexcluding\b",
    "except": r"\bexcept\b",
    "without": r"\bwithout\b",
    "outside": r"\boutside\b",
    "no longer": r"\bno\s+longer\b",
    "n't": r"\b\w+n['’]t\b",
    "non-*": r"\bnon[-‐‑‒–—]\w+",
    "other than": r"\bother\s+than\b",
    "aside from": r"\baside\s+from\b",
}

COMPILED_MARKERS = {
    name: re.compile(pattern, flags=re.IGNORECASE)
    for name, pattern in MARKER_PATTERNS.items()
}

GROUP_COLUMNS = ["dataset", "Q_ID"]
QUESTION_COLUMNS = ("Q1", "Q2", "Q3", "Q4")


@dataclass(frozen=True)
class MetricSpec:
    label: str
    column: str
    idk_columns: tuple[str, ...]
    marker_questions: tuple[str, ...]


METRICS = (
    MetricSpec(
        label="q1_q2_equivalence",
        column="?A1=A2",
        idk_columns=("idk_A1", "idk_A2"),
        marker_questions=("Q1", "Q2"),
    ),
    MetricSpec(
        label="q4_containment",
        column="?A1>A4",
        idk_columns=("idk_A1", "idk_A4"),
        marker_questions=("Q4",),
    ),
    MetricSpec(
        label="complete_partition",
        column="?A4=A1|3",
        idk_columns=("idk_A1", "idk_A3", "idk_A4"),
        marker_questions=("Q4",),
    ),
)


def matched_markers(text: object) -> list[str]:
    """Return all lexical marker labels found in ``text``."""
    value = "" if pd.isna(text) else str(text)
    return [
        name
        for name, pattern in COMPILED_MARKERS.items()
        if pattern.search(value)
    ]


def annotate_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Add transparent marker annotations for every question in a group."""
    required = set(QUESTION_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Question data is missing columns: {sorted(missing)}")

    out = df.copy()
    for question in QUESTION_COLUMNS:
        match_column = f"{question}_matched_markers"
        flag_column = f"{question}_explicit_marker"
        out[match_column] = out[question].apply(matched_markers)
        out[flag_column] = out[match_column].str.len().gt(0)
        out[match_column] = out[match_column].apply(lambda xs: "|".join(xs))
    return out


def load_dataset_audit(path: Path) -> pd.DataFrame:
    """Load and annotate the 600-group ASCB source table."""
    questions = pd.read_csv(path, sep="\t", encoding="utf-8")
    return annotate_questions(questions)


def _unique_evaluation_questions(analysis: pd.DataFrame) -> pd.DataFrame:
    """Return one wording record per evaluation question group."""
    wording_columns = GROUP_COLUMNS + list(QUESTION_COLUMNS)
    wording = analysis[wording_columns].drop_duplicates()

    variants = wording.groupby(GROUP_COLUMNS, dropna=False).size()
    if variants.gt(1).any():
        bad = variants[variants.gt(1)].index.tolist()[:5]
        raise ValueError(
            "Some evaluation groups have multiple Q1-Q4 wordings; "
            f"examples: {bad}"
        )
    return annotate_questions(wording.reset_index(drop=True))


def _validate_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def apply_idk_as_wrong(df: pd.DataFrame, spec: MetricSpec) -> pd.Series:
    """Return a metric copy with relevant IDK cases set to zero."""
    _validate_columns(
        df,
        [spec.column, *spec.idk_columns],
        f"Metric {spec.label}",
    )
    values = pd.to_numeric(df[spec.column], errors="coerce").copy()
    idk = (
        df[list(spec.idk_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(bool)
        .any(axis=1)
    )
    values.loc[values.notna() & idk] = 0.0
    return values


def arcsine_welch_test(
    marked_proportions: pd.Series,
    unmarked_proportions: pd.Series,
) -> tuple[float, float]:
    """Two-sided Welch test on arcsine-square-root proportions."""
    marked = np.arcsin(
        np.sqrt(np.clip(marked_proportions.to_numpy(dtype=float), 0.0, 1.0))
    )
    unmarked = np.arcsin(
        np.sqrt(np.clip(unmarked_proportions.to_numpy(dtype=float), 0.0, 1.0))
    )
    result = ttest_ind(
        marked,
        unmarked,
        equal_var=False,
        alternative="two-sided",
        nan_policy="omit",
    )
    return float(result.statistic), float(result.pvalue)


def analyze_metric(
    base: pd.DataFrame,
    question_annotations: pd.DataFrame,
    spec: MetricSpec,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compute descriptive rates and a question-group-level test."""
    working = base.copy()
    working["outcome"] = apply_idk_as_wrong(working, spec)
    working = working[working["outcome"].notna()].copy()
    marker_flag_columns = [
        f"{question}_explicit_marker" for question in spec.marker_questions
    ]
    marker_match_columns = [
        f"{question}_matched_markers" for question in spec.marker_questions
    ]
    working = working.merge(
        question_annotations[
            GROUP_COLUMNS + marker_flag_columns + marker_match_columns
        ],
        on=GROUP_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    if working[marker_flag_columns].isna().to_numpy().any():
        raise ValueError("Some evaluation rows could not be matched to a question group")

    working["explicit_marker"] = working[marker_flag_columns].any(axis=1)
    working["matched_markers"] = working[marker_match_columns].apply(
        lambda row: "|".join(
            marker
            for marker in MARKER_PATTERNS
            if any(
                marker in value.split("|")
                for value in row.astype(str)
                if value
            )
        ),
        axis=1,
    )

    row_rates = working.groupby("explicit_marker")["outcome"].agg(
        ["sum", "count", "mean"]
    )
    for expected in (False, True):
        if expected not in row_rates.index:
            questions = ", ".join(spec.marker_questions)
            raise ValueError(
                f"Both marked and unmarked groups are required for {questions}"
            )

    group_rates = (
        working.groupby(
            GROUP_COLUMNS
            + marker_flag_columns
            + marker_match_columns
            + ["explicit_marker", "matched_markers"],
            dropna=False,
        )["outcome"]
        .agg(successes="sum", n_models="count", consistency="mean")
        .reset_index()
    )
    marked = group_rates.loc[
        group_rates["explicit_marker"], "consistency"
    ]
    unmarked = group_rates.loc[
        ~group_rates["explicit_marker"], "consistency"
    ]
    statistic, p_value = arcsine_welch_test(marked, unmarked)

    marked_rate = float(row_rates.loc[True, "mean"])
    unmarked_rate = float(row_rates.loc[False, "mean"])
    summary = {
        "metric": spec.label,
        "predicate": spec.column,
        "marker_questions": "|".join(spec.marker_questions),
        "action": str(base["action"].iloc[0]),
        "idk_policy": "relevant IDK responses are assigned outcome 0",
        "marked_question_groups": int(len(marked)),
        "unmarked_question_groups": int(len(unmarked)),
        "marked_model_question_rows": int(row_rates.loc[True, "count"]),
        "unmarked_model_question_rows": int(row_rates.loc[False, "count"]),
        "marked_consistency": marked_rate,
        "unmarked_consistency": unmarked_rate,
        "difference_marked_minus_unmarked": marked_rate - unmarked_rate,
        "test": "two-sided Welch t-test after question-group aggregation "
        "and arcsine-square-root transformation",
        "test_statistic": statistic,
        "p_value": p_value,
    }
    group_rates.insert(0, "metric", spec.label)
    group_rates.insert(1, "predicate", spec.column)
    group_rates.insert(2, "marker_questions", "|".join(spec.marker_questions))
    return summary, group_rates


def run_audit(
    questions_path: Path,
    analysis_path: Path,
    output_dir: Path,
    action: str = "zero-shot",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete lexical and consistency audit and save artifacts."""
    dataset_questions = load_dataset_audit(questions_path)
    analysis = pd.read_csv(analysis_path)
    _validate_columns(
        analysis,
        GROUP_COLUMNS + ["action", *QUESTION_COLUMNS],
        "Evaluation analysis",
    )
    base = analysis[analysis["action"].eq(action)].copy()
    if base.empty:
        raise ValueError(f"No evaluation rows found for action={action!r}")

    evaluation_questions = _unique_evaluation_questions(base)
    dataset_counts = {
        question: int(dataset_questions[f"{question}_explicit_marker"].sum())
        for question in QUESTION_COLUMNS
    }
    evaluation_counts = {
        question: int(evaluation_questions[f"{question}_explicit_marker"].sum())
        for question in QUESTION_COLUMNS
    }
    if dataset_counts != evaluation_counts:
        raise ValueError(
            "Marker counts disagree between the dataset and evaluation files: "
            f"dataset={dataset_counts}, evaluation={evaluation_counts}"
        )

    summaries: list[dict[str, object]] = []
    group_frames: list[pd.DataFrame] = []
    for spec in METRICS:
        summary, group_rates = analyze_metric(base, evaluation_questions, spec)
        summaries.append(summary)
        group_frames.append(group_rates)

    summary_df = pd.DataFrame(summaries)
    group_df = pd.concat(group_frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_questions.to_csv(
        output_dir / "question_marker_audit.csv",
        index=False,
    )
    marker_count_rows = []
    for marker, pattern in MARKER_PATTERNS.items():
        row: dict[str, object] = {"marker": marker, "regex": pattern}
        for question in QUESTION_COLUMNS:
            row[f"{question}_question_groups"] = int(
                dataset_questions[f"{question}_matched_markers"]
                .str.split("|")
                .apply(lambda labels: marker in labels)
                .sum()
            )
        marker_count_rows.append(row)
    pd.DataFrame(marker_count_rows).to_csv(
        output_dir / "marker_counts.csv",
        index=False,
    )
    group_df.to_csv(
        output_dir / "question_group_consistency.csv",
        index=False,
    )
    summary_df.to_csv(
        output_dir / "summary.csv",
        index=False,
    )
    metadata = {
        "questions_path": str(questions_path),
        "analysis_path": str(analysis_path),
        "action": action,
        "n_question_groups": int(len(dataset_questions)),
        "marker_counts": dataset_counts,
        "marker_rates": {
            key: value / len(dataset_questions)
            for key, value in dataset_counts.items()
        },
        "marker_patterns": MARKER_PATTERNS,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Question groups: {len(dataset_questions)}")
    for question in QUESTION_COLUMNS:
        count = dataset_counts[question]
        print(
            f"{question} explicit negation/exclusion: "
            f"{count}/{len(dataset_questions)} ({100 * count / len(dataset_questions):.1f}%)"
        )
    print()
    for row in summaries:
        print(
            f"{row['metric']}: "
            f"{100 * float(row['marked_consistency']):.1f}% with markers vs "
            f"{100 * float(row['unmarked_consistency']):.1f}% without; "
            f"difference={100 * float(row['difference_marked_minus_unmarked']):+.2f} pp; "
            f"p={float(row['p_value']):.6f}"
        )
    print(f"\nArtifacts written to {output_dir}")
    return summary_df, dataset_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help=f"ASCB TSV file (default: {DEFAULT_QUESTIONS})",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=DEFAULT_ANALYSIS,
        help=f"Evaluation analysis CSV (default: {DEFAULT_ANALYSIS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Artifact directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--action",
        default="zero-shot",
        help="Evaluation action to analyze (default: zero-shot/Base)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_audit(
        questions_path=args.questions,
        analysis_path=args.analysis,
        output_dir=args.output_dir,
        action=args.action,
    )

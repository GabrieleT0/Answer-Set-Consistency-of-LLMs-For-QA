"""Paired analysis for the no-IDK prompt ablation.

This script consumes the regular ``output/analysis.csv`` produced by
``src/evaluation/run.py``. It compares each original action with the matching
no-IDK action on identical (dataset, model, question-group) instances.

Before comparing the two prompt conditions, the script applies the Table 3
policy to both: if any answer required by a metric is ``idk``, ``no answer``,
or an empty parsed list, binary consistency and equivalence-style Jaccard
metrics receive 0, while the disjointness Jaccard metric receives 1 (its worst
value). This prevents pairs of null responses from receiving trivial
consistency.

Binary consistency and IDK/empty indicators use a two-sided exact McNemar
test. Continuous Jaccard metrics use a two-sided paired Wilcoxon signed-rank
test. Effect sizes (the original mean, no-IDK mean, and paired difference) are
always reported, including when a significance test is undefined.

The script also writes a compact, paper-oriented table comparing Base, CtE,
and Oracle under no-IDK prompting. Within each model, this table uses only
question groups shared by every currently available no-IDK strategy, so its
rows are directly comparable. Rerunning the script automatically expands the
table as additional model/strategy outputs become available.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]

ACTION_PAIRS = [
    ("zero-shot", "zero-shot-no-idk"),
    ("fixing", "fixing-no-idk"),
    ("classification", "classification-no-idk"),
]

INTERVENTION_PAIRS = [
    ("fixing", "fixing-no-idk"),
    ("classification", "classification-no-idk"),
]

CONSISTENCY_METRICS = [
    "?A1=A2",
    "?A1=A3+A4",
    "?A1>A3",
    "?A1>A4",
    "?A3∅A4",
    "?A4=A1|3",
]

PAPER_CONSISTENCY_METRICS = [
    "?A1=A2",
    "?A1>A3",
    "?A1>A4",
    "?A3∅A4",
    "?A4=A1|3",
]

PAPER_METRIC_NAMES = {
    "?A1=A2": "E_CON_1,2",
    "?A1>A3": "N_CON_3,1",
    "?A1>A4": "N_CON_4,1",
    "?A3∅A4": "D_CON_3,4",
    "?A4=A1|3": r"E_CON_4,1\3",
}

NO_IDK_STRATEGIES = [
    ("zero-shot-no-idk", "Base"),
    ("classification-no-idk", "CtE"),
    ("fixing-no-idk", "Oracle"),
]

ORIGINAL_STRATEGY_NAMES = {
    "zero-shot": "Base",
    "classification": "CtE",
    "fixing": "Oracle",
}

JACCARD_METRICS = [
    "J(A1-A2)",
    "J(A1-A34)",
    "J(A3-A4)",
    "J(A4-A1|3)",
]

IDK_COLUMNS = ["idk_A1", "idk_A2", "idk_A3", "idk_A4"]
JOIN_KEYS = ["Q_ID", "dataset", "llm"]
NULL_ANSWER_TOKENS = {"idk", "no answer"}

# Table 3 treats an IDK/empty answer as the worst possible outcome for every
# metric that uses that answer.
METRIC_IDK_MAP = {
    "?A1=A2": ("idk_A1", "idk_A2"),
    "?A1=A3+A4": ("idk_A1", "idk_A3", "idk_A4"),
    "?A1>A3": ("idk_A1", "idk_A3"),
    "?A1>A4": ("idk_A1", "idk_A4"),
    "?A3∅A4": ("idk_A3", "idk_A4"),
    "?A4=A1|3": ("idk_A1", "idk_A3", "idk_A4"),
    "J(A1-A2)": ("idk_A1", "idk_A2"),
    "J(A1-A34)": ("idk_A1", "idk_A3", "idk_A4"),
    "J(A3-A4)": ("idk_A3", "idk_A4"),
    "J(A4-A1|3)": ("idk_A1", "idk_A3", "idk_A4"),
}

METRIC_WORST_CASE = {
    **{metric: 0.0 for metric in CONSISTENCY_METRICS},
    "J(A1-A2)": 0.0,
    "J(A1-A34)": 0.0,
    # Lower overlap is better for the expected disjoint relation.
    "J(A3-A4)": 1.0,
    "J(A4-A1|3)": 0.0,
}


def _contains_null_answer(value) -> bool:
    """Detect an empty set, IDK, or no-answer token in an answer-set cell."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower().strip("'\"") in NULL_ANSWER_TOKENS:
            return True
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return False
    if isinstance(value, (list, tuple, set)):
        if len(value) == 0:
            return True
        return any(
            str(item).strip().lower().strip("'\"") in NULL_ANSWER_TOKENS
            for item in value
        )
    return False


def _normalize_null_answer_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Combine saved IDK flags with null tokens detectable in answer cells."""
    out = df.copy()
    for answer in ["A1", "A2", "A3", "A4"]:
        flag = f"idk_{answer}"
        if flag not in out.columns:
            raise ValueError(f"Missing required null-answer flag: {flag}")
        saved_flag = (
            pd.to_numeric(out[flag], errors="coerce")
            .fillna(0)
            .astype(bool)
        )
        if answer in out.columns:
            detected = out[answer].apply(_contains_null_answer)
            out[flag] = (saved_flag | detected).astype(int)
        else:
            out[flag] = saved_flag.astype(int)
    return out


def apply_table3_idk_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Impute the worst case for metrics involving a null answer.

    ASCB questions have non-empty answer sets. Therefore ``idk``, ``no
    answer``, and an empty parsed list are all treated as null responses.
    """
    out = _normalize_null_answer_flags(df)
    for metric, idk_columns in METRIC_IDK_MAP.items():
        if metric not in out.columns:
            continue
        missing = [column for column in idk_columns if column not in out.columns]
        if missing:
            raise ValueError(
                f"Cannot apply the Table 3 policy to {metric!r}; "
                f"missing IDK columns: {missing}"
            )
        idk_or_empty = (
            out[list(idk_columns)]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(bool)
            .any(axis=1)
        )
        applicable = out[metric].notna()
        out.loc[applicable & idk_or_empty, metric] = METRIC_WORST_CASE[metric]
    return out


def exact_mcnemar_two_sided(
    original: pd.Series, no_idk: pd.Series
) -> tuple[int, int, float]:
    """Return no-IDK wins, original wins, and the exact two-sided p-value."""
    valid = original.notna() & no_idk.notna()
    original = original.loc[valid].astype(int)
    no_idk = no_idk.loc[valid].astype(int)
    no_idk_wins = int(((no_idk == 1) & (original == 0)).sum())
    original_wins = int(((no_idk == 0) & (original == 1)).sum())
    discordant = no_idk_wins + original_wins
    if discordant == 0:
        return no_idk_wins, original_wins, 1.0
    p_value = binomtest(
        no_idk_wins,
        discordant,
        p=0.5,
        alternative="two-sided",
    ).pvalue
    return no_idk_wins, original_wins, float(p_value)


def paired_wilcoxon_two_sided(
    original: pd.Series, no_idk: pd.Series
) -> tuple[int, int, float]:
    """Return positive/negative paired changes and a two-sided p-value."""
    valid = original.notna() & no_idk.notna()
    differences = (
        no_idk.loc[valid].astype(float) - original.loc[valid].astype(float)
    )
    positive = int((differences > 0).sum())
    negative = int((differences < 0).sum())
    if differences.empty or np.allclose(differences.to_numpy(), 0.0):
        return positive, negative, 1.0
    try:
        p_value = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
        ).pvalue
    except ValueError:
        p_value = np.nan
    return positive, negative, float(p_value)


def pair_actions(
    df: pd.DataFrame, original_action: str, no_idk_action: str
) -> pd.DataFrame:
    """Apply Table 3 scoring and pair rows without mixing datasets."""
    df = apply_table3_idk_policy(df)
    metric_columns = [
        column
        for column in CONSISTENCY_METRICS + JACCARD_METRICS + IDK_COLUMNS
        if column in df.columns
    ]
    original = df.loc[
        df["action"].eq(original_action), JOIN_KEYS + metric_columns
    ].copy()
    ablation = df.loc[
        df["action"].eq(no_idk_action), JOIN_KEYS + metric_columns
    ].copy()

    if original.duplicated(JOIN_KEYS).any():
        raise ValueError(
            f"Duplicate rows found for original action {original_action!r}."
        )
    if ablation.duplicated(JOIN_KEYS).any():
        raise ValueError(
            f"Duplicate rows found for no-IDK action {no_idk_action!r}."
        )

    paired = original.merge(
        ablation,
        on=JOIN_KEYS,
        how="inner",
        suffixes=("_original", "_no_idk"),
        validate="one_to_one",
    )
    if paired.empty:
        return paired

    for suffix in ["original", "no_idk"]:
        for answer in ["A1", "A2", "A3", "A4"]:
            source = f"idk_{answer}_{suffix}"
            if source in paired.columns:
                paired[f"idk_or_empty_{answer}_{suffix}"] = paired[source]
        idk_cols = [
            f"{column}_{suffix}"
            for column in IDK_COLUMNS
            if f"{column}_{suffix}" in paired.columns
        ]
        paired[f"idk_or_empty_any_{suffix}"] = (
            paired[idk_cols].max(axis=1) if idk_cols else np.nan
        )
    return paired


def summarize_group(
    group: pd.DataFrame,
    dataset: str,
    llm: str,
    original_action: str,
    no_idk_action: str,
) -> list[dict]:
    rows: list[dict] = []
    metric_specs = [
        *[(metric, "binary") for metric in CONSISTENCY_METRICS],
        *[(metric, "continuous") for metric in JACCARD_METRICS],
        *[
            (f"idk_or_empty_{answer}", "binary")
            for answer in ["A1", "A2", "A3", "A4", "any"]
        ],
    ]

    for metric, metric_type in metric_specs:
        original_col = f"{metric}_original"
        no_idk_col = f"{metric}_no_idk"
        if original_col not in group or no_idk_col not in group:
            continue

        valid = group[original_col].notna() & group[no_idk_col].notna()
        original = group.loc[valid, original_col].astype(float)
        no_idk = group.loc[valid, no_idk_col].astype(float)
        if original.empty:
            continue

        if metric_type == "binary":
            positive, negative, p_value = exact_mcnemar_two_sided(
                original, no_idk
            )
            test = "exact McNemar (two-sided)"
        else:
            positive, negative, p_value = paired_wilcoxon_two_sided(
                original, no_idk
            )
            test = "paired Wilcoxon (two-sided)"

        rows.append(
            {
                "dataset": dataset,
                "llm": llm,
                "original_action": original_action,
                "no_idk_action": no_idk_action,
                "idk_scoring_policy": "Table 3 worst case",
                "metric": metric,
                "metric_type": metric_type,
                "n_pairs": int(valid.sum()),
                "original_mean": round(float(original.mean()), 6),
                "no_idk_mean": round(float(no_idk.mean()), 6),
                "delta_no_idk_minus_original": round(
                    float((no_idk - original).mean()), 6
                ),
                "n_no_idk_higher": positive,
                "n_original_higher": negative,
                "p_value_two_sided": round(p_value, 6)
                if not np.isnan(p_value)
                else np.nan,
                "test": test,
            }
        )
    return rows


def build_ablation_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-dataset and overall paired reports for available actions."""
    all_rows: list[dict] = []
    paired_action_count = 0

    for original_action, no_idk_action in ACTION_PAIRS:
        paired = pair_actions(df, original_action, no_idk_action)
        if paired.empty:
            continue
        paired_action_count += 1

        for (dataset, llm), group in paired.groupby(["dataset", "llm"]):
            all_rows.extend(
                summarize_group(
                    group,
                    str(dataset),
                    str(llm),
                    original_action,
                    no_idk_action,
                )
            )

        for llm, group in paired.groupby("llm"):
            all_rows.extend(
                summarize_group(
                    group,
                    "overall",
                    str(llm),
                    original_action,
                    no_idk_action,
                )
            )

    if paired_action_count == 0:
        available = sorted(df["action"].dropna().unique().tolist())
        raise ValueError(
            "No matched no-IDK actions were found. Run no-IDK generation and "
            "src/evaluation/run.py first. Available actions: "
            f"{available}"
        )

    return pd.DataFrame(all_rows).sort_values(
        ["original_action", "dataset", "llm", "metric"],
        ignore_index=True,
    )


def summarize_intervention_group(
    group: pd.DataFrame,
    dataset: str,
    llm: str,
    original_action: str,
    no_idk_action: str,
) -> list[dict]:
    """Compare intervention gains within the original and no-IDK regimes."""
    rows: list[dict] = []
    metric_specs = [
        *[(metric, "binary") for metric in CONSISTENCY_METRICS],
        *[(metric, "continuous") for metric in JACCARD_METRICS],
        *[
            (f"idk_or_empty_{answer}", "binary")
            for answer in ["A1", "A2", "A3", "A4", "any"]
        ],
    ]

    for metric, metric_type in metric_specs:
        columns = {
            "original_baseline": f"{metric}_original_baseline",
            "original_method": f"{metric}_original_method",
            "no_idk_baseline": f"{metric}_no_idk_baseline",
            "no_idk_method": f"{metric}_no_idk_method",
        }
        if any(column not in group for column in columns.values()):
            continue

        valid = group[list(columns.values())].notna().all(axis=1)
        values = {
            name: group.loc[valid, column].astype(float)
            for name, column in columns.items()
        }
        if values["original_baseline"].empty:
            continue

        original_gain = (
            values["original_method"] - values["original_baseline"]
        )
        no_idk_gain = (
            values["no_idk_method"] - values["no_idk_baseline"]
        )

        if metric_type == "binary":
            _, _, original_p = exact_mcnemar_two_sided(
                values["original_baseline"], values["original_method"]
            )
            _, _, no_idk_p = exact_mcnemar_two_sided(
                values["no_idk_baseline"], values["no_idk_method"]
            )
            test = "exact McNemar (two-sided)"
        else:
            _, _, original_p = paired_wilcoxon_two_sided(
                values["original_baseline"], values["original_method"]
            )
            _, _, no_idk_p = paired_wilcoxon_two_sided(
                values["no_idk_baseline"], values["no_idk_method"]
            )
            test = "paired Wilcoxon (two-sided)"

        rows.append(
            {
                "dataset": dataset,
                "llm": llm,
                "original_action": original_action,
                "no_idk_action": no_idk_action,
                "idk_scoring_policy": "Table 3 worst case",
                "metric": metric,
                "metric_type": metric_type,
                "n_four_way_pairs": int(valid.sum()),
                "original_baseline_mean": round(
                    float(values["original_baseline"].mean()), 6
                ),
                "original_method_mean": round(
                    float(values["original_method"].mean()), 6
                ),
                "original_gain": round(float(original_gain.mean()), 6),
                "original_gain_p_value_two_sided": round(original_p, 6),
                "no_idk_baseline_mean": round(
                    float(values["no_idk_baseline"].mean()), 6
                ),
                "no_idk_method_mean": round(
                    float(values["no_idk_method"].mean()), 6
                ),
                "no_idk_gain": round(float(no_idk_gain.mean()), 6),
                "no_idk_gain_p_value_two_sided": round(no_idk_p, 6),
                "gain_change_no_idk_minus_original": round(
                    float((no_idk_gain - original_gain).mean()), 6
                ),
                "test": test,
            }
        )
    return rows


def build_intervention_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report whether intervention gains persist after removing the IDK option."""
    baseline = pair_actions(df, "zero-shot", "zero-shot-no-idk")
    if baseline.empty:
        raise ValueError(
            "Matched zero-shot and zero-shot-no-idk rows are required for "
            "the intervention-effect report."
        )

    all_rows: list[dict] = []
    for original_action, no_idk_action in INTERVENTION_PAIRS:
        method = pair_actions(df, original_action, no_idk_action)
        if method.empty:
            continue
        four_way = baseline.merge(
            method,
            on=JOIN_KEYS,
            how="inner",
            suffixes=("_baseline", "_method"),
            validate="one_to_one",
        )
        for (dataset, llm), group in four_way.groupby(["dataset", "llm"]):
            all_rows.extend(
                summarize_intervention_group(
                    group,
                    str(dataset),
                    str(llm),
                    original_action,
                    no_idk_action,
                )
            )
        for llm, group in four_way.groupby("llm"):
            all_rows.extend(
                summarize_intervention_group(
                    group,
                    "overall",
                    str(llm),
                    original_action,
                    no_idk_action,
                )
            )

    if not all_rows:
        raise ValueError(
            "No complete baseline/intervention/no-IDK action sets were found."
        )
    return pd.DataFrame(all_rows).sort_values(
        ["original_action", "dataset", "llm", "metric"],
        ignore_index=True,
    )


def build_no_idk_strategy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compare available no-IDK strategies on common groups within each model.

    A model is included when it has a no-IDK Base run and at least one no-IDK
    intervention. If all three strategies are available, their intersection is
    used. Thus, every strategy row for a model has the same denominator.
    Consistency values are percentages for direct use in a paper table.
    """
    required = {"action", *JOIN_KEYS, *PAPER_CONSISTENCY_METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Cannot build the no-IDK strategy table; missing columns: "
            f"{missing}"
        )

    scored = apply_table3_idk_policy(df)
    columns = [
        "llm",
        "strategy",
        "action",
        "n_common_question_groups",
        "datasets_in_common",
        *PAPER_METRIC_NAMES.values(),
    ]
    rows: list[dict] = []

    for llm in sorted(scored["llm"].dropna().astype(str).unique()):
        model_rows = scored.loc[scored["llm"].astype(str).eq(llm)]
        available = [
            (action, label)
            for action, label in NO_IDK_STRATEGIES
            if model_rows["action"].eq(action).any()
        ]
        available_actions = {action for action, _ in available}
        if "zero-shot-no-idk" not in available_actions or len(available) < 2:
            continue

        strategy_frames: list[pd.DataFrame] = []
        for action, _ in available:
            frame = model_rows.loc[
                model_rows["action"].eq(action),
                JOIN_KEYS + PAPER_CONSISTENCY_METRICS,
            ].copy()
            if frame.duplicated(JOIN_KEYS).any():
                raise ValueError(
                    f"Duplicate rows found for model {llm!r}, "
                    f"action {action!r}."
                )
            frame.rename(
                columns={
                    metric: f"{action}:{metric}"
                    for metric in PAPER_CONSISTENCY_METRICS
                },
                inplace=True,
            )
            strategy_frames.append(frame)

        common = strategy_frames[0]
        for frame in strategy_frames[1:]:
            common = common.merge(
                frame,
                on=JOIN_KEYS,
                how="inner",
                validate="one_to_one",
            )
        if common.empty:
            continue

        datasets = "|".join(
            sorted(common["dataset"].dropna().astype(str).unique())
        )
        for action, label in available:
            row = {
                "llm": llm,
                "strategy": label,
                "action": action,
                "n_common_question_groups": len(common),
                "datasets_in_common": datasets,
            }
            for metric, paper_name in PAPER_METRIC_NAMES.items():
                row[paper_name] = round(
                    100 * float(common[f"{action}:{metric}"].mean()),
                    2,
                )
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def build_cross_model_delta_table(report: pd.DataFrame) -> pd.DataFrame:
    """Average no-IDK minus IDK deltas across fully comparable models."""
    required = {
        "dataset",
        "llm",
        "original_action",
        "metric",
        "delta_no_idk_minus_original",
        "p_value_two_sided",
    }
    missing = sorted(required - set(report.columns))
    if missing:
        raise ValueError(
            "Cannot build the cross-model delta table; missing columns: "
            f"{missing}"
        )

    overall = report.loc[
        report["dataset"].eq("overall")
        & report["metric"].isin(PAPER_CONSISTENCY_METRICS)
        & report["original_action"].isin(ORIGINAL_STRATEGY_NAMES)
    ].copy()
    required_cells = {
        (action, metric)
        for action in ORIGINAL_STRATEGY_NAMES
        for metric in PAPER_CONSISTENCY_METRICS
    }
    complete_models = []
    for llm, group in overall.groupby("llm"):
        cells = set(zip(group["original_action"], group["metric"]))
        if required_cells <= cells:
            complete_models.append(str(llm))

    columns = [
        "strategy",
        "paper_metric",
        "n_models",
        "models",
        "mean_delta_pp",
        "min_delta_pp",
        "max_delta_pp",
        "n_positive",
        "n_significant_p_lt_0.05",
    ]
    if not complete_models:
        return pd.DataFrame(columns=columns)

    comparable = overall.loc[overall["llm"].isin(complete_models)].copy()
    rows: list[dict] = []
    for action, strategy in ORIGINAL_STRATEGY_NAMES.items():
        action_rows = comparable.loc[
            comparable["original_action"].eq(action)
        ]
        for metric in PAPER_CONSISTENCY_METRICS:
            values = action_rows.loc[action_rows["metric"].eq(metric)]
            deltas = 100 * values["delta_no_idk_minus_original"].astype(float)
            rows.append(
                {
                    "strategy": strategy,
                    "paper_metric": PAPER_METRIC_NAMES[metric],
                    "n_models": len(complete_models),
                    "models": "|".join(sorted(complete_models)),
                    "mean_delta_pp": round(float(deltas.mean()), 2),
                    "min_delta_pp": round(float(deltas.min()), 2),
                    "max_delta_pp": round(float(deltas.max()), 2),
                    "n_positive": int((deltas > 0).sum()),
                    "n_significant_p_lt_0.05": int(
                        (values["p_value_two_sided"] < 0.05).sum()
                    ),
                }
            )

        macro_deltas = (
            100
            * action_rows["delta_no_idk_minus_original"].astype(float)
        )
        rows.append(
            {
                "strategy": strategy,
                "paper_metric": "MACRO_AVERAGE",
                "n_models": len(complete_models),
                "models": "|".join(sorted(complete_models)),
                "mean_delta_pp": round(float(macro_deltas.mean()), 2),
                "min_delta_pp": round(float(macro_deltas.min()), 2),
                "max_delta_pp": round(float(macro_deltas.max()), 2),
                "n_positive": int((macro_deltas > 0).sum()),
                "n_significant_p_lt_0.05": int(
                    (action_rows["p_value_two_sided"] < 0.05).sum()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare original ASCB prompts with the no-IDK ablation."
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=REPO_ROOT / "output" / "analysis.csv",
        help="Analysis CSV produced by src/evaluation/run.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "no_idk_ablation.csv",
        help="Destination for the paired long-form report.",
    )
    parser.add_argument(
        "--effects-output",
        type=Path,
        default=REPO_ROOT / "output" / "no_idk_intervention_effects.csv",
        help="Destination for intervention gains within each prompt regime.",
    )
    parser.add_argument(
        "--strategy-table-output",
        type=Path,
        default=(
            REPO_ROOT
            / "output"
            / "no_idk_strategy_comparison_overall.csv"
        ),
        help=(
            "Destination for the compact matched Base/CtE/Oracle no-IDK "
            "comparison table."
        ),
    )
    parser.add_argument(
        "--cross-model-delta-output",
        type=Path,
        default=(
            REPO_ROOT
            / "output"
            / "no_idk_delta_average_across_models.csv"
        ),
        help=(
            "Destination for no-IDK minus IDK deltas averaged across models "
            "with all three strategy comparisons."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.analysis)
    report = build_ablation_report(df)
    effects = build_intervention_report(df)
    strategy_table = build_no_idk_strategy_table(df)
    cross_model_deltas = build_cross_model_delta_table(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    args.effects_output.parent.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.effects_output, index=False)
    args.strategy_table_output.parent.mkdir(parents=True, exist_ok=True)
    strategy_table.to_csv(args.strategy_table_output, index=False)
    args.cross_model_delta_output.parent.mkdir(parents=True, exist_ok=True)
    cross_model_deltas.to_csv(args.cross_model_delta_output, index=False)
    action_pairs = report[["original_action", "no_idk_action"]].drop_duplicates()
    print(f"Wrote {len(report):,} paired metric rows to {args.output}")
    print(
        f"Wrote {len(effects):,} intervention-effect rows to "
        f"{args.effects_output}"
    )
    print(
        f"Wrote {len(strategy_table):,} matched strategy rows to "
        f"{args.strategy_table_output}"
    )
    print(
        f"Wrote {len(cross_model_deltas):,} cross-model delta rows to "
        f"{args.cross_model_delta_output}"
    )
    print(action_pairs.to_string(index=False))


if __name__ == "__main__":
    main()

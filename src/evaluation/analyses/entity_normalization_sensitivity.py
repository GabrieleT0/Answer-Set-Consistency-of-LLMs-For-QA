"""Recompute Table 3 consistency after cached Wikidata alias normalization.

Uses the committed per-item analysis and the saved CtE/Oracle Q3-minus answers.
No network requests are made. Unmapped names retain their original spelling.
"""

from src.evaluation.paths import REPO_ROOT

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from src.evaluation.core.summary_idk_worst import METRIC_IDK_MAP, _add_idk_flags


ROOT = REPO_ROOT
RELATIONS = {
    "E1,2": "?A1=A2",
    "N3,1": "?A1>A3",
    "N4,1": "?A1>A4",
    "D3,4": "?A3∅A4",
    "E4,1\\3": "?A4=A1|3",
}
ADDITIONAL = {
    "E1,*": "?A1=A1*",
    "ESIM1,2": "J(A1-A2)",
    "DSIM3,4": "J(A3-A4)",
    "ESIM4,1\\3": "J(A4-A1|3)",
    "ESIM1,*": "J(A1-A1*)",
}
ACTIONS = {"zero-shot": "Base", "classification": "CtE", "fixing": "Oracle"}
SKIP = {"idk", "no answer", ""}


def parse_set(cell):
    if isinstance(cell, list):
        return set(str(s) for s in cell)
    if not isinstance(cell, str) or not cell.strip():
        return set()
    value = ast.literal_eval(cell)
    return set(str(s) for s in value) if isinstance(value, list) else set()


def load_extra_answers(rows):
    """Read the minus-task Q3 sets that analysis.csv does not preserve."""
    extra = {}
    missing = []
    keys = rows.loc[rows.action.ne("zero-shot"), ["dataset", "action", "llm"]].drop_duplicates()
    for dataset, action, llm in keys.itertuples(index=False, name=None):
        folder = "CtE" if action == "classification" else "Oracle"
        suffix = "classAndAnswer" if action == "classification" else "fixing"
        dataset_dir = "lc-quad" if dataset == "LC-QuAD" else dataset
        path = ROOT / "data" / "answers" / folder / dataset_dir / "minus" / f"Q3_minus_answers_{suffix}_{llm}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open(encoding="utf-8") as fh:
            extra[(dataset, action, llm)] = json.load(fh)
    if missing:
        raise FileNotFoundError("Missing Q3-minus files: " + ", ".join(missing))
    return extra


def load_star_answers(rows):
    star = {}
    keys = rows.loc[rows.action.eq("zero-shot") & rows["?A1=A1*"].notna(),
                    ["dataset", "llm"]].drop_duplicates()
    for dataset, llm in keys.itertuples(index=False, name=None):
        dataset_dir = "lc-quad" if dataset == "LC-QuAD" else dataset
        path = ROOT / "data" / "answers" / "zero-shot" / dataset_dir / "equal" / f"Q1_equal_answers_star_{llm}.json"
        with path.open(encoding="utf-8") as fh:
            star[(dataset, llm)] = json.load(fh)
    return star


def canonical_sets(sets, cache):
    """Merge names within a question row if they share a cached Wikidata ID."""
    strings = set().union(*sets.values())
    parent = {s: s for s in strings}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    owner = {}
    for s in sorted(strings):
        if s.strip().lower() in SKIP:
            continue
        for entity in cache.get(s, ()):
            if entity in owner:
                parent[find(s)] = find(owner[entity])
            else:
                owner[entity] = s
    return {key: {find(s) for s in values} for key, values in sets.items()}


def has_cross_set_alias(sets, cache):
    """Appendix D's predicate, keyed to a full item/strategy row."""
    owner = {}
    for index, key in enumerate(("A1", "A2", "A3", "A4")):
        for name in sets[key]:
            if name.strip().lower() in SKIP:
                continue
            for entity in cache.get(name, ()):
                for earlier_index, earlier_name in owner.get(entity, ()):
                    if earlier_index != index and earlier_name.lower() != name.lower():
                        return True
                owner.setdefault(entity, []).append((index, name))
    return False


def scores(sets, action):
    a1, a2, a3, a4 = (sets[x] for x in ("A1", "A2", "A3", "A4"))
    def jaccard(left, right):
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    if action == "zero-shot":
        scores = {
            "E1,2": int(a1 == a2),
            "N3,1": int(a3 <= a1),
            "N4,1": int(a4 <= a1),
            "D3,4": int(a3.isdisjoint(a4)),
            "E4,1\\3": int(a1 == a3 | a4 and a3.isdisjoint(a4)),
            "ESIM1,2": jaccard(a1, a2),
            "DSIM3,4": jaccard(a3, a4),
            "ESIM4,1\\3": jaccard(a4, a1 - a3),
        }
        if "A1_star" in sets:
            scores["E1,*"] = int(a1 == sets["A1_star"])
            scores["ESIM1,*"] = jaccard(a1, sets["A1_star"])
        return scores
    a1_contain, a1_minus, a3_minus = (sets[x] for x in ("A1*", "A1**", "A3_minus"))
    return {
        "E1,2": int(a1 == a2),
        "N3,1": int(a3 <= a1_contain),
        "N4,1": int(a4 <= a1_minus),
        "D3,4": int(a3_minus.isdisjoint(a4)),
        "E4,1\\3": int(a1_minus == a3_minus | a4 and a3_minus.isdisjoint(a4)),
        "ESIM1,2": jaccard(a1, a2),
        "DSIM3,4": jaccard(a3_minus, a4),
        "ESIM4,1\\3": jaccard(a4, a1_minus - a3_minus),
        "E1,*": int(a1 == a1_contain),
        "ESIM1,*": jaccard(a1, a1_contain),
    }


def run(analysis_path, cache_path, out_dir):
    df = pd.read_csv(analysis_path)
    df = df[df.action.isin(ACTIONS)].copy()
    df = _add_idk_flags(df)
    with cache_path.open(encoding="utf-8") as fh:
        cache = json.load(fh)
    extra = load_extra_answers(df)
    star = load_star_answers(df)
    records = []
    observed = set()
    baseline_mismatches = {relation: 0 for relation in RELATIONS}
    missing_q3 = 0

    for r in df.to_dict("records"):
        action = r["action"]
        sets = {key: parse_set(r[key]) for key in ("A1", "A2", "A3", "A4", "A1*", "A1**")}
        if action != "zero-shot":
            q3 = extra[(r["dataset"], action, r["llm"])].get(str(r["Q_ID"]))
            if q3 is None:
                missing_q3 += 1
                raise ValueError(f"Q3-minus missing for {r['dataset']}, {action}, {r['llm']}, {r['Q_ID']}")
            sets["A3_minus"] = parse_set(q3)
        elif pd.notna(r["?A1=A1*"]):
            star_answer = star[(r["dataset"], r["llm"])].get(str(r["Q_ID"]))
            if star_answer is None:
                raise ValueError(f"Q1-star missing for {r['dataset']}, {r['llm']}, {r['Q_ID']}")
            sets["A1_star"] = parse_set(star_answer)
        observed.update(set().union(*sets.values()))
        raw = scores(sets, action)
        normalized = scores(canonical_sets(sets, cache), action)
        rec = {key: r[key] for key in ("Q_ID", "dataset", "llm", "action")}
        rec["cross_set_alias"] = int(has_cross_set_alias(sets, cache))
        for relation, column in RELATIONS.items():
            if raw[relation] != r[column]:
                baseline_mismatches[relation] += 1
            # Anchor to the committed per-item score. A handful of saved raw
            # answers changed after analysis.csv was generated; adding only the
            # normalization effect preserves the published Table 3 baseline.
            idk = any(r[flag] for flag in METRIC_IDK_MAP[column])
            published = int(r[column])
            adjusted = published + normalized[relation] - raw[relation]
            rec[f"{relation}_raw"] = 0 if idk else published
            rec[f"{relation}_normalized"] = 0 if idk else max(0, min(1, adjusted))
        for metric, column in ADDITIONAL.items():
            if pd.isna(r[column]) or metric not in raw:
                rec[f"{metric}_raw"] = float("nan")
                rec[f"{metric}_normalized"] = float("nan")
                continue
            idk = any(r[flag] for flag in METRIC_IDK_MAP[column])
            worst = 1.0 if metric == "DSIM3,4" else 0.0
            published = float(r[column])
            adjusted = published + normalized[metric] - raw[metric]
            rec[f"{metric}_raw"] = worst if idk else published
            rec[f"{metric}_normalized"] = worst if idk else max(0.0, min(1.0, adjusted))
        records.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    items = pd.DataFrame(records)
    items.to_csv(out_dir / "per_item.csv", index=False)
    metrics = []
    for (dataset, action, llm), group in items.groupby(["dataset", "action", "llm"]):
        for relation in RELATIONS:
            metrics.append({"dataset": dataset, "action": action, "llm": llm,
                            "relation": relation, "n": len(group),
                            "raw": group[f"{relation}_raw"].mean(),
                            "normalized": group[f"{relation}_normalized"].mean()})
    for (action, llm), group in items.groupby(["action", "llm"]):
        for relation in RELATIONS:
            metrics.append({"dataset": "overall", "action": action, "llm": llm,
                            "relation": relation, "n": len(group),
                            "raw": group[f"{relation}_raw"].mean(),
                            "normalized": group[f"{relation}_normalized"].mean()})
    metrics = pd.DataFrame(metrics)
    metrics["delta_pp"] = (metrics.normalized - metrics.raw) * 100
    metrics.to_csv(out_dir / "per_relation.csv", index=False)
    additional = []
    for (dataset, action, llm), group in items.groupby(["dataset", "action", "llm"]):
        for metric in ADDITIONAL:
            valid = group[f"{metric}_raw"].notna()
            if valid.any():
                additional.append({"dataset": dataset, "action": action, "llm": llm,
                                   "metric": metric, "n": int(valid.sum()),
                                   "raw": group.loc[valid, f"{metric}_raw"].mean(),
                                   "normalized": group.loc[valid, f"{metric}_normalized"].mean()})
    for (action, llm), group in items.groupby(["action", "llm"]):
        for metric in ADDITIONAL:
            valid = group[f"{metric}_raw"].notna()
            if valid.any():
                additional.append({"dataset": "overall", "action": action, "llm": llm,
                                   "metric": metric, "n": int(valid.sum()),
                                   "raw": group.loc[valid, f"{metric}_raw"].mean(),
                                   "normalized": group.loc[valid, f"{metric}_normalized"].mean()})
    additional = pd.DataFrame(additional)
    additional["delta_pp"] = (additional.normalized - additional.raw) * 100
    additional.to_csv(out_dir / "additional_metrics.csv", index=False)
    # Table 3's Base control averages the zero-shot repeat-Q1 result and the
    # CtE equal-versus-containment result. The committed overall summary uses
    # that blend, so apply the paired normalization delta to its published value.
    controls = []
    published_summary = pd.read_csv(ROOT / "data/evaluation_results/summary_idk_worst.csv")
    for llm in sorted(items.llm.unique()):
        for metric in ("E1,*", "ESIM1,*"):
            column = ADDITIONAL[metric]
            part = additional[(additional.dataset == "overall") &
                              (additional.llm == llm) &
                              (additional.metric == metric) &
                              (additional.action.isin(["zero-shot", "classification"]))]
            if part.empty:
                continue
            base = published_summary[(published_summary.dataset == "overall") &
                                     (published_summary.action == "zero-shot") &
                                     (published_summary.llm == llm)].iloc[0][column]
            # For models without saved repeated-Q1 answers, the paper uses
            # the CtE cross-context result alone.
            delta = part.delta_pp.mean() / 100
            controls.append({"llm": llm, "metric": metric, "raw": base,
                             "normalized": max(0, min(1, base + delta)),
                             "delta_pp": delta * 100})
    pd.DataFrame(controls).to_csv(out_dir / "table3_control.csv", index=False)
    averages = (metrics.groupby(["dataset", "action", "llm"])[["raw", "normalized"]]
                .mean().reset_index())
    averages["delta_pp"] = (averages.normalized - averages.raw) * 100
    averages["raw_rank"] = averages.groupby(["dataset", "action"]).raw.rank(ascending=False, method="min").astype(int)
    averages["normalized_rank"] = averages.groupby(["dataset", "action"]).normalized.rank(ascending=False, method="min").astype(int)
    averages.to_csv(out_dir / "model_ranks.csv", index=False)
    wide = averages.pivot(index=["dataset", "llm"], columns="action", values=["raw", "normalized"])
    gains = []
    for (dataset, llm), values in wide.iterrows():
        if pd.isna(values.get(("raw", "zero-shot"))):
            continue
        for action in ("classification", "fixing"):
            if pd.isna(values.get(("raw", action))):
                continue
            raw_gain = 100 * (values[("raw", action)] - values[("raw", "zero-shot")])
            norm_gain = 100 * (values[("normalized", action)] - values[("normalized", "zero-shot")])
            gains.append({"dataset": dataset, "llm": llm, "intervention": ACTIONS[action],
                          "raw_gain_pp": raw_gain, "normalized_gain_pp": norm_gain,
                          "gain_change_pp": norm_gain - raw_gain})
    gains_df = pd.DataFrame(gains)
    gains_df.to_csv(out_dir / "intervention_gains.csv", index=False)
    alias_counts = (items.groupby("llm")
                    .agg(item_strategy_rows=("cross_set_alias", "size"),
                         affected_item_strategy_rows=("cross_set_alias", "sum"))
                    .reset_index())
    alias_counts["affected_rate_pct"] = (100 * alias_counts.affected_item_strategy_rows /
                                         alias_counts.item_strategy_rows)
    alias_counts.to_csv(out_dir / "alias_incidence.csv", index=False)
    table = alias_counts.copy()
    overall = averages[averages.dataset == "overall"]
    for action, label in ACTIONS.items():
        part = overall[overall.action == action][["llm", "raw", "normalized", "raw_rank", "normalized_rank"]]
        part = part.rename(columns={col: f"{label}_{col}" for col in part.columns if col != "llm"})
        table = table.merge(part, on="llm", validate="one_to_one")
    for label in ("CtE", "Oracle"):
        part = gains_df[(gains_df.dataset == "overall") & (gains_df.intervention == label)]
        table = table.merge(part[["llm", "raw_gain_pp", "normalized_gain_pp"]]
                            .rename(columns={"raw_gain_pp": f"{label}_raw_gain_pp",
                                             "normalized_gain_pp": f"{label}_normalized_gain_pp"}),
                            on="llm", validate="one_to_one")
    table = table.sort_values("Base_normalized_rank")
    table.to_csv(out_dir / "proposed_table5.csv", index=False)
    lines = ["# Proposed replacement for Appendix D, Table 5", "",
             "Alias affected is the number of observed question-group × strategy rows with at least one cross-set Wikidata alias, divided by the number of evaluated rows for that model. CON cells show original→normalized percentages. Gains are normalized percentage-point gains over Base.", "",
             "| Model | Alias affected | Base CON | CtE CON | Oracle CON | CtE gain | Oracle gain |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in table.to_dict("records"):
        alias = f"{int(r['affected_item_strategy_rows'])}/{int(r['item_strategy_rows'])} ({r['affected_rate_pct']:.1f}%)"
        def score(label):
            return f"{100*r[f'{label}_raw']:.2f}→{100*r[f'{label}_normalized']:.2f}"
        lines.append(f"| {r['llm']} | {alias} | {score('Base')} | {score('CtE')} | {score('Oracle')} | "
                     f"{r['CtE_normalized_gain_pp']:+.2f} | {r['Oracle_normalized_gain_pp']:+.2f} |")
    (out_dir / "proposed_table5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {"source_analysis": str(analysis_path), "source_cache": str(cache_path),
                "n_rows": len(items), "observed_unique_names": len(observed),
                "cached_names": sum(s in cache for s in observed),
                "mapped_names": sum(bool(cache.get(s)) for s in observed),
                "baseline_mismatches_before_idk": baseline_mismatches,
                "missing_q3": missing_q3}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, default=ROOT / "data/evaluation_results/analysis.csv")
    p.add_argument("--cache", type=Path, default=ROOT / "data/wikidata_cache.json")
    p.add_argument("--output", type=Path, default=ROOT / "output/entity_normalization")
    args = p.parse_args()
    run(args.analysis, args.cache, args.output)

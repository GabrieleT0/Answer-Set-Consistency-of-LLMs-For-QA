import re
import math
import unicodedata
import pandas as pd
from typing import Any, Iterable
import os
import json

CANONICAL_LABELS = [
    "Equivalence", "Contains", "ContainedBy", "Overlap", "Disjoint", "Unknown", "Else"
]

# --- helpers ---------------------------------------------------------------

_UNKNOWN_TOKENS = {
    "unknown","unk","n/a","na","none","null","nil","idk","don't know","dont know",
    "cannot determine","can’t determine","cant determine","unsure","uncertain",
    "not sure","not given","not specified","ambiguous"
}

def _first_nonempty_str(it: Iterable[Any]) -> str | None:
    for x in it:
        if x is None: 
            continue
        s = str(x).strip()
        if s:
            return s
    return None

def _pick_from_dict(d: dict) -> str | None:
    # common shapes: {"label": "..."} or {"relation": "..."} or {label: score, ...}
    for k in ("label","relation","pred","class"):
        if k in d and isinstance(d[k], (str, int, float)):
            return str(d[k])
    # try best-score key if numeric
    try:
        numeric = {k: float(v) for k, v in d.items() if isinstance(v, (int, float, str)) and str(v).replace('.','',1).lstrip('-').isdigit()}
        if numeric:
            return max(numeric, key=numeric.get)
    except Exception:
        pass
    # else first key
    if d:
        return str(next(iter(d.keys())))
    return None

def _clean_text(s: str) -> str:
    # Unicode normalize (e.g., different hyphens, spaces)
    s = unicodedata.normalize("NFKC", s)
    # drop parenthetical scores etc: "Equivalence (0.91)" -> "Equivalence"
    s = re.sub(r"\(.*?\)", "", s)
    # collapse spaces & hyphens around keywords
    s = re.sub(r"[-_]+", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _is_unknown_like(s_lower: str) -> bool:
    return s_lower in _UNKNOWN_TOKENS

# --- main normalizer -------------------------------------------------------

def normalize_relation(pred: Any) -> str:
    """
    Map various prediction shapes/phrases/symbols into canonical relation labels.
    Returns one of CANONICAL_LABELS.
    """
    # None / NaN / empty -> Unknown
    if pred is None or (isinstance(pred, float) and math.isnan(pred)):
        return "Unknown"

    # list/tuple: pick first sensible string
    if isinstance(pred, (list, tuple)):
        s = _first_nonempty_str(pred)
        if not s:
            return "Unknown"
        pred = s

    # dict: try to extract label
    if isinstance(pred, dict):
        s = _pick_from_dict(pred)
        if not s:
            return "Unknown"
        pred = s

    # now string
    s_raw = str(pred)
    s = _clean_text(s_raw)
    s_lower = s.casefold()

    # quick exact canonical pass
    if s in CANONICAL_LABELS:
        return s

    # unknown-likes
    if _is_unknown_like(s_lower):
        return "Unknown"

    # guard: special negation patterns first (so "not disjoint" -> Overlap)
    if re.search(r"\bnot\s+disjoint\b", s_lower) or re.search(r"\bnon[-\s]?disjoint\b", s_lower):
        return "Overlap"
    if re.search(r"\bnot\s+overlap(ping)?\b", s_lower):
        return "Disjoint"

    # --- detect by symbols/phrases ---
    # Equivalence
    if re.search(r"\beq(uiv(alent|alence)?)\b", s_lower) or \
       re.search(r"\bequal(s)?\b", s_lower) or \
       re.search(r"\bsame(\s+set)?\b", s_lower) or \
       re.search(r"a\s*=\s*b", s_lower) or "≡" in s or "↔" in s:
        return "Equivalence"

    # Contains (A ⊃ B; superset; includes)
    if "⊃" in s or "⊇" in s or \
       re.search(r"\bsuper\s*set\b", s_lower) or \
       re.search(r"\bsuperset\s+of\b", s_lower) or \
       re.search(r"\bcontain(s|ment)?\b", s_lower) or \
       re.search(r"\bincludes?\b", s_lower) or \
       re.search(r"\b(a|set\s*a)?\s*includes?\s*(b|set\s*b)\b", s_lower):
        return "Contains"

    # ContainedBy (A ⊂ B; subset; contained by; is in)
    if "⊂" in s or "⊆" in s or \
       re.search(r"\bsub\s*set\b", s_lower) or \
       re.search(r"\bsubset\s+of\b", s_lower) or \
       re.search(r"\bcontained\s*by\b", s_lower) or \
       re.search(r"\bis\s+in\b", s_lower) or \
       re.search(r"\bbelongs\s+to\b", s_lower):
        return "ContainedBy"

    # Disjoint (A ∩ B = ∅; no overlap)
    if "∩" in s and ("∅" in s or "= 0" in s_lower) or \
       re.search(r"\bdis[-\s]?joint\b", s_lower) or \
       re.search(r"\bno\s+(overlap|intersection)\b", s_lower) or \
       re.search(r"\bmutual(ly)?\s+exclusive\b", s_lower) or \
       re.search(r"\bnon[-\s]?overlap(ping)?\b", s_lower):
        return "Disjoint"

    # Overlap (A ∩ B ≠ ∅; intersect; partial overlap)
    if "∩" in s and ("≠" in s or "!= " in s_lower) or \
       re.search(r"\boverlap(ping)?\b", s_lower) or \
       re.search(r"\bintersect(s|ion)?\b", s_lower) or \
       re.search(r"\b(non[-\s]?empty|some)\s+intersection\b", s_lower) or \
       re.search(r"\bshare(s)?\s+(elements|items|members)\b", s_lower):
        return "Overlap"

    # If the string literally says "unknown" in any decorative way, catch it late too
    if "unknown" in s_lower:
        return "Unknown"

    # Otherwise:
    return "Else"

# --- convenience wrappers ---------------------------------------------------

def normalize_relation_series(s: pd.Series) -> pd.Series:
    return s.apply(normalize_relation)

def normalize_relation_cols(df: pd.DataFrame, cols: list[str], inplace: bool = False, suffix: str = "_norm") -> pd.DataFrame:
    """
    Normalize multiple relation columns in a DataFrame.
    - If inplace=False, returns a copy with new normalized columns appended (col+suffix).
    - If inplace=True, overwrites the original columns.
    """
    target = df if inplace else df.copy()
    for c in cols:
        norm = target[c].apply(normalize_relation)
        if inplace:
            target[c] = norm
        else:
            target[c + suffix] = norm
    return target



#####Relation Classification #####
def load_relations(root_dir, datasets, llms):
    """
        DataFrame with columns: ["Q_ID", "dataset", "llm", "R(1-2)", "R(1-3)", "R(1-4)", "R(3-4)", "R(1-34)"]
    """
    # find JSON files
    json_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(root_dir)
        for file in files
        if file.startswith("Relation") and file.endswith(".json")
    ]
    print(f"JSON files found: {len(json_files)}")

    # initialize dataframe
    df_relation = pd.DataFrame(
        columns=["Q_ID", "dataset", "llm", "R(1-2)", "R(1-3)", "R(1-4)", "R(3-4)", "R(1-34)"]
    )

    for file in json_files:
        elements = file.replace("_", "/").replace(".json", "").split("/")
        dataset = next((d for d in datasets if d in elements), None)
        llm = next((l for l in llms if l in elements), None)

        if all([dataset, llm]):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # transform dict into rows
            rows = [
                {
                    "dataset": dataset,
                    "llm": llm,
                    "Q_ID": key,
                    "R(1-2)": value[0],
                    "R(1-3)": value[1],
                    "R(1-4)": value[2],
                    "R(3-4)": value[3],
                    "R(1-34)": value[4],
                }
                for key, value in data.items()
            ]
            df_relation = pd.concat([df_relation, pd.DataFrame(rows)], ignore_index=True)

    return df_relation


CANONICAL_LABELS = [
    "Equivalence", "Contains", "ContainedBy", "Overlap", "Disjoint", "Unknown", "Else"
]

GT = {
    "R(1-2)":  "Equivalence",
    "R(1-3)":  "Contains",
    "R(1-4)":  "Contains",
    "R(3-4)":  "Disjoint",
    "R(1-34)": "Equivalence",
}


def _normalize_pred(x: object) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    if s in CANONICAL_LABELS:
        return s
    return "Else"

def per_model_confusions(
    df_relation: pd.DataFrame,
    relation_cols=None,
    include_overall: bool = True,
    round_digits: int = 4,
):
    """
    Build complete confusion matrices per (llm, dataset) and per relation column.
    Adds an 'overall' dataset per llm if include_overall=True.
    
    Returns
    -------
    cms_counts : pd.DataFrame
        MultiIndex rows: (llm, dataset, relation, True)
        Columns: Equivalence, Contains, ContainedBy, Overlap, Disjoint, Unknown, Else (counts)
    cms_ratio : pd.DataFrame
        Same shape, row-normalized ratios.
    """
    if relation_cols is None:
        relation_cols = list(GT.keys())

    needed = {"dataset", "llm", *relation_cols}
    missing = needed - set(df_relation.columns)
    if missing:
        raise ValueError(f"df_relation missing columns: {missing}")

    rows_counts, rows_ratio, idx = [], [], []

    # 1) Per (llm, dataset)
    for (llm, dataset), group in df_relation.groupby(["llm", "dataset"], dropna=False):
        n_group = len(group)
        for rel in relation_cols:
            truth = GT[rel]
            y_pred = group[rel].map(_normalize_pred)
            counts = y_pred.value_counts()
            row_counts = [int(counts.get(lbl, 0)) for lbl in CANONICAL_LABELS]
            row_ratio = [(c / n_group) if n_group > 0 else 0.0 for c in row_counts]
            rows_counts.append(row_counts)
            rows_ratio.append(row_ratio)
            idx.append((llm, dataset, rel, truth))

    # 2) Per llm (overall across datasets)
    if include_overall:
        for llm, group in df_relation.groupby("llm", dropna=False):
            n_group = len(group)
            for rel in relation_cols:
                truth = GT[rel]
                y_pred = group[rel].map(_normalize_pred)
                counts = y_pred.value_counts()
                row_counts = [int(counts.get(lbl, 0)) for lbl in CANONICAL_LABELS]
                row_ratio = [(c / n_group) if n_group > 0 else 0.0 for c in row_counts]
                rows_counts.append(row_counts)
                rows_ratio.append(row_ratio)
                idx.append((llm, "overall", rel, truth))

    index = pd.MultiIndex.from_tuples(idx, names=["llm", "dataset", "relation", "True"])
    cms_counts = pd.DataFrame(rows_counts, index=index, columns=CANONICAL_LABELS)
    cms_ratio  = pd.DataFrame(rows_ratio,  index=index, columns=CANONICAL_LABELS)
    if round_digits is not None:
        cms_ratio = cms_ratio.round(round_digits)

    return cms_counts, cms_ratio


def build_confusion_table(cms_counts: pd.DataFrame,
                          cms_ratio: pd.DataFrame,
                          round_digits: int = 4) -> pd.DataFrame:
    """
    Create one tidy table:
      llm | dataset | relation | True | Accuracy | Size | Equivalence | Contains | ... | Else
    where each label column is 'ratio(count)' and Accuracy is for the True label as 'ratio(count)'.
    """
    records = []
    for idx in cms_counts.index:
        llm, dataset, relation, true_label = idx
        counts_row = cms_counts.loc[idx]
        ratio_row  = cms_ratio.loc[idx]

        N = int(counts_row.sum())
        acc_ratio = float(ratio_row.get(true_label, 0.0))
        acc_count = int(counts_row.get(true_label, 0))

        row = {
            "llm": llm,
            "dataset": dataset,
            "relation": relation,
            "True": true_label,
            "Accuracy": f"{acc_ratio:.{round_digits}f}({acc_count})",
            "Size": N,
        }

        # Add each predicted label as ratio(count)
        for lbl in CANONICAL_LABELS:
            r = float(ratio_row.get(lbl, 0.0))
            c = int(counts_row.get(lbl, 0))
            row[lbl] = f"{r:.{round_digits}f}({c})"

        records.append(row)

    out = pd.DataFrame.from_records(records)
    # nice ordering
    cols = ["llm", "dataset", "relation", "True", "Accuracy", "Size"] + CANONICAL_LABELS
    return out[cols]


def relation_summary(df_relation: pd.DataFrame,
                            relation_cols=None,
                            include_overall: bool = True,
                            round_digits: int = 4) -> pd.DataFrame:
    """
    Convenience wrapper: calls your per_model_confusions(...) then builds the table.
    """
    # uses the per_model_confusions you already have
    cms_counts, cms_ratio = per_model_confusions(
        df_relation,
        relation_cols=relation_cols,
        include_overall=include_overall,
        round_digits=round_digits,
    )
    return build_confusion_table(cms_counts, cms_ratio, round_digits)

def load_relation_clf(root_dir, datasets, llms, tasks) -> pd.DataFrame:
    # Resolve default roo
    task_to_col = {
    "equal":   "R(1-2)",
    "sup-sub": "R(1-3)",
    "minus":   "R(1-34)",}

    # Find JSON files (exclude those starting with 'Q')
    json_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(root_dir)
        for file in files
        if not file.startswith("Q") and file.endswith(".json")
    ]
    rows_map: dict[tuple[str, str, str], dict] = {}

    for file in json_files:
        parts = file.replace("_", "/").replace(".json", "").split("/")
        dataset = next((d for d in datasets if d in parts), None)
        llm     = next((l for l in llms     if l in parts), None)
        task    = next((t for t in tasks    if t in parts), None)
        col     = task_to_col.get(task)

        if not (dataset and llm and col):
            continue

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for qid, rel in data.items():
            # allow value to be list/tuple or scalar
            pred = rel[0] if isinstance(rel, (list, tuple)) and len(rel) else rel
            
            key = (qid, dataset, llm)
            row = rows_map.setdefault(key, {"Q_ID": qid, "dataset": dataset, "llm": llm})
            row[col] = normalize_relation(pred)

    # Materialize dataframe
    df = pd.DataFrame(rows_map.values())
    for c in ["R(1-2)", "R(1-3)", "R(1-34)"]:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[["Q_ID", "dataset", "llm", "R(1-2)", "R(1-3)", "R(1-34)"]]
    df["action"] = "classification"
    return df.sort_values(["llm", "dataset", "Q_ID"]).reset_index(drop=True)


def merge_relations_by_action(df_analysis, df_relation, df_relation_clf):
    keys = ["Q_ID", "dataset", "llm"]
    rel_cols = ["R(1-2)", "R(1-3)", "R(1-4)", "R(3-4)", "R(1-34)"]

    # Ensure target columns exist in df_analysis
    for c in rel_cols:
        if c not in df_analysis.columns:
            df_analysis[c] = pd.NA

    # Deduplicate right tables on keys
    df_rel = df_relation[keys + rel_cols].drop_duplicates(subset=keys)

    # df_relation_clf may have only a subset of rel_cols; align to full set
    clf_cols = [c for c in rel_cols if c in df_relation_clf.columns]
    df_rel_clf_aligned = (
        df_relation_clf[keys + clf_cols]
        .drop_duplicates(subset=keys)
        .reindex(columns=keys + rel_cols)  # add missing relation cols as NaN
    )

    # Fill zero-shot rows from df_relation
    m_zs = df_analysis["action"].eq("zero-shot")
    if m_zs.any():
        zs_merge = df_analysis.loc[m_zs, keys].merge(df_rel, on=keys, how="left")
        df_analysis.loc[m_zs, rel_cols] = zs_merge[rel_cols].values

    # Fill classification rows from df_relation_clf
    m_cls = df_analysis["action"].eq("classification")
    if m_cls.any():
        cls_merge = df_analysis.loc[m_cls, keys].merge(df_rel_clf_aligned, on=keys, how="left")
        df_analysis.loc[m_cls, rel_cols] = cls_merge[rel_cols].values
    df_analysis =  df_analysis.replace({None: pd.NA}).convert_dtypes()
    return df_analysis

def update_summary_by_relations(
    df_analysis: pd.DataFrame,
    df_summary: pd.DataFrame,
    task: str = "zero-shot",
    # Ground truth labels per relation
    relation_truths: dict[str, str] = None,
    # Metrics to average per relation: {relation: [(metric_column_name, output_label_prefix or None), ...]}
    # If output_label_prefix is None, we use the metric column name directly and append (+)/(-).
    metric_spec: dict[str, list[tuple[str, str | None]]] = None,
) -> pd.DataFrame:
    """
    For each (dataset, llm) and 'overall' per llm, compute mean metrics on
    positive vs negative rows for each relation, and write into df_summary.
    
    Returns
    -------
    df_summary : pd.DataFrame (mutated copy)
    """
    # Defaults
    if relation_truths is None:
        relation_truths = {
            "R(1-2)":  "Equivalence",
            "R(1-3)":  "Contains",
            "R(1-4)":  "Contains",
            "R(3-4)":  "Disjoint",
            "R(1-34)": "Equivalence",
        }

    if metric_spec is None:
        metric_spec = {
            "R(1-2)":  [("?A1=A2", None), ("J(A1-A2)", "J(1-2)")],
            "R(1-3)":  [("?A1>A3", None)],
            "R(1-4)":  [("?A1>A4", None), ("J(A1-A4)", "J(1-4)")],
            "R(3-4)":  [("?A3∅A4",None),("J(A3-A4)", "J(3-4)")],  # natural for Disjoint: J should be near 0
            "R(1-34)": [("?A1=A3+A4", None), ("J(A1-A34)", "J(1-34)")],
        }

    # Work on a copy to avoid accidental view issues
    out = df_summary.copy()
    df_temp = df_analysis[df_analysis["action"] == task]

    def set_means(mask, group, rel_col, truth_label):
        # Split pos/neg
        pos = group[group[rel_col] == truth_label]
        neg = group[group[rel_col] != truth_label]

        for metric_col, prefix in metric_spec.get(rel_col, []):
            if metric_col not in group.columns:
                # silently skip missing metrics
                continue

            # Compute means (NaN if empty)
            pos_mean = pos[metric_col].mean() if len(pos) else pd.NA
            neg_mean = neg[metric_col].mean() if len(neg) else pd.NA

            # Build output column names
            if prefix is None:
                col_pos = f"{metric_col}(+)"
                col_neg = f"{metric_col}(-)"
            else:
                col_pos = f"{prefix}+"
                col_neg = f"{prefix}-"

            # Ensure columns exist
            if col_pos not in out.columns:
                out[col_pos] = pd.NA
            if col_neg not in out.columns:
                out[col_neg] = pd.NA

            # Assign
            out.loc[mask, col_pos] = pos_mean
            out.loc[mask, col_neg] = neg_mean

    # Per (dataset, llm)
    for (dataset, llm), group in df_temp.groupby(["dataset", "llm"]):
        mask_common = (
            (out["action"] == task)
            & (out["dataset"] == dataset)
            & (out["llm"] == llm)
        )
        for rel_col, truth_label in relation_truths.items():
            # Skip relation columns that aren't present
            if rel_col not in group.columns:
                continue
            set_means(mask_common, group, rel_col, truth_label)

    # Overall per llm
    for llm, group in df_temp.groupby("llm"):
        mask_overall = ((out["dataset"] == "overall") 
                        & (out["llm"] == llm)
                        & (out["action"] == task))
    
        for rel_col, truth_label in relation_truths.items():
            if rel_col not in group.columns:
                continue
            set_means(mask_overall, group, rel_col, truth_label)

    return out


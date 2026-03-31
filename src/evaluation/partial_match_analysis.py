"""
Partial-match analysis across answer sets using Wikidata as an alias lexicon.

For each pair of answer sets (A1, A2, A3, A4) for the same question, detects
non-exact but semantically equivalent string pairs by querying the Wikidata
SPARQL endpoint:

  Two answer strings a and b are considered a match if they share at least one
  Wikidata entity that lists both strings as rdfs:label or skos:altLabel.
  E.g. "Spain" and "Kingdom of Spain" both appear as labels for Q29 → match.

Outputs (in output/):
  partial_match_cases.csv      — every detected case, for manual inspection
  partial_match_summary.csv    — counts per (llm) across all 4 datasets
  partial_match_by_dataset.csv — counts per (llm, dataset)
"""

import ast
import json
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
SPARQL_BATCH    = 50   
REQUEST_DELAY   = 1.0   
MAX_RETRIES     = 3     
MAX_STR_LEN     = 300 

_HEADERS = {
    "Accept":       "application/sparql-results+json",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent":   "ISWS-RP-partial-match/1.0 (research; contact via GitHub)",
}

SET_PAIRS = [
    ("A1", "A2"),
    ("A1", "A3"),
    ("A1", "A4"),
    ("A2", "A3"),
    ("A2", "A4"),
    ("A3", "A4"),
]

_SKIP = {"idk", "no answer", ""}


# Answer parsing 

def parse_answers(raw) -> list[str]:
    """Parse a stringified list from analysis.csv into a list of strings."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, list):
            return [str(x) for x in result if str(x).strip().lower() not in _SKIP]
        return []
    except Exception:
        return []


# Wikidata SPARQL helpers

def _sparql_escape(s: str) -> str:
    """Escape a string for embedding inside a SPARQL string literal."""
    return (
        s.replace("\\", "\\\\")
         .replace('"',  '\\"')
         .replace("\n", "\\n")
         .replace("\r", "\\r")
         .replace("\t", "\\t")
    )


def _is_queryable(s: str) -> bool:
    """
    Return True if `s` looks like a real entity name worth querying.
    Filters out multi-sentence LLM completions and other garbage.
    """
    if len(s) > MAX_STR_LEN:
        return False
    if "\n" in s:
        return False
    return True


def _fetch_entity_batch(strings: list[str], lang: str) -> dict[str, set[str]]:
    """
    Query Wikidata for one batch of strings via POST (avoids URL-length limits).
    Returns {string: {entity_uri, …}} for strings that matched.
    Retries on transient HTTP errors (429, 502, 503).
    """
    values = " ".join(f'"{_sparql_escape(s)}"@{lang}' for s in strings)
    query = (
        "SELECT DISTINCT ?s ?l WHERE {\n"
        "  ?s rdfs:label|skos:altLabel ?l .\n"
        f"  VALUES ?l {{ {values} }}\n"
        "}"
    )
    result: dict[str, set[str]] = {s: set() for s in strings}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                WIKIDATA_SPARQL,
                data={"query": query},
                headers=_HEADERS,
                timeout=60,
            )
            resp.raise_for_status()
            for binding in resp.json()["results"]["bindings"]:
                entity = binding["s"]["value"]
                label  = binding["l"]["value"]
                if label in result:
                    result[label].add(entity)
            return result
        except requests.RequestException as exc:
            status = getattr(exc.response, "status_code", None) if hasattr(exc, "response") else None
            transient = status in (429, 502, 503) or status is None
            if transient and attempt < MAX_RETRIES:
                wait = REQUEST_DELAY * 2 ** attempt
                print(f"  [wikidata] Attempt {attempt} failed ({status}), retrying in {wait:.0f}s …")
                time.sleep(wait)
            else:
                print(f"  [wikidata] Warning: SPARQL request failed after {attempt} attempt(s) — {exc}")
                break
    return result


# ── Wikidata cache ───────────────────────────────────────────────────────────

class WikidataCache:
    """
    Persistent local cache for Wikidata entity lookups.
    Maps answer string -> list of Wikidata entity URIs.
    """

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        # Store as list[str] in JSON; convert to set[str] on access.
        self._cache: dict[str, list[str]] = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            print(f"  [wikidata] Loaded {len(self._cache):,} cached lookups "
                  f"from {cache_path}")

    def lookup(self, strings: list[str], lang: str = "en") -> dict[str, set[str]]:
        """
        Return {string: set_of_entity_uris} for every string in `strings`.
        Fetches uncached strings from Wikidata in batches and saves the cache.
        """
        missing = [s for s in strings if s not in self._cache]
        # Strings that fail the queryable check are cached immediately as empty
        # so they are never sent to Wikidata.
        unqueryable = [s for s in missing if not _is_queryable(s)]
        if unqueryable:
            print(f"  [wikidata] Skipping {len(unqueryable):,} strings that are "
                  "too long or contain newlines (LLM artifacts).")
            for s in unqueryable:
                self._cache[s] = []
        missing = [s for s in missing if _is_queryable(s)]
        if missing:
            print(f"  [wikidata] Querying {len(missing):,} uncached strings "
                  f"({-(-len(missing)//SPARQL_BATCH)} batches) …")
            for i in range(0, len(missing), SPARQL_BATCH):
                batch   = missing[i: i + SPARQL_BATCH]
                batch_n = i // SPARQL_BATCH + 1
                fetched = _fetch_entity_batch(batch, lang)
                for s, ents in fetched.items():
                    self._cache[s] = sorted(ents)
                if batch_n % 20 == 0:
                    print(f"    {i + len(batch):,} / {len(missing):,}")
                    self.save()
                time.sleep(REQUEST_DELAY)
            self.save()
        return {s: set(self._cache.get(s, [])) for s in strings}

    def save(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)


# Detection

def find_partial_matches(
    df: pd.DataFrame,
    cache: WikidataCache,
    lang: str = "en",
) -> pd.DataFrame:
    """
    For every row and every cross-set pair (A1×A2, …), detect answer strings
    that are non-identical but refer to the same Wikidata entity.
    """
    df = df[df["action"] != "wikidata"].copy()

    # Gather all unique, non-trivial answer strings
    all_strings: set[str] = set()
    for _, row in df.iterrows():
        for col in ("A1", "A2", "A3", "A4"):
            for s in parse_answers(row[col]):
                if s.strip().lower() not in _SKIP:
                    all_strings.add(s)

    unique_strings = sorted(all_strings)
    print(f"  Collected {len(unique_strings):,} unique answer strings.")

    # Wikidata lookup 
    string_to_entities = cache.lookup(unique_strings, lang=lang)

    n_mapped = sum(1 for v in string_to_entities.values() if v)
    print(f"  {n_mapped:,} / {len(unique_strings):,} strings matched "
          "≥1 Wikidata entity.")

    # Detect cross-set alias pairs 
    cases = []
    for _, row in df.iterrows():
        answers = {col: parse_answers(row[col]) for col in ("A1", "A2", "A3", "A4")}
        for left_col, right_col in SET_PAIRS:
            for a in answers[left_col]:
                if a.strip().lower() in _SKIP:
                    continue
                ents_a = string_to_entities.get(a, set())
                if not ents_a:
                    continue
                for b in answers[right_col]:
                    if b.strip().lower() in _SKIP:
                        continue
                    if a.strip().lower() == b.strip().lower():
                        continue  # exact match — skip
                    ents_b = string_to_entities.get(b, set())
                    shared = ents_a & ents_b
                    if shared:
                        cases.append({
                            "Q_ID":            row["Q_ID"],
                            "action":          row["action"],
                            "dataset":         row["dataset"],
                            "llm":             row["llm"],
                            "set_pair":        f"{left_col}×{right_col}",
                            "match_type":      "wikidata_alias",
                            "a_string":        a,
                            "b_string":        b,
                            "shared_entities": "|".join(sorted(shared)),
                        })
    return pd.DataFrame(cases)


# Summaries

def question_level_count(df_cases: pd.DataFrame,
                         group_cols: list[str]) -> pd.DataFrame:
    """
    Count distinct (Q_ID, set_pair) occurrences per group —
    i.e. how many questions have ≥1 Wikidata alias match for each set pair.
    """
    deduped = df_cases.drop_duplicates(subset=group_cols + ["Q_ID", "set_pair"])
    return (
        deduped.groupby(group_cols + ["set_pair"])
        .size()
        .reset_index(name="n_questions_with_partial_match")
    )


def pair_level_count(df_cases: pd.DataFrame,
                     group_cols: list[str]) -> pd.DataFrame:
    """Raw count of all (a_string, b_string) alias pairs."""
    return (
        df_cases.groupby(group_cols + ["set_pair"])
        .size()
        .reset_index(name="n_partial_match_pairs")
    )


def build_summary(df_cases: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    q_counts = question_level_count(df_cases, group_cols)
    p_counts  = pair_level_count(df_cases, group_cols)
    return q_counts.merge(p_counts, on=group_cols + ["set_pair"], how="outer")


def alias_cases_per_model(df_cases: pd.DataFrame) -> pd.DataFrame:
    """
    For each model (llm), count — aggregated across all datasets and set_pairs:
    """
    if df_cases.empty:
        return pd.DataFrame(columns=["llm", "n_questions", "n_alias_pairs", "n_unique_pairs"])

    raw = (
        df_cases.groupby("llm")
        .agg(
            n_questions  =("Q_ID",     "nunique"),
            n_alias_pairs=("a_string", "count"),
        )
        .reset_index()
    )
    unique_pairs = (
        df_cases.drop_duplicates(subset=["llm", "a_string", "b_string"])
        .groupby("llm")
        .size()
        .reset_index(name="n_unique_pairs")
    )

    return raw.merge(unique_pairs, on="llm", how="left").sort_values(
        "n_questions", ascending=False
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Detect alias-based partial matches using Wikidata."
    )
    parser.add_argument(
        "--lang", default="en",
        help="Language tag for Wikidata label lookup (default: en).",
    )
    parser.add_argument(
        "--wikidata-cache", default=None,
        help="Path to Wikidata lookup cache JSON "
             "(default: output/wikidata_cache.json).",
    )
    args = parser.parse_args()

    root_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    analysis_path = os.path.join(root_dir, "output", "analysis.csv")
    output_dir    = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    cache_path = args.wikidata_cache or os.path.join(output_dir, "wikidata_cache.json")

    print(f"Loading {analysis_path} …")
    df = pd.read_csv(analysis_path)
    print(f"  {len(df):,} rows, actions: {sorted(df['action'].unique())}")

    cache = WikidataCache(cache_path)

    print("Scanning for Wikidata alias matches …")
    df_cases = find_partial_matches(df, cache, lang=args.lang)
    print(f"  {len(df_cases):,} alias pairs found "
          f"across {df_cases['Q_ID'].nunique():,} unique Q_IDs")

    # For manual inspection
    cases_path = os.path.join(output_dir, "partial_match_cases.csv")
    df_cases.sort_values(["dataset", "llm", "Q_ID", "set_pair"]).to_csv(
        cases_path, index=False
    )
    print(f"  Cases saved → {cases_path}")

    # Summary by (llm, set_pair) across all datasets
    df_summary = build_summary(df_cases, group_cols=["llm"])
    df_summary.to_csv(os.path.join(output_dir, "partial_match_summary.csv"), index=False)

    # Summary by (dataset, llm, set_pair) ──────────────────────────────
    df_by_dataset = build_summary(df_cases, group_cols=["dataset", "llm"])
    df_by_dataset.to_csv(
        os.path.join(output_dir, "partial_match_by_dataset.csv"), index=False
    )

    # Per-model alias case counts (main result)
    df_per_model = alias_cases_per_model(df_cases)
    per_model_path = os.path.join(output_dir, "partial_match_cnt_summary.csv")
    df_per_model.to_csv(per_model_path, index=False)

    print("\n=== Alias cases per model (all datasets, all set pairs) ===")
    print(df_per_model.to_string(index=False))
    print(f"\n  Saved → {per_model_path}")

    print("\n=== Breakdown by set pair (all LLMs, all datasets) ===")
    sp_totals = (
        df_cases.groupby("set_pair")
        .agg(
            n_alias_pairs=("a_string", "count"),
            n_questions_with_match=("Q_ID", "nunique"),
        )
        .sort_values("n_alias_pairs", ascending=False)
        .reset_index()
    )
    print(sp_totals.to_string(index=False))

    print(f"\nAll CSVs saved to {output_dir}/")


if __name__ == "__main__":
    main()

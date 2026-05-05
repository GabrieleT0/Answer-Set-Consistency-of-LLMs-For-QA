#!/usr/bin/env python3
"""
Convert benchmark questions to JSON format for the GitHub Pages web interface.

By default this script reads ASCB TSV files from data/ASCB/ and outputs the
format expected by docs/index.html.

"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASCB_DIR = REPO_ROOT / "data" / "ASCB"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "questions.json"
QUESTION_COLUMNS = ("Q1", "Q2", "Q3", "Q4")


def ascb_to_json(
    input_dir: str | Path = DEFAULT_ASCB_DIR,
    output_path: str | Path = DEFAULT_OUTPUT,
    language: str = "en",
    question_columns: tuple[str, ...] = QUESTION_COLUMNS,
) -> List[Dict[str, Any]]:
    """
    Convert ASCB benchmark TSV files to docs/questions.json.

    Expected input files:
        data/ASCB/LC-QuAD.tsv
        data/ASCB/qawiki.tsv
        data/ASCB/spinach.tsv
        data/ASCB/synthetic.tsv

    Expected TSV columns:
        ID, Q1, Q2, Q3, Q4

    The ASCB TSV files contain questions, not gold answers. For compatibility
    with docs/index.html, the generated records include an empty `answers` list.
    Extra metadata fields (`group_id`, `source_id`, `question_type`) are kept in
    the JSON for downstream use.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"ASCB input directory not found: {input_dir}")

    tsv_files = sorted(input_dir.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in {input_dir}")

    questions: List[Dict[str, Any]] = []
    group_index = 0
    for tsv_path in tsv_files:
        df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")

        missing = [col for col in ("ID", *question_columns) if col not in df.columns]
        if missing:
            raise ValueError(f"{tsv_path} is missing required columns: {', '.join(missing)}")

        for row_index, row in df.iterrows():
            source_id = str(row["ID"]).strip() or str(row_index)
            group_id = f"ascb_{group_index}"
            group_index += 1

            for question_type in question_columns:
                text = str(row[question_type]).strip()
                if not text:
                    continue

                questions.append({
                    "id": f"{group_id}_{question_type}",
                    "text": text,
                    "language": language,
                    "answers": [],
                    "group_id": group_id,
                    "source_id": source_id,
                    "question_type": question_type,
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(questions)} ASCB questions to {output_path}")
    return questions



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ASCB TSV benchmark questions to docs/questions.json."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_ASCB_DIR),
        help="Directory containing ASCB TSV files. Default: data/ASCB",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSON file. Default: docs/questions.json",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code to store in generated records. Default: en",
    )
    args = parser.parse_args()
    ascb_to_json(args.input_dir, args.output, args.language)


if __name__ == "__main__":
    main()

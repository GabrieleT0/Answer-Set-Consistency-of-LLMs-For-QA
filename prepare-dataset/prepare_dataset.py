from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "Dataset" / "en"
DEFAULT_OUTPUT = DATASET_DIR / "unified_benchmark.tsv"

SOURCE_FILES = {
	"LC-QuAD": DATASET_DIR / "LC-QuAD.tsv",
	"qawiki": DATASET_DIR / "qawiki.tsv",
	"spinach": DATASET_DIR / "spinach.tsv",
	"synthetic": DATASET_DIR / "synthetic.tsv",
}

REQUIRED_COLUMNS = ["ID", "Q1", "Q2", "Q3", "Q4"]


def load_source_table(source_name: str, file_path: Path) -> pd.DataFrame:
	frame = pd.read_csv(file_path, sep="\t", dtype=str)
	missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
	if missing_columns:
		missing = ", ".join(missing_columns)
		raise ValueError(f"{file_path} is missing required columns: {missing}")

	unified = frame[REQUIRED_COLUMNS].copy()
	unified = unified.rename(columns={"ID": "source_id"})
	unified["source"] = source_name
	return unified


def build_unified_dataset() -> pd.DataFrame:
	frames = [load_source_table(source_name, file_path) for source_name, file_path in SOURCE_FILES.items()]
	unified = pd.concat(frames, ignore_index=True)
	unified["index"] = unified.index.astype(str)
	return unified[["index", "source_id", "Q1", "Q2", "Q3", "Q4", "source"]]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Merge the four English benchmark TSV files into one unified TSV."
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT,
		help=f"Output TSV path. Default: {DEFAULT_OUTPUT}",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	unified = build_unified_dataset()
	args.output.parent.mkdir(parents=True, exist_ok=True)
	unified.to_csv(args.output, sep="\t", index=False)
	print(f"Saved {len(unified)} rows to {args.output}")


if __name__ == "__main__":
	main()

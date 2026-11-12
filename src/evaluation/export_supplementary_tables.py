"""Compatibility CLI; implementation lives in evaluation.tables.export_supplementary_tables."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.tables.export_supplementary_tables import main

if __name__ == "__main__":
    main()

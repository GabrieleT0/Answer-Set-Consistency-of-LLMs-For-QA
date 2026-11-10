"""Compatibility entry point for the relation-aware ablation generator.

This module runs live model generation. To score saved outputs afterward, run
``python -m src.evaluation.eval_relation_aware_ablation``.
"""

from src.get_answers.relation_aware_ablation import parse_args, run


if __name__ == "__main__":
    run(parse_args())

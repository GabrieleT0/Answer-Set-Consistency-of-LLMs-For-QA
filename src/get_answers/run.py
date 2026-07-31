# run.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import copy
import json
from get_answers.single_question_benchmark import main as zeroshot_main
from get_answers.relation_classification_and_questions import main as classify_main
from get_answers.try_fix_llm_response import main as fix_main
from get_answers.relation_classification import main as relation_main
from get_answers.logging_utils import setup_logging

# Step 1: Set up logging for the whole pipeline
logger = setup_logging("run_pipeline", "pipeline")


def load_config(config_path=None):
    config_path = config_path or os.path.join(
        os.path.dirname(__file__), "config.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ASCB answers for selected experimental conditions."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.json"),
        help="Path to the JSON experiment configuration.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["relations", "zero-shot", "fixing", "classification"],
        help="Override the generation_steps list in the configuration.",
    )
    return parser.parse_args()


# Step 3: Run main logic
def main(config_path=None, steps=None):
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path or 'config.json'}")
    logger.info(f"{config}")
    # Inject root_dir dynamically
    config["root_dir"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    enabled_steps = steps or config.get(
        "generation_steps",
        ["relations", "zero-shot", "fixing", "classification"],
    )

    llms = list(config["llm_models"])
    for llm in llms:
        model_config = copy.deepcopy(config)
        model_config["llm_models"] = [llm]
        logger.info("=== Starting unified LLM benchmark pipeline ===")

        if "relations" in enabled_steps:
            logger.info("Step: identify question relations")
            relation_main(model_config, logger)
        if "zero-shot" in enabled_steps:
            logger.info("Step: run single-question benchmark")
            zeroshot_main(model_config, logger)
        if "fixing" in enabled_steps:
            logger.info("Step: run fixing benchmark")
            fix_main(model_config, logger)
        if "classification" in enabled_steps:
            logger.info("Step: run classification-then-answer benchmark")
            classify_main(model_config, logger)
        logger.info("All configured tasks completed successfully.")

if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config, steps=args.steps)

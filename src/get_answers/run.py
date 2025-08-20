# run.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from get_answers.single_question_benchmark import main as zeroshot_main
from get_answers.relation_classification_and_questions import main as classify_main
from get_answers.try_fix_llm_response import main as fix_main
from get_answers.relation_classification import main as relation_main
from get_answers.logging_utils import setup_logging

# Step 1: Set up logging for the whole pipeline
logger = setup_logging("run_pipeline", "pipeline")


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config

# Step 3: Run main logic
def main():

    config = load_config()
    logger.info("Loaded config from config.json")
    logger.info(f"{config}")
    # Inject root_dir dynamically
    config["root_dir"] = os.path.dirname(os.path.abspath(__name__))
    # classify_main(config=None, logger=logger)

    llms = config["llm_models"]
    for llm in llms:
        config["llm_models"] = [llm]
        logger.info("=== Starting unified LLM benchmark pipeline ===")
        
        logger.info("Step 0: Identify Relations")
        relation_main(config, logger)
        logger.info("Step 1: Running single question benchmark")
        zeroshot_main(config, logger)

        logger.info("Step 2: Fixing inconsistent LLM responses")
        fix_main(config, logger)

        logger.info("Step 3: Running relation classification and question generation")
        classify_main(config, logger)

    logger.info("All tasks completed successfully.")


if __name__ == "__main__":
    main()


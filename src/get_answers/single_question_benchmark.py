import os
import csv
import json
from langchain_core.prompts import PromptTemplate
from llms import PromptLLMS
import utils 
import yaml
import datetime
import logging

LOGICAL_RELATIONS_MAP = {
                'Q1': 'equal', 'Q2': 'equal', 'Q3': 'sup-sub', 'Q4': 'minus'
            }

# Conditional logging
def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.datetime.now().strftime("single_question_benchmark_%Y-%m-%d_%H-%M.log")
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding='utf-8')
        ]
    )
    for name in logging.root.manager.loggerDict:
        if name not in ["single_question_benchmark"]:  # your custom logger name
            logging.getLogger(name).setLevel(logging.WARNING)

    logger = logging.getLogger("single_question_benchmark")
    return logger
# Load environment variables

root_dir = os.path.dirname(os.path.abspath(__name__))

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(HERE, "prompts.yaml")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)


# === Modules ===

def load_questions(tsv_file, column):
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append(row[column])
    return questions

def get_prompt(prompt_type, language):
    return PromptTemplate(
        input_variables=["question"],
        template=PROMPTS[prompt_type][language]
    )

def process_question(question, llm_model, prompt_template, language, logger):
    try:
        llms = PromptLLMS(model=llm_model, prompt_template=prompt_template, question=question)
        response = llms.execute_single_question()
        if language == 'en':
            return utils.convert_response_to_set(response)
        else:
            return utils.convert_response_to_set_es(response)
    except ValueError as e:
        logger.info(f"Content filter triggered for question: {question}")
        logger.info(f"Error: {e}")
        return None  # fallback response set


def save_answers(answers, dataset, column, language, prompt_type, llm_model, config):
    lang_prefix = '' if language == 'en' else '*'
    relation = LOGICAL_RELATIONS_MAP[column]
    suffix = f"_answers_{'wikidata_' if prompt_type == 'wikidata' else ''}{llm_model}.json"

    out_path = os.path.join(
        root_dir, 'data', 'answers', 'zero-shot',
        dataset.split(".")[0], relation,
        f"{lang_prefix}{column}_{relation}{suffix}"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
    # logger.info(f"Answers saved to {out_path}")


def load_answers(dataset, column, language, prompt_type, llm_model, config):
    lang_prefix = '' if language == 'en' else '*'
    relation = LOGICAL_RELATIONS_MAP[column]
    suffix = f"_answers_{'wikidata_' if prompt_type == 'wikidata' else ''}{llm_model}.json"
    in_file = root_dir + f'/data/answers/zero-shot/{dataset.split(".")[0]}/{relation}/{lang_prefix}{column}_{relation}{suffix}'

    if os.path.exists(in_file):
        with open(in_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}



# === benchmark ===

def run_benchmark_equal(prompt_type, config, logger):
    for language in config["languages"]:
        for llm_model in config["llm_models"]:
            for dataset in config["datasets"]:
                logger.info(f"Processing dataset: {dataset} for model: {llm_model} and language: {language}")
                tsv_file = os.path.join(root_dir, f'data/Dataset/{language}/{dataset}')

                for column in ['Q1', 'Q2', 'Q3', 'Q4']:
                    logger.info(f"Processing column: {column}")
                    questions = load_questions(tsv_file, column)
                    prompt_template = get_prompt(prompt_type, language)

                    answers = load_answers(dataset, column, language, prompt_type, llm_model, config)

                    for index, question in enumerate(questions):
                        if str(index) in answers:
                            continue

                        response_set = process_question(question, llm_model, prompt_template, language, logger)
                        if response_set is None:
                            logger.info(f"Skipping question {index + 1} due to content filter.")
                            continue

                        answers[str(index)] = response_set

                        logger.info(f"Question {index + 1}: {question}")
                        # logger.info(f"LLM Response: {response_set}")

                        save_answers(answers, dataset, column, language, prompt_type, llm_model, config)

                    save_answers(answers, dataset, column, language, prompt_type, llm_model, config)


def main(config = None, logger = setup_logger()):
    if config == None:
        config = {
            "languages": ['en'],
            "llm_models": ['gemini-2.0-flash'],
            "datasets": ['spinach.tsv', 'qawiki.tsv', 'synthetic.tsv'],
            "prompt_types": ['standard', 'wikidata']
        }
    

    for prompt_type in config["prompt_types"]:
        run_benchmark_equal(prompt_type, config, logger)

if __name__ == "__main__":
    main()
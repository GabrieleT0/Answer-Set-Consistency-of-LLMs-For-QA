import os
import csv
import json
from langchain_core.messages import HumanMessage
from llms import return_chat_model
import utils
import yaml
import datetime
import logging

LOGICAL_RELATIONS_MAP = {
    'Q1': 'equal', 'Q2': 'equal', 'Q3': 'sup-sub', 'Q4': 'minus'
}


def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.datetime.now().strftime("chain_of_thought_benchmark_%Y-%m-%d_%H-%M.log")
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
        if name not in ["chain_of_thought_benchmark"]:
            logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("chain_of_thought_benchmark")


root_dir = '../../'

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(HERE, "prompts.yaml")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)


def load_rows(tsv_file):
    rows = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            rows.append({col: row[col] for col in ['Q1', 'Q2', 'Q3', 'Q4']})
    return rows


def parse_cot_response(response_text):
    mapping = {
        'Q1': 'Q1',
        'Q2': 'Q2',
        'Q3': 'Q3',
        'Q4': 'Q4',
    }
    result = {col: None for col in ['Q1', 'Q2', 'Q3', 'Q4']}

    if '∎' in response_text:
        response_text = response_text.split('∎', 1)[1]

    for line in response_text.strip().splitlines():
        line = line.strip()
        for prefix, col in mapping.items():
            if line.startswith(prefix + ':'):
                value = line.split(':', 1)[1].strip()
                if value.lower() == 'idk':
                    result[col] = ['idk']
                else:
                    parts = [p.strip() for p in value.split('|') if p.strip()]
                    result[col] = parts if parts else ['idk']
                break

    return result


def process_row_cot(row_questions, llm_model, language, logger):
    """Single-turn CoT: send all 4 questions at once and parse the structured response."""
    try:
        chat = return_chat_model(llm_model)
        cot_prompt = PROMPTS["chain_of_thought"][language]

        formatted = cot_prompt.format(
            Q1=row_questions['Q1'],
            Q2=row_questions['Q2'],
            Q3=row_questions['Q3'],
            Q4=row_questions['Q4'],
        )
        messages = [HumanMessage(content=formatted)]
        response = chat.invoke(messages)
        # unwrap response if necessary. The selfhosted LLM does not return a structured response but rather a string.
        response = response.content if not isinstance(response,str) else response
        logger.info(f"CoT response: {response}...")
        return parse_cot_response(response)

    except ValueError as e:
        logger.info(f"Content filter triggered for questions: {row_questions}")
        logger.info(f"Error: {e}")
        return None


def save_answers(answers, dataset, column, language, llm_model):
    lang_prefix = '' if language == 'en' else '*'
    relation = LOGICAL_RELATIONS_MAP[column]
    suffix = f"_answers_{llm_model}.json"

    out_path = os.path.join(
        root_dir, 'data', 'answers', 'chain-of-thought',
        dataset.split(".")[0], relation,
        f"{lang_prefix}{column}_{relation}{suffix}"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)


def load_answers(dataset, column, language, llm_model):
    lang_prefix = '' if language == 'en' else '*'
    relation = LOGICAL_RELATIONS_MAP[column]
    suffix = f"_answers_{llm_model}.json"
    in_file = os.path.join(
        root_dir, 'data', 'answers', 'chain-of-thought',
        dataset.split(".")[0], relation,
        f"{lang_prefix}{column}_{relation}{suffix}"
    )

    if os.path.exists(in_file):
        with open(in_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# === Benchmark ===

def run_benchmark(config, logger):
    for language in config["languages"]:
        for llm_model in config["llm_models"]:
            for dataset in config["datasets"]:
                logger.info(f"Processing dataset: {dataset} | model: {llm_model} | language: {language}")
                tsv_file = os.path.join(root_dir, f'data/Dataset/{language}/{dataset}')
                rows = load_rows(tsv_file)

                all_answers = {
                    col: load_answers(dataset, col, language, llm_model)
                    for col in ['Q1', 'Q2', 'Q3', 'Q4']
                }

                for index, row_questions in enumerate(rows):
                    if all(
                        str(index) in all_answers[col] and len(all_answers[col][str(index)]) > 0
                        for col in ['Q1', 'Q2', 'Q3', 'Q4']
                    ):
                        continue

                    result = process_row_cot(row_questions, llm_model, language, logger)
                    if result is None:
                        logger.info(f"Skipping row {index + 1} due to content filter.")
                        continue

                    for col in ['Q1', 'Q2', 'Q3', 'Q4']:
                        if result[col] is not None:
                            all_answers[col][str(index)] = result[col]

                    logger.info(f"Row {index + 1}: {row_questions['Q1'][:80]}")

                    for col in ['Q1', 'Q2', 'Q3', 'Q4']:
                        save_answers(all_answers[col], dataset, col, language, llm_model)

                for col in ['Q1', 'Q2', 'Q3', 'Q4']:
                    save_answers(all_answers[col], dataset, col, language, llm_model)


def main(config=None, logger=None):
    if logger is None:
        logger = setup_logger()
    if config is None:
        config = {
            "languages": ['en'],
            "llm_models": ['llama3.1:8b'],
            "datasets": ['spinach.tsv', 'qawiki.tsv', 'synthetic.tsv'],
        }

    run_benchmark(config, logger)


if __name__ == "__main__":
    main()

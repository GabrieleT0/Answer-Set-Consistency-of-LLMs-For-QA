import os
import csv
import json
import time
import yaml
import logging
import datetime
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import utils
import llms

# === Setup ===

def init_logger():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.datetime.now().strftime("try_fix_llm_response_%Y-%m-%d_%H-%M.log")
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
        if name != "try_fix_llm_response":
            logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("try_fix_llm_response")

logger = init_logger()

def load_prompts():
    here = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(here, "prompts.yaml")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["fix"]


# === Core Fixing Functions ===

def equal_test(config, prompts, llm_model, dataset_name, language='en'):
    chat = llms.return_chat_model(llm_model)
    template = prompts[language]['template']
    fix_prompt = prompts[language]['equal_fix']
    root_dir = config["root_dir"]

    tsv_path = os.path.join(root_dir, f'data/Dataset/{language}/{dataset_name}')
    questions = [(row['Q1'], row['Q2']) for row in csv.DictReader(open(tsv_path, encoding='utf-8'), delimiter='\t')]

    results_q1 = load_answers(config, dataset_name, 'equal', 'Q1', llm_model, language)
    results_q2 = load_answers(config, dataset_name, 'equal', 'Q2', llm_model, language)

    for i, (q1, q2) in enumerate(questions):
        if str(i) in results_q1 and str(i) in results_q2:
            continue
        logger.info(f"[Equal {i}] {q1} | {q2}")
        convo = ConversationChain(llm=chat, memory=ConversationBufferMemory())
        try:
            a1 = utils.convert_response_to_set(convo.predict(input=q1 + "\n" + template))
            a2 = utils.convert_response_to_set(convo.predict(input=q2 + "\n" + template))
            if utils.jaccard_similarity(a1, a2) < 1:
                a2 = utils.convert_response_to_set(convo.predict(input=fix_prompt))
        except Exception as e:
            logger.error(f"[Equal {i}] Error: {e}")
            continue

        results_q1[str(i)] = list(a1)
        results_q2[str(i)] = list(a2)
        if i % 10 == 0:
            save_answers(config, results_q1, dataset_name, 'equal', 'Q1', llm_model, language)
            save_answers(config, results_q2, dataset_name, 'equal', 'Q2', llm_model, language)

    save_answers(config, results_q1, dataset_name, 'equal', 'Q1', llm_model, language)
    save_answers(config, results_q2, dataset_name, 'equal', 'Q2', llm_model, language)


def sup_sub_test(config, prompts, llm_model, dataset_name, language='en'):
    chat = llms.return_chat_model(llm_model)
    template = prompts[language]['template']
    fix_prompt = prompts[language]['sup_sub_fix']
    root_dir = config["root_dir"]

    tsv_path = os.path.join(root_dir, f'data/Dataset/{language}/{dataset_name}')
    questions = [(row['Q1'], row['Q3']) for row in csv.DictReader(open(tsv_path, encoding='utf-8'), delimiter='\t')]

    results_q1 = load_answers(config, dataset_name, 'sup-sub', 'Q1', llm_model, language)
    results_q3 = load_answers(config, dataset_name, 'sup-sub', 'Q3', llm_model, language)

    for i, (q1, q3) in enumerate(questions):
        if str(i) in results_q1 and str(i) in results_q3:
            continue
        logger.info(f"[SupSub {i}] {q1} | {q3}")
        convo = ConversationChain(llm=chat, memory=ConversationBufferMemory())
        try:
            a1 = utils.convert_response_to_set(convo.predict(input=q1 + "\n" + template))
            a3 = utils.convert_response_to_set(convo.predict(input=q3 + "\n" + template))
            if not utils.is_subset(a3, a1) or len(a3) == 0:
                a3 = utils.convert_response_to_set(convo.predict(input=fix_prompt))
        except Exception as e:
            logger.error(f"[SupSub {i}] Error: {e}")
            continue

        results_q1[str(i)] = list(a1)
        results_q3[str(i)] = list(a3)
        if i % 10 == 0:
            save_answers(config, results_q1, dataset_name, 'sup-sub', 'Q1', llm_model, language)
            save_answers(config, results_q3, dataset_name, 'sup-sub', 'Q3', llm_model, language)

    save_answers(config, results_q1, dataset_name, 'sup-sub', 'Q1', llm_model, language)
    save_answers(config, results_q3, dataset_name, 'sup-sub', 'Q3', llm_model, language)


def minus_test(config, prompts, llm_model, dataset_name, language='en', start_index=0, end_index=None):
    chat = llms.return_chat_model(llm_model)
    template = prompts[language]['template']
    fix_prompt = prompts[language]['minus_fix']
    root_dir = config["root_dir"]

    tsv_path = os.path.join(root_dir, f'data/Dataset/{language}/{dataset_name}')
    questions = [(row['Q1'], row['Q3'], row['Q4']) for row in csv.DictReader(open(tsv_path, encoding='utf-8'), delimiter='\t')]
    end_index = min(end_index or len(questions), len(questions))

    results_q1 = load_answers(config, dataset_name, 'minus', 'Q1', llm_model, language)
    results_q3 = load_answers(config, dataset_name, 'minus', 'Q3', llm_model, language)
    results_q4 = load_answers(config, dataset_name, 'minus', 'Q4', llm_model, language)

    for i in range(start_index, end_index):
        if str(i) in results_q1 and str(i) in results_q3 and str(i) in results_q4:
            continue
        q1, q3, q4 = questions[i]
        logger.info(f"[Minus {i}] {q1} | {q3} | {q4}")
        convo = ConversationChain(llm=chat, memory=ConversationBufferMemory())
        try:
            a1 = utils.convert_response_to_set(convo.predict(input=q1 + "\n" + template))
            a2 = utils.convert_response_to_set(convo.predict(input=q3 + "\n" + template))
            a3 = utils.convert_response_to_set(convo.predict(input=q4 + "\n" + template))
            if not utils.is_minus(a1, a2, a3):
                a3 = utils.convert_response_to_set(convo.predict(input=fix_prompt))
        except Exception as e:
            logger.error(f"[Minus {i}] Error: {e}")
            continue

        results_q1[str(i)] = list(a1)
        results_q3[str(i)] = list(a2)
        results_q4[str(i)] = list(a3)

        if i % 10 == 0:
            save_answers(config, results_q1, dataset_name, 'minus', 'Q1', llm_model, language)
            save_answers(config, results_q3, dataset_name, 'minus', 'Q3', llm_model, language)
            save_answers(config, results_q4, dataset_name, 'minus', 'Q4', llm_model, language)

    save_answers(config, results_q1, dataset_name, 'minus', 'Q1', llm_model, language)
    save_answers(config, results_q3, dataset_name, 'minus', 'Q3', llm_model, language)
    save_answers(config, results_q4, dataset_name, 'minus', 'Q4', llm_model, language)


# === I/O Functions ===

def save_answers(config, data, dataset_name, relation, column, llm_model, language):
    root_dir = config["root_dir"]
    prefix = '' if language == 'en' else '*'
    base_dir = os.path.join(root_dir, f'data/answers/follow_up_fixing/{dataset_name.split(".")[0]}/{relation}')
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{prefix}{column}_{relation}_answers_fixing_{llm_model}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved: {path}")


def load_answers(config, dataset_name, relation, column, llm_model, language):
    root_dir = config["root_dir"]
    prefix = '' if language == 'en' else '*'
    path = os.path.join(root_dir, f'data/answers/follow_up_fixing/{dataset_name.split(".")[0]}/{relation}',
                        f"{prefix}{column}_{relation}_answers_fixing_{llm_model}.json")
    return json.load(open(path, 'r', encoding='utf-8')) if os.path.exists(path) else {}


# === Entrypoint ===

def main(config = None):
    load_dotenv()
    prompts = load_prompts()
    
    if not config:
        config = {
            "root_dir": os.path.dirname(os.path.abspath(__name__)),
            "llm_models": ['gpt-4o', 'o3'],
            "languages": ['en'],
            "datasets": ['spinach.tsv', 'qawiki.tsv', 'synthetic.tsv']
        }

    for language in config["languages"]:
        for llm_model in config["llm_models"]:
            for dataset in config["datasets"]:
                logger.info(f"=== Running model {llm_model} on {dataset} ===")
                equal_test(config, prompts, llm_model, dataset, language)
                sup_sub_test(config, prompts, llm_model, dataset, language)
                minus_test(config, prompts, llm_model, dataset, language)
                logger.info(f"=== Finished model {llm_model} on {dataset} ===\n")


if __name__ == "__main__":
    main()

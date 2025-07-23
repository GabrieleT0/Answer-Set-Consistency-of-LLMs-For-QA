import os
import csv
import json
import time
import logging
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import utils
import llms
import yaml

# Setup logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Suppress OpenAI + HTTPX logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain_core").setLevel(logging.WARNING)

logger = logging.getLogger("followup_fixer")
logger.setLevel(logging.INFO)

load_dotenv()

root_dir = os.path.dirname(os.path.abspath(__name__))
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(HERE, "prompts.yaml")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)['fix']


def equal_test(llm_model, dataset_name, language='en'):
    chat = llms.return_chat_model(llm_model)
    template = PROMPTS[language]['template']
    fix_prompt = PROMPTS[language]['equal_fix']

    tsv_path = root_dir + f'/data/Dataset/{language}/{dataset_name}'
    questions = [(row['Q1'], row['Q2']) for row in csv.DictReader(open(tsv_path, encoding='utf-8'), delimiter='\t')]

    results_q1, results_q2 = {}, {}

    for i, (q1, q2) in enumerate(questions):
        convo = ConversationChain(llm=chat, memory=ConversationBufferMemory())
        try:
            a1 = utils.convert_response_to_set(convo.predict(input=q1 + "\n" + template))
            a2 = utils.convert_response_to_set(convo.predict(input=q2 + "\n" + template))
            if utils.jaccard_similarity(a1, a2) < 1:
                logger.info(f"[Equal {i}] Jaccard < 1.0 → applying fix")
                a2 = utils.convert_response_to_set(convo.predict(input=fix_prompt))
        except Exception as e:
            logger.error(f"[Equal {i}] Error with questions: {q1} | {q2} — {e}")
            continue

        results_q1[str(i)] = list(a1)
        results_q2[str(i)] = list(a2)

    _save_json(results_q1, dataset_name, 'equal', 'Q1', llm_model, language)
    _save_json(results_q2, dataset_name, 'equal', 'Q2', llm_model, language)


def sup_sub_test(llm_model, dataset_name, language='en'):
    chat = llms.return_chat_model(llm_model)
    template = PROMPTS[language]['template']
    fix_prompt = PROMPTS[language]['sup_sub_fix']

    tsv_path = root_dir + f'/data/Dataset/{language}/{dataset_name}'
    questions = [(row['Q1'], row['Q3']) for row in csv.DictReader(open(tsv_path, encoding='utf-8'), delimiter='\t')]

    results_q1, results_q3 = {}, {}

    for i, (q1, q3) in enumerate(questions):
        convo = ConversationChain(llm=chat, memory=ConversationBufferMemory())
        try:
            a1 = utils.convert_response_to_set(convo.predict(input=q1 + "\n" + template))
            a3 = utils.convert_response_to_set(convo.predict(input=q3 + "\n" + template))
            if not utils.is_subset(a3, a1) or len(a3) == 0:
                logger.info(f"[SupSub {i}] Not subset or empty → applying fix")
                a3 = utils.convert_response_to_set(convo.predict(input=fix_prompt))
        except Exception as e:
            logger.error(f"[SupSub {i}] Error with questions: {q1} | {q3} — {e}")
            continue

        results_q1[str(i)] = list(a1)
        results_q3[str(i)] = list(a3)

    _save_json(results_q1, dataset_name, 'sup-sub', 'Q1', llm_model, language)
    _save_json(results_q3, dataset_name, 'sup-sub', 'Q3', llm_model, language)


def minus_test(llm_model, dataset_name, language='en', start_index=0, end_index=None):
    chat = llms.return_chat_model(llm_model)
    template = PROMPTS[language]['template']
    fix_prompt = PROMPTS[language]['minus_fix']

    tsv_path = root_dir + f'/data/Dataset/{language}/{dataset_name}'
    questions = [(row['Q1'], row['Q3'], row['Q4']) for row in csv.DictReader(open(tsv_path, encoding='utf-8'), delimiter='\t')]

    if end_index is None or end_index > len(questions):
        end_index = len(questions)

    results_q1, results_q3, results_q4 = {}, {}, {}

    for i in range(start_index, end_index):
        q1, q3, q4 = questions[i]
        convo = ConversationChain(llm=chat, memory=ConversationBufferMemory())
        try:
            a1 = utils.convert_response_to_set(convo.predict(input=q1 + "\n" + template))
            a2 = utils.convert_response_to_set(convo.predict(input=q3 + "\n" + template))
            a3 = utils.convert_response_to_set(convo.predict(input=q4 + "\n" + template))

            if not utils.is_minus(a1, a2, a3):
                logger.info(f"[Minus {i}] Not minus → applying fix")
                a3 = utils.convert_response_to_set(convo.predict(input=fix_prompt))
        except Exception as e:
            logger.error(f"[Minus {i}] Error with questions: {q1}, {q3}, {q4} — {e}")
            continue

        results_q1[str(i)] = list(a1)
        results_q3[str(i)] = list(a2)
        results_q4[str(i)] = list(a3)

        logger.info(f"[Minus {i}] is_minus: {utils.is_minus(a1, a2, a3)}")
        time.sleep(2)

    _save_json(results_q1, dataset_name, 'minus', 'Q1', llm_model, language)
    _save_json(results_q3, dataset_name, 'minus', 'Q3', llm_model, language)
    _save_json(results_q4, dataset_name, 'minus', 'Q4', llm_model, language)


def _save_json(data, dataset_name, relation, column, llm_model, language):
    prefix = '*' if language == 'es' else ''
    base_dir = os.path.join(root_dir, f'data/answers/follow_up_fixing/{dataset_name.split(".")[0]}/{relation}')
    os.makedirs(base_dir, exist_ok=True)
    filename = f"{prefix}{column}_{relation}_answers_fixing_{llm_model}.json"
    path = os.path.join(base_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved file: {path}")


if __name__ == "__main__":
    llm_models = ['o3']
    languages = ['en']
    datasets = ['spinach.tsv']

    for lang in languages:
        for model in llm_models:
            for dataset in datasets:
                logger.info(f"=== Running model {model} on {dataset} ===")
                equal_test(model, dataset, lang) 
                sup_sub_test(model, dataset, lang)
                minus_test(model, dataset, lang)
                logger.info(f"=== Finished model {model} on {dataset} ===\n")

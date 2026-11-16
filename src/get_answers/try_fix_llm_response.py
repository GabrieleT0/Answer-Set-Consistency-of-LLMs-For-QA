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
try:
    from . import utils
    from . import llms
    from .concurrent_requests import completed_requests, chat_model_per_worker
except ImportError:
    import utils
    import llms
    from concurrent_requests import completed_requests, chat_model_per_worker

# === Setup ===

# Conditional logging
def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.datetime.now().strftime("try_fix_llm_response%Y-%m-%d_%H-%M.log")
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
        if name not in ["try_fix_llm_response"]:  # your custom logger name
            logging.getLogger(name).setLevel(logging.WARNING)

    logger = logging.getLogger("try_fix_llm_response")
    return logger

def load_prompts(no_idk=False):
    here = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(here, "prompts.yaml")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_key = "fix_no_idk" if no_idk else "fix"
        return yaml.safe_load(f)[prompt_key]


# === Core Fixing Functions ===

def equal_test(config, prompts, llm_model, dataset_name, language='en', logger=setup_logger()):
    workers = config.get("workers", 1)
    get_chat = chat_model_per_worker(llm_model, llms.return_chat_model, workers)
    template = prompts[language]['template']
    fix_prompt = prompts[language]['equal_fix']
    root_dir = config["root_dir"]

    tsv_path = os.path.join(root_dir, f'data/ASCB/{language}/{dataset_name}')
    questions = [(row['Q1'], row['Q2']) for row in csv.DictReader(open(tsv_path, encoding='utf-8'), delimiter='\t')]

    results_q1 = load_answers(config, dataset_name, 'equal', 'Q1', llm_model, language)
    results_q2 = load_answers(config, dataset_name, 'equal', 'Q2', llm_model, language)

    pending = [(i, pair) for i, pair in enumerate(questions)
               if str(i) not in results_q1 or str(i) not in results_q2]

    def request(item):
        _, (q1, q2) = item
        convo = ConversationChain(llm=get_chat(), memory=ConversationBufferMemory())
        try:
            a1 = utils.convert_response_to_set(convo.predict(input=q1 + "\n" + template))
            a2 = utils.convert_response_to_set(convo.predict(input=q2 + "\n" + template))
            if utils.jaccard_similarity(a1, a2) < 1:
                a2 = utils.convert_response_to_set(convo.predict(input=fix_prompt))
        except Exception as e:
            return e
        return list(a1), list(a2)

    for (i, (q1, q2)), result in completed_requests(pending, request, workers):
        if isinstance(result, Exception):
            logger.error(f"[Equal {i}] Error: {result}")
            continue
        logger.info(f"[Equal {i}] {q1} | {q2}")
        results_q1[str(i)], results_q2[str(i)] = result
        if i % config.get("checkpoint_every", 10) == 0:
            save_answers(config, results_q1, dataset_name, 'equal', 'Q1', llm_model, language)
            save_answers(config, results_q2, dataset_name, 'equal', 'Q2', llm_model, language)

    save_answers(config, results_q1, dataset_name, 'equal', 'Q1', llm_model, language)
    save_answers(config, results_q2, dataset_name, 'equal', 'Q2', llm_model, language)


def sup_sub_test(config, prompts, llm_model, dataset_name, language='en', logger = setup_logger()):
    workers = config.get("workers", 1)
    get_chat = chat_model_per_worker(llm_model, llms.return_chat_model, workers)
    template = prompts[language]['template']
    fix_prompt = prompts[language]['sup_sub_fix']
    root_dir = config["root_dir"]

    tsv_path = os.path.join(root_dir, f'data/ASCB/{language}/{dataset_name}')
    questions = [(row['Q1'], row['Q3']) for row in csv.DictReader(open(tsv_path, encoding='utf-8'), delimiter='\t')]

    results_q1 = load_answers(config, dataset_name, 'sup-sub', 'Q1', llm_model, language)
    results_q3 = load_answers(config, dataset_name, 'sup-sub', 'Q3', llm_model, language)

    pending = [(i, pair) for i, pair in enumerate(questions)
               if str(i) not in results_q1 or str(i) not in results_q3]

    def request(item):
        _, (q1, q3) = item
        convo = ConversationChain(llm=get_chat(), memory=ConversationBufferMemory())
        try:
            a1 = utils.convert_response_to_set(convo.predict(input=q1 + "\n" + template))
            a3 = utils.convert_response_to_set(convo.predict(input=q3 + "\n" + template))
            if not utils.is_subset(a3, a1) or len(a3) == 0:
                a3 = utils.convert_response_to_set(convo.predict(input=fix_prompt))
        except Exception as e:
            return e
        return list(a1), list(a3)

    for (i, (q1, q3)), result in completed_requests(pending, request, workers):
        if isinstance(result, Exception):
            logger.error(f"[SupSub {i}] Error: {result}")
            continue
        logger.info(f"[SupSub {i}] {q1} | {q3}")
        results_q1[str(i)], results_q3[str(i)] = result
        if i % config.get("checkpoint_every", 10) == 0:
            save_answers(config, results_q1, dataset_name, 'sup-sub', 'Q1', llm_model, language)
            save_answers(config, results_q3, dataset_name, 'sup-sub', 'Q3', llm_model, language)

    save_answers(config, results_q1, dataset_name, 'sup-sub', 'Q1', llm_model, language)
    save_answers(config, results_q3, dataset_name, 'sup-sub', 'Q3', llm_model, language)


def minus_test(config, prompts, llm_model, dataset_name, language='en', start_index=0, end_index=None, logger = setup_logger()):
    workers = config.get("workers", 1)
    get_chat = chat_model_per_worker(llm_model, llms.return_chat_model, workers)
    template = prompts[language]['template']
    fix_prompt = prompts[language]['minus_fix']
    root_dir = config["root_dir"]

    tsv_path = os.path.join(root_dir, f'data/ASCB/{language}/{dataset_name}')
    questions = [(row['Q1'], row['Q3'], row['Q4']) for row in csv.DictReader(open(tsv_path, encoding='utf-8'), delimiter='\t')]
    end_index = min(end_index or len(questions), len(questions))

    results_q1 = load_answers(config, dataset_name, 'minus', 'Q1', llm_model, language)
    results_q3 = load_answers(config, dataset_name, 'minus', 'Q3', llm_model, language)
    results_q4 = load_answers(config, dataset_name, 'minus', 'Q4', llm_model, language)

    pending = [i for i in range(start_index, end_index)
               if str(i) not in results_q1 or str(i) not in results_q3 or str(i) not in results_q4]

    def request(i):
        q1, q3, q4 = questions[i]
        convo = ConversationChain(llm=get_chat(), memory=ConversationBufferMemory())
        try:
            a1 = utils.convert_response_to_set(convo.predict(input=q1 + "\n" + template))
            a2 = utils.convert_response_to_set(convo.predict(input=q3 + "\n" + template))
            a3 = utils.convert_response_to_set(convo.predict(input=q4 + "\n" + template))
            if not utils.is_minus(a1, a2, a3):
                a3 = utils.convert_response_to_set(convo.predict(input=fix_prompt))
        except Exception as e:
            return e
        return list(a1), list(a2), list(a3)

    for i, result in completed_requests(pending, request, workers):
        if isinstance(result, Exception):
            logger.error(f"[Minus {i}] Error: {result}")
            continue
        q1, q3, q4 = questions[i]
        logger.info(f"[Minus {i}] {q1} | {q3} | {q4}")
        results_q1[str(i)], results_q3[str(i)], results_q4[str(i)] = result

        if i % config.get("checkpoint_every", 10) == 0:
            save_answers(config, results_q1, dataset_name, 'minus', 'Q1', llm_model, language)
            save_answers(config, results_q3, dataset_name, 'minus', 'Q3', llm_model, language)
            save_answers(config, results_q4, dataset_name, 'minus', 'Q4', llm_model, language)

    save_answers(config, results_q1, dataset_name, 'minus', 'Q1', llm_model, language)
    save_answers(config, results_q3, dataset_name, 'minus', 'Q3', llm_model, language)
    save_answers(config, results_q4, dataset_name, 'minus', 'Q4', llm_model, language)


# === I/O Functions ===

def _output_location(config, llm_model):
    """Store OpenRouter completions in the original GPT-OSS Oracle checkpoints."""
    if config.get("resume_original_answers", False):
        if llm_model not in {"openai/gpt-oss-20b:floor", "openai/gpt-oss-20b:nitro"}:
            raise ValueError(
                "resume_original_answers requires openai/gpt-oss-20b:floor "
                "or openai/gpt-oss-20b:nitro")
        return "Oracle", "gpt-oss:20b"
    return (
        config.get("answer_action_dir") or (
            "fixing-no-idk" if config.get("no_idk_ablation", False)
            else "follow_up_fixing"
        ),
        config.get("output_model", utils.output_model_name(llm_model)),
    )


def save_answers(config, data, dataset_name, relation, column, llm_model, language):
    root_dir = config["root_dir"]
    prefix = '' if language == 'en' else '*'
    action_dir, output_model = _output_location(config, llm_model)
    base_dir = os.path.join(
        root_dir,
        "data",
        "answers",
        action_dir,
        dataset_name.split(".")[0],
        relation,
    )
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{prefix}{column}_{relation}_answers_fixing_{output_model}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    # logger.info(f"Saved: {path}")


def load_answers(config, dataset_name, relation, column, llm_model, language):
    root_dir = config["root_dir"]
    prefix = '' if language == 'en' else '*'
    action_dir, output_model = _output_location(config, llm_model)
    path = os.path.join(
        root_dir,
        "data",
        "answers",
        action_dir,
        dataset_name.split(".")[0],
        relation,
        f"{prefix}{column}_{relation}_answers_fixing_{output_model}.json",
    )
    return json.load(open(path, 'r', encoding='utf-8')) if os.path.exists(path) else {}


# === Entrypoint ===

def main(config = None, logger = setup_logger()):
    load_dotenv()
    
    if not config:
        config = {
            "root_dir": os.path.dirname(os.path.abspath(__name__)),
            "llm_models": ['gemini-2.0-flash','gpt-4o', 'o3'],
            "languages": ['en'],
            "datasets": ['spinach.tsv', 'qawiki.tsv', 'synthetic.tsv']
        }

    prompts = load_prompts(config.get("no_idk_ablation", False))

    for language in config["languages"]:
        for llm_model in config["llm_models"]:
            for dataset in config["datasets"]:
                logger.info(f"=== Running model {llm_model} on {dataset} ===")
                equal_test(config, prompts, llm_model, dataset, language, logger=logger)
                sup_sub_test(config, prompts, llm_model, dataset, language, logger=logger)
                minus_test(config, prompts, llm_model, dataset, language, logger=logger)
                logger.info(f"=== Finished model {llm_model} on {dataset} ===\n")


if __name__ == "__main__":
    main()

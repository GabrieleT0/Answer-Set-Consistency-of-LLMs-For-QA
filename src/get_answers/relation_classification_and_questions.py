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
    log_filename = datetime.datetime.now().strftime("relation_classification_and_question_%Y-%m-%d_%H-%M.log")
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
        if name != "relation_classification":
            logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("relation_classification")

logger = init_logger()

def load_prompts():
    here = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(here, "prompts.yaml")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    return prompts["relation_classification"], prompts["relation_classification_minus"]


# === Core Functions ===

def run_benchmark(config, prompts, llm_model, language, logical_relation, dataset, use_hint=False, start_index=0, end_index=None):
    chat = llms.return_chat_model(llm_model)
    root_dir = config["root_dir"]
    tsv_file = os.path.join(root_dir, f'data/Dataset/{language}/{dataset}')
    
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            if logical_relation == 'Equivalence':
                questions.append((row['Q1'], row['Q2']))
            elif logical_relation == 'Containment':
                questions.append((row['Q1'], row['Q3']))

    end_index = min(end_index or len(questions), len(questions))
    output_prefix = '' if language == 'en' else '*'
    folder_name = 'equal' if logical_relation == 'Equivalence' else 'sup-sub'
    base_output_dir = os.path.join(root_dir, 'data', 'answers', 'rel_classification_and_questions', dataset.split(".")[0], folder_name)
    os.makedirs(base_output_dir, exist_ok=True)

    q1_path = os.path.join(base_output_dir, f'{output_prefix}Q1_{folder_name}_answers_classAndAnswer_{llm_model}.json')
    q2_path = os.path.join(base_output_dir, f'{output_prefix}Q2_{folder_name}_answers_classAndAnswer_{llm_model}.json')
    relation_path = os.path.join(base_output_dir, f'{output_prefix}{logical_relation}_{folder_name}_relation_{llm_model}.json')

    def load_json(path):
        return json.load(open(path, 'r', encoding='utf-8')) if os.path.exists(path) else {}

    answers_ql1 = load_json(q1_path)
    answers_ql2 = load_json(q2_path)
    q12_relation = load_json(relation_path)

    for index in range(start_index, end_index):
        if str(index) in answers_ql1:
            continue
        question = questions[index]
        memory = ConversationBufferMemory()
        try:
            conversation = ConversationChain(llm=chat, memory=memory)
            relation_predicted = conversation.predict(
                input=prompts[language]['template_classification'].format(q1=question[0], q2=question[1])
            ).strip().lower()

            answer1 = conversation.predict(input=question[0] + prompts[language]['template'])

            if use_hint:
                answer2 = conversation.predict(
                    input=question[1] + prompts[language]['hint_prompt'].format(relation=logical_relation) + prompts[language]['template']
                )
            else:
                answer2 = conversation.predict(input=question[1] + prompts[language]['template'])

            answer1 = utils.convert_response_to_set(answer1)
            answer2 = utils.convert_response_to_set(answer2)

            answers_ql1[str(index)] = list(answer1)
            answers_ql2[str(index)] = list(answer2)
            q12_relation[str(index)] = relation_predicted

            for path, data in zip([q1_path, q2_path, relation_path], [answers_ql1, answers_ql2, q12_relation]):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

            logger.info(f"Index {index}: Q1: {question[0]} Q2: {question[1]} Rel: {relation_predicted}")
        except Exception as e:
            logger.error(f"Error at index {index}: {e}")
        time.sleep(1.5)


def run_minus_benchmark(config, prompts, prompts_minus, llm_model, language, test_type, dataset, use_hint=False, start_index=0, end_index=None):
    chat = llms.return_chat_model(llm_model)
    root_dir = config["root_dir"]
    tsv_file = os.path.join(root_dir, f'data/Dataset/{language}/{dataset}')

    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['Q1'], row['Q3'], row['Q4']))

    end_index = min(end_index or len(questions), len(questions))
    output_prefix = '' if language == 'en' else '*'
    base_output_dir = os.path.join(root_dir, 'data', 'answers', 'rel_classification_and_questions', dataset.split(".")[0], 'minus')
    os.makedirs(base_output_dir, exist_ok=True)

    q1_path = os.path.join(base_output_dir, f'{output_prefix}Q1_minus_answers_classAndAnswer_{llm_model}.json')
    q3_path = os.path.join(base_output_dir, f'{output_prefix}Q3_minus_answers_classAndAnswer_{llm_model}.json')
    q4_path = os.path.join(base_output_dir, f'{output_prefix}Q4_minus_answers_classAndAnswer_{llm_model}.json')
    relation_path = os.path.join(base_output_dir, f'{output_prefix}Relation_minus_answers_classAndAnswer_{llm_model}.json')

    def load_json(path):
        return json.load(open(path, 'r', encoding='utf-8')) if os.path.exists(path) else {}

    answers_ql1 = load_json(q1_path)
    answers_ql3 = load_json(q3_path)
    answers_ql4 = load_json(q4_path)
    answer_relation = load_json(relation_path)

    for index in range(start_index, end_index):
        if str(index) in answers_ql1:
            continue
        question = questions[index]
        memory = ConversationBufferMemory()
        try:
            conversation = ConversationChain(llm=chat, memory=memory)

            relation_predicted = conversation.predict(
                input=prompts_minus[language]['template_classification'].format(q1=question[0], q2=question[1], q3=question[2])
            ).strip().lower()

            answer1 = conversation.predict(input=question[0] + prompts[language]['template'])
            answer2 = conversation.predict(input=question[1] + prompts[language]['template'])
            if use_hint:
                answer3 = conversation.predict(
                    input=question[2] + prompts[language]['hint_prompt'].format(relation=test_type) + prompts[language]['template']
                )
            else:
                answer3 = conversation.predict(input=question[2] + prompts[language]['template'])

            answer1 = utils.convert_response_to_set(answer1)
            answer2 = utils.convert_response_to_set(answer2)
            answer3 = utils.convert_response_to_set(answer3)
            relation_set = utils.convert_response_to_set(relation_predicted)

            answers_ql1[str(index)] = list(answer1)
            answers_ql3[str(index)] = list(answer2)
            answers_ql4[str(index)] = list(answer3)
            answer_relation[str(index)] = list(relation_set)

            for path, data in zip([q1_path, q3_path, q4_path, relation_path],
                                  [answers_ql1, answers_ql3, answers_ql4, answer_relation]):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

            logger.info(f"Index {index}: Q1: {question[0]} Q2: {question[1]} Q3: {question[2]} Relation: {relation_predicted}")
        except Exception as e:
            logger.error(f"Error at index {index}: {e}")
        time.sleep(1.5)


# === Entrypoint ===

def main(config = None):
    load_dotenv()
    prompts, prompts_minus = load_prompts()

    if not config:
        config = {
            "root_dir": os.path.dirname(os.path.abspath(__name__)),
            "llm_models": ['gemini-2.0-flash'],
            "languages": ['en'],
            "datasets": ['spinach.tsv', 'qawiki.tsv', 'synthetic.tsv'],
            "relations": ['Minus', 'Containment']
        }

    for language in config["languages"]:
        for llm_model in config["llm_models"]:
            for dataset in config["datasets"]:
                for relation in config["relations"]:
                    logger.info(f"Processing model: {llm_model} | relation: {relation} | lang: {language}")
                    if relation == 'Minus' or relation == 'Resta':
                        run_minus_benchmark(config, prompts, prompts_minus, llm_model, language, relation, dataset)
                    else:
                        run_benchmark(config, prompts, llm_model, language, relation, dataset)


if __name__ == "__main__":
    main()

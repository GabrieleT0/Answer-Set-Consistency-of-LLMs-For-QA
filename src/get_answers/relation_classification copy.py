from langchain_core.prompts import PromptTemplate
from llms import PromptLLMS
import os
import csv
import utils
import json
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import yaml
import datetime
import logging

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


def minus_test(llm_model, language, dataset, logger):
    """
    Run the minus test for the given LLM model and language.
    """
    tsv_file = root_dir + f"/data/Dataset/{language}/{dataset}"
    prompt = PROMPTS["relation_classification_minus"][language]["template_classification"]

    # Load question triples
    question_pairs = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['Q1'], row['Q3'], row['Q4']))
    dataset_name = dataset.split('.')[0]
    output_filename = root_dir + f'/data/answers/zero-shot/{dataset_name}/relation-classification/Minus_Q134_{llm_model}.json'
    answers = {}
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            answers = json.load(f)

    for index, (q1, q3, q4) in enumerate(question_pairs):
        if str(index) in answers:
            continue

        prompt_template = PromptTemplate(
            input_variables=["q1", "q2", "q3"],
            template=prompt
        )
        llms = PromptLLMS(llm_model,prompt_template, q1, q3, q4)
        llm_response = llms.execute_three_question()

        # converted_response = utils.convert_response_to_set_class(llm_response, real_relation)
        converted_response = utils.convert_response_to_set(llm_response)
        answers[index] = list(converted_response)

        logger.info(f"Question {index + 1}: {q1} | {q3} | {q4}")
        logger.info(f"LLM Response: {answers[index]}")

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

def equivalence_test(llm_model, language, dataset, q_pair, relation,logger):
    (q1, q2) = q_pair
    tsv_file = root_dir + f"/data/Dataset/{language}/{dataset}"
    prompt = PROMPTS["relation_classification"][language]["template_classification"]
    # Load questions
    question_pairs = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row[q1], row[q2]))
    dataset_name = dataset.split('.')[0]
    output_filename = root_dir + f'/data/answers/zero-shot/{dataset_name}/relation-classification/{relation}_{q1}_{q2}_{llm_model}.json'
    answers = {}
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            answers = json.load(f)

    for index, (q1, q2) in enumerate(question_pairs):
        if str(index) in answers:
            continue
        prompt_template = PromptTemplate(
            input_variables=["q1", "q2"],
            template=prompt
        )
        llms = PromptLLMS(llm_model, prompt_template, q1, q2)
        llm_response = llms.execute_two_question()
        converted_response = utils.convert_response_to_set(llm_response)

        answers[index] = list(converted_response)

        logger.info(f"Question {index + 1}: {q1} | {q2}")
        logger.info(f"LLM Response: {answers[index]}")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

def sup_sub_test(llm_model, language, dataset):
    chat = PromptLLMS.return_chat_model(llm_model)
    tsv_file = root_dir + f"/data/Dataset/{language}/{dataset}"
    # Load questions
    question_pairs = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['Q1'], row['Q3']))
    dataset_name = dataset.split('.')[0]
    output_filename = root_dir + f'/data/answers/zero-shot/{dataset_name}/relation-classification/{output_prefix}Containment_{llm_model}.json'
    answers = {}
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            answers = json.load(f)

    for index, (q1, q3) in enumerate(question_pairs):
        if str(index) in answers:
            continue
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=chat,
            memory=memory
        )
        relation_predicted = conversation.predict(input=PROMPTS[language].format(q1=q1, q2=q3))
        relation_predicted = relation_predicted.strip().lower()
        print(f"Relation predicted: {relation_predicted}")
        if relation_predicted == 'containment' or relation_predicted == 'contención':
            converted_response = 0.5 # If correctly predict the containment relation, we set the response to 0.5
            direction = conversation.predict(input=PROMPTS["relation_classification_all"][language]["contain_direction"])
            if direction.strip().lower() == 'b':
                converted_response += 0.5 # If p
        else:
            converted_response = 0
        if language == 'en':
            output_prefix = ''
        else:
            output_prefix = '*'

        answers[index] = converted_response

        print(f"Question {index + 1}: q1: {q1} | q2: {q3}")
        print(f"LLM relation: {relation_predicted}, LLM direction: {direction.strip().lower()}")

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)


def main(config = None, logger = setup_logger()):
    
    if config == None: 
        config = {
            "datasets":['qawiki.tsv'],
            "llm_models": ['gpt-4o'],
            "languages": ['en']
        }

    datasets = config["datasets"]
    llm_models = config["llm_models"]
    languages = config['languages']
    for language in languages:
        for llm_model in llm_models:
            for dataset in datasets:
                # Run equivalence test
                logger.info(f"Processing model: {llm_model} for language: {language}")
                q_pair = ("Q1", "Q2")
                equivalence_test(llm_model, language, dataset, q_pair, "Equivalence",logger=logger)
                logger.info(f"Finished processing model: {llm_model} for language: {language})")
                # Run Containment test
                q_pair = ("Q1", "Q3")
                equivalence_test(llm_model, language, dataset, q_pair, "Contains",logger=logger)
                logger.info(f"Finished processing model: {llm_model} for language: {language})")
                q_pair = ("Q1", "Q4")
                equivalence_test(llm_model, language, dataset, q_pair, "Contains",logger=logger)
                logger.info(f"Finished processing model: {llm_model} for language: {language})")
                q_pair = ("Q3", "Q4")
                equivalence_test(llm_model, language, dataset, q_pair, "Disjoint",logger=logger)
                logger.info(f"Finished processing model: {llm_model} for language: {language})")
                # sup_sub_test(llm_model, language, dataset, 'Containment')
                # Run minus test
                logger.info(f"Processing model: {llm_model} for language: {language}")
                minus_test(llm_model, language, dataset, 'Minus', logger=logger)
                logger.info(f"Finished processing model: {llm_model} for language: {language}")

if __name__ == "__main__":
    main()
from langchain_core.prompts import PromptTemplate
from llms import PromptLLMS
import os
import csv
import utils
import json
import yaml
import datetime
import logging

# Conditional logging
def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.datetime.now().strftime("relation_classification_%Y-%m-%d_%H-%M.log")
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
        if name not in ["relation_classification"]:  # your custom logger name
            logging.getLogger(name).setLevel(logging.WARNING)

    logger = logging.getLogger("relation_classification")
    return logger
# Load environment variables

root_dir = os.path.dirname(os.path.abspath(__name__))

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(HERE, "prompts.yaml")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)


def relation_identification(llm_model, language, dataset, logger):
    dataset_name = dataset.split('.')[0]
    tsv_file = root_dir + f"/data/Dataset/{language}/{dataset}"
    # Load questions
    question_pairs = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['Q1'], row['Q2'],row['Q3'], row['Q4']))
    output_filename = root_dir + f'/data/answers/zero-shot/{dataset_name}/relation-classification/Relation_{llm_model}.json'
    answers = {}
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            answers = json.load(f)

    for index, (q1, q2, q3, q4) in enumerate(question_pairs):
        if str(index) in answers:
            continue
        
        prompt_template_1 = PromptTemplate(
            input_variables=["q1", "q2"],
            template=PROMPTS["relation_classification"][language]["template_classification"]
        )

        prompt_template_2 = PromptTemplate(
            input_variables=["q1", "q2", "q3"],
            template=PROMPTS["relation_classification_minus"][language]["template_classification"]
        )
        prompt_template_3 = PromptTemplate(
            input_variables=["q1", "q2", "q3"],
            template=PROMPTS["relation_classification_all"][language]["template_classification"]
        )
        # ?q1=q2
        llms = PromptLLMS(llm_model, prompt_template_1, q1, q2)
        q1_q2 = llms.execute_two_question()

        # llms = PromptLLMS(llm_model, prompt_template_1, q1, q3)
        llms.question1 = q3
        q1_q3 = llms.execute_two_question()

        # llms = PromptLLMS(llm_model, prompt_template_1, q1, q4)
        llms.question1 = q4
        q1_q4 = llms.execute_two_question()
   
        # llms = PromptLLMS(llm_model, prompt_template_1, q3, q4)
        llms.question = q3
        q3_q4 = llms.execute_two_question()

        llms2 = PromptLLMS(llm_model, prompt_template_2, q1, q3, q4)
        q1_q34 = llms2.execute_three_question()

        llms3 = PromptLLMS(llm_model, prompt_template_3, q1, q3, q4)
        relations = llms3.execute_three_question()
        relations = utils.convert_response_to_set(relations)
        answers[index] = [q1_q2, q1_q3, q1_q4, q3_q4, q1_q34, list(relations)]

        logger.info(f"Question {index + 1}")
        logger.info(f"LLM Response: {answers[index]}")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)


def main(config = None, logger = setup_logger()):
    
    if config == None: 
        config = {
            "datasets":['spinach.tsv'],
            "llm_models": ["gpt-4o"],
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
                relation_identification(llm_model, language, dataset, logger=logger)
                logger.info(f"Finished processing model: {llm_model} for language: {language}")

if __name__ == "__main__":
    main()
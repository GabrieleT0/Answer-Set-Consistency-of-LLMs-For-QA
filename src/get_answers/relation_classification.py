from langchain_core.prompts import PromptTemplate
try:
    from .llms import PromptLLMS
    from . import llms
    from . import utils
    from .concurrent_requests import completed_requests, chat_model_per_worker
except ImportError:
    from llms import PromptLLMS
    import llms
    import utils
    from concurrent_requests import completed_requests, chat_model_per_worker
import os
import csv
import json
import yaml
import datetime
import logging
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

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


def _relation_output_path(config, dataset, llm_model):
    project_root = config.get("root_dir", root_dir)
    output_model = config.get("output_model", utils.output_model_name(llm_model))
    return os.path.join(
        project_root, "data", "answers", "zero-shot", dataset.split('.')[0],
        "relation-classification", f"Relation_{output_model}.json",
    )


def relation_identification(llm_model, language, dataset, logger, config=None):
    config = config or {}
    project_root = config.get("root_dir", root_dir)
    tsv_file = os.path.join(project_root, "data", "ASCB", language, dataset)
    PROMPTS_equal = PROMPTS["relation_classification"][language]["template_classification"]
    PROMPTS_minus = PROMPTS["relation_classification_minus"][language]["template_classification"]
    # PROMPTS_all = PROMPTS["relation_classification_all"][language]["template_classification"]
    # Load questions
    question_pairs = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['Q1'], row['Q2'],row['Q3'], row['Q4']))
    output_filename = _relation_output_path(config, dataset, llm_model)
    answers = {}
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            print(output_filename)
            answers = json.load(f)

    workers = config.get("workers", 1)
    pending = [(index, questions) for index, questions in enumerate(question_pairs)
               if str(index) not in answers or not answers[str(index)]]
    get_chat = chat_model_per_worker(llm_model, llms.return_chat_model, workers)

    def request(item):
        _, (q1, q2, q3, q4) = item
        try:
            memory = ConversationBufferMemory()
            conversation = ConversationChain(llm=get_chat(), memory=memory)
            # ?q1=q2
            q1_q2 = conversation.predict(input=PROMPTS_equal.format(q1=q1, q2=q2))
            q1_q3 = conversation.predict(input=PROMPTS_equal.format(q1=q1, q2=q3))
            q1_q4 = conversation.predict(input=PROMPTS_equal.format(q1=q1, q2=q4))
            q3_q4 = conversation.predict(input=PROMPTS_equal.format(q1=q3, q2=q4))
            q1_q34 = conversation.predict(input=PROMPTS_minus.format(q1=q1, q2=q3, q3 = q4))

            return [q1_q2, q1_q3, q1_q4, q3_q4, q1_q34]
        except Exception as e:
            return e

    for (index, _), result in completed_requests(pending, request, workers):
        if isinstance(result, Exception):
            logger.error(f"Error while performing relation identification at {index}: {result}")
            continue
        answers[str(index)] = result
        logger.info(f"Question {index + 1}")
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)


def main(config = None, logger = setup_logger()):
    
    if config == None: 
        config = {
            "datasets":['synthetic.tsv', "spinach.tsv", "qawiki.tsv"],
            "llm_models": ["gpt-5-mini"],
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
                relation_identification(llm_model, language, dataset, logger=logger, config=config)
                logger.info(f"Finished processing model: {llm_model} for language: {language}")

if __name__ == "__main__":
    main()

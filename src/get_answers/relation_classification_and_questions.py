from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import csv
import utils
import json
import time
import llms
import yaml
import datetime
import logging
# Setup logging
# Create a log directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# Log file path (e.g., logs/run_2025-07-17_15-30.log)
log_filename = datetime.datetime.now().strftime("relation_classification_and_question_%Y-%m-%d_%H-%M.log")
log_path = os.path.join(log_dir, log_filename)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(log_path, encoding='utf-8')  # Save to file
    ]
)

# Mute all other libraries except your custom logger
for name in logging.root.manager.loggerDict:
    if name not in ["relation_classification_and_question_"]:  # your custom logger name
        logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger("try_frelation_classification_and_question_ix_llm_response")



load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

root_dir = os.path.dirname(os.path.abspath(__name__))

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(HERE, "prompts.yaml")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)['relation_classification']

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPTS_MINUS = yaml.safe_load(f)['relation_classification_minus']


def run_benchmark(llm_model, language, logical_relation, dataset, use_hint=False, start_index=0, end_index=None):
    chat = llms.return_chat_model(llm_model)
    tsv_file = root_dir + f'/data/Dataset/{language}/{dataset}'
    
    # Read questions
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            if logical_relation == 'Equivalence':
                questions.append((row['Q1'], row['Q2']))
            elif logical_relation == 'Containment':
                questions.append((row['Q1'], row['Q3']))

    if end_index is None or end_index > len(questions):
        end_index = len(questions)

    output_prefix = '*' if language == 'es' else ''
    folder_name = 'equal' if logical_relation == 'Equivalence' else 'sup-sub'

    base_output_dir = root_dir + f'/data/answers/rel_classification_and_questions/{dataset.split(".")[0]}/{folder_name}'
    os.makedirs(base_output_dir, exist_ok=True)

    q1_path = os.path.join(base_output_dir, f'{output_prefix}Q1_{folder_name}_answers_classAndAnswer_{llm_model}.json')
    q2_path = os.path.join(base_output_dir, f'{output_prefix}Q2_{folder_name}_answers_classAndAnswer_{llm_model}.json')
    q12_path = os.path.join(base_output_dir, f'{output_prefix}Q12_{folder_name}_relation_{llm_model}.json')

    # Load previous answers
    def load_json(path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    answers_ql1 = load_json(q1_path)
    answers_ql2 = load_json(q2_path)

    for index in range(start_index, end_index):
        if str(index) in answers_ql1:
            continue  # Skip already processed
        question = questions[index]
        memory = ConversationBufferMemory()
        try: 
            conversation = ConversationChain(llm=chat, memory=memory)
            # Classification prediction
            relation_predicted = conversation.predict(input=PROMPTS[language]['template_classification'].format(q1=question[0], q2=question[1])).strip().lower()
            # Answer generation
            answer1 = conversation.predict(input=question[0] + PROMPTS[language]['template'])

            if use_hint:
                answer2 = conversation.predict(
                    input=question[1] + PROMPTS[language]['hint_prompt'].format(relation=logical_relation) + PROMPTS[language]['template']
                )
            else:
                answer2 = conversation.predict(input=question[1] + PROMPTS[language]['template'])
        except:
            logger.error(f"Error processing question {index}: {question}")
            continue
        logger.info(f"\nOriginal answers: {answer1}, {answer2}")

        # Convert answers
        answer1 = utils.convert_response_to_set(answer1)
        answer2 = utils.convert_response_to_set(answer2)
        # Store
        answers_ql1[str(index)] = list(answer1)
        answers_ql2[str(index)] = list(answer2)

        # Write to files
        with open(q1_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql1, f, ensure_ascii=False, indent=4)
        with open(q2_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

        logger.info(f"Index: {index} Question 1: {question[0]} Question 2: {question[1]}")
        logger.info(f"Answer 1: {answer1} Answer 2: {answer2} Relation: {relation_predicted}\n")
        time.sleep(1.5)


def run_minus_benchmark(llm_model, language, test_type, dataset, use_hint=False, start_index=0, end_index=None):
    chat = llms.return_chat_model(llm_model)
    tsv_file = root_dir + f'/data/Dataset/{language}/{dataset}'

    # Read questions
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['Q1'], row['Q3'], row['Q4']))

    if end_index is None or end_index > len(questions):
        end_index = len(questions)

    output_prefix = '*' if language == 'es' else ''
    # test_type_name = logical_relations[language][test_type]
    base_output_dir = root_dir + f'/data/answers/rel_classification_and_questions/{dataset.split(".")[0]}/minus'
    os.makedirs(base_output_dir, exist_ok=True)

    q1_path = os.path.join(base_output_dir, f'{output_prefix}Q1_minus_answers_classAndAnswer_{llm_model}.json')
    q2_path = os.path.join(base_output_dir, f'{output_prefix}Q3_minus_answers_classAndAnswer_{llm_model}.json')
    q3_path = os.path.join(base_output_dir, f'{output_prefix}Q4_minus_answers_classAndAnswer_{llm_model}.json')

    def load_json(path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    answers_ql1 = load_json(q1_path)
    answers_ql2 = load_json(q2_path)
    answers_ql3 = load_json(q3_path)

    for index in range(start_index, end_index):
        if str(index) in answers_ql1:
            continue  # Skip already processed

        question = questions[index]
        memory = ConversationBufferMemory()
        conversation = ConversationChain(llm=chat, memory=memory)
        try:
            relation_predicted = conversation.predict(
                input=PROMPTS_MINUS[language]['template_classification'].format(q1=question[0], q2=question[1], q3=question[2])
            ).strip().lower()

            answer1 = conversation.predict(input=question[0] + PROMPTS[language]['template'])
            answer2 = conversation.predict(input=question[1] + PROMPTS[language]['template'])

            if use_hint:
                answer3 = conversation.predict(
                    input=question[2] + PROMPTS[language]['hint_prompt'].format(relation=test_type) + PROMPTS[language]['template']
                )
            else:
                answer3 = conversation.predict(input=question[2] + PROMPTS[language]['template'])
        except:
            logger.error(f"Error processing question {index}: {question}")
            continue
        logger.info(f"\nOriginal answers: {answer1}, {answer2}, {answer3}")

        # Convert
        answer1 = utils.convert_response_to_set(answer1)
        answer2 = utils.convert_response_to_set(answer2)
        answer3 = utils.convert_response_to_set(answer3)

        # Save
        answers_ql1[str(index)] = list(answer1)
        answers_ql2[str(index)] = list(answer2)
        answers_ql3[str(index)] = list(answer3)

        # Write after each question
        with open(q1_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql1, f, ensure_ascii=False, indent=4)
        with open(q2_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql2, f, ensure_ascii=False, indent=4)
        with open(q3_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql3, f, ensure_ascii=False, indent=4)

        logger.info(f"\nIndex: {index} Question 1: {question[0]} Question 2: {question[1]} Question 3: {question[2]}")
        logger.info(f"Answer 1: {answer1} Answer 2: {answer2} Relation: {relation_predicted} Answer 3: {answer3}\n")
        time.sleep(1.5)

if __name__ == "__main__":

    llm_models = ['gpt-4o',"o3"]
    languages = ['en']
    datasets = ['spinach.tsv','qawiki.tsv','synthetic.tsv']
    # datasets = ['qawiki.tsv']
    relations = ['Equivalence','Minus','Containment']
    for language in languages:
        for llm_model in llm_models:
            for dataset in datasets:
                for relation in relations:
                    logger.info(f"Processing model: {llm_model} for relation: {relation} in language: {language}")
                    if relation == 'Minus' or relation == 'Resta':
                        run_minus_benchmark(llm_model, language, relation, dataset)
                    else:
                        run_benchmark(llm_model, language, relation, dataset)   
                    
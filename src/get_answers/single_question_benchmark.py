import os
import csv
import json
from langchain_core.prompts import PromptTemplate
from llms import PromptLLMS
import utils 
import yaml

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

def process_question(question, llm_model, prompt_template, language):
    try:
        llms = PromptLLMS(model=llm_model, prompt_template=prompt_template, question=question)
        response = llms.execute_single_question()
        if language == 'en':
            return utils.convert_response_to_set(response)
        else:
            return utils.convert_response_to_set_es(response)
    except ValueError as e:
        print(f"⚠️ Azure content filter triggered for question: {question}")
        print(f"Error: {e}")
        return None  # fallback response set


def save_answers(answers, dataset, column, language, prompt_type, llm_model):
    lang_prefix = '' if language == 'en' else '*'
    relation = LOGICAL_RELATIONS_MAP[column]
    suffix = f"_answers_{'wikidata_' if prompt_type == 'wikidata' else ''}{llm_model}.json"
    out_file = root_dir + f'/data/answers/{dataset.split(".")[0]}/zero-shot/{relation}/{lang_prefix}{column}_{relation}{suffix}'
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
    print(f"Answers saved to {out_file}")

def load_answers(dataset, column, language, prompt_type, llm_model):
    lang_prefix = '' if language == 'en' else '*'
    relation = LOGICAL_RELATIONS_MAP[column]
    suffix = f"_answers_{'wikidata_' if prompt_type == 'wikidata' else ''}{llm_model}.json"
    in_file = root_dir + f'/data/answers/{dataset.split(".")[0]}/zero-shot/{relation}/{lang_prefix}{column}_{relation}{suffix}'

    if os.path.exists(in_file):
        with open(in_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}



# === benchmark ===

def run_benchmark(prompt_type='standard'):
    for language in LANGUAGES:
        for llm_model in LLM_MODELS:
            for dataset in DATASETS:
                print(f"Processing dataset: {dataset} for model: {llm_model} and language: {language}")
                tsv_file = os.path.join(root_dir, f'data/Dataset/{language}/{dataset}')
                
                for column in COLUMNS_MAP[dataset]:
                    print(f"Processing column: {column}")
                    questions = load_questions(tsv_file, column)
                    prompt_template = get_prompt(prompt_type, language)

                    # Load previous answers if available
                    answers = load_answers(dataset, column, language, prompt_type, llm_model)

                    for index, question in enumerate(questions):
                        if str(index) in answers:
                            continue  # Skip already processed

                        response_set = process_question(question, llm_model, prompt_template, language)
                        if response_set is None:
                            print(f"Skipping question {index + 1} due to content filter.")
                            continue

                        answers[str(index)] = response_set

                        print(f"Question {index + 1}: {question}")
                        print(f"LLM Response: {response_set}")
                        
                        if index % 10 == 0:
                            save_answers(answers, dataset, column, language, prompt_type, llm_model)

                    save_answers(answers, dataset, column, language, prompt_type, llm_model)

if __name__ == "__main__":

    COLUMNS_MAP = {'spinach.tsv': ['Q1','Q2', 'Q3', 'Q4']}
    LOGICAL_RELATIONS_MAP = {'Q1': 'equal', 'Q2': 'equal', 'Q3': 'sup-sub', 'Q4': 'minus'}
    LANGUAGES = ['en']
    LLM_MODELS = ['o3']
    DATASETS = ['spinach.tsv']
    # Run the benchmark with the standard prompt type
    # run_benchmark(prompt_type='standard')
    # Run the benchmark with the wikidata prompt type
    run_benchmark(prompt_type='wikidata')

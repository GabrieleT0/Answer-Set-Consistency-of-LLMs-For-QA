from langchain_core.prompts import PromptTemplate
from prompt_llms import *
import os
import csv
import utils
import json

here = os.path.dirname(os.path.abspath(__file__))

PROMPTS = {
    "standard": {
        "en": '''{question} \n
                If you can't answer, return 'idk'. \n
                Return me the answer as a complete list separated by commas, don't add any other text.''',
        "es": '''{question} \n
                Si no puedes responder, devuelve "no sé". \n
                Devuélveme solo la relación lógica entre las tres preguntas, no añadas ningún otro texto.'''
    },
    "wikidata": {
        "en": '''{question} \n
            Just use Wikidata as a source to answer my question. \n
            If you can't answer, return 'idk'. \n
            Return me the answer as a list separated by commas, don't add any other text.''',
        "es": '''{question} \n
            Utiliza Wikidata como fuente para responder a mi pregunta. \n
            Si no puedes responder, devuelve "no sé". \n
            Devuélveme solo la relación lógica entre las tres preguntas; no añadas ningún otro texto.'''
    }
}

columns_map = {
    'equal-wiki.tsv': ['ql1', 'ql2'],
#    'subsetOf-wiki.tsv': ['ql1', 'ql2'],
#    'minus-set.tsv': ['ql3']
}

dataset_map = {
    'equal-wiki.tsv': 'equal',
    'subsetOf-wiki.tsv': 'sup-sub',
    'minus-set.tsv': 'minus'
}

languages = ['en']
llm_models = [CustomServerLLM(PROMPTS["standard"]["en"], "llama3.1:8b")]
datasets_to_process = list(dataset_map.keys())


def run_benchmark(prompt_type='standard'):
        for llm_model in llm_models:
            for dataset in datasets_to_process:
                print(f"Processing dataset: {dataset} for model: {llm_model.__class__.__name__}")
                tsv_file = os.path.join(here, f'../data/Dataset/en/{dataset}')
                for column in columns_map[dataset]:
                    print(f"Processing column: {column}")
                    questions = []
                    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
                        reader = csv.DictReader(tsvfile, delimiter='\t')
                        for row in reader:
                            questions.append(row[column])

                    answers = {}
                    for index, question in enumerate(questions):
                        llm_response = llm_model.query(question=question)
                        converted_response = utils.convert_response_to_set_es(llm_response)
                        answers[index] = converted_response

                        print(f"Question {index + 1}: {question}")
                        print(f"LLM Response: {llm_response}")

                    suffix = f"_answers_{'wikidata_' if prompt_type == 'wikidata' else ''}{llm_model.__class__.__name__}.json"
                    output_filename = os.path.join(
                        here,
                        f'../data/answers/zero-shot/{dataset_map[dataset]}/{column}_{dataset_map[dataset]}{suffix}'
                    )
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(answers, f, ensure_ascii=False, indent=4)

# Run standard benchmark
run_benchmark(prompt_type='standard')

# Run benchmark by using only Wikidata as a source
run_benchmark(prompt_type='wikidata')

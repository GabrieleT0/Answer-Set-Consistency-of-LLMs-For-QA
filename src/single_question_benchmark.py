from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
import os
import csv
import utils
import json

here = os.path.dirname(os.path.abspath(__file__))

PROMPTS = {
    "standard": {
        "en": '''{question} \n
                If you can't answer, return 'idk'. \n
                Return me all answers as a list separated by the symbol '|' don't add any other text.''',
        "es": '''{question} \n
                Si no puedes responder, devuelve "no sé". \n
                Devuélveme la respuesta en forma de lista separada por el símbolo '|' no añadas ningún otro texto.'''
    },
    "wikidata": {
        "en": '''{question} \n
            Just use Wikidata as a source to answer my question. \n
            If you can't answer, return 'idk'. \n
            Return me all answers as a list separated by '|' don't add any other text.''',
        "es": '''{question} \n
            Utiliza Wikidata como fuente para responder a mi pregunta. \n
            Si no puedes responder, devuelve "no sé". \n
            Devuélveme la respuesta en forma de lista separada por el símbolo '|' no añadas ningún otro texto.'''
    }
}


columns_map = {
    'spinach.tsv': ['Q1', 'Q2','Q3','Q4'],
}
languages = ['en']
llm_models = ['gpt-4.1-nano-2025-04-14', 'gpt-4.1-mini-2025-04-14', 'gpt-4.1-2025-04-14']
columns_map = {
    'spinach.tsv': ['Q1', 'Q2','Q3','Q4'],
}
logical_relations_map = {
    'Q1': 'equal',
    'Q2': 'equal',
    'Q3': 'sup-sub',
    'Q4': 'minus',
}
languages = ['en']
llm_models = ['gpt-4.1-2025-04-14']
datasets = ['spinach.tsv']


def run_benchmark(prompt_type='standard'):
    for language in languages:
        for llm_model in llm_models:
            for dataset in datasets:
                print(f"Processing dataset: {dataset} for model: {llm_model} and language: {language}")
                tsv_file = os.path.join(here, f'../data/Dataset/{language}/{dataset}')
                for column in columns_map[dataset]:
                    print(f"Processing column: {column}")
                    questions = []
                    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
                        reader = csv.DictReader(tsvfile, delimiter='\t')
                        for row in reader:
                            questions.append(row[column])

                    answers = {}
                    for index, question in enumerate(questions):
                        prompt = PromptTemplate(
                            input_variables=["question"],
                            template=PROMPTS[prompt_type][language]
                        )
                        llms = PromptLLMS(prompt, question)
                        llm_response = (
                            llms.execute_on_gemini(model=llm_model)
                            if 'gemini' in llm_model
                            else llms.execute_on_openAI_model(openAI_model=llm_model)
                        )
                        if language == 'en':
                            converted_response = utils.convert_response_to_set(llm_response)
                        else:
                            converted_response = utils.convert_response_to_set_es(llm_response)
                        answers[index] = converted_response

                        print(f"Question {index + 1}: {question}")
                        print(f"LLM Response: {llm_response}")

                    lang_prefix = '' if language == 'en' else '*'
                    suffix = f"_answers_{'wikidata_' if prompt_type == 'wikidata' else ''}{llm_model}.json"
                    output_filename = os.path.join(
                        here,
                        f'../data/answers/{dataset.split(".")[0]}/zero-shot/{logical_relations_map[column]}/{lang_prefix}{column}_{logical_relations_map[column]}{suffix}'
                    )
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(answers, f, ensure_ascii=False, indent=4)

# Run standard benchmark
run_benchmark(prompt_type='standard')

# Run benchmark by using only Wikidata as a source
run_benchmark(prompt_type='wikidata')
from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
import os
import csv
import utils
import json
from SPARQLWrapper import SPARQLWrapper, JSON

here = os.path.dirname(os.path.abspath(__file__))
endpoint_url = "https://query.wikidata.org/sparql"
PROMPTS = {
    "en": '''{question} \n
            Give me the SPARQL query to run on Wikidata to answer my question. Use the ?sbj variable in the query to retrieve only the triple's subject.\n
            If you can't answer, return 'idk'. \n
            Return me only the SPARQL query; do not add any other text.''',
    "es": '''{question} \n
            Dame la query SPARQL a ejecutar en Wikidata para responder a mi pregunta. Utilice la variable ?sbj en la query para recuperar sólo el asunto de la triple." \n
            Si no puedes responder, devuelve 'no sé'. \n
            Devuélveme sólo la query SPARQL; no añadas ningún otro texto.'''
}

columns_map = {
    'equal-wiki.tsv': ['ql1', 'ql2'],
    'subsetOf-wiki.tsv': ['ql1', 'ql2'],
    'minus-set.tsv': ['ql3']
}

dataset_map = {
    'equal-wiki.tsv': 'equal',
    'subsetOf-wiki.tsv': 'sup-sub',
    'minus-set.tsv': 'minus'
}

languages = ['en', 'es']
llm_models = ['gpt-4.1-nano-2025-04-14', 'gpt-4.1-mini-2025-04-14', 'gemini-2.5-pro']
datasets_to_process = list(dataset_map.keys())


def run_benchmark():
    for language in languages:
        for llm_model in llm_models:
            for dataset in datasets_to_process:
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
                            template=PROMPTS[language]
                        )
                        llms = PromptLLMS(prompt, question)
                        llm_response = (
                            llms.execute_on_gemini(model=llm_model)
                            if 'gemini' in llm_model
                            else llms.execute_on_openAI_model(openAI_model=llm_model)
                        )
                        query = utils.convert_response_to_set_es(llm_response)
                        sparql = SPARQLWrapper(endpoint_url)
                        sparql.setQuery(query)
                        sparql.setReturnFormat(JSON)
                        results = sparql.query().convert()
                        results_list = []
                        for result in results["results"]["bindings"]:
                            results_list.append(result["sbj"]["value"])
                        answers[index] = results_list

                        print(f"Question {index + 1}: {question}")
                        print(f"LLM Response: {query}")
                        print(f"Query results on wikidata: {results_list}")

                    lang_prefix = '' if language == 'en' else '*'
                    output_filename = os.path.join(
                        here,
                        f'../data/answers/sparql_query/{dataset_map[dataset]}/{lang_prefix}{column}_{dataset_map[dataset]}_answers_sparql_{llm_model}.json'
                    )
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(answers, f, ensure_ascii=False, indent=4)

# Run standard benchmark
run_benchmark()

from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
import os
import csv
import utils
import json
from dotenv import load_dotenv
load_dotenv()
here = os.path.dirname(os.path.abspath(__file__))
openai_api_key = os.getenv('OPENAI_API_KEY')

PROMPTS = {
    "standard": {
        "en": '''I ask you a question with a list of values as answer, return me the answer as a JSON object containing the list of values in the following format:
                {
                    "answer": [answer1, answer2, ...]
                }
                If you can't answer, return the json object in the following format:
                {
                    "answer": 'idk'
                }
                If the answer is an empty list, return the json object in the following format:
                {
                    "answer": []
                }
                In the response, do not use abbreviations or acronyms, but spell out the full terms, i.e. "United States of America" instead of "USA".
                If the response contains numbers or digits, use Arabic numerals. For example, if the answer contains Star Wars V, indicate it with Star Wars 5. Do not use Roman numerals (such as V) or text (such as five).
                Return an exhaustive list.
                ''',
        "es": '''Te haré una pregunta con una lista de valores como respuesta. Devuélveme la respuesta como un objeto JSON que contenga la lista de valores en el siguiente formato:
                {
                    "answer": [respuesta1, respuesta2, ...]
                }
                Si no puedes responder, devuelve el objeto JSON en el siguiente formato:
                {
                    "answer": 'no sé'
                }'''
    },
    "wikidata": {
        "en": '''{question} \n
        I ask you a question with a list of values as answer just use Wikidata as a source to answer my question. Return me the answer as a JSON object containing the list of values in the following format:
                {
                    "answer": [answer1, answer2, ...]
                }
                If you can't answer, return the json object in the following format:
                {
                    "answer": 'idk'
        }
         In the response, do not use abbreviations or acronyms, but spell out the full terms, i.e. "United States of America" instead of "USA".
        ''',
        "es": '''Te haré una pregunta con una lista de valores como respuesta Utiliza Wikidata como fuente para responder a mi pregunta. Devuélveme la respuesta como un objeto JSON que contenga la lista de valores en el siguiente formato:
                {
                    "answer": [respuesta1, respuesta2, ...]
                }
                Si no puedes responder, devuelve el objeto JSON en el siguiente formato:
                {
                    "answer": 'no sé'
                }'''
    }
}

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


def submit_job(language, model, logical_relation, column,type,dataset):
    client = OpenAI(api_key=openai_api_key)
    question_pairs = []
    tsv_file_input = os.path.join(here, f'../data/Dataset/en/{dataset}')
    with open(tsv_file_input, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row[column]))
    tasks = []

    for index, question in enumerate(question_pairs):
        question = f"Question: {question}"
        task = {
            "custom_id": f"task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": 0.1,
                "response_format": { 
                    "type": "json_object"
                },
                "messages": [
                    {
                        "role": "system",
                        "content": PROMPTS[type][language]
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
            }
        }
        tasks.append(task)

    with open(os.path.join(here,f'../data/batch_api/gpt_tasks_{logical_relation}_{model}_{column}_{type}.jsonl'), 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')
    
    batch_file = client.files.create(
        file=open(os.path.join(here,f'../data/batch_api/gpt_tasks_{logical_relation}_{model}_{column}_{type}.jsonl'), "rb"),
        purpose="batch"
    )

    print(batch_file)
    
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    batch_job = client.batches.retrieve(batch_job.id)
    print(batch_job)

    return batch_job.id

def convert_response(job_id,result_path,language, model, logical_relation, column,type):
    client = OpenAI(api_key=openai_api_key)
    batch_job = client.batches.retrieve(job_id)
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content

    result_file_name = os.path.join(here,result_path)
    with open(result_file_name, 'wb') as file:
        file.write(result)

    results = []
    with open(result_file_name, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)
    
    json_results = []
    for result in results:
        json_response = result['response']['body']['choices'][0]['message']['content']
        print(json_response)
        json_results.append((result['custom_id'],json.loads(json_response)))
    
    with open(os.path.join(here,f'../data/batch_api/gpt_results_{language}_{model}_{logical_relation}_{column}_{type}.json'), "w", encoding="utf-8") as file:
        json.dump(json_results, file)
    
    output_json = {}
    for result in json_results:
        if result[1]['answer'] == 'idk' or result[1]['answer'] == 'no sé':
            output_json[result[0].split('-')[1]] = []
        else:
            output_json[result[0].split('-')[1]] = result[1]['answer']

    if language == 'en':
        asterix = ''
    else:
        asterix = '*'

    construct_output_path = os.path.join(here, f'../data/answers/zero-shot/{dataset.split(".")[0]}/{logical_relation}/{asterix}{column}_{logical_relation}_answers_{model}_{type}_{dataset.split(".")[0]}.json')
    with open(construct_output_path, "w", encoding="utf-8") as file:
        json.dump(output_json, file, indent=4, ensure_ascii=False)


# for language in languages:
#     for model in llm_models:
#         for dataset in datasets:
#             for column in columns_map[dataset]:
#                 type = 'wikidata'
#                 job_id = submit_job(language, model, logical_relations_map[column], column, type, dataset)
#                 print(f"Submitted job {job_id} for {language}, {model}, { logical_relations_map[column]}, {column}, {type}, {dataset}")                 

batch_map = {
    #'Q1': 'batch_687a128978a081908531fdc4438e7aa3',
    #'Q2': 'batch_687a128b78848190b51c50cddf539a96',
    'Q3': 'batch_687a128d11d48190980875781c50bd9c',
    #'Q4': 'batch_687a128e9c548190976464ed6f3e0da6'
}

for language in languages:
    for model in llm_models:
        for dataset in datasets:
            for column in columns_map[dataset]: 
                type = 'standard'           
                result_path = f'../data/batch_api/batch_job_results_{language}_{model}_{logical_relations_map[column]}_{column}_{type}_{dataset}.jsonl'
                try:
                    convert_response(batch_map[column], result_path, language, model, logical_relations_map[column], column, type)
                except Exception as e:
                    print(f"Error processing {column} for {language}, {model}, {dataset}: {e}")
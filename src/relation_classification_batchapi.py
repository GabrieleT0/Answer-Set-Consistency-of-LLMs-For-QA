import os
import csv
import json
from dotenv import load_dotenv
import prompt_llms
from openai import OpenAI

here = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

PROMPTS_MINUS = {
    'en': '''
            I prompt you with three questions q1, q2, q3. You need to identify which of the following logical relations holds between the sets of answers for q1, q2 and q3:

            - Equivalence  
            - Containment  
            - Disjoint  
            - Overlap  
            - Minus  
            - Unknown
            Return only the logical relation between the three questions. Return only the first relation that holds.
            Return the response as a JSON object in the following format:
            {
                "answer": "relation"
            }
            If you can't answer, return the json object in the following format:
            {
                "answer": 'idk'
            }
        ''',
    'es': '''
            Te planteo tres preguntas, q1, q2 y q3. Debes identificar cuál de las siguientes relaciones lógicas se cumple entre los conjuntos de respuestas de q1, q2, q3:

            - Equivalencia
            - Contención
            - Disjunto
            - Solapamiento
            - Resta
            - Desconocido

            Estas son las tres preguntas:

            q1: {q1}
            q2: {q2}
            q3: {q3}

            Devuélveme solo la relación lógica entre las tres preguntas. Devuélveme solo la primera relación que se cumple. No añadas ningún otro texto.
            Devuélveme la respuesta como un objeto JSON en el siguiente formato:
            {
                "answer": "relación"
            }
            Si no puedes responder, devuelve el objeto JSON en el siguiente formato:
            {
                "answer": 'no sé'
            }
        '''
    }

PROMPTS = {
    'en': '''
            I prompt you with two questions q1, q2. You need to identify which of the following logical relations holds between the sets of answers for q1 and q2:

            - Equivalence  
            - Containment  
            - Disjoint  
            - Overlap  
            - Complement  
            - Unknown

            Return only the logical relation between the two questions. Return only the first relation that holds. Do not include any additional explanation.
            Return the response as a JSON object in the following format:
            {
                "answer": "relation"
            }
            If you can't answer, return the json object in the following format:
            {
                "answer": 'idk'
            }
        ''',
    'es': '''
            Te planteo dos preguntas, q1, q2. Debes identificar cuál de las siguientes relaciones lógicas se cumple entre los conjuntos de respuestas de q1 y q2:

            - Equivalencia
            - Contención
            - Disjunto
            - Solapamiento
            - Complemento
            - Desconocido

            Devuélveme solo la relación lógica entre las dos preguntas. Devuélveme solo la primera relación que se cumple. No añadas ningún otro texto.
            Si no puedes responder, devuelve el objeto JSON en el siguiente formato:
            {
                "answer": 'no sé'
            }
        '''
    }

llm_models = ['gpt-4.1-mini-2025-04-14','gpt-4.1-2025-04-14']
languages = ['en']
real_relations_en = {'equal-wiki.tsv' : 'Equivalence', 'subsetOf-wiki.tsv': 'Containment', 'minus-set.tsv': 'Minus'}
real_relations_es = {'equal-wiki.tsv' : 'Equivalencia', 'subsetOf-wiki.tsv': 'Contención', 'minus-set.tsv': 'Resta'}
dataset_map = {
    'equal' : 'equal-wiki.tsv',
    'sup-sub' : 'subsetOf-wiki.tsv',
    'minus' : 'minus-set.tsv'
}
output_map = {
    'equal' : 'Equivalence',
    'sup-sub' : 'Containment',
    'minus' : 'minus'
}
datasets = ['qawiki.tsv']

def submit_equivalence_job(language, model,logical_relation, dataset):
    client = OpenAI(api_key=openai_api_key)
    question_pairs = []
    tsv_file_input = os.path.join(here, f'../data/Dataset/en/{dataset}')
    with open(tsv_file_input, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['Q1'], row['Q2']))
    
    tasks = []
    for index, question in enumerate(question_pairs):
        question = f"q1: {question[0]}\nq2: {question[1]}"
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
                        "content": PROMPTS[language]
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
            }
        }
        tasks.append(task)
    with open(os.path.join(here,f'../data/batch_api/gpt_tasks_{logical_relation}_{model}_classification.jsonl'), 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')
    
    batch_file = client.files.create(
        file=open(os.path.join(here,f'../data/batch_api/gpt_tasks_{logical_relation}_{model}_classification.jsonl'), "rb"),
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

def submit_containment_job(language, model,logical_relation, dataset):
    client = OpenAI(api_key=openai_api_key)
    question_pairs = []
    tsv_file_input = os.path.join(here, f'../data/Dataset/en/{dataset}')
    with open(tsv_file_input, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['Q1'], row['Q3']))
    
    tasks = []
    for index, question in enumerate(question_pairs):
        question = f"q1: {question[0]}\nq2: {question[1]}"
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
                        "content": PROMPTS[language]
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
            }
        }
        tasks.append(task)
    with open(os.path.join(here,f'../data/batch_api/gpt_tasks_{logical_relation}_{model}_classification.jsonl'), 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')
    
    batch_file = client.files.create(
        file=open(os.path.join(here,f'../data/batch_api/gpt_tasks_{logical_relation}_{model}_classification.jsonl'), "rb"),
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

def submit_minus_job(language, model,dataset,logical_relation='minus'):
    client = OpenAI(api_key=openai_api_key)
    question_pairs = []
    tsv_file_input = os.path.join(here, f'../data/Dataset/en/{dataset}')
    with open(tsv_file_input, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['Q1'], row['Q3'], row['Q4']))
    
    tasks = []
    for index, question in enumerate(question_pairs):
        question = f"q1: {question[0]}\nq2: {question[1]}\nq3: {question[2]}"
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
                        "content": PROMPTS_MINUS[language]
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
            }
        }
        tasks.append(task)
    with open(os.path.join(here,f'../data/batch_api/gpt_tasks_{logical_relation}_{model}_classification.jsonl'), 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')
    
    batch_file = client.files.create(
        file=open(os.path.join(here,f'../data/batch_api/gpt_tasks_{logical_relation}_{model}_classification.jsonl'), "rb"),
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



def convert_response(job_id,result_path,language, model, dataset, real_relation):
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
        json_results.append((result['custom_id'],json.loads(json_response)))
    
    with open(os.path.join(here,f'../data/batch_api/gpt_results_classification_{language}_{model}_{real_relation}.json'), "w", encoding="utf-8") as file:
        json.dump(json_results, file)
    
    output_json = {}

    for result in json_results:
        if result[1]['answer'] == 'idk' or result[1]['answer'] == 'no sé':
            output_json[result[0].split('-')[1]] = 0
        elif result[1]['answer'].lower().strip() == real_relation.lower().strip():
            output_json[result[0].split('-')[1]] = 1
        else:
            output_json[result[0].split('-')[1]] = 0

    if language == 'en':
        asterix = ''
    else:
        asterix = '*'

    construct_output_path = os.path.join(here, f'../data/answers/zero-shot/{dataset.split(".")[0]}/relation-classification/{asterix}{real_relation}_{model}_{dataset}.json')
    with open(construct_output_path, "w", encoding="utf-8") as file:
        json.dump(output_json, file, indent=4, ensure_ascii=False)

for language in languages:
    for llm_model in llm_models:
        for dataset in datasets:
            print(f"Processing {language} with model {llm_model}")
            # Run equivalence test
            #print(f"Submitting equivalence job for {language} with model {llm_model}")
            #job_id = submit_equvalence_job(language, llm_model, 'equal', dataset)
            #convert_response(job_id, f'../data/batch_api/gpt_results_classification_{language}_{llm_model}_equal.json', language, llm_model, 'equal')
            # Run minus test
            #print(f"Submitting minus job for {language} with model {llm_model}")
            #job_id = submit_minus_job(language, llm_model, dataset, 'minus')
            #convert_response(job_id, f'../data/batch_api/gpt_results_classification_{language}_{llm_model}_minus.json', language, llm_model, 'minus')
            # Run containment test
            #print(f"Submitting containment job for {language} with model {llm_model}")
            #job_id = submit_containment_job(language, llm_model, 'containment', dataset)
            #convert_response(job_id, f'../data/batch_api/gpt_results_classification_{language}_{llm_model}_sup-sub.json', language, llm_model, 'sup-sub')

batch = 'batch_6884ca81c42c8190a629eb8076e892e1'
language = 'en'
llm_models = ['gpt-4.1-2025-04-14']
real_relation = 'Minus'  # Options:
# - Equivalence
# - Containment
# - Minus  
#convert_response(batch, gpt_results, language, model, 'spinach.tsv',real_relation)
for model in llm_models:
    gpt_results = f'../data/batch_api/gpt_results_classification_en_{model}_{real_relation}.json'
    #submit_equivalence_job(language, model, 'equal', 'qawiki.tsv')
    #submit_containment_job(language, model, 'containment', 'qawiki.tsv')
    #submit_minus_job(language, model, 'qawiki.tsv', 'minus')
model = 'gpt-4.1-2025-04-14'
convert_response(batch, gpt_results, language, model, 'qawiki.tsv',real_relation)
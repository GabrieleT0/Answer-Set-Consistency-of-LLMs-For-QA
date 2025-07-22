from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import csv
import utils
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

here = os.path.dirname(os.path.abspath(__file__))

def minus_syntetic_questions_batch():
        """
        Creates a batch job using the OpenAI API to filter the LOD cloud data using the GPT-4o mini model.
        """
        client = OpenAI(api_key=openai_api_key)
        question_prompt = '''
            I prompt to you two questions: question A and question B. A containment relationship hold between the answer of these question, specifically, the responses at the question B is contained in the responses at the question A.\
            Your task now it to generate a question C in a way in which the relationship between the resonses at the three qustions is A-B=C. 
            The answer to the question C must be a list of values, not an ordered list, not a paragraph of text, not a boolean value and not a single number.
            The answer to the questions must be available in Wikidata, and give me also the SPARQL query to retrieve the answer.
            This is an example: question A) Which movies star Uma Thurman? question B) Which science fiction movies star Uma Thurman? question C) Which movies star Uma Thurman excluding those science fiction movies star Uma Thurman?
            Give me only the question C formatted as a json object containing the following information:
            {
                "question": <Question C>
                "sparql_query": <SPARQL query to retrieve the answer to Question C from Wikidata>
            }
        '''
        question_pairs = []
        tsv_file_input = os.path.join(here, f'../data/Dataset/en/filtered_sparql_sup-sub_syntetic.tsv')
        with open(tsv_file_input, newline='', encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for row in reader:
                question_pairs.append((row['ql1'], row['ql2']))
        tasks = []

        for index, (q1, q2) in enumerate(question_pairs):
            questions = f"Question A: {q2}\nQuestion B: {q1}"
            task = {
                "custom_id": f"task-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4.1-mini-2025-04-14",
                    "temperature": 0.7,
                    "response_format": { 
                        "type": "json_object"
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": question_prompt
                        },
                        {
                            "role": "user",
                            "content": questions
                        }
                    ],
                }
            }
            tasks.append(task)
       
        with open(os.path.join(here,'../data/batch_api/gpt_tasks.jsonl'), 'w') as file:
            for obj in tasks:
                file.write(json.dumps(obj) + '\n')
        
        batch_file = client.files.create(
            file=open(os.path.join(here,'../data/batch_api/gpt_tasks.jsonl'), "rb"),
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

def retrieve_and_save_job_result(job_id):
    """
    Retrieves the results of a batch job created with the OpenAI API and saves them to a local JSON file.
    """
    client = OpenAI(api_key=openai_api_key)
    batch_job = client.batches.retrieve(job_id)
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content

    result_file_name = os.path.join(here,'../data/batch_api/batch_job_results.jsonl')
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
        json_results.append(json.loads(json_response))
    
    create_tsv_from_batch_results(json_results)
    
    with open(os.path.join(here,'../data/batch_api/gpt_results.json'), "w", encoding="utf-8") as file:
        json.dump(json_results, file)

def create_tsv_from_batch_results(json_results):
    """
    Creates a TSV file from the results of a batch job.
    """
    question_pairs = []
    tsv_file_input = os.path.join(here, f'../data/Dataset/en/filtered_sparql_sup-sub_syntetic.tsv')
    with open(tsv_file_input, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for index, row in enumerate(reader):
            print(row['ql1'], row['ql2'], json_results[index]['question'], json_results[index]['sparql_query'])
            question_pairs.append((row['ql1'], row['ql2'], json_results[index]['question'], json_results[index]['sparql_query']))

    tsv_output = os.path.join(here, f'../data/Dataset/en/minus_syntetic.tsv')
    file_exists = os.path.exists(tsv_output)
    is_empty = not file_exists or os.stat(tsv_output).st_size == 0
    with open(tsv_output, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        if is_empty:
            writer.writerow(["ql1", "ql2", "ql3", "sparql_ql3"])

        writer.writerows(question_pairs)
    

#minus_syntetic_questions_batch()
retrieve_and_save_job_result('batch_6877b2f016a48190b2ddb76c871812ef')
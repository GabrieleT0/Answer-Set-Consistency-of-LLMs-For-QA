from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
import os
import csv
import utils
import json

# Define editable prompt template
prompt_template_text = ''' {question} \n
Si no puedes responder, devuelve "no sé". \n
Devuélveme solo la relación lógica entre las tres preguntas; no añadas ningún otro texto.
'''

# File and config paths
here = os.path.dirname(os.path.abspath(__file__))
column_name = 'ql3'
input_filename = 'minus-set.tsv'
tsv_file = os.path.join(here, f'../data/Dataset/es/{input_filename}')
llm_model = 'gpt-4.1-nano-2025-04-14'
output_filename = os.path.join(here, f'../data/answers/*{column_name}_minus_answers_wikidata_' + llm_model + '.json')

# Read questions
questions = []
with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in reader:
        questions.append(row[column_name])

# Run prompts
answers = {}
for index, question in enumerate(questions):
    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template_text
    )
    llms = PromptLLMS(prompt, question)
    llm_response = llms.execute_on_openAI_model(openAI_model=llm_model)
    converted_response = utils.convert_response_to_set_es(llm_response)
    answers[index] = converted_response

# Save output
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(answers, f, ensure_ascii=False, indent=4)
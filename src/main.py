from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
import os
import csv
import utils
import json
here = os.path.dirname(os.path.abspath(__file__))

column_name = 'ql1'

input_filename = 'equal-wiki.tsv'
tsv_file = os.path.join(here,f'../data/Dataset/{input_filename}')
llm_model = 'gpt-4.1-nano'
output_filename = os.path.join(here,f'../data/answers/{column_name}_equal_answers_' + llm_model + '.json')

questions = []
with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in reader:
        questions.append(row[column_name])

answers = {}
for index, question in enumerate(questions):
    prompt = PromptTemplate(
            input_variables=["question"],
            template='''{question} \n
            Return me the answer as a list separated by commas, don't add any other text.'''
        )
    llms = PromptLLMS(prompt,question)
    gemini_response = llms.execute_on_openAI_model(openAI_model='gpt-4.1-nano')
    converted_response = utils.convert_response_to_set(gemini_response)
    answers[index] = converted_response



with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(answers, f, ensure_ascii=False, indent=4)
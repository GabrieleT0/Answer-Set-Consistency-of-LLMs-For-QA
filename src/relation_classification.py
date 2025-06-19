from langchain_core.prompts import PromptTemplate
from prompt_llms_class import PromptLLMS
import os
import csv
import utils
import json
here = os.path.dirname(os.path.abspath(__file__))

column_name = 'ql1'
column_name = 'ql2'
column_name = 'ql3'
real_relation = 'Minus'

input_filename = 'minus-set.tsv'
tsv_file = os.path.join(here,f'../data/Dataset/{input_filename}')
llm_model = 'gpt-4.1-nano-2025-04-14'
output_filename = os.path.join(here,f'../data/answers/relation-classification/{real_relation}_{llm_model}.json')

question_pairs = []
with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter=';')
    for row in reader:
        question_pairs.append((row['ql1'], row['ql2'], row['ql3']))

answers = {}
for index, (q1, q2, q3) in enumerate(question_pairs):
    prompt = PromptTemplate(
            input_variables=["q1","q2","q3"],
            template='''
            I prompt you three questions q1, q2, q3 you need to indetify the logical relation of the concept between q1-q2 and q3 \n
             - Equivalence \n
             - Containment \n
             - Disjoint \n
             - Overlap \n
             - Minus \n
             - Unknown \n
            These are the three questions: \n
            q1: {q1} \n
            q2: {q2} \n
            q3: {q3} \n
            Return me only the logical relation between the three questions, don't add any other text. \n '''
        )
    llms = PromptLLMS(prompt,q1,q2,q3)
    llm_response = llms.execute_on_openAI_model_class(openAI_model=llm_model)
    converted_response = utils.convert_response_to_set_class(llm_response,real_relation)
    answers[index] = converted_response

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(answers, f, ensure_ascii=False, indent=4)
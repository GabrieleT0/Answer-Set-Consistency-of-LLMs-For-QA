from langchain_core.prompts import PromptTemplate
from prompt_llms_class import PromptLLMS
import os
import csv
import utils
import json

# Set up paths
here = os.path.dirname(os.path.abspath(__file__))
input_filename = 'minus-set.tsv'
tsv_file = os.path.join(here, f'../data/Dataset/{input_filename}')
llm_model = 'gpt-4.1-nano-2025-04-14'
real_relation_en = 'Minus'
real_relation_es = 'Resta'
output_filename = os.path.join(here, f'../data/answers/relation-classification/{real_relation}_{llm_model}.json')

en_prompt = '''
I prompt you three questions q1, q2, q3 you need to identify the logical relation of the concept between q1-q2 and q3

 - Equivalence  
 - Containment  
 - Disjoint  
 - Overlap  
 - Minus  
 - Unknown

These are the three questions:

q1: {q1}  
q2: {q2}  
q3: {q3}  

Return only the logical relation between the three questions. Do not include any additional explanation.
'''

es_prompt = '''
            Te planteo tres preguntas, q1, q2 y q3. Debes identificar la relación lógica del concepto entre q1-q2 y q3
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
            Devuélveme solo la relación lógica entre las tres preguntas; no añadas ningún otro texto. '''

# Load question triples
question_pairs = []
with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter=';')
    for row in reader:
        question_pairs.append((row['ql1'], row['ql2'], row['ql3']))

answers = {}
for index, (q1, q2, q3) in enumerate(question_pairs):
    prompt = PromptTemplate(
        input_variables=["q1", "q2", "q3"],
        template=prompt_template_text
    )
    llms = PromptLLMS(prompt, q1, q2, q3)
    llm_response = llms.execute_on_openAI_model_class(openAI_model=llm_model)
    converted_response = utils.convert_response_to_set_class(llm_response, real_relation)
    answers[index] = converted_response

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(answers, f, ensure_ascii=False, indent=4)
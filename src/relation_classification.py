from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
import os
import csv
import utils
import json

here = os.path.dirname(os.path.abspath(__file__))

PROMPTS_MINUS = {
    'en': '''
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
        ''',
    'es': '''
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

            Devuélveme solo la relación lógica entre las tres preguntas; no añadas ningún otro texto.
        '''
    }

PROMPTS = {
    'en': '''
            I prompt you two questions q1 and q2 you need to identify the logical relation of the concept between q1 and q2

            - Equivalence  
            - Containment  
            - Disjoint  
            - Overlap  
            - Minus  
            - Unknown

            These are the two questions:

            q1: {q1}  
            q2: {q2}  

            Return only the logical relation between the three questions. Do not include any additional explanation.
        ''',
    'es': '''
            Te planteo dos preguntas, q1 y q2. Debes identificar la relación lógica del concepto entre q1 y q2

            - Equivalencia
            - Contención
            - Disjunto
            - Solapamiento
            - Resta
            - Desconocido

            Estas son las dos preguntas:

            q1: {q1}
            q2: {q2}

            Devuélveme solo la relación lógica entre las tres preguntas; no añadas ningún otro texto.
        '''
    }

llm_models = ['gpt-4.1-nano-2025-04-14', 'gpt-4.1-mini-2025-04-14','gemini-2.5-pro']
languages = ['en', 'es']
real_relations_en = {'equal-wiki.tsv' : 'Equivalence', 'subsetOf-wiki.tsv': 'Containment', 'minus-set.tsv': 'Minus'}
real_relations_es = {'equal-wiki.tsv' : 'Equivalencia', 'subsetOf-wiki.tsv': 'Contención', 'minus-set.tsv': 'Resta'}

def minus_test(llm_model, language):
    """
    Run the minus test for the given LLM model and language.
    """
    input_filename = 'minus-set.tsv'
    tsv_file = utils.get_dataset_path(input_filename, language)
    prompt = PROMPTS_MINUS[language]

    # Load question triples
    question_pairs = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        if language == 'en':
            reader = csv.DictReader(tsvfile, delimiter=';')
        else:
            reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['ql1'], row['ql2'], row['ql3']))

    answers = {}
    for index, (q1, q2, q3) in enumerate(question_pairs):
        prompt_template = PromptTemplate(
            input_variables=["q1", "q2", "q3"],
            template=prompt
        )
        llms = PromptLLMS(prompt_template, q1, q2, q3)
        if 'gemini' in llm_model:
            llm_response = llms.execute_on_gemini_three_question(llm_model)
        else:
            llm_response = llms.execute_on_openAI_three_questions(llm_model)
        if language == 'en':
            real_relation = real_relations_en[input_filename]
            output_prefix = ''
        else:
            real_relation = real_relations_es[input_filename]
            output_prefix = '*'
        converted_response = utils.convert_response_to_set_class(llm_response, real_relation)

        answers[index] = converted_response

        print(f"Question {index + 1}: {q1} | {q2} | {q3}")
        print(f"LLM Response: {llm_response}")
    output_filename = os.path.join(here, f'../data/answers/zero-shot/relation-classification/{output_prefix}Minus_{llm_model}.json')
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

def equivalence_test(llm_model, language):
    input_filename = 'equal-wiki.tsv'
    tsv_file = utils.get_dataset_path(input_filename, language)
    prompt = PROMPTS[language]

    # Load questions
    question_pairs = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['ql1'], row['ql2']))

    answers = {}

    for index, (q1, q2) in enumerate(question_pairs):
        prompt_template = PromptTemplate(
            input_variables=["q1", "q2"],
            template=prompt
        )
        llms = PromptLLMS(prompt_template, q1, q2)
        if 'gemini' in llm_model:
            llm_response = llms.execute_on_gemini_two_question(llm_model)
        else:
            llm_response = llms.execute_on_openAI_two_qeustions(llm_model)
        if language == 'en':
            real_relation = real_relations_en[input_filename]
            output_prefix = ''
        else:
            real_relation = real_relations_es[input_filename]
            output_prefix = '*'
        converted_response = utils.convert_response_to_set_class(llm_response, real_relation)

        answers[index] = converted_response

        print(f"Question {index + 1}: {q1} | {q2}")
        print(f"LLM Response: {llm_response}")
    output_filename = os.path.join(here, f'../data/answers/zero-shot/relation-classification/{output_prefix}Equivalence_{llm_model}.json')
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

def sup_sub_test(llm_model, language):
    input_filename = 'subsetOf-wiki.tsv'
    tsv_file = utils.get_dataset_path(input_filename, language)
    prompt = PROMPTS[language]

    # Load questions
    question_pairs = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['ql1'], row['ql2']))

    answers = {}
    for index, (q1, q2) in enumerate(question_pairs):
        prompt = PromptTemplate(
            input_variables=["q1", "q2"],
            template=prompt
        )
        llms = PromptLLMS(prompt, q1, q2)
        if 'gemini' in llm_model:
            llm_response = llms.execute_on_gemini_two_question(llm_model)
        else:
            llm_response = llms.execute_on_openAI_two_qeustions(llm_model)
        if language == 'en':
            real_relation = real_relations_en[input_filename]
            output_prefix = ''
        else:
            real_relation = real_relations_es[input_filename]
            output_prefix = '*'
        converted_response = utils.convert_response_to_set_class(llm_response, real_relation)

        answers[index] = converted_response

        print(f"Question {index + 1}: {q1} | {q2}")
        print(f"LLM Response: {llm_response}")
    output_filename = os.path.join(here, f'../data/answers/zero-shot/relation-classification/{output_prefix}Containment_{llm_model}.json')
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

for language in languages:
    for llm_model in llm_models:
        # Run equivalence test
        print(f"Processing model: {llm_model} for language: {language}")
        equivalence_test(llm_model, language)
        print(f"Finished processing model: {llm_model} for language: {language})\n")

        # Run Containment test
        print(f"Processing model: {llm_model} for language: {language}")
        sup_sub_test(llm_model, language)
        print(f"Finished processing model: {llm_model} for language: {language}\n")

        # Run minus test
        print(f"Processing model: {llm_model} for language: {language}")
        minus_test(llm_model, language)
        print(f"Finished processing model: {llm_model} for language: {language}\n")
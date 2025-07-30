from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
import os
import csv
import utils
import json
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import prompt_llms

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

            These are the three questions:

            q1: {q1}  
            q2: {q2}  
            q3: {q3}  

            Return only the logical relation between the three questions. Return only the first relation that holds. Do not include any additional explanation.
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

            These are the two questions:

            q1: {q1}  
            q2: {q2}  

            Return only the logical relation between the two questions. Return only the first relation that holds. Do not include any additional explanation.
        ''',
    'es': '''
            Te planteo dos preguntas, q1, q2. Debes identificar cuál de las siguientes relaciones lógicas se cumple entre los conjuntos de respuestas de q1 y q2:

            - Equivalencia
            - Contención
            - Disjunto
            - Solapamiento
            - Complemento
            - Desconocido

            Estas son las dos preguntas:

            q1: {q1}
            q2: {q2}

            Devuélveme solo la relación lógica entre las dos preguntas. Devuélveme solo la primera relación que se cumple. No añadas ningún otro texto.
        '''
    }

PROMPTS_DIRECTION = {
    "en" : '''Now Answer only 'a' or 'b': a) the results of q1 are contained in q2 b) the results of q2 are contained in q1.
        ''',
    "es" : ''' Ahora Responde solo 'a' o 'b': a) los resultados de q1 están contenidos en q2 b) los resultados de q2 están contenidos en q1. '''
}

llm_models = ['deepseek-chat']
languages = ['en']

def minus_test(llm_model, language, dataset, real_relation):
    """
    Run the minus test for the given LLM model and language.
    """
    tsv_file = utils.get_dataset_path(dataset, language)
    prompt = PROMPTS_MINUS[language]

    # Load question triples
    question_pairs = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['Q1'], row['Q3'], row['Q4']))

    answers = {}
    for index, (q1, q3, q4) in enumerate(question_pairs):
        prompt_template = PromptTemplate(
            input_variables=["q1", "q2", "q3"],
            template=prompt
        )
        llms = PromptLLMS(llm_model,prompt_template, q1, q3, q4)
        llm_response = llms.execute_three_question()
        if language == 'en':
            output_prefix = ''
        else:
            output_prefix = '*'
        converted_response = utils.convert_response_to_set_class(llm_response, real_relation)

        answers[index] = converted_response

        print(f"Question {index + 1}: {q1} | {q3} | {q4}")
        print(f"LLM Response: {llm_response}")
    output_filename = os.path.join(here, f'../data/answers/zero-shot/{dataset.split('.')[0]}/relation-classification/{output_prefix}Minus_{llm_model}.json')
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

def equivalence_test(llm_model, language, dataset, real_relation):
    tsv_file = utils.get_dataset_path(dataset, language)
    prompt = PROMPTS[language]

    # Load questions
    question_pairs = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['Q1'], row['Q2']))

    answers = {}

    for index, (q1, q2) in enumerate(question_pairs):
        prompt_template = PromptTemplate(
            input_variables=["q1", "q2"],
            template=prompt
        )
        llms = PromptLLMS(llm_model, prompt_template, q1, q2)
        llm_response = llms.execute_two_question()
        if language == 'en':
            output_prefix = ''
        else:
            output_prefix = '*'
        converted_response = utils.convert_response_to_set_class(llm_response, real_relation)

        answers[index] = converted_response

        print(f"Question {index + 1}: {q1} | {q2}")
        print(f"LLM Response: {llm_response}")
        
    output_filename = os.path.join(here, f'../data/answers/zero-shot/{dataset.split('.')[0]}/relation-classification/{output_prefix}Equivalence_{llm_model}.json')
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

def sup_sub_test(llm_model, language, dataset, real_relation = 'Containment'):
    chat = prompt_llms.return_chat_model(llm_model)
    tsv_file = utils.get_dataset_path(dataset, language)

    # Load questions
    question_pairs = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            question_pairs.append((row['Q1'], row['Q3']))

    answers = {}
    for index, (q1, q3) in enumerate(question_pairs):
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=chat,
            memory=memory
        )
        relation_predicted = conversation.predict(input=PROMPTS[language].format(q1=q1, q2=q3))
        relation_predicted = relation_predicted.strip().lower()
        print(f"Relation predicted: {relation_predicted}")
        if relation_predicted == 'containment' or relation_predicted == 'contención':
            converted_response = 0.5 # If correctly predict the containment relation, we set the response to 0.5
            direction = conversation.predict(input=PROMPTS_DIRECTION[language])
            if direction.strip().lower() == 'b':
                converted_response += 0.5 # If p
        else:
            converted_response = 0
        if language == 'en':
            output_prefix = ''
        else:
            output_prefix = '*'

        answers[index] = converted_response

        print(f"Question {index + 1}: q1: {q1} | q2: {q3}")
        print(f"LLM relation: {relation_predicted}, LLM direction: {direction.strip().lower()}")

    output_filename = os.path.join(here, f'../data/answers/zero-shot/{dataset.split('.')[0]}/relation-classification/{output_prefix}Containment_{llm_model}.json')

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

dataset = 'qawiki.tsv'

for language in languages:
    for llm_model in llm_models:
        # Run equivalence test
        print(f"Processing model: {llm_model} for language: {language}")
        equivalence_test(llm_model, language, dataset, 'Equivalence')
        print(f"Finished processing model: {llm_model} for language: {language})\n")

        # Run Containment test
        print(f"Processing model: {llm_model} for language: {language}")
        #sup_sub_test(llm_model, language, dataset, 'Containment')
        print(f"Finished processing model: {llm_model} for language: {language}\n")

        # Run minus test
        print(f"Processing model: {llm_model} for language: {language}")
        minus_test(llm_model, language, dataset, 'Minus')
        print(f"Finished processing model: {llm_model} for language: {language}\n")
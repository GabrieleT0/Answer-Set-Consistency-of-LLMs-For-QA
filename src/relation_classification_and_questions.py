from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import csv
import utils
import json

here = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

PROMPTS = {
    'en': {
        'template': """\nIf you cannot answer, return \"idk\".\nReturn me all answers as a list separated by comas; don' add any other text.""",
        'template_classification': """
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
                    """,
        'hint_prompt': """Remember that {relation} holds between the answers for this question and the previous question.""",
    },
    'es': {
        'template': """\nSi no puedes responder, devuelve \"idk\".\nDevuélveme la respuesta como una lista separada por comas, sin añadir ningún otro texto""",
        'template_classification': """
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

            Devuélveme solo la relación lógica entre las dos preguntas. Devuélveme solo la primera relación que se cumple. No añadas ningún otro texto.""",
        'hint_prompt': """Recuerda que {relation} se mantiene entre las respuestas de esta pregunta y la anterior."""
    }
}

PROMPTS_MINUS = {
    'en': {'template_classification': '''
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

                Return only the logical relation between the three questions. Return only the first relation that holds. Do not include any additional explanation.
        '''},
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

            Devuélveme solo la relación lógica entre las tres preguntas. Devuélveme solo la primera relación que se cumple. No añadas ningún otro texto.
        '''
    }

dataset_map = {
    'equal-wiki.tsv': 'equal',
    'subsetOf-wiki.tsv': 'sup-sub',
    'minus-set.tsv': 'minus'
}

llm_models = ['gpt-4.1-nano-2025-04-14', 'gpt-4.1-mini-2025-04-14']
languages = ['en', 'es']
logical_relations = {
    'en': {
        'Equivalence': 'equal-wiki.tsv',
        'Containment': 'subsetOf-wiki.tsv',
        'Minus': 'minus-set.tsv',
    },
    'es': {
        'Equivalencia': 'equal-wiki.tsv',
        'Contención': 'subsetOf-wiki.tsv',
        'Resta': 'minus-set.tsv',
    }
}

def run_banchmark(llm_model, language, test_type, use_hint=False):
    chat = ChatOpenAI(model_name=llm_model, openai_api_key=openai_api_key, temperature=0.0)
    tsv_file = os.path.join(here, f'../data/Dataset/{language}/{logical_relations[language][test_type]}')
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['ql2'],row['ql1']))
    answers_ql1 = {}
    answers_ql2 = {}
    for index, question in enumerate(questions):
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=chat,
            memory=memory
        )
        relation_predicted = conversation.predict(input=PROMPTS[language]['template_classification'].format(q1=question[0], q2=question[1]))
        relation_predicted = relation_predicted.strip().lower()
        answer1 = conversation.predict(input=question[0] + PROMPTS[language]['template'])

        if use_hint:
            answer2 = conversation.predict(input=question[1] + PROMPTS[language]['hint_prompt'].format(relation=test_type) + PROMPTS[language]['template'])
        else:
            answer2 = conversation.predict(input=question[1] + PROMPTS[language]['template'])

        answer1 = utils.convert_response_to_set(answer1)
        answer2 = utils.convert_response_to_set(answer2)

        answers_ql1[index] = answer1
        answers_ql2[index] = answer2

    if language == 'es':
        output_prefix = '*'
    else:
        output_prefix = ''

    test_type = logical_relations[language][test_type]

    with open(os.path.join(here, f'../data/answers/rel_classification_and_questions/{test_type}/{output_prefix}ql1_{test_type}_answers_classAndAnswer_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql1, f, ensure_ascii=False, indent=4)

    with open(os.path.join(here, f'../data/answers/rel_classification_and_questions/{test_type}/{output_prefix}ql2_{test_type}_answers_classAndAnswer_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

def run_minus_benchmark(llm_model, language, test_type, use_hint=False):
    chat = ChatOpenAI(model_name=llm_model, openai_api_key=openai_api_key, temperature=0.0)
    tsv_file = os.path.join(here, f'../data/Dataset/{language}/{logical_relations[language][test_type]}')
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['ql2'],row['ql1'], row['ql3']))
    answers_ql1 = {}
    answers_ql2 = {}
    answers_ql3 = {}
    for index, question in enumerate(questions):
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=chat,
            memory=memory
        )
        relation_predicted = conversation.predict(input=PROMPTS[language]['template_classification'].format(q1=question[0], q2=question[1], q3=question[2]))
        relation_predicted = relation_predicted.strip().lower()
        answer1 = conversation.predict(input=question[0] + PROMPTS[language]['template'])
        answer2 = conversation.predict(input=question[1] + PROMPTS[language]['template'])

        if use_hint:
            answer3 = conversation.predict(input=question[2] + PROMPTS[language]['hint_prompt'].format(relation=test_type) + PROMPTS[language]['template'])
        else:
            answer3 = conversation.predict(input=question[2] + PROMPTS[language]['template'])

        answer1 = utils.convert_response_to_set(answer1)
        answer2 = utils.convert_response_to_set(answer2)
        answer3 = utils.convert_response_to_set(answer3)

        answers_ql1[index] = answer1
        answers_ql2[index] = answer2
        answers_ql3[index] = answer3

    if language == 'es':
        output_prefix = '*'
    else:
        output_prefix = ''

    test_type = logical_relations[language][test_type]

    with open(os.path.join(here, f'../data/answers/rel_classification_and_questions/minus/{output_prefix}ql1_minus_answers_classAndAnswer_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql1, f, ensure_ascii=False, indent=4)

    with open(os.path.join(here, f'../data/answers/rel_classification_and_questions/minus/{output_prefix}ql2_minus_answers_classAndAnswer_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

    with open(os.path.join(here, f'../data/answers/rel_classification_and_questions/minus/{output_prefix}ql3_minus_answers_classAndAnswer_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql3, f, ensure_ascii=False, indent=4)

for language in languages:
    for llm_model in llm_models:
        for relation in logical_relations[language].keys():
            print(f"Processing model: {llm_model} for relation: {relation} in language: {language}")
            if relation == 'Minus' or relation == 'Resta':
                run_minus_benchmark(llm_model, language, relation)
            else:
                run_banchmark(llm_model, language, relation)
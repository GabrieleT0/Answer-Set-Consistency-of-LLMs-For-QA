from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import csv
import utils
import json
import time

here = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

PROMPTS = {
    'en': {
        'template': """\nIf you cannot answer, return \"idk\".\nIn the response, do not use abbreviations or acronyms, but spell out the full terms, i.e. "United States of America" instead of "USA".\nReturn me all answers as a list separated by the symbol '|' don' add any other text.""",
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
        'template': """\nSi no puedes responder, devuelve \"idk\".\nDevuélveme la respuesta como una lista separada por el símbolo '|', sin añadir ningún otro texto""",
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

llm_models = ['gpt-4.1-2025-04-14']
languages = ['en']
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

datasets = ['spinach.tsv']

def run_benchmark(llm_model, language, logical_relation, dataset, use_hint=False, start_index=0, end_index=None):
    chat = ChatOpenAI(model_name=llm_model, openai_api_key=openai_api_key, temperature=0.0)
    tsv_file = os.path.join(here, f'../data/Dataset/{language}/{dataset}')
    
    # Read questions
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            if logical_relation == 'Equivalence':
                questions.append((row['Q1'], row['Q2']))
            elif logical_relation == 'Containment':
                questions.append((row['Q1'], row['Q3']))

    if end_index is None or end_index > len(questions):
        end_index = len(questions)

    output_prefix = '*' if language == 'es' else ''
    folder_name = 'equal' if logical_relation == 'Equivalence' else 'sup-sub'

    base_output_dir = os.path.join(here, f'../data/answers/rel_classification_and_questions/{dataset.split(".")[0]}/{folder_name}')
    os.makedirs(base_output_dir, exist_ok=True)

    q1_path = os.path.join(base_output_dir, f'{output_prefix}Q1_{folder_name}_answers_classAndAnswer_{llm_model}.json')
    q2_path = os.path.join(base_output_dir, f'{output_prefix}Q2_{folder_name}_answers_classAndAnswer_{llm_model}.json')

    # Load previous answers
    def load_json(path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    answers_ql1 = load_json(q1_path)
    answers_ql2 = load_json(q2_path)

    for index in range(start_index, end_index):
        if str(index) in answers_ql1:
            continue  # Skip already processed

        question = questions[index]
        memory = ConversationBufferMemory()
        conversation = ConversationChain(llm=chat, memory=memory)

        # Classification prediction
        relation_predicted = conversation.predict(input=PROMPTS[language]['template_classification'].format(q1=question[0], q2=question[1])).strip().lower()

        # Answer generation
        answer1 = conversation.predict(input=question[0] + PROMPTS[language]['template'])

        if use_hint:
            answer2 = conversation.predict(
                input=question[1] + PROMPTS[language]['hint_prompt'].format(relation=logical_relation) + PROMPTS[language]['template']
            )
        else:
            answer2 = conversation.predict(input=question[1] + PROMPTS[language]['template'])

        print("\nOriginal answers:", answer1, answer2)

        # Convert answers
        answer1 = utils.convert_response_to_set(answer1)
        answer2 = utils.convert_response_to_set(answer2)

        # Store
        answers_ql1[str(index)] = list(answer1)
        answers_ql2[str(index)] = list(answer2)

        # Write to files
        with open(q1_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql1, f, ensure_ascii=False, indent=4)
                

        with open(q2_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

        print(f"Index: {index} Question 1: {question[0]} Question 2: {question[1]}")
        print(f"Answer 1: {answer1} Answer 2: {answer2} Relation: {relation_predicted}\n")
        time.sleep(1.5)


def run_minus_benchmark(llm_model, language, test_type, dataset, use_hint=False, start_index=0, end_index=None):
    chat = ChatOpenAI(model_name=llm_model, openai_api_key=openai_api_key, temperature=0.0)
    tsv_file = os.path.join(here, f'../data/Dataset/{language}/{dataset}')

    # Read questions
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['Q1'], row['Q3'], row['Q4']))

    if end_index is None or end_index > len(questions):
        end_index = len(questions)

    output_prefix = '*' if language == 'es' else ''
    test_type_name = logical_relations[language][test_type]
    base_output_dir = os.path.join(here, f'../data/answers/rel_classification_and_questions/{dataset.split(".")[0]}/minus')
    os.makedirs(base_output_dir, exist_ok=True)

    q1_path = os.path.join(base_output_dir, f'{output_prefix}Q1_minus_answers_classAndAnswer_{llm_model}.json')
    q2_path = os.path.join(base_output_dir, f'{output_prefix}Q3_minus_answers_classAndAnswer_{llm_model}.json')
    q3_path = os.path.join(base_output_dir, f'{output_prefix}Q4_minus_answers_classAndAnswer_{llm_model}.json')

    def load_json(path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    answers_ql1 = load_json(q1_path)
    answers_ql2 = load_json(q2_path)
    answers_ql3 = load_json(q3_path)

    for index in range(start_index, end_index):
        if str(index) in answers_ql1:
            continue  # Skip already processed

        question = questions[index]
        memory = ConversationBufferMemory()
        conversation = ConversationChain(llm=chat, memory=memory)

        relation_predicted = conversation.predict(
            input=PROMPTS[language]['template_classification'].format(q1=question[0], q2=question[1], q3=question[2])
        ).strip().lower()

        answer1 = conversation.predict(input=question[0] + PROMPTS[language]['template'])
        answer2 = conversation.predict(input=question[1] + PROMPTS[language]['template'])

        if use_hint:
            answer3 = conversation.predict(
                input=question[2] + PROMPTS[language]['hint_prompt'].format(relation=test_type) + PROMPTS[language]['template']
            )
        else:
            answer3 = conversation.predict(input=question[2] + PROMPTS[language]['template'])

        print("\nOriginal answers:", answer1, answer2, answer3)

        # Convert
        answer1 = utils.convert_response_to_set(answer1)
        answer2 = utils.convert_response_to_set(answer2)
        answer3 = utils.convert_response_to_set(answer3)

        # Save
        answers_ql1[str(index)] = list(answer1)
        answers_ql2[str(index)] = list(answer2)
        answers_ql3[str(index)] = list(answer3)

        # Write after each question
        with open(q1_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql1, f, ensure_ascii=False, indent=4)
        with open(q2_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql2, f, ensure_ascii=False, indent=4)
        with open(q3_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql3, f, ensure_ascii=False, indent=4)

        print(f"\nIndex: {index} Question 1: {question[0]} Question 2: {question[1]} Question 3: {question[2]}")
        print(f"Answer 1: {answer1} Answer 2: {answer2} Relation: {relation_predicted} Answer 3: {answer3}\n")
        time.sleep(1.5)

relations = ['Containment', 'Minus']
for language in languages:
    for llm_model in llm_models:
        for dataset in datasets:
            for relation in relations:
                print(f"Processing model: {llm_model} for relation: {relation} in language: {language}")
                if relation == 'Minus' or relation == 'Resta':
                    run_minus_benchmark(llm_model, language, relation, dataset)
                else:
                    run_benchmark(llm_model, language, relation, dataset, start_index=45)   
                
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import csv
import utils
import json
import time
import prompt_llms

here = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

PROMPTS = {
    'en': {
        'template': """\nIf you cannot answer, return \"idk\".\nIf the question has no answer, return 'no answer'.\nIn the response, do not use abbreviations or acronyms, but spell out the full terms, i.e. "United States of America" instead of "USA".\nIf the response contains numbers or digits, use Arabic numerals. For example, if the answer contains Star Wars V, indicate it with Star Wars 5. Do not use Roman numerals (such as V) or text (such as five).\nPlease, Return me an exhaustive list separated by the symbol '|' don't add any other text.""",
        'equal_fix': """Pay attention, the two questions I asked you before are logically equivalent, but you returned me different values.\nIn the response, do not use abbreviations or acronyms, but spell out the full terms, i.e. "United States of America" instead of "USA".\nIf the response contains numbers or digits, use Arabic numerals. For example, if the answer contains Star Wars V, indicate it with Star Wars 5. Do not use Roman numerals (such as V) or text (such as five).\nPlease, Return me an exhaustive list separated by the symbol '|' don't add any other text.""",
        'sup_sub_fix': """Pay attention, the first question I asked is a more general question than the second question, so the answer of the second question must be a subset of the answer of the first, but the result of the second answer is not contained in the first.\n In the response, do not use abbreviations or acronyms, but spell out the full terms, i.e. "United States of America" instead of "USA".\nIf the response contains numbers or digits, use Arabic numerals. For example, if the answer contains Star Wars V, indicate it with Star Wars 5. Do not use Roman numerals (such as V) or text (such as five).\nPlease, Return me an exhaustive list separated by the symbol '|' don't add any other text.""",
        'minus_fix': """Pay attention, I asked you 3 different questions, the third question should contain the elements of the answer to the first question I asked you, but removing the elements in the answer to the second question I asked you.\nSo the answer to the third question should contain the results that are in the first answer but are not in the answer to the second question.\nIn the response, do not use abbreviations or acronyms, but spell out the full terms, i.e. "United States of America" instead of "USA".\nIf the response contains numbers or digits, use Arabic numerals. For example, if the answer contains Star Wars V, indicate it with Star Wars 5. Do not use Roman numerals (such as V) or text (such as five).\nPlease, Return me an exhaustive list separated by the symbol '|' don't add any other text.""",
    },
    'es': {
        'template': """\nSi no puedes responder, devuelve \"idk\".\nDevuélveme la respuesta como una lista separada por el símbolo '|' sin añadir ningún otro texto""",
        'equal_fix': """Presta atención, las dos preguntas que te hice antes son lógicamente equivalentes, pero me diste respuestas diferentes.\nCorrige la respuesta a la segunda pregunta y devuélvela como una lista separada por el símbolo '|' sin añadir más texto.""",
        'sup_sub_fix': """Presta atención, la primera pregunta que te hice es más general que la segunda tal que cada respuesta a la segunda pregunta deba ser una respuesta a la primera pregunta.\nCorrige la respuesta a la segunda pregunta y devuélvela como una lista separada por el símbolo '|' sin añadir más texto.""",
        'minus_fix': """Presta atención, te hice 3 preguntas diferentes. La tercera debe contener los elementos de la primera respuesta excluyendo los de la segunda.\nLa respuesta a la tercera pregunta debe contener los resultados que están en la primera pero no en la segunda.\nCorrige la respuesta a la segunda pregunta y devuélvela como una lista separada por el símbolo '|' sin añadir más texto."""
    }
}


def equal_test(llm_model, dataset_name, language='en', start_index=0, end_index=None):
    chat = prompt_llms.return_chat_model(llm_model)

    template = PROMPTS[language]['template']
    fix_template_equal = PROMPTS[language]['equal_fix']

    # File and config paths
    here = os.path.dirname(os.path.abspath(__file__))
    tsv_file = utils.get_dataset_path(dataset_name, language)

    # Read questions
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['Q1'], row['Q2']))

    # Limit processing range
    if end_index is None or end_index > len(questions):
        end_index = len(questions)

    # Setup output directory and file paths
    output_prefix = '*' if language == 'es' else ''
    base_output_dir = os.path.join(here, f'../data/answers/follow_up_fixing/{dataset_name.split(".")[0]}/equal')
    os.makedirs(base_output_dir, exist_ok=True)

    q1_path = os.path.join(base_output_dir, f'{output_prefix}Q1_equal_answers_fixing_{llm_model}.json')
    q2_path = os.path.join(base_output_dir, f'{output_prefix}Q2_equal_answers_fixing_{llm_model}.json')

    # Load existing results
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

        answer1 = utils.convert_response_to_set(conversation.predict(input=question[0] + template))
        answer2 = utils.convert_response_to_set(conversation.predict(input=question[1] + template))

        jaccard_similarity = utils.jaccard_similarity(answer1, answer2)

        if jaccard_similarity < 1 or len(answer2) == 0:
            answer3 = utils.convert_response_to_set(conversation.predict(input=fix_template_equal))
        else:
            answer3 = answer2

        # Save results
        answers_ql1[str(index)] = list(answer1)
        answers_ql2[str(index)] = list(answer3)

        # Write incrementally
        with open(q1_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql1, f, ensure_ascii=False, indent=4)
        with open(q2_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

        print(f"Index: {index} Q1: {question[0]} Q2: {question[1]}")
        print(f"A1: {answer1} A2: {answer2} Jaccard: {jaccard_similarity} Final: {answer3}")

    # Final write
    with open(q1_path, 'w', encoding='utf-8') as f:
        json.dump(answers_ql1, f, ensure_ascii=False, indent=4)
    with open(q2_path, 'w', encoding='utf-8') as f:
        json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

def sup_sub_test(llm_model, dataset_name, language='en', start_index=0, end_index=None):
    chat = prompt_llms.return_chat_model(llm_model)

    template = PROMPTS[language]['template']
    fix_template_sup_sub = PROMPTS[language]['sup_sub_fix']

    # File and config paths
    here = os.path.dirname(os.path.abspath(__file__))
    tsv_file = utils.get_dataset_path(dataset_name, language)

    # Read questions
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['Q1'], row['Q3']))

    # Limit the processing range
    if end_index is None or end_index > len(questions):
        end_index = len(questions)

    # Setup output directory and file paths
    output_prefix = '*' if language == 'es' else ''
    base_output_dir = os.path.join(here, f'../data/answers/follow_up_fixing/{dataset_name.split(".")[0]}/sup-sub')
    os.makedirs(base_output_dir, exist_ok=True)

    q1_path = os.path.join(base_output_dir, f'{output_prefix}Q1_sup-sub_answers_fixing_{llm_model}.json')
    q3_path = os.path.join(base_output_dir, f'{output_prefix}Q3_sup-sub_answers_fixing_{llm_model}.json')

    # Load existing partial results if present
    def load_json(path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    answers_ql1 = load_json(q1_path)
    answers_ql2 = load_json(q3_path)

    for index in range(start_index, end_index):
        if str(index) in answers_ql1:
            continue  # Skip already processed entries

        question = questions[index]
        memory = ConversationBufferMemory()
        conversation = ConversationChain(llm=chat, memory=memory)

        answer1 = utils.convert_response_to_set(conversation.predict(input=question[0] + template))
        answer2 = utils.convert_response_to_set(conversation.predict(input=question[1] + template))
        is_subset = utils.is_subset(answer2, answer1)
        jaccard_similarity = utils.jaccard_similarity(answer1, answer2)

        if not is_subset or len(answer2) == 0:
            answer3 = utils.convert_response_to_set(conversation.predict(input=fix_template_sup_sub))
        else:
            answer3 = answer2

        # Save to dicts
        answers_ql1[str(index)] = list(answer1)
        answers_ql2[str(index)] = list(answer3)

        # Write to files immediately
        with open(q1_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql1, f, ensure_ascii=False, indent=4)

        with open(q3_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

        print(f"Index: {index} Q1: {question[0]} Q3: {question[1]}")
        print(f"A1: {answer1} A3: {answer2} isSubset: {is_subset} Jaccard: {jaccard_similarity} Final: {answer3}")

    # Final write
    with open(q1_path, 'w', encoding='utf-8') as f:
        json.dump(answers_ql1, f, ensure_ascii=False, indent=4)

    with open(q3_path, 'w', encoding='utf-8') as f:
        json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

def minus_test(llm_model, dataset_name, language='en', start_index=0, end_index=None):
    chat = prompt_llms.return_chat_model(llm_model)

    template = PROMPTS[language]['template']
    fix_template_minus = PROMPTS[language]['minus_fix']

    # File and config paths
    here = os.path.dirname(os.path.abspath(__file__))
    tsv_file = utils.get_dataset_path(dataset_name, language)

    # Read questions
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['Q1'], row['Q3'], row['Q4']))

    # Limit the processing range
    if end_index is None or end_index > len(questions):
        end_index = len(questions)

    # Setup output directory and file paths
    output_prefix = '*' if language == 'es' else ''
    base_output_dir = os.path.join(here, f'../data/answers/follow_up_fixing/{dataset_name.split(".")[0]}/minus')
    os.makedirs(base_output_dir, exist_ok=True)

    q1_path = os.path.join(base_output_dir, f'{output_prefix}Q1_minus_answers_fixing_{llm_model}.json')
    q3_path = os.path.join(base_output_dir, f'{output_prefix}Q3_minus_answers_fixing_{llm_model}.json')
    q4_path = os.path.join(base_output_dir, f'{output_prefix}Q4_minus_answers_fixing_{llm_model}.json')

    # Load existing partial results if present
    def load_json(path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    answers_ql1 = load_json(q1_path)
    answers_ql2 = load_json(q3_path)
    answers_ql3 = load_json(q4_path)

    for index in range(start_index, end_index):
        if str(index) in answers_ql1:
            continue  # Skip already processed entries

        question = questions[index]
        memory = ConversationBufferMemory()
        conversation = ConversationChain(llm=chat, memory=memory)

        answer1 = utils.convert_response_to_set(conversation.predict(input=question[0] + template))
        answer2 = utils.convert_response_to_set(conversation.predict(input=question[1] + template))
        answer3 = utils.convert_response_to_set(conversation.predict(input=question[2] + template))

        is_minus = utils.is_minus(answer1, answer2, answer3)

        if not is_minus or len(answer3) == 0:
            answer4 = utils.convert_response_to_set(conversation.predict(input=fix_template_minus))
        else:
            answer4 = answer3

        # Save to dict
        answers_ql1[str(index)] = list(answer1)
        answers_ql2[str(index)] = list(answer2)
        answers_ql3[str(index)] = list(answer4)

        # Write to files immediately
        with open(q1_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql1, f, ensure_ascii=False, indent=4)

        with open(q3_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

        with open(q4_path, 'w', encoding='utf-8') as f:
            json.dump(answers_ql3, f, ensure_ascii=False, indent=4)

        print(f"Index: {index} Q1: {question[0]} Q2: {question[1]} Q3: {question[2]}")
        print(f"A1: {answer1} A2: {answer2} A3: {answer3} isMinus: {is_minus} Final: {answer4}")
        

    if language == 'es':
        output_prefix = '*'
    else:
        output_prefix = ''

    with open(os.path.join(here, f'../data/answers/follow_up_fixing/{dataset_name.split('.')[0]}/minus/{output_prefix}Q1_minus_answers_fixing_{llm_model}.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql1, f, ensure_ascii=False, indent=4)

    with open(os.path.join(here, f'../data/answers/follow_up_fixing/{dataset_name.split('.')[0]}/minus/{output_prefix}Q3_minus_answers_fixing_{llm_model}.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

    with open(os.path.join(here, f'../data/answers/follow_up_fixing/{dataset_name.split('.')[0]}/minus/{output_prefix}Q4_minus_answers_fixing_{llm_model}.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql3, f, ensure_ascii=False, indent=4)


llm_models = ['gpt-4.1-2025-04-14']
languages = ['en']
datasets = ['qawiki.tsv']
for language in languages:
    for llm_model in llm_models:
        for dataset_name in datasets:
            # Run logical equivalence test
            print(f"Processing model: {llm_model}")
            equal_test(llm_model, dataset_name, language)
            print(f"Finished processing model: {llm_model}\n")

            # Run subset/superset test
            print(f"Processing model: {llm_model}")
            sup_sub_test(llm_model, dataset_name, language)
            print(f"Finished processing model: {llm_model}\n")

            # Run minus test
            print(f"Processing model: {llm_model}")
            minus_test(llm_model, dataset_name, language)
            print(f"Finished processing model: {llm_model}\n")
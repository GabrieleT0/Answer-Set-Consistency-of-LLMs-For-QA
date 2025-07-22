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
        'template': """\nIf you cannot answer, return \"idk\".\nReturn me all answers as a list separated by the symbol '|' don' add any other text.""",
        'equal_fix': """Pay attention, the two questions I asked you before are logically equivalent, but you returned me different values.\nI'll ask you the questions again, answer correctly according to what I told you and return me all answers as a list separated by the symbol '|' don't add any other text.""",
        'sup_sub_fix': """Pay attention, the first question I asked is a more general question than the second question, so the answer of the second question must be a subset of the answer of the first, but the result of the second answer is not contained in the first. 'll ask you the questions again, answer correctly according to what I told you and return me all answers as a list separated by the symbol '|' don't add any other text.""",
        'minus_fix': """Pay attention, I asked you 3 different questions, the third question should contain the elements of the answer to the first question I asked you, but removing the elements in the answer to the second question I asked you.\nSo the answer to the third question should contain the results that are in the first answer but are not in the answer to the second question. 'll ask you the questions again, answer correctly according to what I told you and return me all answers as a list separated by the symbol '|' don't add any other text."""
    },
    'es': {
        'template': """\nSi no puedes responder, devuelve \"idk\".\nDevuélveme la respuesta como una lista separada por el símbolo '|' sin añadir ningún otro texto""",
        'equal_fix': """Presta atención, las dos preguntas que te hice antes son lógicamente equivalentes, pero me diste respuestas diferentes.\nTe volveré a hacer las preguntas, contéstalas correctamente según lo que te he dicho y devuélvela como una lista separada por el símbolo '|' sin añadir más texto.""",
        'sup_sub_fix': """Presta atención, la primera pregunta que te hice es más general que la segunda tal que cada respuesta a la segunda pregunta deba ser una respuesta a la primera pregunta.\nTe volveré a hacer las preguntas, contéstalas correctamente según lo que te he dicho y devuélvela como una lista separada por el símbolo '|' sin añadir más texto.""",
        'minus_fix': """Presta atención, te hice 3 preguntas diferentes. La tercera debe contener los elementos de la primera respuesta excluyendo los de la segunda.\nLa respuesta a la tercera pregunta debe contener los resultados que están en la primera pero no en la segunda.\nTe volveré a hacer las preguntas, contéstalas correctamente según lo que te he dicho y devuélvela como una lista separada por el símbolo '|' sin añadir más texto."""
    }
}


def equal_test(llm_model,language='en'):
    chat = ChatOpenAI(model_name=llm_model, openai_api_key=openai_api_key, temperature=0.0)

    template = PROMPTS[language]['template']
    fix_template_equal = PROMPTS[language]['equal_fix']

    # File and config paths
    tsv_file = questions = utils.get_dataset_path('equal-wiki.tsv', language)

    # Read questions
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['ql1'],row['ql2']))

    answers_ql1 = {}
    answers_ql2 = {}
    for index, question in enumerate(questions):
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=chat,
            memory=memory
        )
        answer1 = conversation.predict(input=question[0] + template)
        answer1 = utils.convert_response_to_set(answer1)
        answer2 = conversation.predict(input=question[1] + template)
        answer2 = utils.convert_response_to_set(answer2)
        jaccard_similarity = utils.jaccard_similarity(answer1, answer2)
        if jaccard_similarity < 1:
            answer3 = conversation.predict(input=fix_template_equal)
            answer1 = conversation.predict(input=question[0] + template)
            answer1 = utils.convert_response_to_set(answer1)
            answer2 = conversation.predict(input=question[1] + template)
            answer2 = utils.convert_response_to_set(answer2)
        answers_ql1[index] = answer1
        answers_ql2[index] = answer2

        print(f"Index: {index} Question 1: {question[0]} Question 2: {question[0]}")
        print(f"Answer 1: {answer1} Answer 2: {answer2} Jaccard Similarity: {jaccard_similarity}")
    
    if language == 'es':
        output_prefix = '*'
    else:
        output_prefix = ''

    with open(os.path.join(here, f'../data/answers/follow_up_fixing/equal/{output_prefix}ql1_equal_answers_fixing_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql1, f, ensure_ascii=False, indent=4)

    with open(os.path.join(here, f'../data/answers/follow_up_fixing/equal/{output_prefix}ql2_equal_answers_fixing_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

def sup_sub_test(llm_model, language='en'):
    chat = ChatOpenAI(model_name=llm_model, openai_api_key=openai_api_key, temperature=0.0)

    template = PROMPTS[language]['template']
    fix_template_sup_sub = PROMPTS[language]['sup_sub_fix']

    # File and config paths
    here = os.path.dirname(os.path.abspath(__file__))
    tsv_file = utils.get_dataset_path('subsetOf-wiki.tsv', language)

    # Read questions
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['ql1'],row['ql2']))

    answers_ql1 = {}
    answers_ql2 = {}
    for index, question in enumerate(questions):
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=chat,
            memory=memory
        )
        answer1 = conversation.predict(input=question[1] + template)
        answer1 = utils.convert_response_to_set(answer1)
        answer2 = conversation.predict(input=question[0] + template)
        answer2 = utils.convert_response_to_set(answer2)
        is_subset = utils.is_subset(answer2, answer1)
        jaccard_similarity = utils.jaccard_similarity(answer1, answer2)
        if not is_subset or len(answer2) == 0:
            answer3 = conversation.predict(input=fix_template_sup_sub)
            answer1 = conversation.predict(input=question[0] + template)
            answer1 = utils.convert_response_to_set(answer1)
            answer2 = conversation.predict(input=question[1] + template)
            answer2 = utils.convert_response_to_set(answer2)
        answers_ql1[index] = answer1
        answers_ql2[index] = answer2
        
        print(f"Index: {index} Question 1: {question[1]} Question 2: {question[0]}")
        print(f"Answer 1: {answer1} Answer 2: {answer2} isSubset: {is_subset} JaccardSimilarity: {jaccard_similarity}")
        
    if language == 'es':
        output_prefix = '*'
    else:
        output_prefix = ''

    with open(os.path.join(here, f'../data/answers/follow_up_fixing/sup-sub/{output_prefix}ql1_sup-sub_answers_fixing_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql1, f, ensure_ascii=False, indent=4)

    with open(os.path.join(here, f'../data/answers/follow_up_fixing/sup-sub/{output_prefix}ql2_sup-sub_answers_fixing_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql2, f, ensure_ascii=False, indent=4)

def minus_test(llm_model, language='en'):
    chat = ChatOpenAI(model_name=llm_model, openai_api_key=openai_api_key, temperature=0.0)

    template = PROMPTS[language]['template']
    fix_template_minus = PROMPTS[language]['minus_fix']

    # File and config paths
    here = os.path.dirname(os.path.abspath(__file__))
    tsv_file = utils.get_dataset_path('minus-set.tsv', language)

    # Read questions
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        if language == 'en':
            reader = csv.DictReader(tsvfile, delimiter=';')
        else:
            reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['ql1'],row['ql2'],row['ql3']))

    answers_ql1 = {}
    answers_ql2 = {}
    answers_ql3 = {}
    for index, question in enumerate(questions):
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=chat,
            memory=memory
        )
        answer1 = conversation.predict(input=question[1] + template)
        answer1 = utils.convert_response_to_set(answer1)
        answer2 = conversation.predict(input=question[0] + template)
        answer2 = utils.convert_response_to_set(answer2)
        answer3 = conversation.predict(input=question[2] + template)
        answer3 = utils.convert_response_to_set(answer3)
        is_minus = utils.is_minus(answer1, answer2, answer3)
        if not is_minus or len(answer3) == 0:
            answer4 = conversation.predict(input=fix_template_minus)
            answer1 = utils.convert_response_to_set(answer1)
            answer2 = conversation.predict(input=question[0] + template)
            answer2 = utils.convert_response_to_set(answer2)
            answer3 = conversation.predict(input=question[2] + template)
            answer3 = utils.convert_response_to_set(answer3)
        answers_ql1[index] = answer1
        answers_ql2[index] = answer2
        answers_ql3[index] = answer3

        print(f"Index: {index} Question 1: {question[1]} Question 2: {question[0]} Question 3: {question[2]}")
        print(f"Answer 1: {answer1} Answer 2: {answer2} Answer 3: {answer3} isMinus: {is_minus}")

    if language == 'es':
        output_prefix = '*'
    else:
        output_prefix = ''

    with open(os.path.join(here, f'../data/answers/follow_up_fixing/minus/{output_prefix}ql1_minus_answers_fixing_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql1, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(here, f'../data/answers/follow_up_fixing/minus/{output_prefix}ql2_minus_answers_fixing_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql2, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(here, f'../data/answers/follow_up_fixing/minus/{output_prefix}ql3_minus_answers_fixing_' + llm_model + '.json'), 'w', encoding='utf-8') as f:
        json.dump(answers_ql3, f, ensure_ascii=False, indent=4)


llm_models = ['gpt-4.1-nano-2025-04-14', 'gpt-4.1-mini-2025-04-14']
languages = ['en', 'es']

for language in languages:
    for llm_model in llm_models:

        # Run logical equivalence test
        print(f"Processing model: {llm_model}")
        equal_test(llm_model, language)
        print(f"Finished processing model: {llm_model}\n")

        # Run subset/superset test
        print(f"Processing model: {llm_model}")
        sup_sub_test(llm_model, language)
        print(f"Finished processing model: {llm_model}\n")

        # Run minus test
        print(f"Processing model: {llm_model}")
        minus_test(llm_model, language)
        print(f"Finished processing model: {llm_model}\n")

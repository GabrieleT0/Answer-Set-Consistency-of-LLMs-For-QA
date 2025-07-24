from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
import os
import csv
import utils
import json
from gemini_rate_limiter import GeminiRateLimiter
import gemini_rate_limiter
here = os.path.dirname(os.path.abspath(__file__))

PROMPTS = {
    "standard": {
        "en": '''{question} \n
                If you can't answer, return 'idk'. \n
                Return me all answers as a list separated by the symbol '|' don't add any other text.''',
        "es": '''{question} \n
                Si no puedes responder, devuelve "no sé". \n
                Devuélveme la respuesta en forma de lista separada por el símbolo '|' no añadas ningún otro texto.'''
    },
    "wikidata": {
        "en": '''{question} \n
            Just use Wikidata as a source to answer my question. \n
            If you can't answer, return 'idk'. \n
            Return me all answers as a list separated by '|' don't add any other text.''',
        "es": '''{question} \n
            Utiliza Wikidata como fuente para responder a mi pregunta. \n
            Si no puedes responder, devuelve "no sé". \n
            Devuélveme la respuesta en forma de lista separada por el símbolo '|' no añadas ningún otro texto.'''
    }
}


columns_map = {
    'spinach.tsv': ['Q1', 'Q2','Q3','Q4'],
}
languages = ['en']
columns_map = {
    'spinach.tsv': ['Q1', 'Q2','Q3','Q4'],
}
logical_relations_map = {
    'Q1': 'equal',
    'Q2': 'equal',
    'Q3': 'sup-sub',
    'Q4': 'minus',
}
languages = ['en']
llm_models = ['gemini-2.5-pro']
datasets = ['spinach.tsv']

# Initialize rate limiter for Gemini models
rate_limiter = GeminiRateLimiter(rpm=5, tpm=250_000, rpd=100)

def run_benchmark(prompt_type='standard', start_index=0, end_index=None):
    for language in languages:
        for llm_model in llm_models:
            for dataset in datasets:
                print(f"Processing dataset: {dataset} for model: {llm_model} and language: {language}")
                tsv_file = os.path.join(here, f'../data/Dataset/{language}/{dataset}')

                for column in columns_map[dataset]:
                    print(f"Processing column: {column}")
                    questions = []
                    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
                        reader = csv.DictReader(tsvfile, delimiter='\t')
                        for row in reader:
                            questions.append(row[column])

                    # Set output path
                    lang_prefix = '' if language == 'en' else '*'
                    suffix = f"_answers_{'wikidata_' if prompt_type == 'wikidata' else ''}{llm_model}.json"
                    output_dir = os.path.join(here, f'../data/answers/zero-shot/{dataset.split(".")[0]}/{logical_relations_map[column]}')
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = os.path.join(output_dir, f'{lang_prefix}{column}_{logical_relations_map[column]}{suffix}')

                    # Load previously saved answers
                    if os.path.exists(output_filename):
                        with open(output_filename, 'r', encoding='utf-8') as f:
                            answers = json.load(f)
                    else:
                        answers = {}

                    # Adjust end_index
                    if end_index is None or end_index > len(questions):
                        end_index = len(questions)

                    for index in range(start_index, end_index):
                        if str(index) in answers:
                            continue  # Skip already processed

                        question = questions[index]
                        prompt = PromptTemplate(
                            input_variables=["question"],
                            template=PROMPTS[prompt_type][language]
                        )

                        # Needed only for Gemini models to avoid rate limiting issues
                        if 'gemini' in llm_model:
                            estimated_tokens = gemini_rate_limiter.estimate_token_count(question, PROMPTS[prompt_type][language])
                            rate_limiter.wait_if_needed(estimated_tokens)

                        llms = PromptLLMS(model=llm_model, prompt_template=prompt, question=question)
                        llm_response = llms.execute_single_question()

                        if language == 'en':
                            converted_response = utils.convert_response_to_set(llm_response)
                        else:
                            converted_response = utils.convert_response_to_set_es(llm_response)

                        answers[str(index)] = converted_response

                        # Needed only for Gemini models to avoid rate limiting issues
                        if 'gemini' in llm_model:
                            rate_limiter.register_request(estimated_tokens)

                        # Save incrementally
                        with open(output_filename, 'w', encoding='utf-8') as f:
                            json.dump(answers, f, ensure_ascii=False, indent=4)

                        print(f"Question {index + 1}: {question}")
                        print(f"LLM Response: {llm_response}")

                    # Final save to ensure consistency
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(answers, f, ensure_ascii=False, indent=4)


# Run standard benchmark
run_benchmark(prompt_type='standard')

# Run benchmark by using only Wikidata as a source
run_benchmark(prompt_type='wikidata')
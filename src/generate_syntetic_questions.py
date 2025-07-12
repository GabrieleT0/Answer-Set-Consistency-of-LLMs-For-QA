from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
import os
import csv
import utils
import json

here = os.path.dirname(os.path.abspath(__file__))
PROMPTS = {
    "en" : { "equivalence" : '''Generate {number_of_questions_to_generate} pairs of diverse questions about different topics, every pairs of questions must be semantically equivalent \
                        "and the answer to every questions that you formulate must be a list of values, not an ordered list, not a paragraph of text, not a boolean value and not a single number.
                        This is an example of a possible pair of questions: 1. How many regions of France are there? | How many regions does France have?
                        Follow the following format to return the questions: 1. Question1 | Equivalent_Question1
                        Do not add any other kind of text except questions
                        ''',
            "subset-superset" : '''Generate {number_of_questions_to_generate} pairs of diverse questions about different topics, in every pair of questions, the first question must be broader than the second i.e. the answer of the second question must be a subset of the answer of the first \
                                and the answer to every questions that you formulate must be a list of values, not an ordered list, not a paragraph of text, not a boolean value and not a single number. \
                                This is an example of a possible pair of questions: 1. What countries are in the EU? | What countries are in the western EU?
                                Follow the following format to return the questions: 1. Broader_Question | Subset_Question
                                Do not add any other kind of text except questions
                                '''
    }
}

PROMPTS_MINUS = {
    "en" : '''I prompt to you two questions: question A and question B. A containment relationship hold between the answer of these question, specifically, the responses at the question B is contained in the responses at the question A.\
                Your task now it to generate a question C in a way in which the relationship between the resonses at the three qustions is A-B=C. 
                The answer to the question C must be a list of values, not an ordered list, not a paragraph of text, not a boolean value and not a single number.
                This is an example: question A) Which movies star Uma Thurman? question B) Which science fiction movies star Uma Thurman? question C) Which movies star Uma Thurman excluding those science fiction movies star Uma Thurman?
                Follow the following format to return the questions: 1. Question C
                Give me only the question C, do not add any other kind of text.
                Question A: {q1}
                Question B: {q2}
                '''
}   

languages = ["en"]
llm_model = "gemini-2.5-pro"
output_map = {
    'equivalence' : 'equal-syntetic.tsv',
    'subset-superset' : 'sup-sub_syntetic.tsv',
    'minus' : 'minus_syntetic.tsv'
}
number_of_questions_to_generate = 20

def generate_syntetic_questions():

    for language in languages:
        for key in PROMPTS[language]:
            print(f"Generating syntetic questions for {key} experiment using {llm_model} in {language}.")
            tsv_output = os.path.join(here, f'../data/Dataset/{language}/{output_map[key]}')
            prompt = PromptTemplate(
                input_variables=["question"],
                template=PROMPTS[language][key].format(number_of_questions_to_generate=number_of_questions_to_generate)
            )
            llms = PromptLLMS(prompt, PROMPTS[language][key])
            llm_response = (
                llms.execute_on_gemini(model=llm_model)
                if 'gemini' in llm_model
                else llms.execute_on_openAI_model(openAI_model=llm_model)
            )
            print(f"LLM response: {llm_response}")
            parsed = utils.parse_questions_for_tsv(llm_response)
            utils.save_to_tsv(parsed, tsv_output)

def generate_syntetic_questions_minus():

    for language in languages:
        tsv_file_input = os.path.join(here, f'../data/Dataset/{language}/sup-sub_syntetic.tsv')
        question_pairs = []
        with open(tsv_file_input, newline='', encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for row in reader:
                question_pairs.append((row['ql1'], row['ql2']))

        for index, (q1, q2) in enumerate(question_pairs):
            prompt_template = PromptTemplate(
            input_variables=["q1", "q2"],
            template=PROMPTS_MINUS[language]
            )
            llms = PromptLLMS(prompt_template, question1=q2,question2=q1)

            llm_response = (
                llms.execute_on_gemini_two_question(model=llm_model)
                if 'gemini' in llm_model
                else llms.execute_on_gemini_two_question(openAI_model=llm_model)
            )
            parsed = utils.parse_questions_for_tsv(llm_response, input_q1=q1, input_q2=q2, minus=True)
            utils.save_to_tsv(parsed, output_map['minus'], minus=True)


generate_syntetic_questions()
generate_syntetic_questions_minus()
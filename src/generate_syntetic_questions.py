from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
import os
import csv
import utils

here = os.path.dirname(os.path.abspath(__file__))
PROMPTS = {
    "equivalence" : '''Generate {number_of_questions_to_generate} pairs of diverse questions about different topics, every pair of questions must be semantically equivalent. \
                    The answer to every question that you formulate must be a list of values — not an ordered list, not a paragraph of text, not a boolean value, and not a single number. \
                    This is an example of a possible pair of questions: \
                    1. How many regions of France are there? | How many regions does France have? \
                    Follow the following format to return the questions: \
                    1. Question1 | Equivalent_Question1 \
                    Do not add any other kind of text except questions.''',
                    
    "subset-superset" : '''
        {dataset_of_question_pairs}
        Starting from the provided dataset of question pairs, where each pair consists of two semantically equivalent questions, generate a third question whose answer is a subset of the answers to the original two questions.
        The answer to the generated question must be a list of values (not an ordered list, not a descriptive paragraph, not a Boolean value, and not a single number).

        Use the following output format, and provide only the third question:

        1. Broader_Question_from_the_dataset | Subset_Question

        For example, given the pair:
        “What countries are in the EU?” | “What countries are in the western EU?”
        the generated question would be the subset question.

        Return only the formulated subset question and no additional text. '''

} 

languages = ["en"]
llm_model = "gpt-4.1-2025-04-14"
output_map = {
    'equivalence' : 'equal-syntetic.tsv',
    'subset-superset' : 'sup-sub_syntetic.tsv',
    'minus' : 'minus_syntetic.tsv'
}
number_of_questions_to_generate = 51

def generate_syntetic_questions(logical_relationship):
    print(f"Generating syntetic questions for {logical_relationship} experiment using {llm_model} in en.")
    tsv_output = os.path.join(here, f'../data/Dataset/en/{output_map[logical_relationship]}')
    prompt = PromptTemplate(
        input_variables=["question"],
        template=PROMPTS[logical_relationship].format(number_of_questions_to_generate=number_of_questions_to_generate)
    )
    llms = PromptLLMS(prompt, PROMPTS[logical_relationship])
    llm_response = (
        llms.execute_on_gemini(model=llm_model)
        if 'gemini' in llm_model
        else llms.execute_on_openAI_model(openAI_model=llm_model)
    )
    print(f"LLM response: {llm_response}")
    parsed = utils.parse_questions_for_tsv(llm_response)
    
    file_exists = os.path.exists(tsv_output)
    is_empty = not file_exists or os.stat(tsv_output).st_size == 0

    with open(tsv_output, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        if is_empty:
            if logical_relationship == 'equivalence':
                writer.writerow(["ql1", "ql2", "sparql_ql1"])
            else:
                writer.writerow(["ql1", "ql2", "sparql_ql1", "sparql_ql2"])

        writer.writerows(parsed)

generate_syntetic_questions('subset-superset')
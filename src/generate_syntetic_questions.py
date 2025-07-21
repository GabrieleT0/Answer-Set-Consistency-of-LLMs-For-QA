from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
import os
import csv
import utils

here = os.path.dirname(os.path.abspath(__file__))
PROMPTS = {
    "equivalence" : '''Generate {number_of_questions_to_generate} pairs of diverse questions about different topics, every pair of questions must be semantically equivalent. \
                    The answer to every question that you formulate must be a list of values — not an ordered list, not a paragraph of text, not a boolean value, and not a single number. \
                    The answer to the questions must be available in Wikidata, and give me also the SPARQL query to retrieve the answer. \
                    This is an example of a possible pair of questions: \
                    1. How many regions of France are there? | How many regions does France have? | SELECT ?region ?regionLabel WHERE {{{{ ?region wdt:P31 wd:Q36784; wdt:P17 wd:Q142. SERVICE wikibase:label {{{{ bd:serviceParam wikibase:language "en". }}}} }}}} ORDER BY ?regionLabel \
                    Follow the following format to return the questions: \
                    1. Question1 | Equivalent_Question1 | SPARQL_query \
                    Do not add any other kind of text except questions.''',
                    
    "subset-superset" : '''Generate {number_of_questions_to_generate} pairs of diverse questions about different topics, in every pair of questions, the first question must be broader than the second i.e. the answer of the second question must be a subset of the answer of the first \
                             and the answer to every questions that you formulate must be a list of values, not an ordered list, not a paragraph of text, not a boolean value and not a single number. \
                             The answer to the questions must be available in Wikidata, and give me also the SPARQL query to retrieve the answer.
                             This is an example of a possible pair of questions: 1. What countries are in the EU? | What countries are in the western EU? | SELECT ?country ?countryLabel WHERE {{{{?country wdt:P463 wd:Q458.  # member of European Union SERVICE wikibase:label {{{{ bd:serviceParam wikibase:language "en". }}}} ORDER BY ?countryLabel | SELECT ?country ?countryLabel WHERE {{{{ ?country wdt:P463 wd:Q458;  wdt:P30 wd:Q46; wdt:P17 ?sovereign. ?country wdt:P276 ?region. ?region wdt:P279* wd:Q27468. SERVICE wikibase:label {{{{ bd:serviceParam wikibase:language "en". }}}}
                             Follow the following format to return the questions: 1. Broader_Question | Subset_Question | SPARQL_query_Broader_Question | SPARQL_query_Subset_Question
                             Do not add any other kind of text except questions''',
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
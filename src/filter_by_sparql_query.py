
import os
import csv
from SPARQLWrapper import SPARQLWrapper, JSON

language = 'en'

endpoint_url = "https://query.wikidata.org/sparql"
output_filename = 'filtered_sparql_equal_syntetic.tsv'

def filter_equvalence(start=0,end=0,tsv_path_to_filter = f'../data/Dataset/{language}/equal-syntetic.tsv',tsv_path_output = f"../data/Dataset/{language}/filtered_sparql_equal_syntetic.tsv"):
    questions = []
    with open(tsv_path_to_filter, newline='', encoding='utf-8') as tsvfile:
        reader = list(csv.DictReader(tsvfile, delimiter='\t'))
        for row in reader[start:end]:
            questions.append((row['ql1'], row['ql2'], row['sparql_ql1']))

    filetered_questions = []
    for index, (q1,q2,sparql_ql1) in enumerate(questions):
        try:
            sparql = SPARQLWrapper(endpoint_url)
            sparql.setQuery(sparql_ql1)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            results_list = []
            for result in results["results"]["bindings"]:
                first_var = list(result.keys())[0]
                results_list.append(result[first_var]["value"])
            if len(results_list) > 0:
                filetered_questions.append((q1, q2, sparql_ql1))
        except Exception as e:
            continue

    file_exists = os.path.exists(tsv_path_output)
    is_empty = not file_exists or os.stat(tsv_path_output).st_size == 0
    with open(tsv_path_output, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        if is_empty:
            writer.writerow(["ql1", "ql2", "sparql_ql1"])

        writer.writerows(filetered_questions)

def filter_subset_superset(tsv_path_to_filter = f'../data/Dataset/{language}/sup-sub_syntetic.tsv',tsv_path_output = f"../data/Dataset/{language}/filtered_sparql_sup-sub_syntetic.tsv"):
    questions = []
    with open(tsv_path_to_filter, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['ql1'], row['ql2'], row['sparql_ql1'], row['sparql_ql2']))

    filetered_questions = []
    for index, (q1,q2,sparql_ql1,sparql_ql2) in enumerate(questions):
        try:
            sparql = SPARQLWrapper(endpoint_url)
            sparql.setQuery(sparql_ql1)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            sparql = SPARQLWrapper(endpoint_url)
            sparql.setQuery(sparql_ql2)
            sparql.setReturnFormat(JSON)
            results2 = sparql.query().convert()
            results_list = []
            results_list2 = []
            for result in results["results"]["bindings"]:
                first_var = list(result.keys())[0]
                results_list.append(result[first_var]["value"])
            for result in results2["results"]["bindings"]:
                first_var = list(result.keys())[0]
                results_list2.append(result[first_var]["value"])
            if len(results_list) > 0 and len(results_list2) > 0:
                filetered_questions.append((q1, q2, sparql_ql1, sparql_ql2))
        except Exception as e:
            continue

    file_exists = os.path.exists(tsv_path_output)
    is_empty = not file_exists or os.stat(tsv_path_output).st_size == 0
    with open(tsv_path_output, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        if is_empty:
            writer.writerow(["ql1", "ql2", "sparql_ql1", "sparql_ql2"])

        writer.writerows(filetered_questions)

def filter_minus(start,end,tsv_path_to_filter = f'../data/Dataset/{language}/minus_syntetic.tsv',tsv_path_output = f"../data/Dataset/{language}/filtered_sparql_minus_syntetic.tsv"):
    questions = []
    with open(tsv_path_to_filter, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append((row['ql1'], row['ql2'], row['ql3'], row['sparql_ql3']))

    filetered_questions = []
    for index, (q1,q2,q3,sparql_ql3) in enumerate(questions):
        try:
            sparql = SPARQLWrapper(endpoint_url)
            sparql.setQuery(sparql_ql3)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            results_list = []
            for result in results["results"]["bindings"]:
                first_var = list(result.keys())[0]
                results_list.append(result[first_var]["value"])
            if len(results_list) > 0:
                filetered_questions.append((q1, q2, q3, sparql_ql3))
        except Exception as e:
            continue

    file_exists = os.path.exists(tsv_path_output)
    is_empty = not file_exists or os.stat(tsv_path_output).st_size == 0
    with open(tsv_path_output, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        if is_empty:
            writer.writerow(["ql1", "ql2", "ql3", "sparql_ql3"])

        writer.writerows(filetered_questions)

#filter_equvalence(start=240,end=801)
filter_subset_superset()
#filter_minus()
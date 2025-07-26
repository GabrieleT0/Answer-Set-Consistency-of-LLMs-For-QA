import csv
import json

def jaccard_similarity(set1, set2):
    """Calculate the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

def add_answers_to_spinach(llm_model,task_type):
    columns_map = {
        'Q1': 'equal',
        'Q2': 'equal',
        'Q3': 'sup-sub',
        'Q4': 'minus'}
    action = ''
    if task_type == 'follow_up_fixing':
        action = '_fixing'
    if task_type == 'rel_classification_and_questions':
        action = '_classAndAnswer'

    # Load all answers
    all_answers = {}
    for key, value in columns_map.items(): 

        answers_path = f"../data/answers/{task_type}/spinach/{value}/{key}_{value}_answers{action}_{llm_model}.json"
        with open(answers_path, "r", encoding="utf-8") as f:
            all_answers[key] = json.load(f) 

    with open('../data/Dataset/en/spinach.tsv', "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        question_rows = list(reader)
        fieldnames = reader.fieldnames + [k + 'Ans' for k in columns_map.keys()]

    # Add answers to each row
    for index, row in enumerate(question_rows):
        for key in columns_map:
            q1_ans_list = all_answers['Q1'].get(str(index), [])
            q2_ans_list = all_answers['Q2'].get(str(index), [])
            q3_ans_list = all_answers['Q3'].get(str(index), [])
            q4_ans_list = all_answers['Q4'].get(str(index), []) 
            row['Q1Ans'] = q1_ans_list if q1_ans_list else ""
            row['Q2Ans'] = q2_ans_list if q2_ans_list else ""
            row['Q3Ans'] = q3_ans_list if q3_ans_list else ""
            row['Q4Ans'] = q4_ans_list if q4_ans_list else ""
           
    # Write final merged TSV
    with open(f"../data/answers/{task_type}/spinach/spinach_answers_{task_type}_{llm_model}.tsv", "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(question_rows)

llm_models = ['gpt-4.1-mini-2025-04-14']
tasks = ['zero-shot']
for llm_model in llm_models:
    for task in tasks:
        print(f"Adding answers for model: {llm_model} and task: {task}")
        add_answers_to_spinach(llm_model, task)
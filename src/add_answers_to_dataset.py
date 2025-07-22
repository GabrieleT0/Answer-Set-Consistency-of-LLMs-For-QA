import csv
import json

def add_answers_to_spinach(llm_model,task_type):
    columns_map = {
        'Q1': 'equal',
        'Q2': 'equal',
        'Q2S': 'sup-sub',
        'Q4': 'minus'}
    action = ''
    if task_type == 'follow_up_fixing':
        action = '_fixing'
    if task_type == 'rel_classification_and_questions':
        action = '_classAndAnswer'

    # Load all answers
    all_answers = {}
    for key, value in columns_map.items(): 
        if key == 'Q2S':
            answers_path = f"../data/answers/{task_type}/spinach/{value}/Q2_{value}_answers{action}_{llm_model}.json"
        else:
            answers_path = f"../data/answers/{task_type}/spinach/{value}/{key}_{value}_answers{action}_{llm_model}.json"
        with open(answers_path, "r", encoding="utf-8") as f:
            if columns_map[key] == 'sup-sub':
                all_answers['Q3'] = json.load(f)
            else:
                all_answers[key] = json.load(f)

    with open('../data/Dataset/en/spinach.tsv', "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        question_rows = list(reader)
        fieldnames = reader.fieldnames + [k + 'Ans' for k in columns_map.keys()]

    # Add answers to each row
    for index, row in enumerate(question_rows):
        for key in columns_map:
            if key == 'Q2S':
                ans_list = all_answers['Q3'].get(str(index), [])
            else:
                ans_list = all_answers[key].get(str(index), [])
                row[key + 'Ans'] = ans_list[0] if ans_list else ""

    # Write final merged TSV
    with open(f"../data/answers/{task_type}/spinach/spinach_answers_{task_type}_{llm_model}.tsv", "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(question_rows)

llm_models = ['gpt-4.1-nano-2025-04-14','gpt-4.1-mini-2025-04-14','gpt-4.1-2025-04-14']
tasks = ['zero-shot', 'follow_up_fixing','rel_classification_and_questions']
for llm_model in llm_models:
    for task in tasks:
        print(f"Adding answers for model: {llm_model} and task: {task}")
        add_answers_to_spinach(llm_model, task)
import os
import re
import csv

def convert_response_to_set(response):
    """
    Convert the response from LLM.
    """
    if response == 'idk':
        return ['idk']
    if response == 'no answer':
        return []
    response = response.split('|')
    response = [item.strip() for item in response if item.strip()]
    
    return response

def convert_response_to_set_es(response):
    """
    Convert the response from LLM.
    """
    if response == 'no sé':
        return []
    response = response.split('|')
    response = [item.strip() for item in response if item.strip()]
    
    return response

def convert_response_to_set_class(response,real_classification):
    """
    Convert the response from LLM.
    """
    if response.lower().strip() != real_classification.lower().strip():
        return 0
    else: 
        return 1
    
def jaccard_similarity(list1, list2):
    """Calculate the Jaccard similarity between two sets."""
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

def is_subset(set1, set2):
    """Check if set1 is a subset of set2."""
    set1 = set(set1)
    set2 = set(set2)
    return set1.issubset(set2)

def is_minus(set1,set2,set3):
    """Check if set1 minus set2 equals set3."""
    set1 = set(set1)
    set2 = set(set2)
    set3 = set(set3)
    return set1.difference(set2) == set3

def get_dataset_path(filename, language='en'):
    root_dir = os.path.dirname(os.path.abspath(__name__))
    return os.path.join(root_dir, f'/data/Dataset/{language}/{filename}')

def parse_questions_for_tsv(text, minus = False, input_q1 = False, input_q2 = False):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    parsed = []

    for line in lines:
        match = re.match(r'^(\d+)\.\s+(.*)', line)
        if not match:
            continue
        questions_str = match.group(2)
        
        if not minus:
            variants = [q.strip() for q in questions_str.split('|') if q.strip()]
        
            q1 = variants[0] if len(variants) > 0 else ""
            q2 = variants[1] if len(variants) > 1 else ""
            sparql_1 = variants[2] if len(variants) > 2 else ""
            sparql_2 = variants[3] if len(variants) > 3 else ""
            parsed.append((q2, q1, sparql_2, sparql_1))
        else:
            questions_str = questions_str.replace('Question C:', '').strip()
            parsed.append((input_q1, input_q2, questions_str))

    return parsed

def save_to_tsv(parsed_data, filename="questions.tsv", minus = False):
    file_exists = os.path.exists(filename)
    is_empty = not file_exists or os.stat(filename).st_size == 0

    with open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        if not minus:
            if is_empty:
                writer.writerow(["ql1", "ql2"])
        else:
            if is_empty:
                writer.writerow(["ql1", "ql2", "ql3"])

        writer.writerows(parsed_data)
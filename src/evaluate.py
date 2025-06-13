import os
import json
import pandas as pd
here = os.path.dirname(os.path.abspath(__file__))

# For the equivalence check
def jaccard_similarity(set1, set2):
    """Calculate the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

# For the subset-superset check
def is_subset(set1, set2):
    """Check if set1 is a subset of set2."""
    return set1.issubset(set2)

# Evaluate the percentace of equvalence of answers
def evaluate_percentage_equivalence(set1, set2):
    """Evaluate the percentage of equivalence between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return (intersection / union) * 100

# Evaluate the minus set equivalence
def evaluate_minus_set_equivalence(a_set, b_set,llm_c_set):
    c_set = a_set - b_set
    jaccard_similarity_value = jaccard_similarity(c_set, llm_c_set)

    return jaccard_similarity_value
    
model = 'gpt-4.1-nano-2025-04-14'
task = 'equal'
action = 'ORIGINAL'
# Evaluate the minus-set
ql1_file = os.path.join(here, f'../data/answers/sup-sub/ql1_subsup_answers_{model}.json')
ql2_file = os.path.join(here, f'../data/answers/sup-sub/ql2_subsup_answers_{model}.json')
ql3_file = os.path.join(here, f'../data/answers/minus/ql3_minus_answers_{model}.json')

with open(ql1_file, 'r', encoding='utf-8') as f:
    ql1_answers = json.load(f)
with open(ql2_file, 'r', encoding='utf-8') as f:
    ql2_answers = json.load(f)
with open(ql3_file, 'r', encoding='utf-8') as f:
    ql3_answers = json.load(f)

minus_set_similarity_scores = {}
minus_set_similarity_scores['info'] = ['minus','jaccard','gpt-4.1-nano-2025-04-14']
minus_set_binary_scores = {}
minus_set_binary_scores['info'] = ['minus','binary equivalence','gpt-4.1-nano-2025-04-14']
for question_id in ql1_answers.keys():
    set_a = set(ql1_answers[question_id])
    set_b = set(ql2_answers.get(question_id, [])) 
    set_c_llm = set(ql3_answers.get(question_id, []))
    similarity = evaluate_minus_set_equivalence(set_a, set_b, set_c_llm)
    minus_set_similarity_scores[question_id] = similarity
    c_set = set_a - set_b
    minus_set_binary_scores[question_id] = int(c_set == set_c_llm)

output_file = os.path.join(here, f'../data/evaluation_results/minus_jaccard_set_results_{model}.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(minus_set_similarity_scores, f, ensure_ascii=False, indent=4)
output_file = os.path.join(here, f'../data/evaluation_results/eq_jaccard_set_results_{model}.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(minus_set_binary_scores, f, ensure_ascii=False, indent=4)

# Evaluate the equivalence of answers
ql1_file = os.path.join(here, f'../data/answers/equal/ql1_equal_answers_{model}.json')
ql2_file = os.path.join(here, f'../data/answers/equal/ql2_equal_answers_{model}.json')
with open(ql1_file, 'r', encoding='utf-8') as f:
    ql1_answers = json.load(f)
with open(ql2_file, 'r', encoding='utf-8') as f:
    ql2_answers = json.load(f)

# Calculate Jaccard similarity for each question and percentage of equivalence
similarity_scores = {}
equivalence_percentage_scores = {}
for question_id in ql1_answers.keys():
    set1 = set(ql1_answers[question_id])
    set2 = set(ql2_answers.get(question_id, [])) 
    similarity = jaccard_similarity(set1, set2)
    is_empty_set = 0
    if len(set1) == 0 and len(set2) == 0:
        is_empty_set = 1
    binary_count = 0
    if set1 == set2:
        binary_count = 1
    similarity_scores[question_id] = (similarity,is_empty_set,binary_count)

sim_df = pd.DataFrame.from_dict(
    similarity_scores, 
    orient='index', 
    columns=['JaccardSimilarity', 'IsEmptySet','binary-count'])
sim_df.to_csv(f"{task}-{model}-{action}.tsv",sep='\t', index=False)

# Evaluate the subset-superset relationship
ql1_file = os.path.join(here, f'../data/answers/sup-sub/ql1_subsup_answers_wikidata_{model}.json')
ql2_file = os.path.join(here, f'../data/answers/sup-sub/ql2_subsup_answers_wikidata_{model}.json')
with open(ql1_file, 'r', encoding='utf-8') as f:
    ql1_answers = json.load(f)
with open(ql2_file, 'r', encoding='utf-8') as f:
    ql2_answers = json.load(f)

# Check subset-superset relationship for each question
subset_superset_results = {}
for question_id in ql1_answers.keys():
    sub_set = set(ql1_answers[question_id])
    sup_set = set(ql2_answers.get(question_id, [])) 
    is_subset_result = sub_set.issubset(sup_set)
    subset_superset_results[question_id] = int(is_subset_result)

# Save the subset-superset results to a JSON file
output_file = os.path.join(here, f'../data/evaluation_results/subset_superset_wikidata_results_{model}.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(subset_superset_results, f, ensure_ascii=False, indent=4)
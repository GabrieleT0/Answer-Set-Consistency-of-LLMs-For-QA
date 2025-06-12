import os
import json
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

model = 'gpt-4.1-nano'

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
    similarity_scores[question_id] = similarity
    equivalence_percentage = evaluate_percentage_equivalence(set1, set2)
    equivalence_percentage_scores[question_id] = equivalence_percentage

# Save the similarity and percentage scores to a JSON file
similarity = os.path.join(here, f'../data/evaluation_results/eq_jaccard_similarity_scores_{model}.json')
percentage = os.path.join(here, f'../data/evaluation_results/eq_percentage_scores_{model}.json')
with open(similarity, 'w', encoding='utf-8') as f:
    json.dump(similarity_scores, f, ensure_ascii=False, indent=4)
with open(percentage, 'w', encoding='utf-8') as f:
    json.dump(equivalence_percentage_scores, f, ensure_ascii=False, indent=4)



# Evaluate the subset-superset relationship
ql1_file = os.path.join(here, f'../data/answers/sup-sub/ql1_subsup_answers_{model}.json')
ql2_file = os.path.join(here, f'../data/answers/sup-sub/ql2_subsup_answers_{model}.json')
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
output_file = os.path.join(here, f'../data/evaluation_results/subset_superset_results_{model}.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(subset_superset_results, f, ensure_ascii=False, indent=4)
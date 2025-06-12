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


# Evaluate the equivalence of answers
ql1_file = os.path.join(here, '../data/answers/equal/ql1_equal_answers_gpt-4.1-nano.json')
ql2_file = os.path.join(here, '../data/answers/equal/ql2_equal_answers_gpt-4.1-nano.json')
with open(ql1_file, 'r', encoding='utf-8') as f:
    ql1_answers = json.load(f)
with open(ql2_file, 'r', encoding='utf-8') as f:
    ql2_answers = json.load(f)

# Calculate Jaccard similarity for each question
similarity_scores = {}
for question_id in ql1_answers.keys():
    set1 = set(ql1_answers[question_id])
    set2 = set(ql2_answers.get(question_id, [])) 
    similarity = jaccard_similarity(set1, set2)
    similarity_scores[question_id] = similarity

# Save the similarity scores to a JSON file
output_file = os.path.join(here, '../data/evaluation_results/eq_jaccard_similarity_scores.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(similarity_scores, f, ensure_ascii=False, indent=4)


# Evaluate the subset-superset relationship
ql1_file = os.path.join(here, '../data/answers/sup-sub/ql1_subsup_answers_gpt-4.1-nano.json')
ql2_file = os.path.join(here, '../data/answers/sup-sub/ql2_subsup_answers_gpt-4.1-nano.json')
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
output_file = os.path.join(here, '../data/evaluation_results/subset_superset_results.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(subset_superset_results, f, ensure_ascii=False, indent=4)
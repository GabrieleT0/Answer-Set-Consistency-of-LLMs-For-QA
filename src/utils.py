import os

def convert_response_to_set(response):
    """
    Convert the response from LLM.
    """
    if response == 'idk':
        return []
    response = response.split(',')
    response = [item.strip() for item in response if item.strip()]
    
    return response

def convert_response_to_set_es(response):
    """
    Convert the response from LLM.
    """
    if response == 'no sé':
        return []
    response = response.split(',')
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
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, f'../data/Dataset/{language}/{filename}')
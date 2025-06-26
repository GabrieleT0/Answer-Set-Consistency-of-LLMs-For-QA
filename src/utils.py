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
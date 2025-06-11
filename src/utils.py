def convert_response_to_set(response):
    """
    Convert the response from watchman query to a set of file paths.
    """
    response = response.split(',')
    response = [item.strip() for item in response if item.strip()]
    
    return response
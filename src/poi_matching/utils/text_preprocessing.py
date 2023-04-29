import re 


def remove_emoji(string):
    """Remove emojis from text

    Args:
        string (str): target string to be transformed

    Returns:
        str: string without emojis
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    
    if string:
        return emoji_pattern.sub(r'', string)
    else:
        return None


import re

def normalize_transcript(text):
    """Whitespace only: no speculative correction of drug names or numbers."""
    if not isinstance(text,str): raise TypeError('Transcript must be a string')
    return re.sub(r'\s+',' ',text).strip()

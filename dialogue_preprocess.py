# dialogue_preprocessing.py
"""
Preprocessing utilities for dialogue datasets.
"""

def preprocess_dialogue(dialogue_list):
    """
    Given a dialogue list (list of dicts with 'role' and 'content'),
    take the first 3 turns, format them as 'role: content',
    and append 'continue the dialogue for few turn'.
    
    Returns:
        str: preprocessed prompt or None if not enough turns
    """
    if not isinstance(dialogue_list, list) or len(dialogue_list) < 3:
        return None
    
    # Take first 3 turns
    turns = dialogue_list[:3]
    
    # Format each turn
    formatted_turns = [f"{turn['role']}: {turn['content']}" for turn in turns]
    
    # Join with newlines and add the continue line
    prompt = "\n".join(formatted_turns) + "\ncontinue the dialogue for few turn"
    
    return prompt
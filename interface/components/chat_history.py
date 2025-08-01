# interface/components/chat_history.py

def append_to_history(history, user_message, bot_response, user_emotion=None):
    """
    Append a new turn to conversation history.
    
    Args:
        history (list of tuples): [(user_msg, bot_response, emotion), ...]
        user_message (str)
        bot_response (str)
        user_emotion (str, optional)
        
    Returns:
        Updated history list.
    """
    if history is None:
        history = []
    history.append((user_message, bot_response, user_emotion))
    return history

def format_history_for_display(history):
    """
    Format chat history to a string suitable for UI display.
    """
    formatted = ""
    for i, (u, b, e) in enumerate(history):
        emotion_text = f" [{e}]" if e else ""
        formatted += f"**User{emotion_text}:** {u}\n\n**Bot:** {b}\n\n"
    return formatted

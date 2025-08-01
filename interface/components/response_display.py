# interface/components/response_display.py

def format_response_for_display(emotion, confidence, response):
    """
    Format a response string with emotion and confidence nicely for markdown rendering.
    """
    return f"**Detected Emotion:** `{emotion}` _(Confidence: {confidence:.2f})_\n\n**Bot:** {response}"

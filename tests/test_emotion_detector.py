# tests/test_emotion_detector.py

import pytest
from core.emotion_detector import EmotionDetector

@pytest.fixture(scope="module")
def emotion_detector():
    # Initialize the EmotionDetector once for all tests in this module
    return EmotionDetector()

@pytest.mark.parametrize("text,expected", [
    ("I am so happy today!", "happy"),
    ("This makes me really sad.", "sad"),
    ("Why did you do that?!", "angry"),
    ("It's just another ordinary day.", "neutral"),
])
def test_predict_emotion_basic(emotion_detector, text, expected):
    """Test that the detector returns the correct emotion for typical inputs."""
    result = emotion_detector.predict(text)
    assert isinstance(result, str), "Prediction should be a string label"
    assert result == expected

@pytest.mark.parametrize("text", [
    "", 
    "    ",
    "\n"
])
def test_predict_empty_or_whitespace(emotion_detector, text):
    """Test that empty or whitespace input defaults to 'neutral' (or your default)."""
    result = emotion_detector.predict(text)
    assert isinstance(result, str)
    assert result == "neutral"  # Change if your default differs

@pytest.mark.parametrize("invalid_input", [None, 123, 3.14, [], {}])
def test_predict_non_string_input(emotion_detector, invalid_input):
    """Test that detector raises error for non-string inputs."""
    with pytest.raises(TypeError):
        emotion_detector.predict(invalid_input)

def test_predict_special_characters(emotion_detector):
    """Test handling of text with emojis or special symbols."""
    result = emotion_detector.predict("I'm so excited!!! 😁🎉")
    assert isinstance(result, str)
    assert result in {"happy", "excited", "surprised"}  # Adjust to your possible outputs

@pytest.mark.parametrize("text", [
    "Wow!",
    "Oh.",
    "Hmm..."
])
def test_predict_short_expressions(emotion_detector, text):
    """Test detector's behavior with very short, non-specific expressions."""
    result = emotion_detector.predict(text)
    assert isinstance(result, str)
    # You might just check it's any known label
    assert result in {"happy", "sad", "angry", "neutral"}


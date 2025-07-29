# core/emotion_detector.py

from transformers import pipeline
from typing import Dict, Any


class EmotionDetector:
    """
    A class for detecting emotions from user text using a pre-trained
    Hugging Face transformer model.
    """

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    ):
        """
        Initialize the Hugging Face pipeline for emotion classification.
        """
        self.emotion_classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True
        )

    def detect_emotion(self, text: str) -> Dict[str, Any]:
        """
        Detect the primary emotion and probability distribution for the given text.

        Args:
            text (str): User input or message.

        Returns:
            dict: {
                'emotion': str (most probable emotion),
                'scores': dict of {emotion_label: score}
            }
        """
        if not text or len(text.strip()) == 0:
            return {"emotion": "neutral", "scores": {}}

        # Run prediction
        output = self.emotion_classifier(text)

        # Convert list of {'label', 'score'} to a dictionary
        label_scores = {item["label"]: item["score"] for item in output[0]}
        top_emotion = max(label_scores, key=label_scores.get)

        return {
            "emotion": top_emotion.lower(),
            "scores": label_scores
        }


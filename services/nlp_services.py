import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Dict, Tuple


class EmotionClassifierService:
    """
    Service for predicting emotion label and confidence from text input
    using a fine-tuned Hugging Face transformer (e.g., MELD).
    """
    def __init__(
        self,
        model_name: str,
        label_map: Optional[Dict[int, str]] = None,
        hf_token: Optional[str] = None,
        local_files_only: bool = True,
    ):
        self.model_name = model_name

        # Load label map from model directory if not provided
        if label_map is None:
            label_path = os.path.join(model_name, "id2label.json")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    label_map = {int(k): v for k, v in json.load(f).items()}
            else:
                try:
                    config_path = os.path.join(model_name, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, "r") as f:
                            config = json.load(f)
                        if "id2label" in config:
                            label_map = {int(k): v for k, v in config["id2label"].items()}
                        else:
                            raise ValueError
                    else:
                        raise ValueError
                except Exception:
                    raise ValueError(
                        "label_map must be provided or id2label.json/config.json with id2label must exist in model path."
                    )
        self.label_map = label_map

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,token=hf_token, local_files_only=local_files_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,token=hf_token, local_files_only=local_files_only
        )

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict the most probable emotion label and confidence for a given text.

        Args:
            text: Input text string.

        Returns:
            Tuple of (predicted_label, confidence_score).
        """
        if not isinstance(text, str) or not text.strip():
            return "neutral", 1.0

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            pred_label_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_label_id].item()

        label = self.label_map.get(pred_label_id, "unknown")
        return label, confidence

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Return a dictionary of {label: probability} for all labels for the input text.

        Args:
            text: Input text string.

        Returns:
            Dictionary mapping emotion labels to predicted probabilities.
        """
        if not isinstance(text, str) or not text.strip():
            if "neutral" in self.label_map.values():
                return {lbl: 1.0 if lbl == "neutral" else 0.0 for lbl in self.label_map.values()}
            return {}

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)
            probs = F.softmax(output.logits, dim=-1).squeeze(0).cpu().tolist()

        return {self.label_map.get(i, "unknown"): prob for i, prob in enumerate(probs)}


if __name__ == "__main__":
    # Example usage

    MODEL_NAME = "../data/models/meld_emotion_model"
    HF_TOKEN = None

    LABEL_MAP = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise"
    }

    classifier = EmotionClassifierService(model_name=MODEL_NAME, label_map=LABEL_MAP, hf_token=HF_TOKEN)
    test_sentence = "I can't believe you just said that. I'm furious!"
    emotion, confidence = classifier.predict(test_sentence)
    print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2f})")

    probs = classifier.predict_proba(test_sentence)
    print("Class probabilities:")
    for emotion_label, prob in probs.items():
        print(f"{emotion_label}: {prob:.3f}")

import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionClassifierService:
    """
    Service for predicting emotion label and confidence from text input
    using a fine-tuned Hugging Face transformer (e.g., MELD).
    """

    def __init__(self, model_name, label_map=None, hf_token=None):
        self.model_name = model_name

        # Load label map from model directory if not provided
        if label_map is None:
            label_path = os.path.join(model_name, "id2label.json")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    label_map = {int(k): v for k, v in json.load(f).items()}
            else:
                # Try loading from the model's config, if present (Hugging Face convention)
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        """
        Predict the most probable emotion label and confidence for a given text.
        """
        if not isinstance(text, str) or not text.strip():
            return "neutral", 1.0

        # Tokenize input text
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            output = self.model(**encoded)
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            pred_label_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_label_id].item()

        label = self.label_map.get(pred_label_id, "unknown")
        return label, confidence

    def predict_proba(self, text):
        """
        Return a dictionary of {label: probability} for all labels for the text.
        """
        if not isinstance(text, str) or not text.strip():
            # Return all 0 except 'neutral' if available, else empty
            if "neutral" in self.label_map.values():
                return {lbl: 1.0 if lbl == "neutral" else 0.0 for lbl in self.label_map.values()}
            return {}

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)
            probs = F.softmax(output.logits, dim=-1).squeeze(0).cpu().tolist()

        return {self.label_map.get(i, "unknown"): prob for i, prob in enumerate(probs)}

# ------------------ For standalone testing ------------------

if __name__ == "__main__":
    # Path to fine-tuned MELD model
    MODEL_NAME = "../data/models/meld_emotion_model"
    HF_TOKEN = None  # Set token if needed

    # MELD label map (use if id2label.json/config.json is missing)
    LABEL_MAP = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise"
    }

    # Initialize classifier
    classifier = EmotionClassifierService(model_name=MODEL_NAME, label_map=LABEL_MAP, hf_token=HF_TOKEN)

    # Test prediction
    test_sentence = "I can't believe you just said that. I'm furious!"
    emotion, confidence = classifier.predict(test_sentence)
    print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2f})")

    # Show probability distribution
    proba = classifier.predict_proba(test_sentence)
    print("Class probabilities:")
    for e, p in proba.items():
        print(f"{e}: {p:.3f}")

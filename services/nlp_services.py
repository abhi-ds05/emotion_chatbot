import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class EmotionClassifierService:
    def __init__(self, model_name, label_map, hf_token=None):
        self.label_map = label_map

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        # Tokenize input
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(**encoded)
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            pred_label_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_label_id].item()

        label = self.label_map.get(pred_label_id, "unknown")
        return label, confidence

# Configuration
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
HF_TOKEN = None  # Optional: os.getenv("HF_TOKEN") or hardcode token if needed

# Label map for 28 classes (from the model card)
LABEL_MAP = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval", 5: "caring",
    6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment", 10: "disapproval",
    11: "disgust", 12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude",
    16: "grief", 17: "joy", 18: "love", 19: "nervousness", 20: "optimism", 21: "pride",
    22: "realization", 23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}

# For testing
if __name__ == "__main__":
    classifier = EmotionClassifierService(model_name=MODEL_NAME, label_map=LABEL_MAP, hf_token=HF_TOKEN)
    test_sentence = "I'm feeling really happy and thankful today!"
    emotion, confidence = classifier.predict(test_sentence)
    print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2f})")

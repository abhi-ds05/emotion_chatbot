# services/nlp_services.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmotionClassifierService:
    """
    Service for detecting emotions from text using a Transformers-based classifier.
    """

    def __init__(self, model_name: str, hf_token: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token).to(self.device)
        self.model.eval()
        self.label_map = self.model.config.id2label  # Auto from model

    def predict(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
        pred_label = torch.argmax(output.logits, dim=1).item()
        return self.label_map[pred_label]

    def predict_proba(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
            probs = torch.softmax(output.logits, dim=1).cpu().squeeze().tolist()
        return {self.label_map[i]: float(probs[i]) for i in range(len(probs))}


# Example usage
if __name__ == "__main__":
    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
    HF_TOKEN = None  # Or os.getenv("HF_TOKEN")

    classifier = EmotionClassifierService(model_name=MODEL_NAME, hf_token=HF_TOKEN)
    test_sentence = "I'm so happy to see you!"
    emotion = classifier.predict(test_sentence)
    proba = classifier.predict_proba(test_sentence)
    print(f"Emotion: {emotion}")
    print(f"Probabilities: {proba}")

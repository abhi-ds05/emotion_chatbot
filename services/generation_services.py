# services/generation_service.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ResponseGeneratorService:
    """
    Generates responses using a large language model (e.g., Mistral).
    """

    def __init__(self, model_name: str, hf_token: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token).to(self.device)
        self.model.eval()

    def generate_response(self, user_message: str, emotion: str) -> str:
        prompt = (
            f"The user seems to be feeling {emotion}.\n"
            f"User: {user_message}\n"
            f"Bot:"
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**encoded, max_new_tokens=100, do_sample=True, temperature=0.7)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded.split("Bot:")[-1].strip()

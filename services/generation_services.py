from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

class ResponseGeneratorService:
    """
    Generates chatbot-style responses using a causal language model like DialoGPT.
    """

    def __init__(self, model_name: str, hf_token: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Detect if it's a local path or remote repo ID
        is_local = os.path.isdir(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            local_files_only=is_local
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            local_files_only=is_local
        ).to(self.device)

        self.model.eval()

    def generate_response(self, user_message: str, emotion: str) -> str:
        """
        Generate a chatbot response conditioned on user message (and optionally emotion).

        Args:
            user_message: Text input from the user.
            emotion: Detected emotion (optional context for future versions).

        Returns:
            Generated response string.
        """
        # Currently emotion isn't injected directly into prompt (DialoGPT doesn't use instruction format)
        prompt = f"{user_message}"
        inputs = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt").to(self.device)

        with torch.no_grad():
            reply_ids = self.model.generate(
                inputs,
                max_new_tokens=100,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
            )

        # Only decode the new tokens
        response = self.tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return response.strip()

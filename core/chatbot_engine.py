from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from dotenv import load_dotenv
import os

class ChatbotEngine:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initializes the chatbot engine using a Hugging Face model and token.
        HF_TOKEN must be present in the .env file.
        """
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")

        if hf_token is None:
            raise ValueError("HF_TOKEN not found in environment variables. Please set it in .env")

        if not torch.cuda.is_available():
            print("⚠️ Warning: CUDA is not available. Running this model on CPU may be very slow.")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            token=hf_token
        )

        # ✅ DON'T manually set `device` if using `device_map="auto"`
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        """
        Generates a response to the given prompt using the language model.
        """
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            **kwargs
        )
        return outputs[0]["generated_text"].replace(prompt, "").strip()

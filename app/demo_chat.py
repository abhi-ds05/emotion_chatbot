# app/demo_chat.py

from services.nlp_services import EmotionClassifierService
from services.generation_services import ResponseGeneratorService

if __name__ == "__main__":
    emotion_model = "j-hartmann/emotion-english-distilroberta-base"
    generator_model = "mistralai/Mistral-7B-Instruct-v0.2"
    HF_TOKEN = None  # Add your token if required

    emotion_service = EmotionClassifierService(model_name=emotion_model, hf_token=HF_TOKEN)
    generator_service = ResponseGeneratorService(model_name=generator_model, hf_token=HF_TOKEN)

    user_input = input("User: ")
    emotion = emotion_service.predict(user_input)
    bot_response = generator_service.generate_response(user_input, emotion)

    print(f"\n[Detected Emotion: {emotion}]")
    print(f"Bot: {bot_response}")

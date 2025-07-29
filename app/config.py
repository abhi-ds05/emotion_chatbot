# app/config.py

from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Emotion-Aware Chatbot"
    LOG_LEVEL: str = "INFO"
    # Add keys for API tokens or database urls if needed
    # For example:
    # HUGGINGFACE_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()

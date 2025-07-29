# app/main.py

from fastapi import FastAPI
from app.routes import router
from dotenv import load_dotenv
import os

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI app.
    Loads environment variables from .env file.
    """
    # Load environment variables
    load_dotenv()

    app = FastAPI(
        title="Emotion-Aware Multimodal Chatbot",
        description="A chatbot capable of detecting emotion from text, audio, and images.",
        version="1.0.0"
    )
    app.include_router(router)
    
    # You can add middleware here if needed
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


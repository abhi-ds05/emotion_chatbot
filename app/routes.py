# app/routes.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import io
import numpy as np
from PIL import Image

from core.response_generator import ResponseGenerator
from core.multimodal_emotion import MultimodalEmotionDetector

router = APIRouter()

# Initialize core components once per app
response_generator = ResponseGenerator()
multimodal_detector = MultimodalEmotionDetector()

# --------- Request / Response Models ---------

class ChatRequest(BaseModel):
    user_id: str
    message: str
    emotion: Optional[str] = None  # Optional: if emotion detected externally

class ChatResponse(BaseModel):
    response: str
    context_emotion: str
    chat_history: List[str]

# --------- Text-only Chat Endpoint ---------

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Basic chat endpoint that accepts user text and optional emotion,
    returns chatbot reply and emotion context.
    """
    try:
        result = response_generator.generate_response(
            user_id=request.user_id,
            user_message=request.message,
            emotion=request.emotion
        )
        return {
            "response": result["response"],
            "context_emotion": result.get("context_emotion", "neutral"),
            "chat_history": result.get("chat_history", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# --------- Multimodal Chat Endpoint ---------

@router.post("/multimodal_chat")
async def multimodal_chat(
    user_id: str = Form(...),
    user_message: str = Form(...),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    """
    Chat endpoint that accepts text, and optionally audio and/or image files,
    runs multimodal emotion detection, and returns the chatbot's response
    augmented with multimodal emotion information.
    """
    try:
        # Process audio file if provided
        audio_path = None
        if audio is not None:
            audio_path = f"/tmp/{audio.filename}"
            # Write uploaded audio to a temp file for processing
            with open(audio_path, "wb") as f:
                f.write(await audio.read())

        # Process image file if provided
        image_array = None
        if image is not None:
            image_bytes = await image.read()
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_array = np.array(image_pil)

        # Run multimodal emotion detection
        emotions = multimodal_detector.detect_multimodal(
            text=user_message,
            image=image_array,
            audio_path=audio_path
        )

        # Generate chatbot response using the fused emotion
        response = response_generator.generate_response(
            user_id=user_id,
            user_message=user_message,
            emotion=emotions["final_emotion"]
        )

        # Optional: clean up temp audio file if you want (not shown here)

        return {
            "response": response["response"],
            "detected_emotions": emotions,
            "context_emotion": response.get("context_emotion", None),
            "chat_history": response.get("chat_history", [])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in multimodal chat: {str(e)}")

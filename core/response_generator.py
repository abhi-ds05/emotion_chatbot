# core/response_generator.py

from typing import Optional, Dict, Any, List
from core.chatbot_engine import ChatbotEngine  # Handles LLM-based response generation
from core.memory_manager import MemoryManager  # Stores session-level chat history
from core.user_state_tracker import UserStateTracker  # Tracks emotion history

class ResponseGenerator:
    def __init__(self):
        self.engine = ChatbotEngine()  # Load a lightweight model inside this engine
        self.memory = MemoryManager()
        self.user_tracker = UserStateTracker()
    
    def generate_response(
        self,
        user_id: str,
        user_message: str,
        emotion: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an LLM-based chatbot response using history and detected emotion.
        """
        # Load past context
        chat_history = self.memory.get_history(user_id)
        user_emotion_history = self.user_tracker.get_emotion_history(user_id)

        # Track emotion
        if emotion:
            self.user_tracker.update_emotion(user_id, emotion)

        # Build conversational prompt
        prompt = self._build_prompt(user_message, chat_history, emotion)

        try:
            # Get response from the LLM engine
            response_text = self.engine.generate(prompt)
        except Exception as e:
            response_text = f"Sorry, I encountered an error generating a response. ({e})"

        # Log the conversation turn
        self.memory.update_history(user_id, {"role": "user", "message": user_message, "emotion": emotion})
        self.memory.update_history(user_id, {"role": "bot", "message": response_text})

        return {
            "response": response_text,
            "context_emotion": emotion or "neutral",
            "chat_history": self.memory.get_history(user_id)
        }

    def _build_prompt(
        self,
        user_message: str,
        chat_history: List[Dict[str, Any]],
        emotion: Optional[str]
    ) -> str:
        """
        Construct the full prompt using memory and emotion context.
        """
        prompt_parts = []

        for turn in chat_history[-10:]:  # Keep last 10 turns
            role = turn.get("role", "user")
            message = turn.get("message", "")
            prompt_parts.append(f"{role}: {message}")

        if emotion:
            prompt_parts.append(f"[User is feeling {emotion}]")

        prompt_parts.append(f"user: {user_message}")
        prompt_parts.append("bot:")

        return "\n".join(prompt_parts)

# core/response_generator.py

from typing import Optional, Dict, Any, List
from core.chatbot_engine import ChatbotEngine  # Your LLM handler
from core.memory_manager import MemoryManager  # Persistent chat history
from core.user_state_tracker import UserStateTracker  # Tracks user emotion state

class ResponseGenerator:
    def __init__(self):
        self.engine = ChatbotEngine()
        self.memory = MemoryManager()
        self.user_tracker = UserStateTracker()
    
    def generate_response(
        self,
        user_id: str,
        user_message: str,
        emotion: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a chatbot response informed by prior memory and user emotion.
        
        Args:
            user_id (str): Unique identifier for the user/session.
            user_message (str): The input message from the user.
            emotion (Optional[str]): The detected emotion of the user.

        Returns:
            Dict[str, Any]: Contains chatbot response text, context emotion, and chat history.
        """

        # Retrieve previous chat history & memory for context
        chat_history = self.memory.get_history(user_id)
        user_emotion_history = self.user_tracker.get_emotion_history(user_id)

        # Update user emotion state tracker
        if emotion:
            self.user_tracker.update_emotion(user_id, emotion)
        
        # Compose prompt/input for the chatbot engine, including memory and emotion context
        prompt = self._build_prompt(user_message, chat_history, emotion)

        # Generate response from LLM/chat engine
        response_text = self.engine.generate(prompt)

        # Update memory with new turn
        self.memory.update_history(user_id, {"role": "user", "message": user_message, "emotion": emotion})
        self.memory.update_history(user_id, {"role": "bot", "message": response_text})

        # Optional: track or integrate emotion into response generation, analytics, etc.
        context_emotion = emotion or "neutral"

        return {
            "response": response_text,
            "context_emotion": context_emotion,
            "chat_history": self.memory.get_history(user_id)
        }
    
    def _build_prompt(
        self,
        user_message: str,
        chat_history: List[Dict[str, Any]],
        emotion: Optional[str]
    ) -> str:
        """
        Create a prompt string for the chatbot engine using conversation history and detected emotion.
        """
        # Example simplistic prompt builder: concatenate recent messages + emotion label
        prompt_parts = []
        for turn in chat_history[-10:]:  # last 10 turns
            role = turn.get("role", "user")
            msg = turn.get("message", "")
            prompt_parts.append(f"{role}: {msg}")

        if emotion:
            prompt_parts.append(f"[User is feeling {emotion}]")

        prompt_parts.append(f"user: {user_message}")
        prompt_parts.append("bot:")

        return "\n".join(prompt_parts)

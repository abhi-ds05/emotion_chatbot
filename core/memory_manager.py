# core/memory_manager.py

import datetime
from typing import Dict, List, Optional


class UserMemory:
    """
    Stores conversation and emotional state for a specific user.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.chat_history: List[Dict[str, str]] = []  # Each item: {"role": "user/bot", "message": "text"}
        self.emotion_history: List[Dict[str, str]] = []  # Each item: {"timestamp": "2025-07-22T...", "emotion": "happy"}

    def add_message(self, role: str, message: str):
        self.chat_history.append({"role": role, "message": message})

    def add_emotion(self, emotion: str):
        self.emotion_history.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "emotion": emotion
        })

    def get_recent_history(self, turns: int = 5) -> List[Dict[str, str]]:
        return self.chat_history[-turns:]

    def get_last_emotion(self) -> Optional[str]:
        return self.emotion_history[-1]["emotion"] if self.emotion_history else None

    def clear(self):
        self.chat_history.clear()
        self.emotion_history.clear()


class MemoryManager:
    """
    Manages memory for multiple users in the chatbot system.
    """

    def __init__(self):
        self.user_sessions: Dict[str, UserMemory] = {}

    def get_user_memory(self, user_id: str) -> UserMemory:
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = UserMemory(user_id)
        return self.user_sessions[user_id]

    def reset_user(self, user_id: str):
        if user_id in self.user_sessions:
            self.user_sessions[user_id].clear()

    def delete_user(self, user_id: str):
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]

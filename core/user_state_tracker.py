# core/user_state_tracker.py

import datetime
from typing import Dict, List, Optional
from collections import Counter
from statistics import mode


class UserState:
    """
    Represents a tracked user's overall state and emotional trends across sessions.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.first_seen = datetime.datetime.utcnow()
        self.last_seen = datetime.datetime.utcnow()
        self.messages: List[str] = []
        self.emotions: List[str] = []
    
    def update_state(self, message: str, emotion: Optional[str] = None):
        """
        Updates the user state with a new message and optional emotion.

        Args:
            message (str): Message from user.
            emotion (str, optional): Detected emotion.
        """
        self.last_seen = datetime.datetime.utcnow()
        self.messages.append(message)
        if emotion:
            self.emotions.append(emotion)
    
    def message_count(self) -> int:
        return len(self.messages)

    def most_common_emotions(self, top_k: int = 3) -> Dict[str, int]:
        """
        Returns the top K most common emotions expressed by the user.

        Returns:
            Dict of emotion labels to their counts.
        """
        emotion_counts = Counter(self.emotions)
        return dict(emotion_counts.most_common(top_k))

    def current_emotion(self) -> Optional[str]:
        """
        Returns the latest recorded emotion.
        """
        return self.emotions[-1] if self.emotions else None

    def dominant_emotion(self) -> Optional[str]:
        """
        Returns the most frequently detected emotion.
        """
        if not self.emotions:
            return None
        try:
            return mode(self.emotions)
        except:
            # Fallback to most common if multimodal
            return Counter(self.emotions).most_common(1)[0][0]

    def time_active(self) -> str:
        delta = self.last_seen - self.first_seen
        return str(delta)


class UserStateTracker:
    """
    Manages state for all users in the system.
    Can be used to personalize behavior across sessions.
    """

    def __init__(self):
        self.users: Dict[str, UserState] = {}

    def get_or_create_user(self, user_id: str) -> UserState:
        if user_id not in self.users:
            self.users[user_id] = UserState(user_id)
        return self.users[user_id]

    def update_user(
        self,
        user_id: str,
        message: str,
        emotion: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Updates the state tracker with a new message and emotion.

        Returns:
            Dict: summary of key stats.
        """
        user = self.get_or_create_user(user_id)
        user.update_state(message, emotion)

        return {
            "user_id": user_id,
            "last_emotion": user.current_emotion() or "unknown",
            "dominant_emotion": user.dominant_emotion() or "n/a",
            "total_messages": str(user.message_count()),
        }

    def get_user_summary(self, user_id: str) -> Optional[Dict[str, str]]:
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]

        return {
            "user_id": user.user_id,
            "total_messages": str(user.message_count()),
            "dominant_emotion": user.dominant_emotion() or "n/a",
            "recent_emotion": user.current_emotion() or "n/a",
            "most_common": str(user.most_common_emotions()),
            "time_active": user.time_active(),
        }

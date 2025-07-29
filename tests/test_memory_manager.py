# tests/test_memory_manager.py

import pytest
from core.memory_manager import MemoryManager, UserMemory

@pytest.fixture(scope="module")
def memory_manager():
    # Initialize MemoryManager once for all tests
    return MemoryManager()

def test_get_user_memory_creates_new(memory_manager):
    """It should create and return a new UserMemory if user_id not known."""
    user_id = "user123"
    user_memory = memory_manager.get_user_memory(user_id)
    assert isinstance(user_memory, UserMemory)
    assert user_memory.user_id == user_id

def test_get_user_memory_returns_existing(memory_manager):
    """It should return existing UserMemory instance for repeated calls."""
    user_id = "user_existing"
    memory1 = memory_manager.get_user_memory(user_id)
    memory2 = memory_manager.get_user_memory(user_id)
    assert memory1 is memory2  # Same instance returned

def test_add_and_retrieve_messages(memory_manager):
    user_id = "user_messages"
    user_mem = memory_manager.get_user_memory(user_id)
    user_mem.clear()
    user_mem.add_message("user", "Hello")
    user_mem.add_message("bot", "Hi there!")
    recent = user_mem.get_recent_history(2)
    assert len(recent) == 2
    assert recent[0]["role"] == "user"
    assert recent[0]["message"] == "Hello"
    assert recent[1]["role"] == "bot"
    assert recent[1]["message"] == "Hi there!"

def test_add_and_get_last_emotion(memory_manager):
    user_id = "user_emotions"
    user_mem = memory_manager.get_user_memory(user_id)
    user_mem.clear()
    assert user_mem.get_last_emotion() is None  # no emotion yet
    user_mem.add_emotion("happy")
    last_emotion = user_mem.get_last_emotion()
    assert last_emotion == "happy"

def test_get_recent_history_limit(memory_manager):
    user_id = "user_history_limit"
    user_mem = memory_manager.get_user_memory(user_id)
    user_mem.clear()
    # Add 10 messages
    for i in range(10):
        user_mem.add_message("user", f"msg{i}")
    # Retrieve last 5
    recent = user_mem.get_recent_history(5)
    assert len(recent) == 5
    assert recent[0]["message"] == "msg5"  # 6th message indexed at 5
    assert recent[-1]["message"] == "msg9"

def test_clear_clears_all(memory_manager):
    user_id = "user_clear_test"
    user_mem = memory_manager.get_user_memory(user_id)
    user_mem.add_message("user", "Hello")
    user_mem.add_emotion("sad")
    user_mem.clear()
    assert user_mem.get_recent_history() == []
    assert user_mem.get_last_emotion() is None

def test_reset_user_clears_memory(memory_manager):
    user_id = "user_reset"
    user_mem = memory_manager.get_user_memory(user_id)
    user_mem.add_message("bot", "Hello")
    user_mem.add_emotion("neutral")
    memory_manager.reset_user(user_id)
    assert user_mem.get_recent_history() == []
    assert user_mem.get_last_emotion() is None

def test_delete_user_removes_user(memory_manager):
    user_id = "user_delete"
    _ = memory_manager.get_user_memory(user_id)
    assert user_id in memory_manager.user_sessions
    memory_manager.delete_user(user_id)
    assert user_id not in memory_manager.user_sessions

@pytest.mark.parametrize("invalid_role, content", [
    (None, "hi"),
    ("", "hello"),
    (123, "test"),
])
def test_add_message_with_invalid_role_raises(memory_manager, invalid_role, content):
    user_id = "user_invalid_role"
    user_mem = memory_manager.get_user_memory(user_id)
    user_mem.clear()
    with pytest.raises(Exception):
        user_mem.add_message(invalid_role, content)

@pytest.mark.parametrize("invalid_emotion", [None, "", 123, [], {}])
def test_add_emotion_invalid_types(memory_manager, invalid_emotion):
    user_id = "user_invalid_emotion"
    user_mem = memory_manager.get_user_memory(user_id)
    user_mem.clear()
    with pytest.raises(Exception):
        user_mem.add_emotion(invalid_emotion)

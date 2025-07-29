import pytest
from core.chatbot_engine import ChatbotEngine

@pytest.fixture(scope="module")
def chatbot_engine():
    return ChatbotEngine()

def test_generate_response(chatbot_engine):
    prompt = "user: I am feeling happy today.\nbot:"
    response = chatbot_engine.generate(prompt)
    assert isinstance(response, str)
    assert response.strip() != ""

def test_generate_empty_prompt(chatbot_engine):
    prompt = ""
    response = chatbot_engine.generate(prompt)
    assert isinstance(response, str)
    assert response.strip() != ""

@pytest.mark.parametrize("invalid_input", [None, 123, 3.14, [], {}, True])
def test_generate_non_string_input(chatbot_engine, invalid_input):
    with pytest.raises(TypeError):
        chatbot_engine.generate(invalid_input)

def test_generate_whitespace_prompt(chatbot_engine):
    prompt = "     "
    response = chatbot_engine.generate(prompt)
    assert isinstance(response, str)

def test_generate_special_characters(chatbot_engine):
    prompt = "user: 😊🤖💬! ¿Qué tal?\nbot:"
    response = chatbot_engine.generate(prompt)
    assert isinstance(response, str)
    assert len(response.strip()) > 0

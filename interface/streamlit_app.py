import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Ensure imports from root-level directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Service Imports ===
from services.nlp_services import EmotionClassifierService
from services.generation_services import ResponseGeneratorService

# === Component Utilities ===
from interface.components.response_display import format_response_for_display
from interface.components.emotion_probability_plot import plot_emotion_probabilities

# === Load environment variables ===
load_dotenv()

# === Configuration ===
EMOTION_MODEL_PATH = os.path.abspath(os.environ.get("EMOTION_MODEL_PATH", "./data/models/meld_emotion_model"))
GENERATION_MODEL_NAME = os.environ.get("GENERATION_MODEL_NAME", "microsoft/DialoGPT-medium")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

LABEL_MAP = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}

# === Model Loaders (with caching) ===
@st.cache_resource(show_spinner="Loading emotion classifier...")
def get_emotion_classifier():
    return EmotionClassifierService(
        model_name=EMOTION_MODEL_PATH,
        label_map=LABEL_MAP,
        hf_token=HF_TOKEN
    )

@st.cache_resource(show_spinner="Loading response generator...")
def get_response_generator():
    return ResponseGeneratorService(
        model_name=GENERATION_MODEL_NAME,
        hf_token=HF_TOKEN
    )

emotion_classifier = get_emotion_classifier()
response_generator = get_response_generator()

# === Streamlit UI ===
st.set_page_config(page_title="Emotion Chatbot", layout="centered")
st.title("🤖 Emotion-Aware Chatbot")
st.markdown("Enter a message and get an empathetic reply based on the detected emotion.")

user_message = st.text_area("Your Message", "", height=100)

if st.button("Send") and user_message.strip():
    with st.spinner("Analyzing..."):
        # Emotion prediction
        emotion, confidence = emotion_classifier.predict(user_message)
        response = response_generator.generate_response(user_message, emotion)
        probabilities = emotion_classifier.predict_proba(user_message)

        # Display response
        st.markdown(format_response_for_display(emotion, confidence, response))

        # Plot using component
        fig = plot_emotion_probabilities(probabilities, return_fig=True)
        st.pyplot(fig)

st.markdown("---")
st.caption("Built with ❤️ using MELD, DialoGPT-medium, Transformers, and Streamlit.")

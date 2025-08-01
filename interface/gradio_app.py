import os
from dotenv import load_dotenv
import gradio as gr

from services.nlp_services import EmotionClassifierService
from services.generation_services import ResponseGeneratorService

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

# === Initialize services ===
emotion_classifier = EmotionClassifierService(
    model_name=EMOTION_MODEL_PATH,
    label_map=LABEL_MAP,
    hf_token=HF_TOKEN,
    local_files_only=True
)

response_generator = ResponseGeneratorService(
    model_name=GENERATION_MODEL_NAME,
    hf_token=HF_TOKEN
)

# === Main chatbot logic ===
def chatbot_interface(user_message):
    if not user_message or not user_message.strip():
        return "Please enter a message.", None

    # Step 1: Detect emotion
    emotion, confidence = emotion_classifier.predict(user_message)

    # Step 2: Generate response
    reply = response_generator.generate_response(user_message, emotion)

    # Step 3: Prepare display
    formatted_output = format_response_for_display(emotion, confidence, reply)
    fig = plot_emotion_probabilities(emotion_classifier.predict_proba(user_message), return_fig=True)

    return formatted_output, fig

# === Gradio Interface ===
demo = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(lines=2, placeholder="Type your message here...", label="You"),
    outputs=[
        gr.Markdown(label="Chatbot Output"),
        gr.Plot(label="Emotion Confidence Distribution")  # Changed to Plot for matplotlib support
    ],
    title="Emotion-Aware Chatbot",
    description="A chatbot that detects emotion in your message and responds empathetically using DialoGPT.",
    theme="default",
)

if __name__ == "__main__":
    demo.launch()

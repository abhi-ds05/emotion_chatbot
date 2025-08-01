# interface/components/input_widgets.py

import gradio as gr

def create_user_input_box():
    """
    Creates a Gradio Textbox with consistent styling to be used in chatbot UI.
    """
    return gr.Textbox(
        lines=2, 
        placeholder="Type your message here...",
        label="You",
        elem_id="user-input-box"
    )

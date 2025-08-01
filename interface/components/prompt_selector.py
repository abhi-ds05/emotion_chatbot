# interface/components/prompt_selector.py

def create_prompt_dropdown(prompt_options, default=None):
    """
    Create a dropdown menu to select prompt templates.

    Args:
        prompt_options (list): list of prompt keys or names.
        default (str): default selected prompt key.

    Returns:
        The selected prompt key.
    """
    import gradio as gr
    if default is None and prompt_options:
        default = prompt_options[0]
    return gr.Dropdown(label="Prompt Template", choices=prompt_options, value=default)

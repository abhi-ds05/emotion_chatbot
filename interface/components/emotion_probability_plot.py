# interface/components/emotion_probability_plot.py

import matplotlib.pyplot as plt

def plot_emotion_probabilities(prob_dict, return_fig=False):
    """
    Create a bar chart from a dictionary of emotion probabilities.

    Args:
        prob_dict (dict): {emotion_label: probability}
        return_fig (bool): if True, return matplotlib Figure object

    Returns:
        None or matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(prob_dict.keys())
    values = list(prob_dict.values())

    ax.bar(labels, values, color='skyblue')
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Emotion Prediction Confidence")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if return_fig:
        return fig
    plt.show()

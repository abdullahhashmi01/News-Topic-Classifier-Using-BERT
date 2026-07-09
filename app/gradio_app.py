"""
Gradio app for live interaction with the fine-tuned News Topic Classifier.

Run with:
    python app/gradio_app.py
"""

import os
import sys

import gradio as gr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.predict import NewsTopicClassifier

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = NewsTopicClassifier(config.MODEL_OUTPUT_DIR)
    return _classifier


def classify(text: str):
    if not text or not text.strip():
        return {}
    clf = get_classifier()
    _, probs = clf.predict(text)
    return probs


examples = [
    "Stocks rally as tech earnings beat expectations",
    "Local team wins championship after dramatic overtime finish",
    "Scientists discover new exoplanet that may support life",
    "United Nations holds emergency summit over trade tensions",
]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        lines=4,
        label="News headline / short article",
        placeholder="Paste a headline or a few sentences here...",
    ),
    outputs=gr.Label(num_top_classes=4, label="Predicted Topic Probabilities"),
    title="📰 News Topic Classifier (Fine-tuned BERT)",
    description=(
        "Fine-tuned bert-base-uncased on the AG News dataset. "
        "Classifies text into World, Sports, Business, or Sci/Tech."
    ),
    examples=examples,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()

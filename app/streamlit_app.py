"""
Streamlit app for live interaction with the fine-tuned News Topic Classifier.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import streamlit as st
import pandas as pd

# Allow importing the `src` package when run as `streamlit run app/streamlit_app.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.predict import NewsTopicClassifier

st.set_page_config(
    page_title="News Topic Classifier (BERT)",
    page_icon="📰",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading fine-tuned BERT model ...")
def load_classifier():
    return NewsTopicClassifier(config.MODEL_OUTPUT_DIR)


st.title("📰 News Topic Classifier")
st.caption(
    "Fine-tuned `bert-base-uncased` on the AG News dataset — "
    "classifies a headline/article into World, Sports, Business, or Sci/Tech."
)

try:
    clf = load_classifier()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        "Could not load a fine-tuned model from "
        f"`{config.MODEL_OUTPUT_DIR}`.\n\n"
        "Train it first with:\n```\npython -m src.train\n```\n\n"
        f"Underlying error: {e}"
    )

text_input = st.text_area(
    "Enter a news headline or short article:",
    placeholder="e.g. NASA's new telescope captures stunning images of a distant galaxy",
    height=120,
)

col1, col2 = st.columns([1, 3])
predict_clicked = col1.button("Classify", type="primary", disabled=not model_loaded)

if predict_clicked:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        label, probs = clf.predict(text_input)
        st.success(f"Predicted Topic: **{label}**")

        df = pd.DataFrame(
            {"Topic": list(probs.keys()), "Probability": list(probs.values())}
        ).sort_values("Probability", ascending=False)

        st.bar_chart(df.set_index("Topic"))
        st.dataframe(df, hide_index=True, use_container_width=True)

st.divider()
with st.expander("ℹ️ About this app"):
    st.write(
        """
        - **Model**: bert-base-uncased, fine-tuned via Hugging Face `Trainer`
        - **Dataset**: AG News (World / Sports / Business / Sci-Tech)
        - **Metrics tracked during training**: accuracy, F1 (macro & weighted)
        - Run `python -m src.train` to (re)train, then `python -m src.evaluate`
          to generate the confusion matrix and metrics report.
        """
    )

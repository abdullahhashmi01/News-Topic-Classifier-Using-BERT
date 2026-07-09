import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/bert-ag-news"
MAX_LEN = 128
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

st.set_page_config(page_title="News Topic Classifier", page_icon="📰", layout="centered")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


def predict(texts, tokenizer, model):
    """Runs inference on a list of texts, returns a list of probability tensors."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    return probs


st.title("📰 News Topic Classifier")
st.caption("BERT fine-tuned on AG News — classifies text into World, Sports, Business, or Sci/Tech")

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(
        f"Could not load the model from `{MODEL_PATH}`. "
        f"Make sure the model files (config.json, model.safetensors, tokenizer.json, "
        f"tokenizer_config.json, vocab.txt) are in that folder.\n\nError: {e}"
    )
    st.stop()

tab_single, tab_batch = st.tabs(["🔤 Single text", "📋 Batch (multiple texts)"])

# ---------------- Single text mode ----------------
with tab_single:
    text = st.text_area(
        "Enter a news headline or article snippet",
        placeholder="e.g. Apple unveils new AI chip that boosts iPhone performance",
        height=100,
    )

    if st.button("Classify", type="primary", key="single_btn"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            probs = predict([text], tokenizer, model)[0]
            pred_idx = int(probs.argmax())
            pred_label = LABEL_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

            st.success(f"**Predicted category: {pred_label}**  ({confidence:.1%} confidence)")

            st.write("Confidence across all categories:")
            for label, p in zip(LABEL_NAMES, probs.tolist()):
                st.write(f"{label}")
                st.progress(p, text=f"{p:.1%}")

# ---------------- Batch mode ----------------
with tab_batch:
    st.write("Paste multiple texts, one per line, or upload a CSV with a `text` column.")

    batch_text = st.text_area(
        "One text per line",
        placeholder="Local team wins championship after dramatic overtime finish\nCentral bank raises interest rates amid inflation concerns",
        height=150,
    )

    uploaded_csv = st.file_uploader("Or upload a CSV file with a 'text' column", type=["csv"])

    if st.button("Classify batch", type="primary", key="batch_btn"):
        texts = []

        if uploaded_csv is not None:
            df_in = pd.read_csv(uploaded_csv)
            if "text" not in df_in.columns:
                st.error("CSV must have a column named 'text'.")
            else:
                texts = df_in["text"].dropna().astype(str).tolist()
        elif batch_text.strip():
            texts = [line.strip() for line in batch_text.split("\n") if line.strip()]

        if not texts:
            st.warning("Please enter some lines of text or upload a CSV.")
        else:
            with st.spinner(f"Classifying {len(texts)} texts..."):
                probs = predict(texts, tokenizer, model)

            rows = []
            for text_item, p in zip(texts, probs):
                pred_idx = int(p.argmax())
                row = {
                    "text": text_item,
                    "predicted_category": LABEL_NAMES[pred_idx],
                    "confidence": float(p[pred_idx]),
                }
                for label, prob in zip(LABEL_NAMES, p.tolist()):
                    row[f"prob_{label}"] = prob
                rows.append(row)

            result_df = pd.DataFrame(rows)
            st.dataframe(result_df, use_container_width=True)

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
            )

st.divider()
st.caption(f"Model loaded from: `{MODEL_PATH}`")

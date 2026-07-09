"""
Lightweight inference wrapper around the fine-tuned model.
Used by both the Streamlit app and the Gradio app for live predictions.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config


class NewsTopicClassifier:
    def __init__(self, model_dir: str = config.MODEL_OUTPUT_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str):
        """Returns (predicted_label, dict of {label: probability})."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

        pred_id = int(probs.argmax())
        pred_label = config.ID2LABEL[pred_id]
        prob_dict = {config.ID2LABEL[i]: float(p) for i, p in enumerate(probs)}
        return pred_label, prob_dict

    @torch.no_grad()
    def predict_batch(self, texts):
        results = []
        for t in texts:
            results.append(self.predict(t))
        return results


if __name__ == "__main__":
    clf = NewsTopicClassifier()
    sample = "Apple unveils new AI chip that boosts iPhone performance"
    label, probs = clf.predict(sample)
    print(f"Text: {sample}")
    print(f"Predicted topic: {label}")
    print(f"Probabilities: {probs}")

"""
Standalone evaluation script: loads a fine-tuned model from disk and
reports accuracy, F1 (macro/weighted), precision, recall, a full
classification report, and a confusion matrix plot.

Usage:
    python -m src.evaluate
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from . import config
from .data_preprocessing import get_tokenized_datasets


def evaluate_model(model_dir: str = config.MODEL_OUTPUT_DIR):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"Loading tokenized test set ...")
    tokenized_datasets, _ = get_tokenized_datasets()
    test_dataset = tokenized_datasets["test"]

    print(f"Loading fine-tuned model from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    args = TrainingArguments(
        output_dir="/tmp/eval_scratch",
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")

    report = classification_report(
        labels, preds, target_names=config.LABEL_NAMES, digits=4
    )

    print("\n===== Evaluation Results =====")
    print(f"Accuracy      : {acc:.4f}")
    print(f"F1 (macro)    : {f1_macro:.4f}")
    print(f"F1 (weighted) : {f1_weighted:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print("\nClassification report:\n", report)

    # Save metrics to JSON
    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision,
        "recall_macro": recall,
    }
    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.LABEL_NAMES,
        yticklabels=config.LABEL_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - AG News Topic Classifier")
    plt.tight_layout()
    cm_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\nSaved confusion matrix to {cm_path}")
    print(f"Saved metrics to {os.path.join(config.RESULTS_DIR, 'metrics.json')}")

    return metrics


if __name__ == "__main__":
    evaluate_model()

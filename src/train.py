"""
Fine-tunes bert-base-uncased on the AG News topic classification dataset
using the Hugging Face Trainer API.

Usage:
    python -m src.train
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from . import config
from .data_preprocessing import get_tokenized_datasets


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }


def main():
    set_seed(config.SEED)

    print(f"Loading & tokenizing dataset ({config.DATASET_NAME}) ...")
    tokenized_datasets, tokenizer = get_tokenized_datasets()

    print(f"Loading base model: {config.MODEL_CHECKPOINT}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_CHECKPOINT,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=config.MODEL_OUTPUT_DIR,
        logging_dir=config.LOGS_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        report_to="none",
        seed=config.SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training ...")
    trainer.train()

    print("Evaluating best model ...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving final model + tokenizer to {config.MODEL_OUTPUT_DIR}")
    trainer.save_model(config.MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(config.MODEL_OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""
Loads the AG News dataset from the Hugging Face Hub, tokenizes it with the
BERT tokenizer, and returns train / test splits ready for the Trainer API.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

# Clean absolute import from the project package root
from src import config


def load_raw_dataset():
    """Downloads (or loads from local cache) the AG News dataset."""
    dataset = load_dataset(config.DATASET_NAME)

    if config.TRAIN_SUBSET_SIZE:
        dataset["train"] = dataset["train"].shuffle(seed=config.SEED).select(
            range(config.TRAIN_SUBSET_SIZE)
        )
    if config.EVAL_SUBSET_SIZE:
        dataset["test"] = dataset["test"].shuffle(seed=config.SEED).select(
            range(config.EVAL_SUBSET_SIZE)
        )
    return dataset


def get_tokenizer():
    return AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)


def tokenize_dataset(dataset, tokenizer):
    """Applies tokenization to every split of the dataset."""

    def _tokenize_fn(batch):
        return tokenizer(
            batch[config.TEXT_COLUMN],
            truncation=True,
            padding="max_length",
            max_length=config.MAX_SEQ_LENGTH,
        )

    tokenized = dataset.map(_tokenize_fn, batched=True)

    # Rename label column to "labels" -> what HF Trainer expects
    tokenized = tokenized.rename_column(config.LABEL_COLUMN, "labels")

    # Keep only the columns the model needs
    keep_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized["train"].column_names:
        keep_cols.append("token_type_ids")

    all_cols = tokenized["train"].column_names
    remove_cols = [c for c in all_cols if c not in keep_cols]
    tokenized = tokenized.remove_columns(remove_cols)

    tokenized.set_format("torch")
    return tokenized


def get_tokenized_datasets():
    """Convenience function: raw dataset -> tokenizer -> tokenized dataset."""
    raw = load_raw_dataset()
    tokenizer = get_tokenizer()
    tokenized = tokenize_dataset(raw, tokenizer)
    return tokenized, tokenizer


if __name__ == "__main__":
    print("⏳ Loading and tokenizing dataset...")
    tokenized, tokenizer = get_tokenized_datasets()
    
    print("\n✅ Dataset Structure:")
    print(tokenized)
    
    print("\n📝 Sample PyTorch Tensors (First Row):")
    sample = tokenized["train"][0]
    for key, val in sample.items():
        # Prints shapes to verify everything is correctly padded and tensorized
        print(f"  {key}: shape={val.shape} type={val.dtype}")
"""
Central configuration file for the News Topic Classifier project.
Change hyperparameters, paths, and dataset settings here in one place.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models", "bert-ag-news")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "ag_news"          # Hugging Face Hub dataset id
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# AG News has 4 topic classes
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_LABELS = len(LABEL_NAMES)

ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}

# ---------------------------------------------------------------------------
# Model / Tokenizer
# ---------------------------------------------------------------------------
MODEL_CHECKPOINT = "bert-base-uncased"
MAX_SEQ_LENGTH = 128

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LOGGING_STEPS = 50

# Set to a small number (e.g. 2000) for a quick smoke test on CPU,
# or None to use the full ~120k training rows / 7.6k test rows.
TRAIN_SUBSET_SIZE = None
EVAL_SUBSET_SIZE = None

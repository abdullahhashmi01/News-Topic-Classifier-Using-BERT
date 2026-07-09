# 📰 News Topic Classifier — Fine-tuned BERT on AG News

End-to-end NLP project: fine-tunes `bert-base-uncased` (Hugging Face Transformers)
on the **AG News** dataset to classify news text into 4 topics — **World, Sports,
Business, Sci/Tech** — evaluated with accuracy & F1, and deployed via **Streamlit**
and **Gradio** for live interaction.

---

## 📁 Project Structure

```
news-topic-classifier/
├── src/
│   ├── config.py               # All hyperparameters, paths, label maps
│   ├── data_preprocessing.py   # Loads AG News + tokenizes with BERT tokenizer
│   ├── train.py                # Fine-tunes bert-base-uncased with Trainer API
│   ├── evaluate.py             # Accuracy, F1, precision/recall, confusion matrix
│   └── predict.py              # Inference wrapper used by the apps
├── app/
│   ├── streamlit_app.py        # Streamlit UI for live predictions
│   └── gradio_app.py           # Gradio UI (alternative) for live predictions
├── notebooks/
│   └── run_on_colab.ipynb      # One-click, GPU-ready notebook (recommended)
├── models/
│   └── bert-ag-news/           # Fine-tuned model gets saved here after training
├── results/                    # metrics.json + confusion_matrix.png land here
├── requirements.txt
└── README.md
```

---

## ⚠️ Important note about where to actually train

Fine-tuning BERT needs: (1) internet access to download the model/dataset from
Hugging Face, and (2) ideally a GPU. If you're running this on a restricted or
offline machine, use **`notebooks/run_on_colab.ipynb`** on
[Google Colab](https://colab.research.google.com) (free GPU, Runtime → Change
runtime type → GPU) — it runs the whole pipeline top to bottom in ~10–15 minutes.

On your own machine/server with internet + GPU, just use the `src/` scripts
directly as described below — both paths produce the same model.

---

## 🚀 Setup (local machine / server)

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Step 1 — Train (fine-tune BERT)

```bash
python -m src.train
```

This will:
- Download AG News from the Hugging Face `datasets` hub (~120k train / ~7.6k test rows)
- Tokenize with `bert-base-uncased`'s tokenizer (max length 128)
- Fine-tune for 3 epochs (`src/config.py` to change epochs, LR, batch size, etc.)
- Evaluate each epoch, keep the best checkpoint by F1-macro (early stopping enabled)
- Save the final model + tokenizer to `models/bert-ag-news/`

**Quick smoke test on CPU:** set `TRAIN_SUBSET_SIZE = 2000` and
`EVAL_SUBSET_SIZE = 500` in `src/config.py` to verify the pipeline runs before
committing to a full run.

---

## 📊 Step 2 — Evaluate

```bash
python -m src.evaluate
```

Outputs:
- Console report: accuracy, F1 (macro & weighted), precision, recall, full
  per-class classification report
- `results/metrics.json` — machine-readable metrics
- `results/confusion_matrix.png` — confusion matrix heatmap

Typical AG News fine-tuned BERT accuracy is **~94–95%** with F1-macro in the
same range, though your exact numbers depend on epochs/seed/hardware.

---

## 🎛️ Step 3 — Deploy for live interaction

### Option A: Streamlit
```bash
streamlit run app/streamlit_app.py
```
Opens a browser UI with a text box → paste a headline → click **Classify** →
see predicted topic + a probability bar chart across all 4 classes.

### Option B: Gradio
```bash
python app/gradio_app.py
```
Opens a Gradio UI (also gives a shareable public link if `share=True` is set
in `demo.launch()`) with the same live-classification experience plus example
headlines to try.

Both apps load the model from `models/bert-ag-news/` (via `src/predict.py`),
so make sure Step 1 has completed and that folder exists before launching.

---

## 🧠 Skills demonstrated

| Area | What's used |
|---|---|
| NLP with Transformers | `AutoTokenizer`, `AutoModelForSequenceClassification` (BERT) |
| Transfer learning & fine-tuning | Hugging Face `Trainer` / `TrainingArguments`, early stopping |
| Evaluation metrics | Accuracy, F1 (macro & weighted), precision, recall, confusion matrix |
| Lightweight deployment | Streamlit app + Gradio app, both backed by the same saved model |

---

## 🔧 Customization

- **Change hyperparameters** → edit `src/config.py` (epochs, batch size, LR, max seq length)
- **Swap the base model** → change `MODEL_CHECKPOINT` in `src/config.py` (e.g. `distilbert-base-uncased`, `roberta-base`)
- **Use your own dataset** → replace `load_raw_dataset()` in `src/data_preprocessing.py`, update `LABEL_NAMES` in `config.py`

---

## 📦 Dataset

**AG News** — 4-class news topic classification dataset, loaded automatically
via `datasets.load_dataset("ag_news")`. Classes: `World`, `Sports`, `Business`,
`Sci/Tech`.

# Sentiment Analysis of Movie Reviews
### Comparing LSTM, LSTM + Attention, and BERT on combined movie review datasets

---

## Overview

This project builds and compares three deep learning models for binary sentiment classification of movie reviews (positive / negative). All models are trained and evaluated on the same combined dataset of ~79,000 short movie reviews.

---

## Dataset

Two publicly available datasets are combined and re-split:

| Dataset | Source | Samples |
|---------|--------|---------|
| Rotten Tomatoes (Cornell MR) | Hugging Face | ~10,660 |
| SST-2 (Stanford Sentiment Treebank) | Hugging Face | ~68,000 |
| **Combined** | | **~78,881** |

Split: **80% train / 10% validation / 10% test**

Both datasets are loaded automatically via the Hugging Face `datasets` library.

---

## Project Structure

```
sentiment_analysis/
├── dataset_prep.py          # data loading, cleaning, tokenising, encoding, padding
├── bert_data.py             # BERT-specific data pipeline (uses HF tokenizer)
├── lstm_baseline.py         # BiLSTM model architecture
├── lstm_attention.py        # BiLSTM + Attention model architecture
├── bert_model.py            # DistilBERT fine-tuning model
├── config.py                # hyperparameters
├── train_lstm.py            # training script for LSTM models
├── train_attention.py       # training script for LSTM + Attention
├── train_bert.py            # training script for BERT
├── evaluate.py              # evaluation script for LSTM
├── evaluate_attention.py    # evaluation script for LSTM + Attention
├── evaluate_bert.py         # evaluation script for BERT
└── glove.6B.100d.txt        # GloVe vectors (download separately, see below)
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install datasets transformers scikit-learn
pip install -r requirements.txt
```



### 2. Download GloVe embeddings (for LSTM models)

Download from https://nlp.stanford.edu/data/glove.6B.zip, unzip, and place `glove.6B.100d.txt` in the project root.

---

## Usage

### Train a model

```
python train_lstm.py         # LSTM with GloVe
python train_attention.py    # LSTM + Attention with GloVe
python train_bert.py         # DistilBERT fine-tuning
```

Each script saves the best checkpoint based on validation accuracy:
- `best_lstm_glove.pt`
- `best_lstm_attention.pt`
- `best_bert.pt`

### Evaluate a model

```
python evaluate.py            # LSTM
python evaluate_attention.py  # LSTM + Attention
python evaluate_bert.py       # BERT
```

---

## Hyperparameters

All hyperparameters are centralised in `config.py`:

```python
EMBEDDING_DIM = 100
HIDDEN_DIM    = 128
NUM_LAYERS    = 2
DROPOUT       = 0.4
LEARNING_RATE = 0.001      # LSTM; BERT uses 2e-5 hardcoded in train_bert.py
EPOCHS        = 20
BATCH_SIZE    = 64        # BERT uses 16 hardcoded in bert_data.py
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- `datasets`, `transformers`, `scikit-learn`

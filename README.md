# Sentiment Analysis of Movie Reviews
### Comparing LSTM, LSTM + Attention, and BERT on combined movie review datasets

---

## Overview

This project builds and compares three deep learning models for binary sentiment classification of movie reviews (positive / negative). It was developed as a university NLP project to gain hands-on experience with sequence modelling, pretrained embeddings, and transformer-based transfer learning.

All models are trained and evaluated on the same combined dataset of ~79,000 short movie reviews.

---

## Results

| Model | Test Accuracy |
|-------|--------------|
| LSTM (random embeddings) | 91.78% |
| LSTM (GloVe embeddings) | 93.06% |
| LSTM + Attention (GloVe, tuned) | 93.35% |
| DistilBERT (fine-tuned) | **94.02%** |

**Key takeaway**: BERT achieves the best accuracy but shows strong overfitting after epoch 2. A well-tuned LSTM with GloVe embeddings offers a compelling trade-off between performance and computational cost — especially on short text like movie reviews (average ~10 tokens).

---

## Dataset

Two publicly available datasets are combined and re-split:

| Dataset | Source | Samples |
|---------|--------|---------|
| Rotten Tomatoes (Cornell MR) | Hugging Face | ~10,660 |
| SST-2 (Stanford Sentiment Treebank) | Hugging Face | ~68,000 |
| **Combined** | | **~78,881** |

Split: **80% train / 10% validation / 10% test** (shuffled with seed 42)

Both datasets are loaded automatically via the Hugging Face `datasets` library — no manual downloads required.

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
```

> For CPU-only: `pip install torch` without the index URL.

### 2. Download GloVe embeddings (for LSTM models)

Download from https://nlp.stanford.edu/data/glove.6B.zip, unzip, and place `glove.6B.100d.txt` in the project root.

> GloVe is optional — the LSTM will fall back to random embeddings if the file is absent.

---

## Usage

### Train a model

```bash
python train_lstm.py         # LSTM with GloVe
python train_attention.py    # LSTM + Attention with GloVe
python train_bert.py         # DistilBERT fine-tuning
```

Each script saves the best checkpoint based on validation accuracy:
- `best_lstm_glove.pt`
- `best_lstm_attention.pt`
- `best_bert.pt`

### Evaluate a model

```bash
python evaluate.py            # LSTM
python evaluate_attention.py  # LSTM + Attention
python evaluate_bert.py       # BERT
```

---

## Model Architectures

### LSTM (Baseline)
- Embedding layer (random or GloVe initialised)
- 2-layer Bidirectional LSTM (hidden dim: 128)
- Dropout (0.3)
- Fully connected layer → Sigmoid output

### LSTM + Attention
- Same as baseline but replaces the final hidden state with an attention-weighted sum of all hidden states
- Learns to focus on sentiment-bearing words (e.g. "terrible", "brilliant")

### DistilBERT
- Pretrained `distilbert-base-uncased` from Hugging Face (66M parameters)
- Fine-tuned with a classification head on top of the `[CLS]` token
- Learning rate: 2e-5 (much lower than LSTM to preserve pretrained weights)
- Best performance reached at epoch 2 — overfits quickly beyond that

---

## Hyperparameters

All hyperparameters are centralised in `config.py`:

```python
EMBEDDING_DIM = 100
HIDDEN_DIM    = 264
NUM_LAYERS    = 3
DROPOUT       = 0.4
LEARNING_RATE = 0.001      # LSTM; BERT uses 2e-5 hardcoded in train_bert.py
EPOCHS        = 20
BATCH_SIZE    = 128        # BERT uses 16 hardcoded in bert_data.py
```

---

## Discussion

- **GloVe vs random embeddings**: +1.28% accuracy — pretrained word vectors provide a meaningful head start
- **Attention on short text**: only +0.29% gain — attention helps most on long sequences; our reviews average just 10 tokens
- **BERT overfitting**: train loss reaches ~0.02 while val loss climbs to ~0.30 after epoch 2 — the model is over-parameterised for this dataset size
- **Overall gap**: only 2.24% separates the simplest and most powerful model, suggesting short opinionated text is well-handled even by simpler architectures

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU recommended (tested on RTX 4060)
- `datasets`, `transformers`, `scikit-learn`

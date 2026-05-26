from datasets import load_dataset
from collections import Counter
import re
import random
from collections import Counter

MAX_VOCAB_SIZE = 25000
MAX_SEQ_LEN = 64

#####################################################################################

#cleaning the text 
def clean_text(text: str) -> str:
    text = text.lower()                            # lowercase
    text = re.sub(r"<[^>]+>", " ", text)           # remove HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text

#####################################################################################

#turning words into tokens
def tokenise(text: str) -> list:
    return text.split()

#####################################################################################

#encoding of the strings
def encode(tokens: list, vocab: dict) -> list:
    return [vocab.get(token, 1) for token in tokens]  # 1 = <UNK>

#####################################################################################

def pad(encoded: list, max_len: int) -> list:
    if len(encoded) >= max_len:
        return encoded[:max_len]        # truncate if too long
    return encoded + [0] * (max_len - len(encoded))  # pad with 0s

#####################################################################################

def prep_data():

    rt   = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    sst2 = load_dataset("stanfordnlp/sst2")

    combined = []
    # Rotten Tomatoes
    for split in ["train", "validation", "test"]:
        for sample in rt[split]:
            combined.append((sample["text"], sample["label"]))

    # SST-2 (only train and validation labeled) 
    for split in ["train", "validation"]:
        for sample in sst2[split]:
            combined.append((sample["sentence"], sample["label"]))

    cleaned = [(clean_text(text), label) for text, label in combined]
    tokenised = [(tokenise(text), label) for text, label in cleaned]
    tokenised = [(tokens, label) for tokens, label in tokenised if len(tokens) > 0]

    random.seed(42)
    random.shuffle(tokenised)

    n = len(tokenised)
    n_test  = int(n * 0.1)
    n_val   = int(n * 0.1)
    n_train = n - n_test - n_val

    train_data = tokenised[:n_train]
    val_data   = tokenised[n_train:n_train + n_val]
    test_data  = tokenised[n_train + n_val:]

    word_counts = Counter(word for tokens, label in train_data for word in tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_counts.most_common(MAX_VOCAB_SIZE - 2):  # -2 for PAD and UNK
        vocab[word] = len(vocab)

    train_encoded = [(encode(tokens, vocab), label) for tokens, label in train_data]
    val_encoded   = [(encode(tokens, vocab), label) for tokens, label in val_data]
    test_encoded  = [(encode(tokens, vocab), label) for tokens, label in test_data]

    train_final = [(pad(encoded, MAX_SEQ_LEN), label) for encoded, label in train_encoded]
    val_final   = [(pad(encoded, MAX_SEQ_LEN), label) for encoded, label in val_encoded]
    test_final  = [(pad(encoded, MAX_SEQ_LEN), label) for encoded, label in test_encoded]

    return train_final, val_final, test_final, vocab



train, val, test, vocab = prep_data()
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"Vocab size: {len(vocab)}")
print(f"Sample: {train[0]}")
    


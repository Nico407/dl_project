import re
import random
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from dataset_prep import clean_text

MAX_SEQ_LEN = 64
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def get_raw_splits():

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
    cleaned = [(text, label) for text, label in cleaned if len(text) > 0]
    random.seed(42)
    random.shuffle(cleaned)

    n = len(cleaned)
    n_test  = int(n * 0.1)
    n_val   = int(n * 0.1)
    n_train = n - n_test - n_val

    train = cleaned[:n_train]
    val   = cleaned[n_train:n_train + n_val]
    test  = cleaned[n_train + n_val:]

    return train, val, test

class BertDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoded = tokenizer(
            text,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.float)
        }
    
def get_dataloaders():
    train, val, test = get_raw_splits()

    train_loader = DataLoader(BertDataset(train), batch_size=16, shuffle=True)
    val_loader   = DataLoader(BertDataset(val),   batch_size=16, shuffle=False)
    test_loader  = DataLoader(BertDataset(test),  batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader
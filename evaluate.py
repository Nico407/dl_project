import torch
import torch.nn as nn
import config
from dataset_prep import prep_data
from lstm_baseline import baseline_lstm
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load data
train, val, test, vocab = prep_data()

# create model and load saved weights
model = baseline_lstm(
    vocab_size=len(vocab),
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim=config.HIDDEN_DIM,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT
)
model = model.to(device)
model.load_state_dict(torch.load("best_lstm_glove.pt", map_location=device))

criterion = nn.BCELoss()

# test data only
test_sequences = torch.tensor([s for s, l in test], dtype=torch.long)
test_labels    = torch.tensor([l for s, l in test], dtype=torch.float)

test_loader = DataLoader(
    TensorDataset(test_sequences, test_labels),
    batch_size=config.BATCH_SIZE,
    shuffle=False
)

# evaluate
model.eval()
total_test_loss = 0
total_test_acc  = 0

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        labels    = labels.to(device)
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        acc  = (torch.round(predictions) == labels).float().mean()
        total_test_loss += loss.item()
        total_test_acc  += acc.item()

avg_test_loss = total_test_loss / len(test_loader)
avg_test_acc  = total_test_acc  / len(test_loader)

print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f}")
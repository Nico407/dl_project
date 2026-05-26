import torch
import torch.nn as nn
from bert_data import get_dataloaders
from bert_model import Bert_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load data
train_loader, val_loader, test_loader = get_dataloaders()

# create model and load saved weights
model = Bert_model(dropout=0.3).to(device)
model.load_state_dict(torch.load("best_bert.pt", map_location=device))

criterion = nn.BCELoss()

# evaluate
model.eval()
total_test_loss = 0
total_test_acc  = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        predictions    = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)
        acc  = (torch.round(predictions) == labels).float().mean()
        total_test_loss += loss.item()
        total_test_acc  += acc.item()

avg_test_loss = total_test_loss / len(test_loader)
avg_test_acc  = total_test_acc  / len(test_loader)

print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f}")
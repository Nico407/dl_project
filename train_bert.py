import torch
import torch.nn as nn
from bert_data import get_dataloaders
from bert_model import Bert_model
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


train_loader, val_loader, test_loader = get_dataloaders()
model = Bert_model(dropout=0.3).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


best_val_acc = 0.0

for epoch in range(config.EPOCHS):
    
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    #Validation
    model.eval()
    total_val_loss = 0
    total_val_acc  = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)
            acc  = (torch.round(predictions) == labels).float().mean()
            total_val_loss += loss.item()
            total_val_acc  += acc.item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc  = total_val_acc  / len(val_loader)

    print(f"Epoch {epoch+1}/{config.EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {avg_val_acc:.4f}")

    # after the print statement, still inside the epoch loop
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        torch.save(model.state_dict(), "best_bert.pt")
        print(f"  → New best model saved!")
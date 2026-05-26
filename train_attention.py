import torch
import torch.nn as nn
import config
from dataset_prep import prep_data
from lstm_attention import attention_lstm
from torch.utils.data import DataLoader, TensorDataset

print(torch.cuda.is_available()) 
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#load data
train, val, test, vocab = prep_data()

#call model
model = attention_lstm(
    vocab_size= len(vocab),
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim = config.HIDDEN_DIM,
    num_layers = config.NUM_LAYERS,
    dropout = config.DROPOUT
)
model = model.to(device)
model.load_glove(vocab)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
 

#convert to tensors
train_sequences = torch.tensor([s for s, l in train], dtype=torch.long)
train_labels    = torch.tensor([l for s, l in train], dtype=torch.float)
test_sequences = torch.tensor([s for s, l in test], dtype=torch.long)
test_labels    = torch.tensor([l for s, l in test], dtype=torch.float)
val_sequences = torch.tensor([s for s, l in val], dtype=torch.long)
val_labels    = torch.tensor([l for s, l in val], dtype=torch.float)

#wrap in DataLoader
train_loader = DataLoader(
    TensorDataset(train_sequences, train_labels),
    batch_size=config.BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    TensorDataset(test_sequences, test_labels),
    batch_size=config.BATCH_SIZE,
    shuffle=False
)
val_loader = DataLoader(
    TensorDataset(val_sequences, val_labels),
    batch_size=config.BATCH_SIZE,
    shuffle=False
)

#####################################################################################
#training loop
best_val_acc = 0.0
for epoch in range(config.EPOCHS):
    
    model.train()
    total_loss = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels    = labels.to(device)
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()        # ← accumulate
    
    avg_train_loss = total_loss / len(train_loader)  # ← average

    #Validation
    model.eval()
    total_val_loss = 0
    total_val_acc  = 0

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels    = labels.to(device)
            predictions = model(sequences)
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
        torch.save(model.state_dict(), "best_lstm_attention_1.pt")
        print(f"  → New best model saved!")

#####################################################################################

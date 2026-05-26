import torch
import torch.nn as nn



class attention_lstm(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 because bidirectional!!!
        self.sigmoid = nn.Sigmoid()
    
    #this function is entierly AI generated in order to add the GloVe emebder
    def load_glove(self, vocab, glove_path="glove.6B.100d.txt"):
        print("Loading GloVe vectors...")
        glove = {}
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                word  = parts[0]
                vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
                glove[word] = vector

        found = 0
        for word, idx in vocab.items():
            if word in glove:
                self.embedding.weight.data[idx] = glove[word]
                found += 1

        print(f"  {found}/{len(vocab)} words initialised from GloVe")

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        scores = self.attention(output)           # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)    # normalise to sum to 1
        context = (weights * output).sum(dim=1)  # weighted sum
        context_cat_dropout = self.dropout(context)
        context_cat_fc = self.fc(context_cat_dropout)  
        context_sigmoid = self.sigmoid(context_cat_fc)
        return torch.squeeze(context_sigmoid, 1)



import torch
import torch.nn as nn



class baseline_lstm(nn.Module):
    
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
        hidden_fwd = hidden[-2, :, :]   #last forward layer
        hidden_bwd = hidden[-1, :, :]   #last backward layer
        hidden_cat = torch.cat((hidden_fwd, hidden_bwd), dim=1) #added both sentiments (back and forward)
        hidden_cat_dropout = self.dropout(hidden_cat)
        hidden_cat_fc = self.fc(hidden_cat_dropout)  
        hidden_sigmoid = self.sigmoid(hidden_cat_fc)
        return torch.squeeze(hidden_sigmoid, 1)

if __name__ == "__main__":
    # create a dummy model
    model = baseline_lstm(
        vocab_size=17669,
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )

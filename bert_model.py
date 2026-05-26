import torch
import torch.nn as nn
from transformers import DistilBertModel

class Bert_model(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        cls_token = output.last_hidden_state [: , 0, :]
        dropout = self.dropout(cls_token)
        fc = self.fc(dropout)
        return torch.squeeze(self.sigmoid(fc), 1)
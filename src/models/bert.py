from typing import List, Dict, Any

from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel
import torch

from utils import accuracy, plot_results

class Bert(nn.Module):
    def __init__(self, num_categorical_features, hidden_size=100, output_size=1, dropout=0.1):
        super(Bert, self).__init__()

        config = AutoConfig.from_pretrained("bert-base-uncased")

        self.bert = AutoModel.from_pretrained("bert-base-uncased", config=config)

        self.linear1 = nn.Linear(config.hidden_size + num_categorical_features, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, attention_mask, categorical_features):
        # Process textual input through GPT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        seq_output = bert_output[0]  # (bs, seq_len, dim)
        # mean pooling, i.e. getting average representation of all tokens
        pooled_output = seq_output.mean(axis=1)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        # Concatenate text features with categorical features
        x = torch.cat((pooled_output, categorical_features), dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

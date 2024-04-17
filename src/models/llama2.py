from typing import List

from torch import nn, Tensor
from torch.utils.data import Dataset
from transformers import AutoModel, BitsAndBytesConfig
import torch


class Llama2Dataset(Dataset):
    def __init__(self, descriptions: list[str], categorical_features: List[List[Tensor]], labels: List[float], tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.categorical_features = categorical_features
        self.labels = labels
        self.descriptions = descriptions

        # Tokenize descriptions here
        self.encodings = tokenizer(descriptions, truncation=True, padding=True, max_length=max_seq_length, return_tensors="pt")

    def __getitem__(self, idx):
        # Retrieve the tokenized text data
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]

        # Retrieve and concatenate the categorical features
        categorical_features = torch.cat(self.categorical_features[idx])

        # Retrieve the label
        label = self.labels[idx]
        description = self.descriptions[idx]
        # Return a dictionary to match expected input format of models and trainers
        return {
            'text': description,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'categorical_features': categorical_features,
            'labels': torch.tensor(label, dtype=torch.float)  # Ensure label is a tensor
        }

    def __len__(self):
        return len(self.labels)


class Llama2(nn.Module):
    def __init__(self, num_categorical_features):
        super(Llama2, self).__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=False,
        )

        device_map = {"": 0}

        self.llama = AutoModel.from_pretrained(
            'NousResearch/Llama-2-7b-chat-hf',
            quantization_config=bnb_config,
            device_map=device_map
            )

        self.llama.config.use_cache = False
        self.llama.config.pretraining_tp = 1

        self.fc1 = nn.Linear(self.llama.config.hidden_size + num_categorical_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)  # Output layer for salary prediction

    def forward(self, input_ids, attention_mask, categorical_features):
        # Process textual input through LLaMA-2
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token's representation

        # Concatenate text features with categorical features
        combined_features = torch.cat((text_features, categorical_features), dim=1)
        combined_features = self.dropout(combined_features)

        # Passing through the additional dense layers
        x = torch.relu(self.fc1(combined_features))
        x = torch.relu(self.fc2(x))
        x = self.output(x)

        return x


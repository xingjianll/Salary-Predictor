from typing import List

from torch import nn, optim, Tensor
from torch.utils.data import Dataset
from transformers import AutoModel
from peft import LoraConfig
import torch

from src.models.utils import plot_results


class Llama2Dataset(Dataset):
    def __init__(self, text_encodings, categorical_features: List[List[Tensor]], labels: List[float]):
        self.text_encodings = text_encodings
        self.categorical_features = categorical_features
        self.labels = labels

    def __getitem__(self, idx):
        one_hot_vectors = self.categorical_features[idx]
        label = self.labels[idx]
        categorical_features = torch.cat(one_hot_vectors)  # Concatenation along the feature dimension
        return categorical_features, label

    def __len__(self):
        return len(self.labels)


class Llama2(nn.Module):
    def __init__(self, num_categorical_features):
        super(Llama2, self).__init__()
        # Load the pre-trained LLaMA-2 model with LoRA configuration
        self.llama = AutoModel.from_pretrained(
            'NousResearch/Llama-2-7b-chat-hf',
            lora_config=LoraConfig(
                r=64,  # Rank of the adaptation matrices
                lora_alpha=16,  # Scaling factor for the adaptation matrices
                lora_dropout=0.1,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
        )

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


def _collate_batch(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    categorical_features = torch.stack([item['categorical_features'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float)
    return input_ids, attention_masks, categorical_features, labels


def train_model(model, train_loader, val_loader, learning_rate=0.01, num_epochs=10, plot_every=50, plot=True):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    iters, losses, train_losses, val_losses = [], [], [], []
    iter_count = 0

    for epoch in range(num_epochs):
        model.train()
        for input_ids, attention_masks, categorical_features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_masks, categorical_features)
            outputs = outputs.squeeze()  # Ensure outputs match the labels' shape
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (iter_count + 1) % plot_every == 0:
                iters.append(iter_count)
                train_losses.append(evaluate(model, train_loader, criterion))
                val_losses.append(evaluate(model, val_loader, criterion))
                print(f"Iteration {iter_count+1}: Train Loss {train_losses[-1]}, Validation Loss {val_losses[-1]}")
            iter_count += 1

    if plot:
        plot_results(iters, train_losses, val_losses)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attention_masks, categorical_features, labels in data_loader:
            outputs = model(input_ids, attention_masks, categorical_features)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)
from typing import List, Dict, Any

from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel
import torch

from utils import accuracy, plot_results


class GPT1Dataset(Dataset):
    def __init__(self, input_ids: list[Tensor],
                 attention_mask: list[Tensor],
                 categorical_features: list[Tensor],
                 labels: list[float]):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.categorical_features = categorical_features
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'categorical_features': self.categorical_features[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)


   
class GPT1(nn.Module):
    def __init__(self, num_categorical_features, hidden_size=100, output_size=1, dropout=0.1, fine_tune=True):
        super(GPT1, self).__init__()

        config = AutoConfig.from_pretrained("openai-gpt")

        self.bert = AutoModel.from_pretrained("openai-gpt", config=config)
        if not fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.linear1 = nn.Linear(config.hidden_size + num_categorical_features, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _collate_batch(batch):
    """Custom collate function for handling batches of data where all input tensors are of the same length."""

    # Separate and stack the data directly since all tensors are already of the same length
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    categorical_features = torch.stack([item['categorical_features'] for item in batch]).float()
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float).to(device)

    return input_ids, attention_mask, categorical_features, labels


def train_model(model,
                train_data: GPT1Dataset,
                val_data: GPT1Dataset,
                learning_rate=0.01,
                batch_size=100,
                eval_batch_size=100,
                num_epochs=10,
                plot_every=50,
                plot=True):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch)
    train_loader2 = DataLoader(train_data, batch_size=eval_batch_size, shuffle=False, collate_fn=_collate_batch)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    iters, losses, train_mae, val_mae = [], [], [], []
    iter_count = 0

    for epoch in range(num_epochs):
        model.train()
        for input_ids, attention_mask, categorical_features, label in train_loader:
            input_ids = input_ids
            attention_mask = attention_mask
            categorical_features = categorical_features
            label = label

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, categorical_features)
            outputs = outputs.squeeze()
            loss = criterion(outputs, label.float())
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            if (iter_count + 1) % plot_every == 0:
                iters.append(iter_count)
                losses.append(float(loss))
                train_mae.append(calculate_mae(model, train_loader2))
                val_mae.append(calculate_mae(model, val_loader))
                print(
                    f"Iter {iter_count + 1}: Loss: {losses[-1]} Train mae {train_mae[-1]}, Validation mae {val_mae[-1]}")
            iter_count += 1

    if plot:
        plot_results(iters, losses, train_mae, val_mae)


def train_classifier(model,
                train_data: GPT1Dataset,
                val_data: GPT1Dataset,
                learning_rate=0.01,
                batch_size=100,
                num_epochs=10,
                plot_every=50,
                plot=True):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch)
    train_loader2 = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    iters, losses, train_mae, val_mae = [], [], [], []
    iter_count = 0

    for epoch in range(num_epochs):
        model.train()
        for input_ids, attention_mask, categorical_features, label in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, categorical_features)
            outputs = outputs.squeeze()
            loss = criterion(outputs, label.type(torch.long))
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            if (iter_count + 1) % plot_every == 0:
                iters.append(iter_count)
                losses.append(float(loss))
                train_mae.append(calculate_accuracy(model, train_loader2))
                val_mae.append(calculate_accuracy(model, val_loader))
                print(
                    f"Iter {iter_count + 1}: Loss: {losses[-1]} Train Acc: {train_mae[-1]}, Validation Acc: {val_mae[-1]}")
            iter_count += 1

    if plot:
        plot_results(iters, losses, train_mae, val_mae)


def calculate_accuracy(model, dataloader: DataLoader) -> float:
    """
    Calculate the accuracy for a model over a given dataloader.

    Args:
        model: The model to evaluate.
        dataloader (DataLoader): The DataLoader containing the dataset.

    Returns:
        float: The accuracy of the model.
    """
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for input_ids, attention_mask, categorical_features, labels in dataloader:
            outputs = model(input_ids, attention_mask, categorical_features)
            outputs = outputs.squeeze()  # Adjust shape if necessary

            predictions = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(predictions == labels).item()
            total_count += labels.size(0)
            if total_count >= 1500:
                break

    return total_correct / total_count


def calculate_mae(model, dataloader: DataLoader) -> float:
    """
    Calculate the mean absolute error for a model over a given dataloader.

    Args:
        model: The model to evaluate.
        dataloader (DataLoader): The DataLoader containing the dataset.

    Returns:
        float: The mean absolute error of the model.
    """
    total_distance = 0
    total_count = 0

    with torch.no_grad():
        for input_ids, attention_mask, categorical_features, labels in dataloader:
            outputs = model(input_ids, attention_mask, categorical_features)
            outputs = outputs.squeeze()  # Adjust shape if necessary

            distances = torch.abs(labels - outputs)
            total_distance += distances.sum().item()
            total_count += labels.size(0)
            if total_count >= 1500:
                break

    return total_distance / total_count
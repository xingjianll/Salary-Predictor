from typing import List, Dict, Any

from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import OpenAIGPTModel
import torch

from utils import plot_results


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
    def __init__(self, num_categorical_features):
        super(GPT1, self).__init__()

        self.gpt = OpenAIGPTModel.from_pretrained("openai-gpt")

        self.fc1 = nn.Linear(self.gpt.config.hidden_size + num_categorical_features, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(100, 1)  # Output layer for salary prediction

    def forward(self, input_ids, attention_mask, categorical_features):
        # Process textual input through GPT
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)

        text_features = outputs.last_hidden_state[:, -1, :]  # Use the last token's representation

        # Concatenate text features with categorical features
        combined_features = torch.cat((text_features, categorical_features), dim=1)
        combined_features = self.fc1(combined_features)

        x = self.relu(combined_features)
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
                num_epochs=10,
                plot_every=50,
                plot=True):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch)

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
                train_mae.append(accuracy(model, train_data))
                val_mae.append(accuracy(model, val_data))
                print(
                    f"Iter {iter_count + 1}: Loss: {losses[-1]} Train mae {train_mae[-1]}, Validation mae {val_mae[-1]}")
            iter_count += 1

    if plot:
        plot_results(iters, losses, train_mae, val_mae)


def accuracy(model, dataset: Dataset) -> float:
    """
    copied from csc413 lab 1
    Compute the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model` - A torch.nn model. We will only be passing `nn.Linear` models.
                  However, to make your code more generally useful, do not access
                  `model.weight` and `model.bias` parameters directly. These
                  class attributes may not exist for other kinds of models.
        `dataset` - A list of 2-tuples of the form (x, t), where `x` is a PyTorch
                  tensor of shape [1, 28, 28] representing an MNIST image,
                  and `t` is the corresponding target label

    Returns: a floating-point value between 0 and 1.
    """
    total = 0
    distance = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for i in range(500):
        data = dataset[i]

        input_ids = data['input_ids'].unsqueeze(0)
        attention_mask = data['attention_mask'].unsqueeze(0)
        categorical_features = data['categorical_features'].unsqueeze(0)
        label = data['labels']

        output = model(input_ids, attention_mask, categorical_features)
        output = output.item()

        distance += float(abs(label-output))
        total += 1

    return distance / total
from typing import List, Any, Tuple

import torch
import torch.nn as nn
from torch import optim, Tensor
from torch.utils.data import DataLoader, Dataset
from src.models.utils import accuracy, plot_results


class MLPDataset(Dataset):
    def __init__(self, categorical_features: List[List[Tensor]], labels: List[float]):
        # Data is a list of tuples, where each tuple is (list of one-hot vectors, label)
        self.categorical_features = categorical_features
        self.labels = labels

    def __getitem__(self, idx: int) -> tuple[Tensor, float]:
        one_hot_vectors = self.categorical_features[idx]
        label = self.labels[idx]
        # Combine the individual feature tensors into a single tensor before passing it to the model
        categorical_features = torch.cat(one_hot_vectors)  # Concatenation along the feature dimension
        return categorical_features, label

    def __len__(self):
        return len(self.categorical_features)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def _collate_batch(batch):
    """Collate batch of data."""
    inputs = torch.stack([item[0] for item in batch]).float()
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float)
    return inputs, labels


def train_model(model,
                train_data: MLPDataset,
                val_data: MLPDataset,
                learning_rate=0.01,
                batch_size=100,
                num_epochs=10,
                plot_every=50,
                plot=True):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    iters, losses, train_losses, val_losses = [], [], [], []
    iter_count = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, label in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, label.float())
            loss.backward()
            optimizer.step()

            if (iter_count + 1) % plot_every == 0:
                iters.append(iter_count)
                train_losses.append(accuracy(model, train_data))
                val_losses.append(accuracy(model, val_data))
                print(f"Iteration {iter_count + 1}: Train Loss {train_losses[-1]}, Validation Loss {val_losses[-1]}")
            iter_count += 1

    if plot:
        plot_results(iters, train_losses, val_losses)

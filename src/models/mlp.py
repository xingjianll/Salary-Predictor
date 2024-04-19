from typing import List

import torch
import torch.nn as nn
from torch import optim, Tensor
from torch.utils.data import DataLoader, Dataset
from utils import accuracy, plot_results, mae


class MLPDataset(Dataset):
    def __init__(self, categorical_features: List[Tensor], labels: List[float]):
        # Data is a list of tuples, where each tuple is (list of one-hot vectors, label)
        self.categorical_features = categorical_features
        self.labels = labels

    def __getitem__(self, idx: int) -> tuple[Tensor, float]:
        categorical_feature = self.categorical_features[idx]
        label = self.labels[idx]
        # Combine the individual feature tensors into a single tensor before passing it to the model
        return categorical_feature, label

    def __len__(self):
        return len(self.categorical_features)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class MLP_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


def _collate_batch(batch):
    """Collate batch of data."""
    inputs = torch.stack([item[0] for item in batch]).float()
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float)
    return inputs, labels


def train_classifier(model,
                     train_data: MLPDataset,
                     val_data: MLPDataset,
                     learning_rate=0.01,
                     batch_size=100,
                     num_epochs=10,
                     plot_every=50,
                     plot=True):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    iters, losses, train_accuracy, val_accuracy = [], [], [], []
    iter_count = 0

    for epoch in range(num_epochs):
        model.train()

        for inputs, label in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label.type(torch.long))
            loss.backward()
            optimizer.step()

            if (iter_count + 1) % plot_every == 0:
                iters.append(iter_count)
                losses.append(float(loss))
                train_accuracy.append(accuracy(model, train_data))
                val_accuracy.append(accuracy(model, val_data))
                print(
                    f"Iter {iter_count + 1}: Loss: {losses[-1]} Train Acc: {train_accuracy[-1]}, Validation Acc: {val_accuracy[-1]}")
            iter_count += 1

    if plot:
        plot_results(iters, losses, train_accuracy, val_accuracy)


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

    iters, losses, train_mae, val_mae = [], [], [], []
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
                losses.append(float(loss))
                train_mae.append(mae(model, train_data))
                val_mae.append(mae(model, val_data))
                print(
                    f"Iter {iter_count + 1}: Loss: {losses[-1]} Train mae {train_mae[-1]}, Validation mae {val_mae[-1]}")
            iter_count += 1

    if plot:
        plot_results(iters, losses, train_mae, val_mae)

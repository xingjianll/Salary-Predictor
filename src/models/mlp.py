import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from src.models.utils import accuracy, plot_results
import matplotlib.pyplot as plt


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


def train_model(model, train_data, val_data, learning_rate=0.01, batch_size=100, num_epochs=10, plot_every=50, plot=True):
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
                print(f"Iteration {iter_count+1}: Train Loss {train_losses[-1]}, Validation Loss {val_losses[-1]}")
            iter_count += 1

    if plot:
        plot_results(iters, train_losses, val_losses)


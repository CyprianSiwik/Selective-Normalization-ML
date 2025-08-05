# train.py — Shared training loop for all models and methods

import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
import os

from evaluate import evaluate_simple as evaluate
from visualize import plot_training_curves
from utils import unpack_batch


def train(model, train_loader, test_loader, epochs=10, lr=0.001, log_file='training_log.csv', plot_dir='plots'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare logging
    os.makedirs(plot_dir, exist_ok=True)
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc (%)', 'Test Acc (%)', 'Epoch Time (s)'])

    train_losses, train_accuracies, test_accuracies, times = [], [], [], []

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for batch in train_loader:
            x, y = unpack_batch(batch)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        end_time = time.time()
        epoch_time = end_time - start_time

        train_acc = 100 * correct / total
        test_acc = evaluate(model, test_loader)
        avg_loss = running_loss / len(train_loader)

        # Log results
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_loss, train_acc, test_acc, epoch_time])

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        times.append(epoch_time)

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, Time={epoch_time:.2f}s")

    # Plot loss and accuracy
    plot_training_curves(train_losses, train_accuracies, test_accuracies, plot_dir)

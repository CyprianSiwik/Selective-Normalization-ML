# train.py — Shared training loop for all models and methods

import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
import os
import sys
import resource

from evaluate import evaluate_detailed
from visualize import plot_training_curves, plot_extra_metrics
from utils import unpack_batch
from models.norm_utils import STUDY_SITE_TYPES

LOG_COLUMNS = [
    'Epoch', 'Train Loss', 'Train Acc (%)', 'Test Loss', 'Test Acc (%)',
    'Epoch Time (s)', 'Inference Time (s/batch)', 'Grad Norm',
    'Activation Mean', 'Activation Std', 'Peak Memory (MB)',
]


def _peak_memory_mb():
    """Peak resident set size so far, in MB. ru_maxrss is KB on Linux, bytes on macOS."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == 'darwin' else peak / 1024


def _find_study_site(model):
    """
    Find the normalization site closest to the classifier head (the last
    matching module in registration order). This is the layer whose
    activation statistics are most relevant to the dropout/normalization
    interaction under study: it's an nn.Identity when norm=None, or the
    actual norm layer (batch/layer/group/selective) otherwise, so activation
    stats are comparable in meaning across all five methods.
    """
    site = None
    for module in model.modules():
        if isinstance(module, STUDY_SITE_TYPES):
            site = module
    return site


def _attach_activation_hook(model):
    stats = {'sum_mean': 0.0, 'sum_std': 0.0, 'count': 0}
    site = _find_study_site(model)

    def hook(module, inputs, output):
        if module.training:
            with torch.no_grad():
                stats['sum_mean'] += output.mean().item()
                stats['sum_std'] += output.std().item()
                stats['count'] += 1

    handle = site.register_forward_hook(hook) if site is not None else None
    return stats, handle


def train(model, train_loader, test_loader, epochs=10, lr=0.001, log_file='training_log.csv', plot_dir='plots'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    activation_stats, hook_handle = _attach_activation_hook(model)

    # Prepare logging
    os.makedirs(plot_dir, exist_ok=True)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(LOG_COLUMNS)

    history = {col: [] for col in LOG_COLUMNS}

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        grad_norm_sum = 0.0
        activation_stats['sum_mean'] = activation_stats['sum_std'] = activation_stats['count'] = 0

        for batch in train_loader:
            x, y = unpack_batch(batch)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            grad_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_sq += p.grad.detach().float().norm(2).item() ** 2
            grad_norm_sum += grad_norm_sq ** 0.5

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        epoch_time = time.time() - start_time

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        avg_grad_norm = grad_norm_sum / len(train_loader)
        avg_activation_mean = activation_stats['sum_mean'] / max(activation_stats['count'], 1)
        avg_activation_std = activation_stats['sum_std'] / max(activation_stats['count'], 1)

        test_metrics = evaluate_detailed(model, test_loader, verbose=False)
        peak_memory = _peak_memory_mb()

        row = [
            epoch, avg_loss, train_acc, test_metrics['loss'], test_metrics['accuracy'],
            epoch_time, test_metrics['inference_time'], avg_grad_norm,
            avg_activation_mean, avg_activation_std, peak_memory,
        ]

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        for col, val in zip(LOG_COLUMNS, row):
            history[col].append(val)

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Test Acc={test_metrics['accuracy']:.2f}%, Grad Norm={avg_grad_norm:.4f}, "
              f"Time={epoch_time:.2f}s")

    if hook_handle is not None:
        hook_handle.remove()

    # Plot loss/accuracy and the extra diagnostic metrics
    plot_training_curves(history['Train Loss'], history['Train Acc (%)'], history['Test Acc (%)'], plot_dir)
    plot_extra_metrics(history['Grad Norm'], history['Activation Std'], plot_dir)

    return history

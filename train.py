# train.py — Shared training loop for all models and methods

import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
import os
import sys
import resource
import numpy as np

from evaluate import evaluate_detailed
from visualize import plot_training_curves, plot_extra_metrics
from utils import unpack_batch
from models.norm_utils import STUDY_SITE_TYPES

LOG_COLUMNS = [
    'Epoch', 'Train Loss', 'Train Acc (%)', 'Test Loss', 'Test Acc (%)',
    'Epoch Time (s)', 'Inference Time (s/batch)', 'Grad Norm', 'Grad Norm Std',
    'Activation Mean', 'Activation Std', 'Eval Activation Mean', 'Eval Activation Std',
    'Peak Memory (MB)',
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
    """
    Track running mean/std of the study-site activations, separately for
    training-mode and eval-mode forward passes, and keep a raw sample of
    its training-mode output from the most recent batch — used to plot
    activation-distribution histograms (see plot_activation_histograms.py)
    that make the dropout/normalization distortion the README describes
    directly visible, rather than only summarized as a scalar.

    The train/eval split also exposes a mismatch specific to selective
    normalization: its running stats are accumulated only from
    dropout-surviving activations during training, but eval-mode forward
    passes see every activation (dropout is off), so the two can diverge
    more than for the other methods.
    """
    stats = {
        'train_sum_mean': 0.0, 'train_sum_std': 0.0, 'train_count': 0,
        'eval_sum_mean': 0.0, 'eval_sum_std': 0.0, 'eval_count': 0,
        'last_raw': None,
    }
    site = _find_study_site(model)

    def hook(module, inputs, output):
        with torch.no_grad():
            if module.training:
                stats['train_sum_mean'] += output.mean().item()
                stats['train_sum_std'] += output.std().item()
                stats['train_count'] += 1
                stats['last_raw'] = output.detach().cpu().numpy().ravel()
            else:
                stats['eval_sum_mean'] += output.mean().item()
                stats['eval_sum_std'] += output.std().item()
                stats['eval_count'] += 1

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
        grad_norms = []
        activation_stats['train_sum_mean'] = activation_stats['train_sum_std'] = activation_stats['train_count'] = 0

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
            grad_norms.append(grad_norm_sq ** 0.5)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        epoch_time = time.time() - start_time

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        # Mean gradient norm (convergence proxy) and its batch-to-batch std
        # within the epoch (training-stability proxy: a method that's
        # equally accurate but swings harder between batches is still less
        # stable).
        avg_grad_norm = float(np.mean(grad_norms))
        grad_norm_std = float(np.std(grad_norms))
        avg_activation_mean = activation_stats['train_sum_mean'] / max(activation_stats['train_count'], 1)
        avg_activation_std = activation_stats['train_sum_std'] / max(activation_stats['train_count'], 1)

        activation_stats['eval_sum_mean'] = activation_stats['eval_sum_std'] = activation_stats['eval_count'] = 0
        test_metrics = evaluate_detailed(model, test_loader, verbose=False)
        avg_eval_activation_mean = activation_stats['eval_sum_mean'] / max(activation_stats['eval_count'], 1)
        avg_eval_activation_std = activation_stats['eval_sum_std'] / max(activation_stats['eval_count'], 1)
        peak_memory = _peak_memory_mb()

        row = [
            epoch, avg_loss, train_acc, test_metrics['loss'], test_metrics['accuracy'],
            epoch_time, test_metrics['inference_time'], avg_grad_norm, grad_norm_std,
            avg_activation_mean, avg_activation_std, avg_eval_activation_mean, avg_eval_activation_std,
            peak_memory,
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

    # Save a raw activation sample from the final training batch, for
    # cross-method distribution comparisons (plot_activation_histograms.py).
    if activation_stats['last_raw'] is not None:
        log_dir = os.path.dirname(log_file) or '.'
        np.save(os.path.join(log_dir, 'activation_sample.npy'), activation_stats['last_raw'])

    # Plot loss/accuracy and the extra diagnostic metrics
    plot_training_curves(history['Train Loss'], history['Train Acc (%)'], history['Test Acc (%)'], plot_dir)
    plot_extra_metrics(history['Grad Norm'], history['Activation Std'], plot_dir)

    return history

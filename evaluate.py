# evaluate.py — Lightweight + detailed evaluation functions

import torch
import torch.nn.functional as F
import time

from utils import unpack_batch


def evaluate_simple(model, data_loader):
    """Quick evaluation used inside training loop (returns accuracy only)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            x, y = unpack_batch(batch)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total


def evaluate_detailed(model, data_loader):
    """Full evaluation: accuracy, loss, and inference time."""
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    inference_times = []

    with torch.no_grad():
        for batch in data_loader:
            x, y = unpack_batch(batch)

            start = time.time()
            outputs = model(x)
            end = time.time()

            loss = F.cross_entropy(outputs, y)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            inference_times.append(end - start)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    avg_time = sum(inference_times) / len(inference_times)

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Avg Inference Time per Batch: {avg_time:.4f}s")

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'inference_time': avg_time
    }

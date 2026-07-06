# Visualization utilities for results and analysis

import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(train_losses, train_accuracies, test_accuracies, plot_dir):
    """
    Plot and save training loss and accuracy curves.

    Args:
        train_losses (list of float): Training loss per epoch.
        train_accuracies (list of float): Training accuracy per epoch.
        test_accuracies (list of float): Test accuracy per epoch.
        plot_dir (str): Directory to save plots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    # Plot Loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'loss.png'))
    plt.close()

    # Plot Accuracy
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'accuracy.png'))
    plt.close()


def plot_extra_metrics(grad_norms, activation_stds, plot_dir):
    """
    Plot gradient norm and activation std per epoch — diagnostics for
    training stability and the dropout/normalization interaction under study.

    Args:
        grad_norms (list of float): Mean gradient L2 norm per epoch.
        activation_stds (list of float): Mean activation std (at the norm
            study site) per epoch.
        plot_dir (str): Directory to save plots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure()
    plt.plot(grad_norms, label='Grad Norm')
    plt.title('Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'grad_norm.png'))
    plt.close()

    plt.figure()
    plt.plot(activation_stds, label='Activation Std')
    plt.title('Activation Std at Study Site')
    plt.xlabel('Epoch')
    plt.ylabel('Std')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'activation_std.png'))
    plt.close()


def load_log_and_plot(log_file, plot_dir='plots'):
    """
    Load a CSV log file and plot training curves.

    Args:
        log_file (str): Path to CSV log file with columns:
                        'Epoch', 'Train Loss', 'Train Acc (%)', 'Test Acc (%)', ...
        plot_dir (str): Directory to save plots.
    """
    df = pd.read_csv(log_file)

    train_losses = df['Train Loss'].tolist()
    train_accuracies = df['Train Acc (%)'].tolist()
    test_accuracies = df['Test Acc (%)'].tolist()

    plot_training_curves(train_losses, train_accuracies, test_accuracies, plot_dir)

# Your proposed selective normalization method
# selective_norm.py — Dropout + Selective Normalization (norm only on active neurons)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn import CNN
from models.mlp import MLP
from models.rnn import RNNClassifier
from models.light_cnn import LightCNN
from models.light_mlp import LightMLP
from models.light_rnn import LightRNN

from data import cifar
from data import mnist
from data import imdb
from data import uci_adult

from train import train


class SelectiveNormalization(nn.Module):
    """
    Selective Normalization Layer - normalizes only active (non-dropped) neurons.

    This is the core innovation: instead of normalizing all neurons and then applying
    dropout, we apply dropout first and then normalize only the remaining active neurons.
    """

    def __init__(self, num_features, dropout_rate=0.5, norm_type='batch', eps=1e-5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type
        self.eps = eps

        # Standard normalization parameters (will be applied selectively)
        if norm_type == 'batch':
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
            # Running stats for batch norm (used during inference)
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.momentum = 0.1
        elif norm_type == 'layer':
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x):
        if not self.training:
            # During inference, use standard normalization (no dropout)
            if self.norm_type == 'batch':
                return F.batch_norm(x, self.running_mean, self.running_var,
                                    self.weight, self.bias, False, 0.0, self.eps)
            elif self.norm_type == 'layer':
                return F.layer_norm(x, x.shape[1:], self.weight, self.bias, self.eps)

        # TRAINING: Apply selective normalization
        # Step 1: Generate dropout mask
        mask = torch.bernoulli((1 - self.dropout_rate) * torch.ones_like(x))

        # Step 2: Apply dropout
        x_dropped = x * mask

        # Step 3: Normalize only active neurons
        if self.norm_type == 'batch':
            # Find active neurons (non-zero after dropout)
            active_mask = (mask == 1)

            if active_mask.any():
                # Calculate mean and var only from active neurons
                active_values = x_dropped[active_mask]
                if len(active_values) > 1:  # Need at least 2 values for variance
                    mean_active = active_values.mean()
                    var_active = active_values.var(unbiased=False)

                    # Update running statistics with active neurons only
                    with torch.no_grad():
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_active
                        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_active

                    # Apply normalization only to active neurons
                    normalized_active = (active_values - mean_active) / torch.sqrt(var_active + self.eps)

                    # Reconstruct the tensor
                    result = torch.zeros_like(x_dropped)
                    result[active_mask] = normalized_active * self.weight.expand_as(
                        normalized_active) + self.bias.expand_as(normalized_active)
                    return result

            # Fallback: if not enough active neurons, return dropped input
            return x_dropped

        elif self.norm_type == 'layer':
            # For layer norm, work with active dimensions
            active_mask = (mask == 1)
            if active_mask.any():
                # Calculate statistics only from active neurons
                active_values = x_dropped[active_mask]
                if len(active_values) > 1:
                    mean_active = active_values.mean()
                    var_active = active_values.var(unbiased=False)

                    # Apply normalization to active neurons
                    normalized_active = (active_values - mean_active) / torch.sqrt(var_active + self.eps)

                    # Reconstruct
                    result = torch.zeros_like(x_dropped)
                    result[active_mask] = normalized_active * self.weight.expand_as(
                        normalized_active) + self.bias.expand_as(normalized_active)
                    return result

            return x_dropped


class SelectiveNormWrapper(nn.Module):
    """
    Wrapper to add selective normalization to existing models.
    """

    def __init__(self, base_model, dropout_rate=0.5, norm_type='batch'):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type

        # Replace existing dropout and add selective norm layers
        self._replace_dropout_with_selective_norm()

    def _replace_dropout_with_selective_norm(self):
        """
        Replace dropout layers in the base model with selective normalization.
        """
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Dropout):
                # Get the input size for the normalization layer
                # This is a simplified approach - you might need to adjust based on your model structure
                parent_modules = name.split('.')[:-1]
                parent = self.base_model
                for parent_name in parent_modules:
                    parent = getattr(parent, parent_name)

                # Replace dropout with selective normalization
                # Note: This is a simplified replacement - you may need to adjust the num_features
                # based on the specific layer and model architecture
                setattr(parent, name.split('.')[-1],
                        SelectiveNormalization(num_features=128, dropout_rate=self.dropout_rate,
                                               norm_type=self.norm_type))

    def forward(self, x):
        return self.base_model(x)


def run_selective_norm(model_type='cnn', dataset='cifar10', dropout_rate=0.3, normalization='selective',
                       lightweight=False):
    """
    Run training with dropout + selective normalization (norm only on active neurons).

    Args:
        model_type (str): 'cnn', 'mlp', or 'rnn'
        dataset (str): dataset name
        dropout_rate (float): dropout probability
        normalization (str): should be 'selective' for this method
        lightweight (bool): Use lightweight model variants if True
    """
    input_channels = None
    num_classes = None
    input_size = None
    vocab_size = 0

    # === Dataset loading ===
    if dataset == 'cifar10':
        train_loader = cifar.get_cifar_dataset('cifar10', train=True)
        test_loader = cifar.get_cifar_dataset('cifar10', train=False)
        input_channels = 3
        num_classes = 10

    elif dataset == 'cifar100':
        train_loader = cifar.get_cifar_dataset('cifar100', train=True)
        test_loader = cifar.get_cifar_dataset('cifar100', train=False)
        input_channels = 3
        num_classes = 100

    elif dataset == 'mnist':
        train_loader = mnist.get_mnist_dataset(train=True)
        test_loader = mnist.get_mnist_dataset(train=False)
        input_size = 784
        num_classes = 10

    elif dataset == 'imdb':
        train_loader, test_loader, vocab = imdb.get_imdb_dataset()
        vocab_size = len(vocab)
        num_classes = 2

    elif dataset == 'uci_adult':
        train_loader, test_loader, _, _ = uci_adult.get_adult_dataloaders()
        input_size = 105
        num_classes = 2

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # === Model Selection with Selective Normalization ===
    if model_type == 'cnn':
        if lightweight:
            base_model = LightCNN(input_channels=input_channels, num_classes=num_classes,
                                  dropout=0.0)  # No dropout in base
        else:
            base_model = CNN(input_channels=input_channels, num_classes=num_classes, dropout=0.0)  # No dropout in base

        # Wrap with selective normalization
        model = SelectiveNormWrapper(base_model, dropout_rate=dropout_rate, norm_type='batch')

    elif model_type == 'mlp':
        if lightweight:
            base_model = LightMLP(input_size=input_size, hidden_sizes=[128, 64], num_classes=num_classes, dropout=0.0)
        else:
            base_model = MLP(input_size=input_size, hidden_sizes=[512, 256], num_classes=num_classes, dropout=0.0)

        # Wrap with selective normalization
        model = SelectiveNormWrapper(base_model, dropout_rate=dropout_rate, norm_type='batch')

    elif model_type == 'rnn':
        if lightweight:
            base_model = LightRNN(
                vocab_size=vocab_size,
                embed_dim=64,
                hidden_dim=64,
                num_classes=num_classes,
                dropout=0.0,  # No dropout in base
                bidirectional=False
            )
        else:
            base_model = RNNClassifier(
                vocab_size=vocab_size,
                embed_dim=128,
                hidden_dim=256,
                num_classes=num_classes,
                dropout=0.0,  # No dropout in base
                bidirectional=False
            )

        # Wrap with selective normalization
        model = SelectiveNormWrapper(base_model, dropout_rate=dropout_rate, norm_type='layer')

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # === Train ===
    train(model, train_loader, test_loader, epochs=10)


if __name__ == '__main__':
    # Test standard models with selective normalization
    run_selective_norm(model_type='cnn', dataset='cifar10', dropout_rate=0.3, normalization='selective',
                       lightweight=False)

    # Test lightweight models with selective normalization
    run_selective_norm(model_type='mlp', dataset='mnist', dropout_rate=0.5, normalization='selective', lightweight=True)
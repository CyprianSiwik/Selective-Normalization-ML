# Lightweight MLP model definition
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64], num_classes=10, dropout=0.5):
        """
        Lightweight MLP model with smaller hidden layers.

        Args:
            input_size (int): Number of input features (e.g. 28*28=784 for MNIST).
            hidden_sizes (list): List of hidden layer sizes. Default is smaller than standard MLP.
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability.
        """
        super(LightMLP, self).__init__()

        self.layers = nn.ModuleList()
        in_dim = input_size

        for hidden_dim in hidden_sizes:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.output_layer = nn.Linear(in_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten if necessary

        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        return self.output_layer(x)